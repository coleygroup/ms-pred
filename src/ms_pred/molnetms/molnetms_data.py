import logging
import json
import numpy as np

import torch
from rdkit import Chem
from torch.utils.data.dataset import Dataset
import dgl

import ms_pred.common as common


class MolMSFeaturizer:
    """Create a 3D mol featurizer"""

    # Hardcoded
    char_to_vec = {i: j.tolist() for i, j in common.element_to_position.items()}
    num_atom_feats = 3 + char_to_vec["C"].__len__()
    num_atom_feats_full = 7 + num_atom_feats

    def __init__(self, num_points: int = 300):

        self.valid_letters = set(self.char_to_vec.keys())

        # Max atoms
        self.num_points = num_points

    @classmethod
    def atom_feats(cls):
        return cls.num_atom_feats_full

    def parse_mol_block(self, mol_block):
        """Parse the mol block to get the atom points and bonds

        Taken from 3dmolms code

        Args:
            mol_block (list): the lines of mol block

        Returns:
            points (list): the atom points, (npoints, num feats)
            bonds (list): the atom bonds, (npoints, 4)
        """
        points = []
        bonds = []
        for d in mol_block:
            if len(d) == 69:  # the format of molecular block is fixed
                atom = [i for i in d.split()]
                # atom: [x, y, z, atom_type, charge, stereo_care_box, valence]
                # sdf format (atom block): https://docs.chemaxon.com/display/docs/mdl-molfiles-rgfiles-sdfiles-rxnfiles-rdfiles-formats.md

                # TODO: Replace with ENCODE ATOM
                if len(atom) == 16 and atom[3] in self.valid_letters:
                    # only x-y-z coordinates
                    # point = [float(atom[0]), float(atom[1]), float(atom[2])]

                    # x-y-z coordinates and atom type
                    point = [
                        float(atom[0]),
                        float(atom[1]),
                        float(atom[2]),
                    ] + self.char_to_vec[atom[3]]
                    points.append(point)
                elif len(atom) == 16:  # check the atom type
                    raise ValueError(
                        f"Error: {atom[3]} is not in {self.valid_letters}, please check the dataset."
                    )

            elif len(d) == 12:
                bond = [int(i) for i in d.split()]
                if len(bond) == 4:
                    bonds.append(bond)

        points = np.array(points)
        assert points.shape[1] == self.num_atom_feats

        # center the points
        points_xyz = points[:, :3]
        centroid = np.mean(points_xyz, axis=0)
        points_xyz -= centroid

        points = np.concatenate((points_xyz, points[:, 3:]), axis=1)

        return points.tolist(), bonds

    def get_3d_graph(self, mol):
        """Get the 3D graph of the molecule.

        Args:
            smiles (str): the smiles of the molecule
        """
        mol_block = Chem.MolToMolBlock(mol).split("\n")

        # 1. x,y,z-coordinates; 2. atom type (one-hot);
        point_set, bonds = self.parse_mol_block(mol_block)
        for idx, atom in enumerate(mol.GetAtoms()):
            # 3. number of immediate neighbors who are “heavy” (nonhydrogen) atoms;
            point_set[idx].append(atom.GetDegree())
            # 4. valence minus the number of hydrogens;
            point_set[idx].append(atom.GetExplicitValence())
            point_set[idx].append(atom.GetMass() / 100)  # 5. atomic mass;
            # 6. atomic charge;
            point_set[idx].append(atom.GetFormalCharge())
            # 7. number of implicit hydrogens;
            point_set[idx].append(atom.GetNumImplicitHs())
            point_set[idx].append(int(atom.GetIsAromatic()))  # 8. is aromatic;
            point_set[idx].append(int(atom.IsInRing()))  # 9. is in a ring;

        point_set = np.array(point_set).astype(np.float32)

        # generate mask
        point_mask = np.ones_like(point_set[0])

        point_set = torch.cat(
            (
                torch.Tensor(point_set),
                torch.zeros((self.num_points - point_set.shape[0], point_set.shape[1])),
            ),
            dim=0,
        )
        point_mask = torch.cat(
            (
                torch.Tensor(point_mask),
                torch.zeros((self.num_points - point_mask.shape[0])),
            ),
            dim=0,
        )
        assert point_set.shape[1] == self.num_atom_feats_full

        # 3D graph has 3 xyz poositions followed by atom features (1 hot and 7 others)
        return point_set, point_mask

    @staticmethod
    def collate_3d(batch):
        raise NotImplementedError()


class BinnedDataset(Dataset):
    """SmiDataset."""

    def __init__(
        self,
        df,
        data_dir,
        num_bins,
        graph_featurizer,
        num_workers=0,
        upper_limit=1500,
        form_dir_name: str = "subform_20",
        use_ray=False,
        **kwargs,
    ):
        self.df = df
        self.num_bins = num_bins
        self.num_workers = num_workers
        self.upper_limit = upper_limit
        self.bins = np.linspace(0, self.upper_limit, self.num_bins)
        self.name_to_adduct = dict(self.df[["spec", "ionization"]].values)

        # Read in all molecules
        self.smiles = self.df["smiles"].values

        self.graph_featurizer = graph_featurizer

        if self.num_workers == 0:
            self.mols = [Chem.MolFromSmiles(i) for i in self.smiles]
            self.weights = [common.ExactMolWt(i) for i in self.mols]
            self.mol_graphs = [self.graph_featurizer.get_3d_graph(i) for i in self.mols]
        else:

            def mol_from_smi(x):
                return Chem.MolFromSmiles(x)

            self.mols = common.chunked_parallel(
                self.smiles,
                mol_from_smi,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=600,
                max_retries=3,
                use_ray=use_ray,
            )
            self.weights = common.chunked_parallel(
                self.mols,
                lambda x: common.ExactMolWt(x),
                chunks=100,
                max_cpu=self.num_workers,
                timeout=600,
                max_retries=3,
                use_ray=use_ray,
            )
            self.mol_graphs = common.chunked_parallel(
                self.mols,
                self.graph_featurizer.get_3d_graph,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
                use_ray=use_ray,
            )

        self.weights = np.array(self.weights)

        # Read in all specs
        self.spec_names = self.df["spec"].values
        spec_files = [
            (data_dir / "subformulae" / f"{form_dir_name}" / f"{spec_name}.json")
            for spec_name in self.spec_names
        ]

        def process_spec_file(x):
            return common.bin_from_file(x, num_bins=num_bins, upper_limit=upper_limit)

        if self.num_workers == 0:
            spec_outputs = [process_spec_file(i) for i in spec_files]
        else:
            spec_outputs = common.chunked_parallel(
                spec_files,
                process_spec_file,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
                use_ray=use_ray,
            )

        self.metas, self.spec_ars = zip(*spec_outputs)
        mask = np.array([i is not None for i in self.spec_ars])
        logging.info(f"Could not find tables for {np.sum(~mask)} spec")

        # Self.weights, self. mol_graphs
        self.metas = np.array(self.metas)[mask].tolist()
        self.spec_ars = np.array(self.spec_ars, dtype=object)[mask].tolist()
        self.df = self.df[mask]
        self.spec_names = np.array(self.spec_names)[mask].tolist()
        self.weights = np.array(self.weights)[mask].tolist()
        self.mol_graphs = np.array(self.mol_graphs, dtype=object)[mask].tolist()

        self.adducts = [
            common.ion2onehot_pos[self.name_to_adduct[i]] for i in self.spec_names
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        name = self.spec_names[idx]
        meta = self.metas[idx]
        ar = self.spec_ars[idx]
        graph, graph_mask = self.mol_graphs[idx]
        full_weight = self.weights[idx]
        adduct = self.adducts[idx]
        outdict = {
            "name": name,
            "binned": ar,
            "full_weight": full_weight,
            "adduct": adduct,
            "graph": graph,
            "graph_mask": graph,
            "_meta": meta,
        }
        return outdict

    @classmethod
    def get_collate_fn(cls):
        return BinnedDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        """collate_fn"""
        names = [j["name"] for j in input_list]
        spec_ars = [j["binned"] for j in input_list]
        graphs = [j["graph"] for j in input_list]
        graph_masks = [j["graph_mask"] for j in input_list]
        full_weight = [j["full_weight"] for j in input_list]
        adducts = [j["adduct"] for j in input_list]

        # Now pad everything else to the max channel dim
        spectra_tensors = torch.stack([torch.tensor(spectra) for spectra in spec_ars])
        full_weight = torch.FloatTensor(full_weight)

        # Stack graphs and graph mask
        batched_graphs = torch.stack(graphs, 0)
        batched_mask = torch.stack(graph_masks, 0)
        adducts = torch.FloatTensor(adducts)

        return_dict = {
            "spectra": spectra_tensors,
            "graphs": batched_graphs,
            "graph_masks": batched_mask,
            "names": names,
            "adducts": adducts,
            "full_weight": full_weight,
        }
        return return_dict


class MolDataset(Dataset):
    """MolDataset."""

    def __init__(self, df, graph_featurizer, num_workers: int = 0, **kwargs):

        self.df = df
        self.num_workers = num_workers
        self.graph_featurizer = graph_featurizer
        self.name_to_adduct = dict(self.df[["spec", "ionization"]].values)

        # Read in all molecules
        self.smiles = self.df["smiles"].values
        self.spec_names = ["" for i in self.smiles]
        if "spec" in list(self.df.keys()):
            self.spec_names = self.df["spec"].values

        if self.num_workers == 0:
            self.mols = [Chem.MolFromSmiles(i) for i in self.smiles]
            self.weights = [common.ExactMolWt(i) for i in self.mols]
            self.mol_graphs = [self.graph_featurizer.get_3d_graph(i) for i in self.mols]
        else:
            self.mols = common.chunked_parallel(
                self.smiles,
                Chem.MolFromSmiles,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
            )
            self.weights = common.chunked_parallel(
                self.mols,
                common.ExactMolWt,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
            )
            self.mol_graphs = common.chunked_parallel(
                self.mols,
                self.graph_featurizer.get_3d_graph,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
            )

        # Extract
        self.weights = np.array(self.weights)
        self.adducts = [
            common.ion2onehot_pos[self.name_to_adduct[i]] for i in self.spec_names
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        smi = self.smiles[idx]
        graph, mask = self.mol_graphs[idx]
        full_weight = self.weights[idx]
        spec_name = self.spec_names[idx]
        adduct = self.adducts[idx]
        outdict = {
            "smi": smi,
            "graph": graph,
            "mask": mask,
            "adduct": adduct,
            "full_weight": full_weight,
            "spec_name": spec_name,
        }
        return outdict

    @classmethod
    def get_collate_fn(cls):
        return MolDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        """collate_fn"""
        names = [j["smi"] for j in input_list]
        spec_names = [j["spec_name"] for j in input_list]
        graphs = [j["graph"] for j in input_list]
        masks = [j["mask"] for j in input_list]
        full_weight = [j["full_weight"] for j in input_list]
        adducts = [j["adduct"] for j in input_list]
        adducts = torch.FloatTensor(adducts)

        batched_graph = torch.stack(graphs, 0)
        batched_mask = torch.stack(masks, 0)

        full_weight = torch.FloatTensor(full_weight)
        return_dict = {
            "graphs": batched_graph,
            "masks": batched_mask,
            "spec_names": spec_names,
            "names": names,
            "full_weight": full_weight,
            "adducts": adducts,
        }
        return return_dict
