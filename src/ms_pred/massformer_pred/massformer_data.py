import logging
import json
import numpy as np

import torch
from rdkit import Chem
from torch.utils.data.dataset import Dataset
import dgl

import ms_pred.common as common
from ._massformer_graph_featurizer import MassformerGraphFeaturizer


class BinnedDataset(Dataset):
    """BinnedDataset."""

    def __init__(
        self,
        df,
        data_dir,
        num_bins,
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
        self.subform_path = data_dir / "subformulae" / f"{form_dir_name}"
        self.subform_h5 = None

        # Read in all molecules & specs
        self.graph_featurizer = MassformerGraphFeaturizer()

        def process_df(df_row):
            smi, spec_name, ces = df_row["smiles"], df_row['spec'], df_row['collision_energies']
            mol = Chem.MolFromSmiles(smi)
            mw = common.ExactMolWt(mol)
            graph = self.graph_featurizer(mol)

            # load spectrum
            all_spec_output = []
            for ce in eval(ces):
                ce = int(ce)  # collision energies are integers in NIST
                all_spec_output.append((spec_name, smi, mol, mw, graph, ce))
            return all_spec_output

        dfrows = [i for _, i in self.df.iterrows()]
        if self.num_workers == 0:
            outputs = [process_df(i) for i in dfrows]
        else:
            outputs = common.chunked_parallel(
                dfrows,
                process_df,
                chunks=100,
                max_cpu=self.num_workers,
            )
        outputs = [j for i in outputs for j in i]  # unroll

        self.spec_names, self.smiles, self.mols, self.weights, self.mol_graphs, self.collision_energies = zip(*outputs)

        # collision energy statistics
        self.collision_energy_mean = common.NIST_COLLISION_ENERGY_MEAN
        self.collision_energy_std = common.NIST_COLLISION_ENERGY_STD

        # Self.weights, self. mol_graphs
        self.adducts = [
            common.ion2onehot_pos[self.name_to_adduct[i]] for i in self.spec_names
        ]

    def __len__(self):
        return len(self.spec_names)

    def __getitem__(self, idx: int):
        name = self.spec_names[idx]
        collision_energy = self.collision_energies[idx]
        graph = self.mol_graphs[idx]
        full_weight = self.weights[idx]
        adduct = self.adducts[idx]

        if self.subform_h5 is None:
            self.subform_h5 = common.HDF5Dataset(self.subform_path)
        json_str = self.subform_h5.read_str(f'{name}_collision {collision_energy}.json')
        meta, spec_ar = common.bin_from_str(json_str, num_bins=self.num_bins, upper_limit=self.upper_limit)

        norm_collision_energy = (collision_energy - self.collision_energy_mean) / (self.collision_energy_std + 1e-6)  # it's not "NCE" on Thermo instruments!!
        outdict = {
            "name": name,
            "binned": spec_ar,
            "full_weight": full_weight,
            "adduct": adduct,
            "gf_v2_data": graph,
            "collision_energy": collision_energy,
            "norm_collision_energy": norm_collision_energy,   # it's not "NCE" on Thermo instruments!!
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
        graphs = MassformerGraphFeaturizer.collate_func(
            [j["gf_v2_data"] for j in input_list]
        )
        full_weight = [j["full_weight"] for j in input_list]
        adducts = [j["adduct"] for j in input_list]
        collision_energies = [j["collision_energy"] for j in input_list]
        norm_collision_energies = [j["norm_collision_energy"] for j in input_list]  # it's not "NCE" on Thermo instruments!!

        # Now pad everything else to the max channel dim
        spectra_tensors = torch.stack([torch.tensor(spectra) for spectra in spec_ars])
        full_weight = torch.FloatTensor(full_weight)

        # frag_batch.set_n_initializer(dgl.init.zero_initializer)
        # frag_batch.set_e_initializer(dgl.init.zero_initializer)

        adducts = torch.FloatTensor(adducts)
        collision_energies = torch.FloatTensor(collision_energies)
        norm_collision_energies = torch.FloatTensor(norm_collision_energies)

        return_dict = {
            "spectra": spectra_tensors,
            "gf_v2_data": graphs,
            "names": names,
            "adducts": adducts,
            "collision_energies": collision_energies,
            "norm_collision_energies": norm_collision_energies,
            "full_weight": full_weight,
        }
        return return_dict


class MolDataset(Dataset):
    """MolDataset."""

    def __init__(self, df, num_workers: int = 0, **kwargs):

        self.df = df
        self.num_workers = num_workers
        self.graph_featurizer = MassformerGraphFeaturizer()
        self.name_to_adduct = dict(self.df[["spec", "ionization"]].values)

        # Read in all molecules & specs
        self.graph_featurizer = MassformerGraphFeaturizer()

        def process_df(df_row):
            smi, spec_name, ces = df_row["smiles"], df_row['spec'], df_row['collision_energies']
            mol = Chem.MolFromSmiles(smi)
            mw = common.ExactMolWt(mol)
            graph = self.graph_featurizer(mol)

            # load spectrum
            all_spec_output = []
            for ce in eval(ces):
                ce = int(ce)  # collision energies are integers in NIST
                all_spec_output.append((spec_name, smi, mol, mw, graph, ce))
            return all_spec_output

        dfrows = [i for _, i in self.df.iterrows()]
        if self.num_workers == 0:
            outputs = [process_df(i) for i in dfrows]
        else:
            outputs = common.chunked_parallel(
                dfrows,
                process_df,
                chunks=100,
                max_cpu=self.num_workers,
            )
        outputs = [j for i in outputs for j in i]  # unroll

        self.spec_names, self.smiles, self.mols, self.weights, self.mol_graphs, self.collision_energies = zip(*outputs)

        # Extract
        self.adducts = [
            common.ion2onehot_pos[self.name_to_adduct[i]] for i in self.spec_names
        ]

        # collision energy statistics
        self.collision_energy_mean = common.NIST_COLLISION_ENERGY_MEAN
        self.collision_energy_std = common.NIST_COLLISION_ENERGY_STD

    def __len__(self):
        return len(self.spec_names)

    def __getitem__(self, idx: int):
        smi = self.smiles[idx]
        spec_name = self.spec_names[idx]
        collision_energy = self.collision_energies[idx]
        graph = self.mol_graphs[idx]
        full_weight = self.weights[idx]
        adduct = self.adducts[idx]
        norm_collision_energy = (collision_energy - self.collision_energy_mean) / (self.collision_energy_std + 1e-6)  # it's not "NCE" on Thermo instruments!!

        outdict = {
            "smi": smi,
            "gf_v2_data": graph,
            "adduct": adduct,
            "full_weight": full_weight,
            "spec_name": spec_name,
            "collision_energy": collision_energy,
            "norm_collision_energy": norm_collision_energy,  # it's not "NCE" on Thermo instruments!!
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
        graphs = MassformerGraphFeaturizer.collate_func(
            [j["gf_v2_data"] for j in input_list]
        )
        full_weight = [j["full_weight"] for j in input_list]
        adducts = [j["adduct"] for j in input_list]
        collision_energies = [j["collision_energy"] for j in input_list]
        norm_collision_energies = [j["norm_collision_energy"] for j in input_list]  # it's not "NCE" on Thermo instruments!!
        adducts = torch.FloatTensor(adducts)
        collision_energies = torch.FloatTensor(collision_energies)
        norm_collision_energies = torch.FloatTensor(norm_collision_energies)

        full_weight = torch.FloatTensor(full_weight)
        return_dict = {
            "gf_v2_data": graphs,
            "spec_names": spec_names,
            "names": names,
            "full_weight": full_weight,
            "adducts": adducts,
            "collision_energies": collision_energies,
            "norm_collision_energies": norm_collision_energies,
        }
        return return_dict
