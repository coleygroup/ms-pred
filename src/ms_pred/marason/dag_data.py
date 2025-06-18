""" dag_data.py

Fragment dataset to build out model class

"""
import logging
from pathlib import Path
from typing import List
import json
import copy
import functools
import random
import pygmtools as pygm

import numpy as np
import pandas as pd

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
import dgl.nn as dgl_nn


import torch
import dgl
from torch.utils.data.dataset import Dataset

import ms_pred.common as common
import ms_pred.nn_utils as nn_utils
import ms_pred.magma.fragmentation as fragmentation

class TreeProcessor:
    """TreeProcessor.

    Hold key functionalities to read in a magma dag and proces it.

    """

    def __init__(
        self,
        pe_embed_k: int = 10,
        root_encode: str = "gnn",
        binned_targs: bool = False,
        add_hs: bool = False,
        embed_elem_group: bool = False,
        include_frag_morgan=False,
        include_draw_dict=False,
    ):
        """ """
        self.pe_embed_k = pe_embed_k
        self.root_encode = root_encode
        self.binned_targs = binned_targs
        self.add_hs = add_hs
        self.embed_elem_group = embed_elem_group
        self.counter = 1
        self.include_frag_morgan = include_frag_morgan
        self.include_draw_dict=include_draw_dict

        # Hard coded bins (for now)
        self.bins = np.linspace(0, 1500, 15000)

    def get_frag_info(
        self,
        frag: int,
        engine: fragmentation.FragmentEngine,
    ):
        num_atoms = engine.natoms
        # Need to find all kept atoms and all kept bonds between
        kept_atom_inds, kept_atom_symbols = engine.get_present_atoms(frag)
        draw_dict = engine.get_draw_dict(frag) if self.include_draw_dict else None

        # H count
        form = engine.formula_from_kept_inds(kept_atom_inds)

        # Need to re index the targets to match the new graph size
        num_kept = len(kept_atom_inds)
        new_inds = np.arange(num_kept)
        old_inds = kept_atom_inds

        old_to_new = np.zeros(num_atoms, dtype=int)
        old_to_new[old_inds] = new_inds

        # Keep new_to_old for autoregressive predictions
        new_to_old = np.zeros(num_kept, dtype=int)
        new_to_old[new_inds] = old_inds

        return {
            "new_to_old": new_to_old,
            "old_to_new": old_to_new,
            "form": form,
            "draw_dict":draw_dict
        }


    def featurize_frag(
        self,
        frag: int,
        engine: fragmentation.FragmentEngine,
        add_random_walk: bool = False,
    ) -> False:
        """featurize_frag.

        Prev.  dgl_from_frag

        """
        kept_atom_inds, kept_atom_symbols = engine.get_present_atoms(frag)
        atom_symbols = engine.atom_symbols
        kept_bond_orders, kept_bonds = engine.get_present_edges(frag)

        info = self.get_frag_info(frag, engine)
        old_to_new = info['old_to_new']

        # Remap new bond inds
        new_bond_inds = np.empty((0, 2), dtype=int)
        if len(kept_bonds) > 0:
            new_bond_inds = old_to_new[np.vstack(kept_bonds)]

        if self.add_hs:
            h_adds = np.array(engine.atom_hs)[kept_atom_inds]
        else:
            h_adds = None

        # Make dgl graphs for new targets
        graph = self.dgl_featurize(
            np.array(atom_symbols)[kept_atom_inds],
            h_adds=h_adds,
            bond_inds=new_bond_inds,
            bond_types=np.array(kept_bond_orders),
            embed_elem_group=self.embed_elem_group,
        )

        if add_random_walk:
            self.add_pe_embed(graph)

        frag_feature_dict = {
            "graph": graph,
        }
        frag_feature_dict.update(info)
        if self.include_frag_morgan:
            try:
                frag_feature_dict["morgan"] = self.MorganFromGraphs(np.array(atom_symbols)[kept_atom_inds], new_bond_inds, np.array(kept_bond_orders))
            except:
                frag_feature_dict["morgan"] = np.zeros((2048), dtype=np.bool_)
        else:
            frag_feature_dict["morgan"] = None
        return frag_feature_dict

    def _convert_to_dgl(
        self,
        tree: dict,
        include_targets: bool = True,
        last_row=False,
        include_dag_graph=False
    ):
        """_convert_to_dgl.

        Args:
            tree (dict): tree dictionary
            include_targets (bool): Try to add inten targets for supervising
                the inten model
            last_row:
        """
        root_smiles = tree["root_canonical_smiles"]
        engine = fragmentation.FragmentEngine(mol_str=root_smiles, mol_str_type="smiles", mol_str_canonicalized=True)
        # bottom_depth = engine.max_broken_bonds
        bottom_depth = engine.max_tree_depth
        if self.root_encode == "gnn":
            root_frag = engine.get_root_frag()
            root_graph_dict = self.featurize_frag(
                frag=root_frag,
                engine=engine,
            )
            root_repr = root_graph_dict["graph"]
        elif self.root_encode == "fp":
            root_repr = common.get_morgan_fp_smi(root_smiles)
        else:
            raise ValueError()

        root_form = common.form_from_smi(root_smiles)

        # Two types of mass shifts: +adduct or +-electron
        adduct_mass_shift = np.array([
            common.ion2mass[tree["adduct"]],
            -common.ELECTRON_MASS if common.is_positive_adduct(tree["adduct"]) else common.ELECTRON_MASS
        ])
        root_morgan = common.get_morgan_fp_smi(root_smiles, isbool=True)

        # Need to include mass and inten targets here, maybe not necessary in
        # all cases?
        masses, inten_frag_ids, dgl_inputs, inten_targets, frag_targets, max_broken, frag_morgans, draw_dicts = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        forms, max_remove_hs, max_add_hs = [], [], []
        parent_dag_ids = []
        frag_hash_ids = []
        for k, sub_frag in tree["frags"].items():
            max_broken_num = sub_frag["max_broken"]
            tree_depth = sub_frag["tree_depth"]
            frag_hash_ids.append(sub_frag["frag_hash"])
            parent_dag_ids.append(sub_frag["parents"])
            inten_frag_ids.append(k)

            # Skip because we never fragment last row
            if (not last_row) and (tree_depth == bottom_depth):
                continue

            binary_targs = sub_frag["atoms_pulled"]
            frag = sub_frag["frag"]

            # Get frag dict and target
            frag_dict = self.featurize_frag(
                frag,
                engine,
            )
            draw_dicts.append(frag_dict["draw_dict"])
            frag_morgans.append(frag_dict["morgan"])
            forms.append(frag_dict["form"])
            old_to_new = frag_dict["old_to_new"]
            graph = frag_dict["graph"]
            max_broken.append(max_broken_num)

            max_remove_hs.append(sub_frag["max_remove_hs"])
            max_add_hs.append(sub_frag["max_add_hs"])

            # if include_targets and not self.binned_targs:
            #     inten_targs = sub_frag["intens"]
            #     inten_targets.append(inten_targs)


            # For gen model only!!
            targ_vec = np.zeros(graph.num_nodes())
            for j in old_to_new[binary_targs]:
                targ_vec[j] = 1

            graph = frag_dict["graph"]

            # Define targ vec
            dgl_inputs.append(graph)
            masses.append(sub_frag["base_mass"])
            frag_targets.append(torch.from_numpy(targ_vec))
        if include_targets:
            inten_targets = np.array(tree["raw_spec"])

        masses = engine.shift_bucket_masses[None, None, :] + \
                 adduct_mass_shift[None, :, None] + \
                 np.array(masses)[:, None, None]
        max_remove_hs = np.array(max_remove_hs)
        max_add_hs = np.array(max_add_hs)
        max_broken = np.array(max_broken)

        # Feat each form
        all_form_vecs = [common.formula_to_dense(i) for i in forms]
        all_form_vecs = np.array(all_form_vecs)
        root_form_vec = common.formula_to_dense(root_form)
        id_node_map = {node:i for i, node in enumerate(frag_hash_ids)}
        edge_set = set()
        src, dst = [], []
        edge_feats = []
        connectivities = []
        for i, parents in enumerate(parent_dag_ids):
            for parent in parents:
                if parent in id_node_map:
                    parent_id = id_node_map[parent]
                    if (i, parent_id) not in edge_set:
                        src.append(i)
                        dst.append(parent_id)
                        edge_set.add((i, parent_id))
                        edge_feats.append(1)
                        edge_feats.append(-1)
                        connectivities.append([parent_id, i])
                        connectivities.append([i, parent_id])
        
        src = torch.tensor(src)
        dst = torch.tensor(dst)
        # dag_graph = dgl.heterograph({
        #     # ('node', 'parent', 'node'): (src, dst),
        #     ('node', 'child', 'node'): (dst, src),
        # }, num_nodes_dict={'node':masses.shape[0]})
        dag_graph = dgl.graph(
            (src, dst), num_nodes=masses.shape[0]
        ) if include_dag_graph else None

        out_dict = {
            "root_repr": root_repr,
            "dgl_frags": dgl_inputs,
            "masses": masses,
            "inten_targs": np.array(inten_targets) if include_targets else None,
            "max_remove_hs": max_remove_hs,
            "max_add_hs": max_add_hs,
            "max_broken": max_broken,
            "targs": frag_targets,
            "form_vecs": all_form_vecs,
            "root_form_vec": root_form_vec,
            "root_morgan":root_morgan,
            "frag_morgans":np.stack(frag_morgans, axis = 0) if self.include_frag_morgan else None,
            "dag_graph":dag_graph,
            "connectivities":connectivities,
            "edge_feats":edge_feats,
            "draw_dicts":draw_dicts,
            "inten_frag_ids":inten_frag_ids
        }
        return out_dict

    def _process_tree(
        self,
        tree: dict,
        include_targets: bool = True,
        last_row=False,
        convert_to_dgl=True,
        include_dag_graph=False
    ):
        """_process_tree.

        Args:
            tree (dict): tree dictionary
            include_targets (bool): Try to add inten targets for supervising
                the inten model
            last_row:
            pickle_input: If pickle_input, this
        """
        if convert_to_dgl:
            out_dict = self._convert_to_dgl(tree, include_targets, last_row, include_dag_graph)
            if "collision_energy" in tree:
                out_dict["collision_energy"] = tree["collision_energy"]
            if "instrument" in tree:
                out_dict["instrument"] = tree["instrument"]
        else:
            out_dict = tree

        dgl_inputs = out_dict["dgl_frags"]
        root_repr = out_dict["root_repr"]

        if self.pe_embed_k > 0:
            for graph in dgl_inputs:
                self.add_pe_embed(graph)

            if isinstance(root_repr, dgl.DGLGraph):
                self.add_pe_embed(root_repr)

        if include_targets and self.binned_targs:
            intens = out_dict["inten_targs"]
            bin_posts = np.clip(np.digitize(intens[:, 0], self.bins), 0, len(self.bins) - 1)
            new_out = np.zeros_like(self.bins)
            for bin_post, inten in zip(bin_posts, intens[:, 1]):
                new_out[bin_post] = max(new_out[bin_post], inten)
            inten_targets = new_out
            out_dict["inten_targs"] = inten_targets
        
        return out_dict

    def add_pe_embed(self, graph):
        pe_embeds = nn_utils.random_walk_pe(
            graph, k=self.pe_embed_k, eweight_name="e_ind"
        )
        graph.ndata["h"] = torch.cat((graph.ndata["h"], pe_embeds), -1).float()
        return graph

    def process_tree_gen(self, tree: dict, convert_to_dgl=True):
        proc_out = self._process_tree(
            tree, include_targets=False, last_row=False, convert_to_dgl=convert_to_dgl, include_dag_graph=False
        )
        keys = {
            "root_repr",
            "dgl_frags",
            "targs",
            "max_broken",
            "form_vecs",
            "root_form_vec",
            "collision_energy",
        }
        dgl_tree = {i: proc_out[i] for i in keys}
        return {"dgl_tree": dgl_tree, "tree": tree}

    def process_tree_inten(self, tree, convert_to_dgl=True):
        proc_out = self._process_tree(
            tree, include_targets=True, last_row=True, convert_to_dgl=convert_to_dgl, include_dag_graph=True
        )
        keys = {
            "root_repr",
            "dgl_frags",
            "masses",
            "inten_targs",
            "max_remove_hs",
            "max_add_hs",
            "max_broken",
            "form_vecs",
            "root_form_vec",
            "collision_energy",
            "root_morgan",
            "frag_morgans",
            "dag_graph",
            "connectivities",
            "edge_feats",
            "draw_dicts",
            "inten_frag_ids"
        }
        dgl_tree = {i: proc_out[i] for i in keys}
        return {"dgl_tree": dgl_tree, "tree": tree}

    def process_tree_inten_pred(self, tree: dict, convert_to_dgl=True):
        proc_out = self._process_tree(
            tree, include_targets=False, last_row=True, convert_to_dgl=convert_to_dgl, include_dag_graph=True
        )
        keys = {
            "root_repr",
            "dgl_frags",
            "masses",
            "max_remove_hs",
            "inten_targs",
            "max_add_hs",
            "max_broken",
            "form_vecs",
            "root_form_vec",
            "collision_energy",
            "root_morgan",
            "frag_morgans",
            "dag_graph",
            "connectivities",
            "edge_feats",
            "inten_frag_ids"
        }
        dgl_tree = {i: proc_out[i] for i in keys}
        return {"dgl_tree": dgl_tree, "tree": tree}
    
    def add_charges(self, mol):
        mol.UpdatePropertyCache(strict=False)
        ps = Chem.DetectChemistryProblems(mol)
        if not ps:
            Chem.SanitizeMol(mol)
            return mol
        for p in ps:
            if p.GetType()=='AtomValenceException':
                at = mol.GetAtomWithIdx(p.GetAtomIdx())
                if at.GetAtomicNum()==7 and at.GetFormalCharge()==0 and at.GetExplicitValence()==4:
                    at.SetFormalCharge(1)
                if at.GetAtomicNum()==8 and at.GetFormalCharge()==0 and at.GetExplicitValence()==3:
                    at.SetFormalCharge(1)
                if at.GetAtomicNum()==8 and at.GetFormalCharge()==0 and at.GetExplicitValence()==1:
                    at.SetFormalCharge(-1)
                if at.GetAtomicNum()==9 and at.GetFormalCharge()==0 and at.GetExplicitValence()==2:
                    at.SetFormalCharge(1)
        Chem.SanitizeMol(mol)
        return mol
    def MorganFromGraphs(self, node_list, edge_list, edge_type_list):
        mol = Chem.RWMol()
        # add atoms to mol and keep track of index
        node_to_idx = {}
        for i in range(len(node_list)):
            a = Chem.Atom(node_list[i])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx
            a.SetNoImplicit(True)
        for i in range(len(edge_list)):
            if edge_type_list[i] == 1:
                bond_type = Chem.rdchem.BondType.SINGLE
                mol.AddBond(node_to_idx[edge_list[i][0]], node_to_idx[edge_list[i][1]], bond_type)
            elif edge_type_list[i] == 2:
                bond_type = Chem.rdchem.BondType.DOUBLE
                mol.AddBond(node_to_idx[edge_list[i][0]], node_to_idx[edge_list[i][1]], bond_type)
        mol = self.add_charges(mol.GetMol())
        radical_fp = common.get_morgan_fp(mol, isbool=True)
        return radical_fp
    def dgl_featurize(
        self,
        atom_symbols: List[str],
        h_adds: np.ndarray,
        bond_inds: np.ndarray,
        bond_types: np.ndarray,
        embed_elem_group: bool,
    ):
        """dgl_featurize.

        Args:
            atom_symbols (List[str]): node_types
            h_adds (np.ndarray): h_adds
            bond_inds (np.ndarray): bond_inds
            bond_types (np.ndarray)
            embed_elem_group (bool): embed the element group in the periodic table
        """
        node_types = [common.element_to_position[el] for el in atom_symbols]
        node_types = np.vstack(node_types)
        num_nodes = node_types.shape[0]

        src, dest = bond_inds[:, 0], bond_inds[:, 1]
        src_tens_, dest_tens_ = torch.from_numpy(src), torch.from_numpy(dest)
        bond_types = torch.from_numpy(bond_types)
        src_tens = torch.cat([src_tens_, dest_tens_])
        dest_tens = torch.cat([dest_tens_, src_tens_])
        bond_types = torch.cat([bond_types, bond_types])
        bond_featurizer = torch.eye(fragmentation.MAX_BONDS)

        bond_types_onehot = bond_featurizer[bond_types.long()]
        node_data = torch.from_numpy(node_types)

        if embed_elem_group:
            node_groups = [common.element_to_group[el] for el in atom_symbols]
            node_groups = np.vstack(node_groups)
            node_data = torch.hstack([node_data, torch.from_numpy(node_groups)])

        # H data is defined, add that
        if h_adds is None:
            zero_vec = torch.zeros((node_data.shape[0], common.MAX_H))
            node_data = torch.hstack([node_data, zero_vec])
        else:
            h_featurizer = torch.eye(common.MAX_H)
            h_adds_vec = torch.from_numpy(h_adds)
            node_data = torch.hstack([node_data, h_featurizer[h_adds_vec]])

        g = dgl.graph(data=(src_tens, dest_tens), num_nodes=num_nodes)
        g.ndata["h"] = node_data.float()
        g.edata["e"] = bond_types_onehot.float()
        g.edata["e_ind"] = bond_types.long()
        return g

    def get_node_feats(self):
        if self.embed_elem_group:
            return self.pe_embed_k + common.ELEMENT_DIM + common.MAX_H + common.ELEMENT_GROUP_DIM
        else:
            return self.pe_embed_k + common.ELEMENT_DIM + common.MAX_H


class DAGDataset(Dataset):
    """DAGDataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        magma_h5: Path,
        magma_map: dict,
        num_workers=0,
        use_ray: bool = False,
        **kwargs,
    ):
        """__init__.

        Args:
            df (pd.DataFrame): df
            magma_map (dict): magma_map
            num_workers:
            use_ray (bool): use_ray
            kwargs:
        """
        self.df = df
        self.num_workers = num_workers
        self.magma_h5 = magma_h5
        self.magma_map = magma_map
        valid_spec_ids = set([self.rm_collision(i) for i in self.magma_map])
        valid_specs = [(i in valid_spec_ids and inst in common.instrument2onehot_pos) for i, inst in self.df[["spec", "instrument"]].values]
        self.df_sub = self.df[valid_specs]
        self.name_to_idx = {}
        if len(self.df_sub) == 0:
            self.spec_names = []
            self.name_to_dict = {}
        else:
            ori_label_map = self.df_sub.set_index("spec").drop("collision_energies", axis=1).to_dict(orient="index") # labels without collision energy
            self.spec_names = []
            self.name_to_dict = {}
            for i in self.magma_map:
                ori_id = self.rm_collision(i)
                if ori_id in ori_label_map:
                    self.spec_names.append(i)
                    self.name_to_dict[i] = copy.deepcopy(ori_label_map[ori_id])
            for idx, spec_name in enumerate(self.spec_names):
                self.name_to_idx[spec_name] = idx


        for i in self.name_to_dict:
            self.name_to_dict[i]["magma_file"] = self.magma_map[i]

        logging.info(f"{len(self.df_sub)} of {len(self.df)} entries have {len(self.spec_names)} trees.")

        adduct_map = dict(self.df[["spec", "ionization"]].values)
        self.name_to_adduct = {
            i: adduct_map[self.rm_collision(i)] for i in self.spec_names
        }
        self.name_to_adducts = {
            i: common.ion2onehot_pos[self.name_to_adduct[i]] for i in self.spec_names
        }
        instrument_map = dict(self.df[["spec", "instrument"]].values)
        self.name_to_instrument = {
            i: instrument_map[self.rm_collision(i)] for i in self.spec_names
        }

        self.name_to_instruments = {
            i: common.instrument2onehot_pos[self.name_to_instrument[i]] for i in self.spec_names
        }
        self.name_to_smiles = {k: v['smiles'] for k, v in self.name_to_dict.items()}
        self.name_to_precursors = {k: v['precursor'] for k, v in self.name_to_dict.items()}

    def load_tree(self, x):
        filename = self.name_to_dict[x]["magma_file"]
        if not type(self.magma_h5) is common.HDF5Dataset:
            self.magma_h5 = common.HDF5Dataset(self.magma_h5)
        fp = self.magma_h5.read_str(filename)
        return json.loads(fp)

    def read_fn(self, x):
        return self.read_tree(self.load_tree(x))

    def __len__(self):
        return len(self.spec_names)

    def __getitem__(self, idx: int):
        """__getitem__."""

        # name = copy.deepcopy(self.spec_names[idx])
        # adduct = copy.deepcopy(self.name_to_adducts[name])
        # dgl_entry = copy.deepcopy(self.dgl_trees[idx])

        name = self.spec_names[idx]
        adduct = self.name_to_adducts[name]
        precursor = self.name_to_precursors[name]
        instrument = self.name_to_instruments[name]

        dgl_entry = self.read_fn(name)["dgl_tree"]
        # dgl_entry = self.dgl_trees[idx]

        outdict = {"name": name, "adduct": adduct, "precursor": precursor, "instrument": instrument}

        # Convert this into a list of graphs with a list of targets
        outdict.update(dgl_entry)
        return outdict

    def get_node_feats(self) -> int:
        """get_node_feats."""
        return self.tree_processor.get_node_feats()

    @staticmethod
    def rm_collision(key: str) -> str:
        """remove `_collision VALUE` from the string"""
        keys = key.split('_collision')
        if len(keys) == 2:
            return keys[0]
        elif len(keys) == 1:
            return key
        else:
            raise ValueError(f'Unrecognized key: {key}')


class GenDataset(DAGDataset):
    """GenDatset."""

    def __init__(
        self,
        df: pd.DataFrame,
        tree_processor: TreeProcessor,
        magma_h5: Path,
        magma_map: dict,
        num_workers=0,
        use_ray: bool = False,
        **kwargs,
    ):
        """__init__.

        Args:
        """
        self.tree_processor = tree_processor
        self.read_tree = self.tree_processor.process_tree_gen
        super().__init__(
            df=df,
            magma_h5=magma_h5,
            magma_map=magma_map,
            num_workers=num_workers,
            use_ray=use_ray,
        )

    @classmethod
    def get_collate_fn(cls):
        return GenDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        """collate_fn.

        Batch a list of graphs and convert.

        Should return dict containing:
            1. Vector of names.
            2. Batched root reprs
            3. Batched fragment graphs
            4. Batched target atoms
            6. Number of atoms per frag graph
            7. Indices mapping batched fragment graphs to roots
            8. Conversion mapping indices between root and children?

        """
        names = [j["name"] for j in input_list]
        frag_graphs = [j["dgl_frags"] for j in input_list]
        frag_graphs_e = [j for i in frag_graphs for j in i]

        num_frags = torch.LongTensor([len(i) for i in frag_graphs])
        frag_atoms = torch.LongTensor([j.num_nodes() for i in frag_graphs for j in i])

        targs = [j for i in input_list for j in i["targs"]]
        targs_padded = torch.nn.utils.rnn.pad_sequence(targs, batch_first=True)

        batched_reprs = _collate_root(input_list)

        frag_batch = dgl.batch(frag_graphs_e)
        root_inds = torch.arange(len(frag_graphs)).repeat_interleave(num_frags)

        max_broken = [torch.LongTensor(i["max_broken"]) for i in input_list]
        max_broken = torch.cat(max_broken)

        supply_adduct = "adduct" in input_list[0]
        if supply_adduct:
            adducts = [j["adduct"] for j in input_list]
            adducts = torch.FloatTensor(adducts)

        supply_instrument = "instrument" in input_list[0]
        if supply_instrument:
            instruments = [j["instrument"] for j in input_list]
            instruments = torch.FloatTensor(instruments)

        collision_engs = [float(j["collision_energy"]) for j in input_list]
        collision_engs = torch.FloatTensor(collision_engs)

        precursor_mzs = [j["precursor"] for j in input_list]
        precursor_mzs = torch.FloatTensor(precursor_mzs)

        # Collate forms
        form_vecs = torch.LongTensor([j for i in input_list for j in i["form_vecs"]])
        root_vecs = torch.LongTensor([i["root_form_vec"] for i in input_list])
        output = {
            "names": names,
            "root_reprs": batched_reprs,
            "frag_graphs": frag_batch,
            "targ_atoms": targs_padded,
            "frag_atoms": frag_atoms,
            "inds": root_inds,
            "broken_bonds": max_broken,
            "adducts": adducts if supply_adduct else None,
            "collision_engs": collision_engs,
            "instruments": instruments if supply_instrument else None,
            "precursor_mzs": precursor_mzs,
            "root_form_vecs": root_vecs,
            "frag_form_vecs": form_vecs,
        }
        return output


class IntenDataset(DAGDataset):
    """GenDatset."""

    def __init__(
        self,
        df: pd.DataFrame,
        tree_processor: TreeProcessor,
        magma_h5: Path,
        magma_map: dict,
        num_workers=0,
        use_ray: bool = False,
        closest=None,
        closest_distances = None,
        valid_ref_count = None,
        max_ref_count = 10,
        engs_db = None,
        specs_db = None,
        specs=None,
        ref_spec_names = None,
        **kwargs,
    ):
        """__init__.

        Args:
        """
        self.ref_spec_names = ref_spec_names
        self.specs=specs

        self.tree_processor = tree_processor
        self.read_tree = self.tree_processor.process_tree_inten
        if self.specs is not None:
            self.read_tree = self.tree_processor.process_tree_inten_pred
        self.closest = closest
        self.distances = closest_distances
        self.max_ref_count = max_ref_count
        self.ref_counts = valid_ref_count
        self.engs_db = engs_db
        self.specs_db = specs_db
        super().__init__(
            df=df,
            magma_h5=magma_h5,
            magma_map=magma_map,
            num_workers=num_workers,
            use_ray=use_ray,
        )
    def __getitem__(self, idx):
        name = self.spec_names[idx]
        adduct = self.name_to_adducts[name]
        precursor = self.name_to_precursors[name]
        dataset_smi = self.name_to_smiles[name]
        instrument = self.name_to_instruments[name]
        outdict = {"name": name, "adduct": adduct, "precursor": precursor, "smiles":dataset_smi, "instrument":instrument}
        dgl_entry = self.read_fn(name)["dgl_tree"]
        outdict.update(dgl_entry)
        if self.specs is not None:
            outdict["inten_targs"] = self.specs[idx, :].toarray().reshape(-1)            
        distance = None
        ref_count = min(self.ref_counts[idx], self.max_ref_count) if self.ref_counts is not None else None
        refs = ref_specs = ref_engs = None
        if self.ref_spec_names is not None:
            if self.closest is not None:
                ref_idx = self.closest[idx, 0].item()
                ref_name = self.ref_spec_names[ref_idx]
                gen_pred = json.loads(self.magma_h5.read_str(f"{ref_name}.json"))

                smi = gen_pred['root_canonical_smiles']
                trees = self.tree_processor.process_tree_inten_pred(gen_pred)
                refs = trees['dgl_tree']
                colli_eng = float(common.get_collision_energy(ref_name))
                refs.update(
                {"name": gen_pred['name'], "adduct": adduct, "precursor": precursor, "collision_energy": colli_eng,
                    "smiles": smi})
                ref_engs, ref_specs = [], []
                spec_indices = self.closest[idx, :ref_count]
                ref_engs = self.engs_db[spec_indices]
                ref_specs = self.specs_db[spec_indices, :].toarray()
                refs["inten_targs"] = ref_specs[0]
                if self.distances is not None:
                    distance = self.distances[idx]     
        outdict["distance"] = distance
        outdict["ref"] = refs
        outdict["ref_count"] = ref_count
        outdict["ref_collision_engs"] = ref_engs
        outdict["ref_inten_targs"] = ref_specs
        return outdict  
    
    @classmethod
    def get_collate_fn(cls):
        return IntenDataset.collate_fn
    
    @classmethod
    def get_sub_collate_fn(cls):
        return IntenDataset.sub_collate_fn
    
    @staticmethod
    def sub_collate_fn(
        input_list, type = ""
    ):
        """collate_fn.

        Batch a list of graphs and convert.

        Should return dict containing:
            1. Vector of names.
            2. Batched root graphs
            3. Batched fragment graphs
            5. Number of atoms per frag graph
            6. Conversion mapping indices between root and children
            7. Inten targets
            8. Num broken bonds per atom
            9. Num frags in each mols dag

        """
        names = [j["name"] for j in input_list]
        frag_graphs = [j["dgl_frags"] for j in input_list]
        frag_graphs_e = [j for i in frag_graphs for j in i]
        draw_dicts = [j["draw_dicts"] if "draw_dicts" in j else None for j in input_list] 
        connectivities = [j["connectivities"] for j in input_list]
        num_edges = torch.LongTensor([len(i) for i in connectivities])
        connectivities = _unroll_pad(input_list, "connectivities").long()
        edge_feats = _unroll_pad(input_list, "edge_feats")
        num_frags = torch.LongTensor([len(i) for i in frag_graphs])
        frag_atoms = torch.LongTensor([i.num_nodes() for i in frag_graphs_e])
        batched_reprs = _collate_root(input_list)
        if input_list[0]["dag_graph"] == None:
            dag_graphs=None
        else:
            dag_graphs = _collate_root(input_list, name="dag_graph")
        frag_batch = dgl.batch(frag_graphs_e)
        root_inds = torch.arange(len(frag_graphs)).repeat_interleave(num_frags)


        targs_padded = None
        if "inten_targs" in input_list[0] and input_list[0]["inten_targs"] is not None:
            targs_padded = _unroll_pad(input_list, "inten_targs")

        masses_padded = _unroll_pad(input_list, "masses")
        max_remove_hs_padded = _unroll_pad(input_list, "max_remove_hs")
        max_add_hs_padded = _unroll_pad(input_list, "max_add_hs")


        max_broken = [torch.LongTensor(i["max_broken"]) for i in input_list]
        broken_padded = torch.nn.utils.rnn.pad_sequence(max_broken, batch_first=True)

        supply_adduct = "adduct" in input_list[0]
        if supply_adduct:
            adducts = [j["adduct"] for j in input_list]
            adducts = torch.FloatTensor(adducts)
        supply_instrument = "instrument" in input_list[0]
        if supply_instrument:
            instruments = [j["instrument"] for j in input_list]
            instruments = torch.FloatTensor(instruments)

        collision_engs = [float(j["collision_energy"]) for j in input_list]
        collision_engs = torch.FloatTensor(collision_engs)

        inten_frag_ids = None
        if input_list[0]["inten_frag_ids"] is not None:
            inten_frag_ids = [i["inten_frag_ids"] for i in input_list]

        precursor_mzs = [j["precursor"] for j in input_list]
        precursor_mzs = torch.FloatTensor(precursor_mzs)

        # Collate fomrs
        form_vecs = _unroll_pad(input_list, "form_vecs")
        root_vecs = _unroll_pad(input_list, "root_form_vec")


        root_morgans = np.stack([j["root_morgan"] for j in input_list]) if type == "" else None
        frag_morgans = None
        if input_list[0]["frag_morgans"] is not None:
            max_frag_morgans = max([j["frag_morgans"].shape[0] for j in input_list])
            frag_morgans = [np.pad(j["frag_morgans"], ((0, max_frag_morgans - j["frag_morgans"].shape[0]), (0, 0)), mode = 'constant') for j in input_list]
            frag_morgans = torch.from_numpy(np.stack(frag_morgans, axis = 0))
            
            
        output = {
            f"{type}names": names,
            f"{type}root_reprs": batched_reprs,
            f"{type}frag_graphs": frag_batch,
            f"{type}frag_atoms": frag_atoms,
            f"{type}inds": root_inds,
            f"{type}num_frags": num_frags,
            f"{type}inten_targs": targs_padded,
            f"{type}inten_frag_ids": inten_frag_ids,
            f"{type}masses": masses_padded,
            f"{type}broken_bonds": broken_padded,
            f"{type}max_add_hs": max_add_hs_padded,
            f"{type}max_remove_hs": max_remove_hs_padded,
            f"{type}adducts": adducts if supply_adduct else None,
            f"{type}collision_engs": collision_engs,
            f"{type}instruments": instruments if supply_instrument else None,
            f"{type}precursor_mzs": precursor_mzs,
            f"{type}root_form_vecs": root_vecs,
            f"{type}frag_form_vecs": form_vecs,
            f"{type}root_morgans":root_morgans,
            f"{type}frag_morgans":frag_morgans,
            f"{type}num_edges":num_edges,
            f"{type}edge_feats":edge_feats,
            f"{type}connectivities":connectivities,
            f"{type}dag_graphs":dag_graphs,
            f"{type}draw_dicts":draw_dicts,
        }
        if type == "":
            if "distance" in input_list[0] and input_list[0]["distance"] is not None:
                output["distances"] = torch.FloatTensor([j["distance"] for j in input_list])
            else:
                output["distances"] = None
            if "ref_count" in input_list[0] and input_list[0]["ref_count"] is not None:
                output["ref_counts"] = torch.IntTensor([j["ref_count"] for j in input_list])
            else:
                output["ref_counts"] = None
            if "ref_inten_targs" in input_list[0] and input_list[0]["ref_inten_targs"] is not None:
                inten_targs_list = [j["ref_inten_targs"] for j in input_list]
                output["ref_inten_targs"] = torch.from_numpy(np.concatenate(inten_targs_list))
            else:
                output["ref_inten_targs"] = None
            if "ref_collision_engs" in input_list[0] and input_list[0]["ref_collision_engs"] is not None:
                inten_engs_list = [j["ref_collision_engs"] for j in input_list]
                output["ref_collision_engs"] = torch.from_numpy(np.concatenate(inten_engs_list))
            else:
                output["ref_collision_engs"] = None
        return output

    @staticmethod
    def collate_fn(
        input_list,
    ):
        sub_collate_fn = IntenDataset.get_sub_collate_fn()
        result = sub_collate_fn(input_list)
        closest_dict = {}
        if 'ref' in input_list[0] and input_list[0]['ref'] is not None:
            ref_input_list = [j["ref"] for j in input_list]
            closest_dict = sub_collate_fn(ref_input_list, type = "closest_")
        else:
            for key in result:
                closest_dict['closest_' + key] = None 
            result['ref'] = None 
        result.update(closest_dict)
        return result

class IntenPredDataset(DAGDataset):
    """IntenPredDatset."""

    def __init__(
        self,
        df: pd.DataFrame,
        tree_processor: TreeProcessor,
        magma_h5: Path,
        magma_map: dict,
        num_workers=0,
        use_ray: bool = False,
        closest=None,
        closest_distances = None,
        valid_ref_count = None,
        max_ref_count = 10,
        engs_db = None,
        specs_db = None,
        ref_spec_names = None,
        **kwargs,
    ):
        """__init__.

        Args:
        """
        self.tree_processor = tree_processor
        self.read_tree = self.tree_processor.process_tree_inten_pred
        self.closest = closest
        self.distances = closest_distances
        self.max_ref_count = max_ref_count
        self.ref_counts = valid_ref_count
        self.engs_db = engs_db
        self.specs_db = specs_db
        self.ref_spec_names = ref_spec_names
        super().__init__(
            df=df,
            magma_h5=magma_h5,
            magma_map=magma_map,
            num_workers=num_workers,
            use_ray=use_ray,
        )
    def __getitem__(self, idx):
        name = self.spec_names[idx]
        adduct = self.name_to_adducts[name]
        precursor = self.name_to_precursors[name]
        dataset_smi = self.name_to_smiles[name]
        dgl_entry = self.read_fn(name)["dgl_tree"]
        instrument = self.name_to_instruments[name]
        # dgl_entry = self.dgl_trees[idx]

        outdict = {"name": name, "adduct": adduct, "precursor": precursor, "smiles":dataset_smi, "instrument":instrument}

        # Convert this into a list of graphs with a list of targets
        outdict.update(dgl_entry)
        distance = None
        ref_count = min(self.ref_counts[idx], self.max_ref_count) if self.ref_counts is not None else None
        refs = ref_specs = ref_engs = None
        if self.ref_spec_names is not None:
            if self.closest is not None:
                ref_idx = self.closest[idx, 0].item()
                ref_name = self.ref_spec_names[ref_idx]
                gen_pred = json.loads(self.magma_h5.read_str(f"pred_{ref_name}.json"))
                smi = gen_pred['root_canonical_smiles']
                trees = self.tree_processor.process_tree_inten_pred(gen_pred)
                refs = trees['dgl_tree']
                colli_eng = float(common.get_collision_energy(ref_name))
                refs.update(
                {"name": gen_pred['name'], "adduct": adduct, "precursor": precursor, "collision_energy": colli_eng,
                    "smiles": smi})
                spec_indices = self.closest[idx, :ref_count]
                ref_engs = self.engs_db[spec_indices]
                ref_specs = self.specs_db[spec_indices, :].toarray()
                if self.distances is not None:
                    distance = self.distances[idx]     
        outdict["distance"] = distance
        outdict["ref"] = refs
        outdict["ref_count"] = ref_count
        outdict["ref_collision_engs"] = ref_engs
        outdict["ref_inten_targs"] = ref_specs
        return outdict  
        

    @classmethod
    def get_collate_fn(cls):
        return IntenDataset.collate_fn


def _unroll_pad(input_list, key):
    if input_list[0][key] is not None:
        out = [torch.FloatTensor(i[key]) if i[key] is not None
               else torch.FloatTensor(input_list[0][key] * float('nan'))
               for i in input_list]
        out_padded = torch.nn.utils.rnn.pad_sequence(out, batch_first=True)
        return out_padded
    return None


def _collate_root(input_list, name = 'root_repr'):
    reprs = [j[name] for j in input_list]
    if isinstance(reprs[0], dgl.DGLGraph):
        batched_reprs = dgl.batch(reprs)
    elif isinstance(reprs[0], np.ndarray):
        batched_reprs = torch.FloatTensor(np.vstack(reprs)).float()
    else:
        raise NotImplementedError()
    return batched_reprs

