"""Inten and Frag dataset and featurization utilities for Iceberg Transformer.

- TreeProcessor: builds root representations (DGL or Graphormer tensors) and fragment targets
- Optimized Graphormer input creation with vectorized BFS and multi-hop edge features
- IntenDataset: merges generation inputs with simplified intensity targets
"""

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import dgl

from rdkit import Chem  # type: ignore
from rdkit.Chem import rdPartialCharges  # type: ignore

import ms_pred.common as common
import ms_pred.nn_utils as nn_utils
import ms_pred.magma.fragmentation as fragmentation
from ms_pred.dag_pred.dag_data import DAGDataset, _collate_root
import torch.nn.functional as F
import json


class TreeProcessor:
    def __init__(
        self,
        pe_embed_k: int = 10,
        root_encode: str = "gnn",
        binned_targs: bool = False,
        add_hs: bool = False,
        embed_elem_group: bool = False,
        multi_hop_max_dist: int = 5,
    ):
        self.pe_embed_k = pe_embed_k
        self.root_encode = root_encode
        self.binned_targs = binned_targs
        self.add_hs = add_hs
        self.embed_elem_group = embed_elem_group
        self.bins = np.linspace(0, 1500, 15000)
        self.multi_hop_max_dist = multi_hop_max_dist

    # ---------- lightweight fragment ----------
    def featurize_frag_lite(self, frag: int, engine: fragmentation.FragmentEngine) -> Dict[str, Any]:
        kept_atom_inds, _ = engine.get_present_atoms(frag)
        form = engine.formula_from_kept_inds(kept_atom_inds)
        return {"kept_indices": np.asarray(kept_atom_inds, dtype=int), "form": form}

    # ---------- main tree ----------
    def featurize_tree(self, tree: Dict[str, Any]) -> Dict[str, Any]:
        root_smiles = tree["root_canonical_smiles"]
        engine = fragmentation.FragmentEngine(mol_str=root_smiles, mol_str_type="smiles", mol_str_canonicalized=True)

        atom_symbols = engine.atom_symbols
        total_atom_weights = engine.atom_weights_h
        num_hs = engine.atom_hs
        total_hs = engine.total_hs

        atom_form_vecs_np = [common.formula_to_dense(f"{s}H{h}") for s, h in zip(atom_symbols, num_hs)]
        atom_form_vecs = torch.from_numpy(np.stack(atom_form_vecs_np))

        mol = Chem.MolFromSmiles(root_smiles)
        if mol is None:
            raise ValueError(f"Cannot create RDKit molecule from SMILES: {root_smiles}")

        if self.root_encode == "gnn":
            root_repr = self.rdkit_featurize(mol)
        elif self.root_encode == "graphormer":
            root_repr = None
        else:
            raise ValueError(f"Unsupported root_encode: {self.root_encode}")

        root_form = common.form_from_smi(root_smiles)
        adduct_mass_shift = np.array([
            common.ion2mass[tree["adduct"]],
            -common.ELECTRON_MASS if common.is_positive_adduct(tree["adduct"]) else common.ELECTRON_MASS,
        ])

        # fragment targets
        frag_targs_list = []
        if isinstance(tree, dict):       
            for _, sub_frag in tree["frags"].items():
                frag = sub_frag["frag"]
                info = self.featurize_frag_lite(frag, engine)
                kept = info["kept_indices"]
                if kept.size == 0:
                    continue
                mask = np.zeros((len(total_atom_weights),), dtype=bool)
                mask[kept] = True
                frag_targs_list.append(torch.from_numpy(mask))
            inten_targets = np.asarray(tree["raw_spec"])  # [N,2]
        elif isinstance(tree, common.MassSpec):
            for frag in tree.int_frags:
                info = self.featurize_frag_lite(
                    frag,
                    engine,
                )
                kept = info["kept_indices"]
                if kept.size == 0:
                    continue
                mask = np.zeros((len(total_atom_weights),), dtype=bool)
                mask[kept] = True
                frag_targs_list.append(torch.from_numpy(mask))
            inten_targets = np.array(tree.info["raw_spec"])
        else:
            raise TypeError(f'Unknown type of {tree}')
        if len(frag_targs_list) == 0:
            frag_targs_list.append(torch.zeros((len(total_atom_weights),), dtype=torch.bool))
        frag_targs = torch.stack(frag_targs_list, dim=0)

        bin_posts = np.clip(np.digitize(inten_targets[:, 0], self.bins), 0, len(self.bins) - 1)
        new_out = np.zeros_like(self.bins)
        for b, inten in zip(bin_posts, inten_targets[:, 1]):
            new_out[b] = max(new_out[b], inten)

        if self.pe_embed_k > 0 and self.root_encode == "gnn":
            root_repr = self.add_pe_embed(root_repr)

        root_form_vec = torch.from_numpy(common.formula_to_dense(root_form))

        adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True)
        adj_matrix = torch.from_numpy(adj_matrix).float()

        graphormer_input = self.create_graphormer_input(mol, multi_hop_max_dist=self.multi_hop_max_dist) if self.root_encode == "graphormer" else None

        return {
            "root_repr": root_repr,
            "root_smiles": root_smiles,
            "root_form": root_form,
            "adduct_mass_shift": adduct_mass_shift,
            "frag_targs": frag_targs,
            "inten_targs": new_out,
            "weights": torch.tensor(total_atom_weights, dtype=torch.float32),
            "collision_energy": float(tree["collision_energy"]) if "collision_energy" in tree else 0.0,
            "atom_symbols": atom_symbols,
            "root_form_vec": root_form_vec,
            "atom_form_vecs": atom_form_vecs,
            "adj_matrix": adj_matrix,
            "atom_hs": torch.tensor(num_hs, dtype=torch.long),
            "total_hs": total_hs,
            "graphormer_input": graphormer_input,
        }

    # ---------- PE helpers ----------
    def add_pe_embed(self, graph: Any):
        pe_embeds = nn_utils.random_walk_pe(graph, k=self.pe_embed_k, eweight_name="e_ind")
        graph.ndata["h"] = torch.cat((graph.ndata["h"], pe_embeds), -1).float()
        return graph

    def get_pe_for_tensor(self, mol: Any, node_features: torch.Tensor):
        adj = Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True)
        src, dst = np.nonzero(adj)
        eweights = adj[(src, dst)]
        tmp = dgl.graph((src, dst), num_nodes=adj.shape[0])
        tmp.ndata["h"] = node_features
        tmp.edata["e_ind"] = torch.tensor(eweights, dtype=torch.float32)
        pe_embeds = nn_utils.random_walk_pe(tmp, k=self.pe_embed_k, eweight_name="e_ind")
        return pe_embeds

    # ---------- RDKit featurize for DGL ----------
    def rdkit_featurize(self, mol: Any):
        num_atoms = mol.GetNumAtoms()
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
        except Exception:
            pass

        # Precompute SSSR ring info for ring count features
        sssr = Chem.GetSymmSSSR(mol)
        atom_ring_counts = [0] * num_atoms
        bond_ring_counts = {}
        for bond in mol.GetBonds():
            key = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
            bond_ring_counts[key] = 0
        for ring in sssr:
            ring_atoms = list(ring)
            r_len = len(ring_atoms)
            for a in ring_atoms:
                atom_ring_counts[a] += 1
            for k in range(r_len):
                a1 = ring_atoms[k]
                a2 = ring_atoms[(k + 1) % r_len]
                key = tuple(sorted((a1, a2)))
                if key in bond_ring_counts:
                    bond_ring_counts[key] += 1

        node_features = []
        for atom_idx in range(num_atoms):
            atom = mol.GetAtomWithIdx(atom_idx)
            feats = []
            atomic_num = atom.GetAtomicNum()
            feats.extend([float(atomic_num == x) for x in common.VALID_ATOM_NUM])
            feats.append(float(atomic_num not in common.VALID_ATOM_NUM))
            degree = atom.GetTotalDegree()
            deg_oh = [float(degree == x) for x in range(common.MAX_COMMON_DEGREE + 1)]
            deg_oh.append(float(degree > common.MAX_COMMON_DEGREE))
            feats.extend(deg_oh)
            formal_charge = atom.GetFormalCharge()
            chg_oh = [float(formal_charge == x) for x in range(-common.MAX_ABS_FORMAL_CHARGE, common.MAX_ABS_FORMAL_CHARGE + 1)]
            chg_oh.append(float(abs(formal_charge) > common.MAX_ABS_FORMAL_CHARGE))
            feats.extend(chg_oh)
            hyb = atom.GetHybridization()
            hyb_oh = [float(hyb == x) for x in common.COMMON_HYBRIDIZATION]
            hyb_oh.append(float(hyb not in common.COMMON_HYBRIDIZATION))
            feats.extend(hyb_oh)
            feats.append(float(atom.GetIsAromatic()))
            nH = atom.GetTotalNumHs()
            h_oh = [float(nH == x) for x in range(common.COMMON_MAX_HYDROGEN_COUNTS + 1)]
            h_oh.append(float(nH >= common.COMMON_MAX_HYDROGEN_COUNTS))
            feats.extend(h_oh)
            # Ring presence one-hot
            in_ring = atom.IsInRing()
            feats.extend([float(not in_ring), float(in_ring)])
            # Ring size one-hot
            for rsz in common.COMMON_RING_SIZES:
                feats.append(float(atom.IsInRingSize(rsz)))
                feats.append(float(not atom.IsInRingSize(rsz)))
            # Fused ring one-hot (>=2 rings)
            arc = atom_ring_counts[atom_idx]
            fused = arc >= 2
            feats.extend([float(not fused), float(fused)])
            chi = atom.GetChiralTag()
            feats.extend([float(chi == x) for x in common.COMMON_CHIRALITY])
            try:
                charge = float(atom.GetProp("_GasteigerCharge"))
                if np.isnan(charge) or np.isinf(charge):
                    charge = 0.0
            except Exception:
                charge = 0.0
            feats.append(charge)
            if self.embed_elem_group:
                sym = atom.GetSymbol()
                if sym in common.ELEMENT_TO_GROUP:
                    feats.extend(common.element_to_group[sym].tolist())
                else:
                    feats.extend([0.0] * common.ELEMENT_GROUP_DIM)
            node_features.append(feats)

        edge_indices = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.extend([(i, j), (j, i)])
            feats = []
            # Bond type one-hot (existing)
            bt = bond.GetBondType()
            bt_oh = [float(bt == x) for x in common.COMMON_BOND_TYPES]
            bt_oh.append(float(bt not in common.COMMON_BOND_TYPES))
            feats.extend(bt_oh)
            # Conjugation one-hot [not, yes]
            conj = bond.GetIsConjugated()
            feats.extend([float(not conj), float(conj)])
            # Ring presence one-hot [not, in ring]
            in_ring = bond.IsInRing()
            feats.extend([float(not in_ring), float(in_ring)])
            # Ring size one-hot: a bond can belong to multiple ring sizes in fused systems
            for rsz in common.COMMON_RING_SIZES:
                feats.append(float(bond.IsInRingSize(rsz)))
                feats.append(float(not bond.IsInRingSize(rsz)))
            # Fused ring (>=2 rings) one-hot [not fused, fused]
            brc = bond_ring_counts.get(tuple(sorted((i, j))), 0)
            fused = brc >= 2
            feats.extend([float(not fused), float(fused)])
            # Stereo one-hot (existing)
            stereo = bond.GetStereo()
            feats.extend([float(stereo == x) for x in common.VALID_BOND_STEREO])
            edge_features.extend([feats, feats])

        if edge_indices:
            src_nodes, dst_nodes = zip(*edge_indices)
            assert dgl is not None, "dgl is required for GNN root encoding"
            g = dgl.graph((src_nodes, dst_nodes), num_nodes=num_atoms)
            g.edata["e"] = torch.tensor(edge_features, dtype=torch.float32)
        else:
            assert dgl is not None, "dgl is required for GNN root encoding"
            g = dgl.graph(([], []), num_nodes=num_atoms)
            g.edata["e"] = torch.empty((0, len(edge_features[0]) if edge_features else 12), dtype=torch.float32)
        g.ndata["h"] = torch.tensor(node_features, dtype=torch.float32)
        return g

    def get_node_feats(self) -> int:
        if self.root_encode in ("gnn", "graphormer"):
            nf = 0
            nf += len(common.VALID_ELEMENTS) + 1
            nf += common.MAX_COMMON_DEGREE + 2
            nf += common.MAX_ABS_FORMAL_CHARGE * 2 + 2
            nf += len(common.COMMON_HYBRIDIZATION) + 1
            nf += 1
            nf += common.COMMON_MAX_HYDROGEN_COUNTS + 2
            nf += 2  # ring presence one-hot
            nf += len(common.COMMON_RING_SIZES) * 2  # ring size one-hot
            nf += 2  # fused ring one-hot
            nf += len(common.COMMON_CHIRALITY)
            nf += 1
            if self.embed_elem_group:
                nf += common.ELEMENT_GROUP_DIM
            nf += self.pe_embed_k
            return nf
        raise NotImplementedError(f"Unsupported root encoding method: {self.root_encode}")

    def get_edge_feats(self) -> int:
            nf = 0
            nf += len(common.COMMON_BOND_TYPES) + 1  # bond type one-hot (+ other)
            nf += 2  # conjugation one-hot
            nf += 2  # ring presence one-hot
            nf += 2*len(common.COMMON_RING_SIZES)  # ring size multi-hot (no extra 'no ring' slot)
            nf += 2  # fused ring one-hot
            nf += len(common.VALID_BOND_STEREO) + 1  # stereo one-hot
            return nf

    # ---------- Graphormer input ----------
    def create_graphormer_input(
        self,
        mol: Any,
        max_nodes: int = 128,
        spatial_pos_max: int = 1024,
        multi_hop_max_dist: int = 5,
    ) -> Dict[str, Any]:
        num_atoms = mol.GetNumAtoms()
        if num_atoms > max_nodes:
            raise ValueError(f"Molecule has {num_atoms} atoms, exceeding max_nodes={max_nodes}")

        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
        except Exception:
            pass
        # Precompute SSSR once for both atom and bond ring count features
        sssr = Chem.GetSymmSSSR(mol)
        atom_ring_counts = [0] * num_atoms
        bond_ring_counts = {}
        for bond in mol.GetBonds():
            key = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
            bond_ring_counts[key] = 0
        for ring in sssr:
            ring_atoms = list(ring)
            r_len = len(ring_atoms)
            for a in ring_atoms:
                atom_ring_counts[a] += 1
            for k in range(r_len):
                a1 = ring_atoms[k]
                a2 = ring_atoms[(k + 1) % r_len]
                key = tuple(sorted((a1, a2)))
                if key in bond_ring_counts:
                    bond_ring_counts[key] += 1

        node_features = []
        for atom_idx in range(num_atoms):
            atom = mol.GetAtomWithIdx(atom_idx)
            feats = []
            atomic_num = atom.GetAtomicNum()
            feats.extend([float(atomic_num == x) for x in common.VALID_ATOM_NUM])
            feats.append(float(atomic_num not in common.VALID_ATOM_NUM))
            degree = atom.GetTotalDegree()
            deg_oh = [float(degree == x) for x in range(common.MAX_COMMON_DEGREE + 1)]
            deg_oh.append(float(degree > common.MAX_COMMON_DEGREE))
            feats.extend(deg_oh)
            formal_charge = atom.GetFormalCharge()
            chg_oh = [float(formal_charge == x) for x in range(-common.MAX_ABS_FORMAL_CHARGE, common.MAX_ABS_FORMAL_CHARGE + 1)]
            chg_oh.append(float(abs(formal_charge) > common.MAX_ABS_FORMAL_CHARGE))
            feats.extend(chg_oh)
            hyb = atom.GetHybridization()
            hyb_oh = [float(hyb == x) for x in common.COMMON_HYBRIDIZATION]
            hyb_oh.append(float(hyb not in common.COMMON_HYBRIDIZATION))
            feats.extend(hyb_oh)
            feats.append(float(atom.GetIsAromatic()))
            nH = atom.GetTotalNumHs()
            h_oh = [float(nH == x) for x in range(common.COMMON_MAX_HYDROGEN_COUNTS + 1)]
            h_oh.append(float(nH > common.COMMON_MAX_HYDROGEN_COUNTS))
            feats.extend(h_oh)
            # Ring presence one-hot
            in_ring = atom.IsInRing()
            feats.extend([float(not in_ring), float(in_ring)])
            # Ring size multi-hot
            for rsz in common.COMMON_RING_SIZES:
                feats.append(float(atom.IsInRingSize(rsz)))
                feats.append(float(not atom.IsInRingSize(rsz)))
            # Fused ring one-hot
            arc = atom_ring_counts[atom_idx]
            fused = arc >= 2
            feats.extend([float(not fused), float(fused)])
            chi = atom.GetChiralTag()
            feats.extend([float(chi == x) for x in common.COMMON_CHIRALITY])
            if self.embed_elem_group:
                sym = atom.GetSymbol()
                if sym in common.ELEMENT_TO_GROUP:
                    feats.extend(common.element_to_group[sym].tolist())
                else:
                    feats.extend([0.0] * common.ELEMENT_GROUP_DIM)
            try:
                charge = float(atom.GetProp("_GasteigerCharge"))
                if np.isnan(charge) or np.isinf(charge):
                    charge = 0.0
            except Exception:
                charge = 0.0
            feats.append(charge)
            node_features.append(feats)
        x = torch.tensor(node_features, dtype=torch.float32)
        if self.pe_embed_k > 0:
            pe_embeds = self.get_pe_for_tensor(mol, x)
            x = torch.cat((x, pe_embeds), -1).float()

        adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=False).astype(np.bool_)
        n = num_atoms
        UNREACHABLE = np.int16(32767)
        dist_matrix = np.full((n, n), UNREACHABLE, dtype=np.int16)
        parents = np.full((n, n), -1, dtype=np.int16)
        for src in range(n):
            visited = np.zeros(n, dtype=bool)
            frontier = np.zeros(n, dtype=bool)
            frontier[src] = True
            visited[src] = True
            dist_matrix[src, src] = 0
            parents[src, src] = src
            depth = np.int16(0)
            while frontier.any():
                next_frontier = adj_matrix[frontier].any(axis=0)
                next_frontier &= ~visited
                if not next_frontier.any():
                    break
                depth = np.int16(depth + 1)
                dist_matrix[src, next_frontier] = depth
                candidates = adj_matrix[:, next_frontier] & frontier[:, None]
                parent_idxs = candidates.argmax(axis=0).astype(np.int16)
                parents[src, next_frontier] = parent_idxs
                visited |= next_frontier
                frontier = next_frontier
        dist_matrix = np.clip(dist_matrix, 0, spatial_pos_max - 1)
        spatial_pos = torch.from_numpy(dist_matrix).long()

        bond_features_dict = {}
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            feats = []
            # Bond type one-hot
            bt = bond.GetBondType()
            bt_oh = [float(bt == x) for x in common.COMMON_BOND_TYPES]
            bt_oh.append(float(bt not in common.COMMON_BOND_TYPES))
            feats.extend(bt_oh)
            # Conjugation one-hot
            conj = bond.GetIsConjugated()
            feats.extend([float(not conj), float(conj)])
            # Ring presence one-hot
            in_ring = bond.IsInRing()
            feats.extend([float(not in_ring), float(in_ring)])
            # Ring size one-hot
            for rsz in common.COMMON_RING_SIZES:
                feats.append(float(bond.IsInRingSize(rsz)))
                feats.append(float(not bond.IsInRingSize(rsz)))
            # Fused ring one-hot
            brc = bond_ring_counts.get(tuple(sorted((i, j))), 0)
            fused = brc >= 2
            feats.extend([float(not fused), float(fused)])
            # Stereo one-hot
            stereo = bond.GetStereo()
            feats.extend([float(stereo == x) for x in common.VALID_BOND_STEREO])
            feats.append(float(stereo not in common.VALID_BOND_STEREO))
            bond_features_dict[(i, j)] = feats
            bond_features_dict[(j, i)] = feats

        bond_feat_dim = self.get_edge_feats()
        attn_edge_type = torch.zeros([num_atoms, num_atoms, bond_feat_dim], dtype=torch.float32)
        for (i, j), feats in bond_features_dict.items():
            attn_edge_type[i, j] = torch.tensor(feats, dtype=torch.float32)

        in_degree = torch.tensor([atom.GetDegree() for atom in mol.GetAtoms()], dtype=torch.long)
        out_degree = in_degree.clone()

        edge_input = torch.zeros([num_atoms, num_atoms, multi_hop_max_dist, bond_feat_dim], dtype=torch.long)
        bond_feat_tensor_long = attn_edge_type.long()
        for src in range(num_atoms):
            for tgt in range(num_atoms):
                if tgt == src or parents[src, tgt] == -1:
                    continue
                path_nodes = [tgt]
                cur = tgt
                while cur != src:
                    cur = int(parents[src, cur])
                    path_nodes.append(cur)
                path_nodes.reverse()
                max_hops = min(len(path_nodes) - 1, multi_hop_max_dist)
                for hop in range(max_hops):
                    u = path_nodes[hop]
                    v = path_nodes[hop + 1]
                    edge_input[src, tgt, hop] = bond_feat_tensor_long[u, v]

        attn_bias = torch.where(spatial_pos >= spatial_pos_max, float("-inf"), 0.0)
        attn_bias = F.pad(attn_bias, (1, 0, 1, 0), value=0.0)

        return {
            "x": x,
            "attn_bias": attn_bias,
            "attn_edge_type": attn_edge_type,
            "spatial_pos": spatial_pos,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "edge_input": edge_input,
            "num_atoms": num_atoms,
        }

class IntenDataset(DAGDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        magma_h5: Path,
        magma_map: dict,
        root_encode: str = "gnn",
        binned_targs: bool = False,
        add_hs: bool = False,
        embed_elem_group: bool = False,
        tree_processor: TreeProcessor = None,
        datatype: str = "PredSpecDB",
        **kwargs,
    ):
        super().__init__(df, magma_h5, magma_map, **kwargs)
        self.root_encode = root_encode
        self.binned_targs = binned_targs
        self.add_hs = add_hs
        self.embed_elem_group = embed_elem_group
        self.tree_processor = tree_processor
        self.read_tree = self.tree_processor.featurize_tree
        self.datatype = datatype

    def __getitem__(self, idx: int):
        name = self.spec_names[idx]
        adduct = self.name_to_adducts[name]
        precursor = self.name_to_precursors[name]
        entry = self.read_fn(name)
        outdict = {"name": name, "adduct": adduct, "precursor": precursor}
        outdict.update(entry)
        return outdict

    def get_node_feats(self) -> int:
        return self.tree_processor.get_node_feats()

    @classmethod
    def get_collate_fn(cls):
        return IntenDataset.collate_fn

    def load_tree(self, x):
        if self.datatype == "PredSpecDB":
            filekeys = self.name_to_dict[x]["magma_file"]
            if not type(self.magma_h5) is common.PredSpecDB:
                self.magma_h5 = common.PredSpecDB(self.magma_h5)
            spec = self.magma_h5.read(*filekeys)
            return spec
        elif self.datatype == "HDF5":
            filename = self.name_to_dict[x]["magma_file"]
            if not type(self.magma_h5) is common.HDF5Dataset:
                self.magma_h5 = common.HDF5Dataset(self.magma_h5)
            fp = self.magma_h5.read_str(filename)
            return json.loads(fp)
        else:
            raise ValueError(f"Unsupported datatype: {self.datatype}")

    @staticmethod
    def collate_fn(batch):
        names = [item["name"] for item in batch]
        smis = [item["root_smiles"] for item in batch]
        weights = [item["weights"] for item in batch]
        weights_padded = torch.nn.utils.rnn.pad_sequence(weights, batch_first=True)
        adduct_mass_shifts = torch.from_numpy(np.stack([item["adduct_mass_shift"] for item in batch])).float()

        if dgl is not None and isinstance(batch[0]["root_repr"], dgl.DGLGraph):
            batched_reprs = _collate_root(batch)
        else:
            batched_reprs = None
        adducts = torch.FloatTensor([item["adduct"] for item in batch])
        collision_engs = torch.FloatTensor([float(item["collision_energy"]) for item in batch])

        frag_targs = [item["frag_targs"] for item in batch]
        num_frags = torch.LongTensor([t.shape[0] for t in frag_targs])
        max_atoms = max(t.shape[1] for t in frag_targs)
        padded_frag_targs = []
        for t in frag_targs:
            if t.shape[1] < max_atoms:
                pad_width = max_atoms - t.shape[1]
                padded = torch.nn.functional.pad(t, (0, pad_width), value=0)
            else:
                padded = t
            padded_frag_targs.append(padded)

        atom_form_vecs = torch.cat([item["atom_form_vecs"] for item in batch], dim=0)
        root_form_vecs = torch.stack([item["root_form_vec"] for item in batch], dim=0)

        atom_hs_list = [item["atom_hs"] for item in batch]
        atom_hs_padded = torch.nn.utils.rnn.pad_sequence(atom_hs_list, batch_first=True)
        total_hs_list = torch.LongTensor([item["total_hs"] for item in batch])

        adj_matrices = [item["adj_matrix"] for item in batch]
        max_nodes = max(adj.shape[0] for adj in adj_matrices)
        padded_adj_matrices = []
        for adj in adj_matrices:
            if adj.shape[0] < max_nodes:
                pad_size = max_nodes - adj.shape[0]
                padded_adj = torch.nn.functional.pad(adj, (0, pad_size, 0, pad_size), value=0)
            else:
                padded_adj = adj
            padded_adj_matrices.append(padded_adj)
        adj_matrices_batch = torch.stack(padded_adj_matrices, dim=0)

        graphormer_inputs = [item["graphormer_input"] for item in batch if item["graphormer_input"] is not None]
        graphormer_batch = None
        if graphormer_inputs:
            max_nodes_gf = max(gf['x'].shape[0] for gf in graphormer_inputs)
            max_dist = max(gf['edge_input'].shape[2] for gf in graphormer_inputs)
            node_feat_dim = graphormer_inputs[0]['x'].shape[1]
            edge_feat_dim = graphormer_inputs[0]['attn_edge_type'].shape[2]
            batch_size = len(graphormer_inputs)

            x_batch = torch.zeros([batch_size, max_nodes_gf, node_feat_dim], dtype=torch.float32)
            attn_bias_batch = torch.full([batch_size, max_nodes_gf + 1, max_nodes_gf + 1], -99999, dtype=torch.float32)
            attn_edge_type_batch = torch.zeros([batch_size, max_nodes_gf, max_nodes_gf, edge_feat_dim], dtype=torch.float32)
            spatial_pos_batch = torch.zeros([batch_size, max_nodes_gf, max_nodes_gf], dtype=torch.long)
            in_degree_batch = torch.zeros([batch_size, max_nodes_gf], dtype=torch.long)
            out_degree_batch = torch.zeros([batch_size, max_nodes_gf], dtype=torch.long)
            edge_input_batch = torch.zeros([batch_size, max_nodes_gf, max_nodes_gf, max_dist, edge_feat_dim], dtype=torch.float32)

            for i, gf_input in enumerate(graphormer_inputs):
                num_nodes = gf_input['x'].shape[0]
                edge_dist = gf_input['edge_input'].shape[2]
                x_batch[i, :num_nodes] = gf_input['x']
                attn_bias_batch[i, :num_nodes+1, :num_nodes+1] = gf_input['attn_bias']
                attn_edge_type_batch[i, :num_nodes, :num_nodes] = gf_input['attn_edge_type']
                spatial_pos_batch[i, :num_nodes, :num_nodes] = gf_input['spatial_pos']
                in_degree_batch[i, :num_nodes] = gf_input['in_degree']
                out_degree_batch[i, :num_nodes] = gf_input['out_degree']
                edge_input_batch[i, :num_nodes, :num_nodes, :edge_dist] = gf_input['edge_input']

            graphormer_batch = {
                'x': x_batch,
                'attn_bias': attn_bias_batch,
                'attn_edge_type': attn_edge_type_batch,
                'spatial_pos': spatial_pos_batch,
                'in_degree': in_degree_batch,
                'out_degree': out_degree_batch,
                'edge_input': edge_input_batch,
            }
            num_atoms = torch.tensor([gf['num_atoms'] for gf in graphormer_inputs], dtype=torch.long)
        else:
            num_atoms = batched_reprs.batch_num_nodes()

        frag_targs = torch.cat(padded_frag_targs, dim=0)
        inten_targs_padded = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(item['inten_targs']).float() for item in batch], batch_first=True)
        precursor_mzs = torch.FloatTensor([j["precursor"] for j in batch])

        output = {
            "smis": smis,
            "names": names,
            "root_reprs": batched_reprs,
            "frag_targs": frag_targs,
            "adducts": adducts,
            "collision_engs": collision_engs,
            "weights": weights_padded,
            "adduct_mass_shifts": adduct_mass_shifts,
            "inten_targs": inten_targs_padded,
            "precursor_mzs": precursor_mzs,
            "num_frag_targs": num_frags,
            "root_form_vecs": root_form_vecs,
            "atom_form_vecs": atom_form_vecs,
            "adj_matrices": adj_matrices_batch,
            "atom_hs": atom_hs_padded,
            "total_hs": total_hs_list,
            "graphormer_input": graphormer_batch,
            "num_atoms": num_atoms,
        }
        return output
    
class GenDataset(DAGDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        magma_h5: Path,
        magma_map: dict,
        **kwargs,
    ):
        super().__init__(df, magma_h5, magma_map, **kwargs)
        self.read_tree = self.load_tree

    def __getitem__(self, idx: int):
        name = self.spec_names[idx]
        adduct = self.name_to_adducts[name]
        precursor = self.name_to_precursors[name]
        entry = self.read_fn(name)
        outdict = {"name": name, "adduct": adduct, "precursor": precursor}
        outdict.update(entry)
        return outdict

    @classmethod
    def get_collate_fn(cls):
        return GenDataset.collate_fn

    def load_tree(self, x):
        filekeys = self.name_to_dict[x]["magma_file"]
        if not type(self.magma_h5) is common.PredSpecDB:
            self.magma_h5 = common.PredSpecDB(self.magma_h5)
        spec = self.magma_h5.read(*filekeys)
        return spec

    @staticmethod
    def collate_fn(batch):
        names = [item["name"] for item in batch]
        smis = [item["root_smiles"] for item in batch]
        adducts = torch.FloatTensor([item["adduct"] for item in batch])
        precursor_mzs = torch.FloatTensor([item["precursor"] for item in batch])
        output = {
            "smis": smis,
            "names": names,
            "adducts": adducts,
            "precursor_mzs": precursor_mzs,
        }
        return output
