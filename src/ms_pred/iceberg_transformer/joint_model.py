""" joint_model. """
from collections import defaultdict
import ms_pred
import numpy as np
import pytorch_lightning as pl
import torch

import ms_pred.common as common
import ms_pred.iceberg_transformer.frag_model as frag_model
import ms_pred.iceberg_transformer.inten_model as inten_model
import ms_pred.iceberg_transformer.dataset as dataset
from ms_pred import nn_utils
from ms_pred.magma.fragmentation import FRAGMENT_ENGINE_PARAMS
import ms_pred.magma.fragmentation as fragmentation
from rdkit import Chem
MAX_BROKEN_BONDS = FRAGMENT_ENGINE_PARAMS['max_broken_bonds']

class JointModel(pl.LightningModule):
    def __init__(
        self,
        frag_model_obj: frag_model.FragOnlyModel,
        inten_model_obj: inten_model.IntenModel,
    ):
        """__init__.

        Args:
            frag_model_obj (frag_model.FragOnlyModel): frag_model_obj
            inten_model_obj (inten_model.IntenModel): inten_model_obj
        """

        super().__init__()
        self.frag_model_obj = frag_model_obj
        self.inten_model_obj = inten_model_obj
        self.inten_collate_fn = ms_pred.iceberg_transformer.dataset.IntenDataset.get_collate_fn()

        root_enc_gen = self.frag_model_obj.root_encode
        pe_embed_gen = self.frag_model_obj.pe_embed_k
        embed_elem_group_gen = self.frag_model_obj.embed_elem_group
        multi_hop_max_dist_frag = self.frag_model_obj.multi_hop_max_dist

        root_enc_inten = self.inten_model_obj.root_encode
        pe_embed_inten = self.inten_model_obj.pe_embed_k
        embed_elem_group_inten = self.inten_model_obj.embed_elem_group
        multi_hop_max_dist_inten = self.inten_model_obj.multi_hop_max_dist

        self.frag_tp = dataset.TreeProcessor(
            root_encode=root_enc_gen, multi_hop_max_dist=multi_hop_max_dist_frag, pe_embed_k=pe_embed_gen, embed_elem_group=embed_elem_group_gen,
        )

        self.inten_tp = dataset.TreeProcessor(
            root_encode=root_enc_inten, multi_hop_max_dist=multi_hop_max_dist_inten, pe_embed_k=pe_embed_inten, embed_elem_group=embed_elem_group_inten,
        )

    @classmethod
    def from_checkpoints(cls, frag_checkpoint, inten_checkpoint):
        """from_checkpoints.

        Args:
            frag_checkpoint
            inten_checkpoint
        """

        frag_model_obj = frag_model.FragOnlyModel.load_from_checkpoint(frag_checkpoint)
        inten_model_obj = inten_model.IntenModel.load_from_checkpoint(inten_checkpoint)
        return cls(frag_model_obj, inten_model_obj)

    def predict_mol(
        self,
        smi: str,
        collision_eng: float,
        adduct: str,
        device: str,
        binned_out: bool = False,
    ) -> dict:
        """predict_mol.

        Args:
            smi (str): smi
            adduct
            threshold (float): threshold
            device (str): device
            binned_out
        """
        self.eval()
        self.freeze()

        # Run tree gen model
        # Defines exact tree
        root_smi = smi
        if type(root_smi) is str:
            batched_input = False
            root_smi = [root_smi]
            collision_eng = [collision_eng]
            precursor_mz = [precursor_mz]
            adduct = [adduct]
        else:
            batched_input = True
        batch_size = len(root_smi)
        
        fragment_prediction = self.frag_model_obj.predict(
            root_smi=root_smi,
            collision_eng=collision_eng,
            adduct=adduct,
            device=device,
        )
        fragments = fragment_prediction['fragment_patterns']
        num_frags = fragment_prediction['patterns_count']
        
       
        to_tensor = lambda x: torch.tensor(x, device=device, dtype=torch.float) if x is not None else x
        adducts = to_tensor([common.ion2onehot_pos[a] for a in adduct])
        collision_engs = to_tensor(collision_eng)
        mols = [Chem.MolFromSmiles(rsmi) for rsmi in root_smi]
        graphormer_inputs = [self.inten_tp.create_graphormer_input(mol=m, multi_hop_max_dist=self.inten_tp.multi_hop_max_dist) for m in mols]
        num_atoms = torch.tensor([gf['num_atoms'] for gf in graphormer_inputs], dtype=torch.long).to(device)
        max_nodes_gf = max(gf['x'].shape[0] for gf in graphormer_inputs)
        max_dist = max(gf['edge_input'].shape[2] for gf in graphormer_inputs)
        node_feat_dim = graphormer_inputs[0]['x'].shape[1]
        edge_feat_dim = graphormer_inputs[0]['attn_edge_type'].shape[2]
        batch_size = len(graphormer_inputs)
        x_batch = torch.zeros([batch_size, max_nodes_gf, node_feat_dim], dtype=torch.float32)
        attn_bias_batch = torch.full([batch_size, max_nodes_gf + 1, max_nodes_gf + 1], -99999, dtype=torch.float32)
        attn_edge_type_batch = torch.zeros([batch_size, max_nodes_gf, max_nodes_gf, edge_feat_dim], dtype=torch.float32)
        spatial_pos_batch = torch.zeros([batch_size, max_nodes_gf, max_nodes_gf], dtype=torch.long)
        degree_batch = torch.zeros([batch_size, max_nodes_gf], dtype=torch.long)
        edge_input_batch = torch.zeros([batch_size, max_nodes_gf, max_nodes_gf, max_dist, edge_feat_dim], dtype=torch.float32)
        adj_matrices = [Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True) for mol in mols]
        adj_matrices = [torch.from_numpy(adj_matrix).float() for adj_matrix in adj_matrices]
        max_nodes = torch.max(num_atoms).item()
        padded_adj_matrices = []
        for adj in adj_matrices:
            if adj.shape[0] < max_nodes:
                pad_size = max_nodes - adj.shape[0]
                padded_adj = torch.nn.functional.pad(adj, (0, pad_size, 0, pad_size), value=0)
            else:
                padded_adj = adj
            padded_adj_matrices.append(padded_adj)
        adj_matrices_batch = torch.stack(padded_adj_matrices, dim=0).to(device)
        for i, gf_input in enumerate(graphormer_inputs):
            num_nodes = gf_input['x'].shape[0]
            edge_dist = gf_input['edge_input'].shape[2]
            x_batch[i, :num_nodes] = gf_input['x']
            attn_bias_batch[i, :num_nodes+1, :num_nodes+1] = gf_input['attn_bias']
            attn_edge_type_batch[i, :num_nodes, :num_nodes] = gf_input['attn_edge_type']
            spatial_pos_batch[i, :num_nodes, :num_nodes] = gf_input['spatial_pos']
            degree_batch[i, :num_nodes] = gf_input['degree']
            edge_input_batch[i, :num_nodes, :num_nodes, :edge_dist] = gf_input['edge_input']

        graphormer_batch = {
            'x': x_batch,
            'attn_bias': attn_bias_batch,
            'attn_edge_type': attn_edge_type_batch,
            'spatial_pos': spatial_pos_batch,
            'degree': degree_batch,
            'edge_input': edge_input_batch,
        }
        graphormer_batch = {k: v.to(device) for k, v in graphormer_batch.items()}
        adduct_mass_shift = torch.tensor([[
            common.ion2mass[mol_adduct],
            -common.ELECTRON_MASS if common.is_positive_adduct(mol_adduct) else common.ELECTRON_MASS,
        ] for mol_adduct in adduct]).to(device)
        engines = [fragmentation.FragmentEngine(mol_str=rsmi, mol_str_type="smiles", mol_str_canonicalized=True) for rsmi in root_smi]
        total_atom_masses = [torch.from_numpy(engine.atom_weights_h).to(device) for engine in engines]
        masses_padded = torch.nn.utils.rnn.pad_sequence(total_atom_masses, batch_first=True)
        root_forms = [common.form_from_smi(rsmi) for rsmi in root_smi]
        root_form_vecs = torch.stack([torch.from_numpy(common.formula_to_dense(root_form)) for root_form in root_forms]).to(device)
        atom_hs_list = [torch.tensor(engine.atom_hs) for engine in engines]
        atom_hs_padded = torch.nn.utils.rnn.pad_sequence(atom_hs_list, batch_first=True).to(device)
        total_hs = torch.LongTensor([engine.total_hs for engine in engines]).to(device)

        atom_symbols_batch = [engine.atom_symbols for engine in engines]
        atom_form_vecs_np = [[common.formula_to_dense(f"{s}H{h}") for s, h in zip(atom_symbols, num_hs)] for atom_symbols, num_hs in zip(atom_symbols_batch, atom_hs_list)]
        atom_form_vecs_padded = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(np.stack(atom_form_vec_np, axis=0)) for atom_form_vec_np in atom_form_vecs_np], batch_first=True).to(device)
        atom_form_vecs = nn_utils.pack_padded_tensor(atom_form_vecs_padded, lengths=num_atoms)
        frag_targs = nn_utils.pack_padded_tensor(fragments, lengths=num_frags)
        inten_preds = self.inten_model_obj.predict(
            None,
            collision_engs,
            adducts,
            masses_padded,
            adduct_mass_shift,
            root_form_vecs,
            atom_form_vecs,
            num_atoms,
            adj_matrices=adj_matrices_batch,
            frag_targs=frag_targs,
            num_frag_targs=num_frags,
            atom_hs=atom_hs_padded,
            total_hs=total_hs,
            graphormer_input=graphormer_batch,  # Pass Graphormer input if available
            binned_out=binned_out,
        )

        if binned_out:
            out = inten_preds
        else:
            out = {"spec": [], "frag": []}
            max_broken_bonds = self.inten_model_obj.max_broken_bonds
            num_shifts = len(adduct_mass_shift[0].reshape(-1)) * (1 + max_broken_bonds * 2)  # number of shifts,
            hydrogen_shift = torch.arange(-max_broken_bonds, max_broken_bonds + 1, device=device) * common.ELEMENT_TO_MASS["H"]
            # num_shifts = len(masses[0, 0, :, :].reshape(-1))  # number of shifts,
            #                                                   # (1 + h_shift * 2) * 2 if include_unshifted_mz==True
            # if not self.inten_model_obj.include_unshifted_mz:
            #     masses = masses[:, :, :1, :].contiguous()  # only keep m/z with adduct shift
            masses_frag = torch.sum(masses_padded.unsqueeze(1)*fragments, dim=-1)
            masses_frag_shifted = (
                masses_frag[:, :, None, None]
                + hydrogen_shift[None, None, None, :]
                + adduct_mass_shift[:, None, :, None]
            )
            for i, (inten_pred, mass, n) in \
                    enumerate(zip(inten_preds["spec"], masses_frag_shifted, num_frags)):
                out_mass = mass[:n].reshape(-1)
                out_inten = inten_pred.reshape(-1)
                out_frag = fragments[i, :n].repeat_interleave(num_shifts, dim=0)

                # add to output dict
                out["spec"].append(torch.stack((out_mass, out_inten), dim=1))
                
                out["frag"].append(out_frag)
                    
        if batched_input:
            return out
        else:
            return {k: v[0] for k, v in out.items()}
