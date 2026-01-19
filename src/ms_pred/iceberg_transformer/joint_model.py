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
import ms_pred.magma.fragmentation as fragmentation
from rdkit import Chem

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
    
    def predict(self, graphormer_input, num_atoms, adducts, collision_engs, root_form_vecs, masses, 
                adduct_mass_shifts, atom_form_vecs, adj_matrices, atom_hs, total_hs):
        breakpoints_pred = self.frag_model_obj(graphormer_input, num_atoms, adducts, collision_engs, root_form_vecs)
        inten_mol_embeddings = self.inten_model_obj.molecular_embedding(adducts, collision_engs, None, graphormer_input=graphormer_input)
        inten_root_tokens = inten_mol_embeddings["root_tokens"]
        inten_node_embeddings = inten_mol_embeddings["node_embeddings"]
        frag_logits = breakpoints_pred["frag_logits"][-1]
        frag_card_logits = breakpoints_pred["frag_card_logits"][-1]
        with torch.no_grad():
            breakpoints = self.frag_model_obj.breakpoint_inference(frag_logits, frag_card_logits, num_atoms)
            fragments, fragment_count = self.inten_model_obj.breakpoints_to_patterns(breakpoints, adj_matrices, num_atoms)
            fragments = nn_utils.pack_padded_tensor(fragments, fragment_count).bool()
        inten_pred_end_to_end = self.inten_model_obj.inten_calculation(
                                inten_root_tokens,
                                inten_node_embeddings,
                                fragments,
                                fragment_count,
                                root_form_vecs,
                                atom_form_vecs,
                                num_atoms, atom_hs, 
                                total_hs, 
                                adj_matrices, 
                                adduct_mass_shifts,
                                masses,
                            )
        frags_pred = {"fragments":fragments, "fragment_count":fragment_count}
        return {"breakpoints_pred":breakpoints_pred, "inten_pred_end_to_end":inten_pred_end_to_end, "frags_pred":frags_pred}

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
        to_tensor = lambda x: torch.tensor(x, device=device, dtype=torch.float) if x is not None else x
        adducts = to_tensor([common.ion2onehot_pos[a] for a in adduct])
        collision_engs = to_tensor(collision_eng)
        mols = [Chem.MolFromSmiles(rsmi) for rsmi in root_smi]
        graphormer_inputs = [self.frag_model_obj.tree_processor.create_graphormer_input(mol=m, multi_hop_max_dist=self.frag_model_obj.tree_processor.multi_hop_max_dist) for m in mols]
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
            'x': x_batch.to(device),
            'attn_bias': attn_bias_batch.to(device),
            'attn_edge_type': attn_edge_type_batch.to(device),
            'spatial_pos': spatial_pos_batch.to(device),
            'degree': degree_batch.to(device),
            'edge_input': edge_input_batch.to(device),
        }
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
        with torch.no_grad():
            pred = self.predict(
                graphormer_batch,
                num_atoms,
                adducts,
                collision_engs, 
                root_form_vecs=root_form_vecs,
                masses=masses_padded,
                adduct_mass_shifts=adduct_mass_shift,
                atom_form_vecs=atom_form_vecs,
                adj_matrices=adj_matrices_batch,
                atom_hs=atom_hs_padded,
                total_hs=total_hs,
                training=False
            )
            out=pred["inten_pred_end_to_end"]
            frags_pred = pred["frags_pred"]
            output = out["output"]
            out_preds_binned = out["output_binned"]
            out_preds = [
                pred[:num_frag, :]
                for pred, num_frag in zip(output, frags_pred["fragment_count"])
            ]

            if binned_out:
                inten_preds = {
                    "spec": out_preds_binned,
                }
                out = inten_preds
            else:
                inten_preds = {
                    "spec": out_preds,
                }
                out = {"spec": [], "frag": []}
                fragments = nn_utils.pad_packed_tensor(frags_pred["fragments"], frags_pred["fragment_count"], 0)
                max_broken_bonds = self.inten_model_obj.max_broken_bonds
                num_shifts = len(adduct_mass_shift[0].reshape(-1)) * (1 + max_broken_bonds * 2)  # number of shifts,
                hydrogen_shift = torch.arange(-max_broken_bonds, max_broken_bonds + 1, device=device) * common.ELEMENT_TO_MASS["H"]
                # num_shifts = len(masses[0, 0, :, :].reshape(-1))  # number of shifts,
                #                                                   # (1 + h_shift * 2) * 2 if include_unshifted_mz==True
                masses_frag = torch.sum(masses_padded.unsqueeze(1)*fragments, dim=-1)
                masses_frag_shifted = (
                    masses_frag[:, :, None, None]
                    + hydrogen_shift[None, None, None, :]
                    + adduct_mass_shift[:, None, :, None]
                )
                for i, (inten_pred, mass, n) in \
                        enumerate(zip(inten_preds["spec"], masses_frag_shifted, frags_pred["fragment_count"])):
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