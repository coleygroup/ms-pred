"""Fragmentation-only model using Graphormer/GNN encoder and Transformer-based decoder.

Predicts fragment masks per molecule without intensity modeling.
"""
from typing import Dict, Any
import logging

import torch
import torch.nn as nn
import pytorch_lightning as pl
import pygmtools as pygm
import dgl.nn as dgl_nn

import ms_pred.magma.fragmentation as fragmentation
import ms_pred.nn_utils as nn_utils
from ms_pred.graphormer.graphormer_graph_encoder import GraphormerGraphEncoder
import torch.nn.functional as F
import copy
import ms_pred.common as common
from LinSATNet import linsat_layer, init_constraints
import math
import numpy as np
from ms_pred.iceberg_transformer.dataset import TreeProcessor
import dgl
from rdkit import Chem  # type: ignore
from ms_pred.iceberg_transformer.inten_model import IntenModel
from ms_pred.iceberg_transformer.frag_model import FragOnlyModel
from ms_pred.common import CompositeMassSpec, bin_spectra
import functools

class JointModel(pl.LightningModule):
    def __init__(self, frag_hidden_size: int = 512,
            frag_graphormer_layers: int = 6,
            frag_decoder_layers: int = 3,
            frag_encoder_layers: int = 0,
            frag_dropout: float = 0.1,
            node_feats: int = 128,
            edge_feats: int = 12,
            multi_hop_max_dist: int = 5,
            num_edge_dis: int = 10,
            max_breakpoints: int = 40,
            embed_adduct: bool = False,
            embed_collision: bool = False,
            embed_elem_group: bool = False,
            encode_forms: bool = False,
            linsat_tau: float = 0.01,
            max_broken_bonds: int = 6,
            pe_embed_k: int = 0,
            enable_aux_loss: bool = False,
            enable_decoder_norm: bool = False,
            inten_hidden_size: int = 512,
            inten_graphormer_layers: int = 2,
            inten_decoder_layers: int = 3,
            inten_encoder_layers: int = 3,
            inten_dropout: float = 0,
            inten_loss_fn: str = "cosine",
            binned_targs:bool = False,
            sk_tau: float = 0.01,
            ppm_tol: float = 20,
            contr_weight: float = 1.0,
            contr_loss_fn: str = "entropy",
            inten_weight: float = 1,
            frag_weight: float = 0.1,
            magma_warmup_steps: int = 10000,
            magma_decay_rate: float = 0.9,
            magma_decay_steps: int = 2000,
            lr: float = 1e-4,
            lr_decay_rate: float = 0.9,
            weight_decay: float = 0,
            warmup: int = 1000,
            **kwargs,
        ):
        super().__init__()
        self.save_hyperparameters()
        self.frag_graphormer_layer = frag_graphormer_layers
        self.frag_hidden_size = frag_hidden_size
        self.frag_graphormer_layers = frag_graphormer_layers
        self.frag_decoder_layers = frag_decoder_layers
        self.frag_encoder_layers = frag_encoder_layers
        self.frag_dropout = frag_dropout
        self.node_feats = node_feats
        self.edge_feats = edge_feats
        self.multi_hop_max_dist = multi_hop_max_dist
        self.num_edge_dis = num_edge_dis
        self.max_breakpoints = max_breakpoints
        self.embed_adduct = embed_adduct
        self.embed_collision = embed_collision
        self.embed_elem_group = embed_elem_group
        self.encode_forms = encode_forms
        self.linsat_tau = linsat_tau
        self.max_broken_bonds = max_broken_bonds
        self.pe_embed_k = pe_embed_k
        self.enable_aux_loss = enable_aux_loss
        self.enable_decoder_norm = enable_decoder_norm

        self.inten_hidden_size = inten_hidden_size
        self.inten_graphormer_layers = inten_graphormer_layers
        self.inten_decoder_layers = inten_decoder_layers
        self.inten_encoder_layers = inten_encoder_layers
        self.inten_dropout = inten_dropout
        self.inten_loss_fn = inten_loss_fn
        self.binned_targs = binned_targs
        self.sk_tau = sk_tau
        self.ppm_tol = ppm_tol
        self.contr_weight = contr_weight
        self.contr_loss_fn = contr_loss_fn
        self.frag_predictor = FragOnlyModel(hidden_size=self.frag_hidden_size, 
                                            layers=self.frag_graphormer_layers, decoder_layers=self.frag_decoder_layers,
                                            encoder_layers=self.frag_encoder_layers, dropout=self.frag_dropout, node_feats=self.node_feats, 
                                            edge_feats=self.edge_feats, multi_hop_max_dist=self.multi_hop_max_dist, num_edge_dis=self.num_edge_dis,
                                            max_breakpoints=self.max_breakpoints, embed_adduct=self.embed_adduct, embed_collision=self.embed_collision, 
                                            embed_elem_group=self.embed_elem_group, encode_forms=self.encode_forms, linsat_tau=self.linsat_tau,
                                            max_broken_bonds=self.max_broken_bonds, pe_embed_k=self.pe_embed_k,
                                            enable_aux_loss=self.enable_aux_loss, enable_decoder_norm=self.enable_decoder_norm, root_encode="graphormer")
        self.inten_predictor = IntenModel(hidden_size=self.inten_hidden_size, layers=self.inten_graphormer_layers, decoder_layers=self.inten_decoder_layers, 
                                          encoder_layers=self.inten_encoder_layers, dropout=self.inten_dropout, node_feats=self.node_feats, edge_feats=self.edge_feats, pe_embed_k=self.pe_embed_k,
                                          max_broken_bonds=self.max_broken_bonds, root_encode="graphormer", embed_adduct=self.embed_adduct, embed_collision=self.embed_collision, 
                                          embed_elem_group=self.embed_elem_group, encode_forms=self.encode_forms, loss_fn=self.inten_loss_fn, binned_targs=self.binned_targs, sk_tau=self.sk_tau, 
                                          ppm_tol=self.ppm_tol, multi_hop_max_dist=self.multi_hop_max_dist, num_edge_dis=self.num_edge_dis, contr_weight=self.contr_weight, contr_loss_fn=self.contr_loss_fn)
        self.inten_weight = inten_weight
        self.frag_weight = frag_weight
        self.magma_warmup_steps = magma_warmup_steps
        self.magma_decay_rate = magma_decay_rate
        self.magma_decay_steps = magma_decay_steps
        self.step=0
        self.lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.weight_decay = weight_decay
        self.warmup = warmup

    def forward(self, graphormer_input, num_atoms, adducts, collision_engs, root_form_vecs, masses, 
                adduct_mass_shifts, atom_form_vecs, adj_matrices, atom_hs, total_hs, 
                training=False, frag_targs=None, num_frag_targs=None):
        breakpoints_pred = self.frag_predictor(graphormer_input, num_atoms, adducts, collision_engs, root_form_vecs)
        inten_mol_embeddings = self.inten_predictor.molecular_embedding(adducts, collision_engs, None, graphormer_input=graphormer_input)
        inten_root_tokens = inten_mol_embeddings["root_tokens"]
        inten_node_embeddings = inten_mol_embeddings["node_embeddings"]
        inten_pred_magma = self.inten_predictor.inten_calculation(
                                inten_root_tokens,
                                inten_node_embeddings,
                                frag_targs,
                                num_frag_targs,
                                root_form_vecs,
                                atom_form_vecs,
                                num_atoms, atom_hs, 
                                total_hs, 
                                adj_matrices, 
                                adduct_mass_shifts,
                                masses,
                            ) if training else None
        frag_logits = breakpoints_pred["frag_logits"][-1]
        frag_card_logits = breakpoints_pred["frag_card_logits"][-1]
        with torch.no_grad():
            breakpoints = self.frag_predictor.breakpoint_inference(frag_logits, frag_card_logits, num_atoms)
            fragments, fragment_count = self.frag_predictor.breakpoints_to_patterns(breakpoints, adj_matrices, num_atoms)
            fragments = nn_utils.pack_padded_tensor(fragments, fragment_count).bool()
        inten_pred_end_to_end = self.inten_predictor.inten_calculation(
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
        return {"breakpoints_pred":breakpoints_pred, "inten_pred_end_to_end":inten_pred_end_to_end, "inten_pred_magma":inten_pred_magma, "frags_pred":frags_pred}
    

    def _common_step(self, batch, name="train"):
        pred = self.forward(
            graphormer_input=batch.get("graphormer_input"),
            num_atoms=batch["num_atoms"],
            adducts=batch["adducts"],
            collision_engs=batch["collision_engs"],
            root_form_vecs=batch["root_form_vecs"],
            masses=batch["masses"],
            adduct_mass_shifts=batch["adduct_mass_shifts"],
            atom_form_vecs=batch["atom_form_vecs"],
            adj_matrices=batch["adj_matrices"],
            atom_hs=batch["atom_hs"],
            total_hs=batch["total_hs"],
            training=name=="train",
            frag_targs=batch["frag_targs"] if name=="train" else None,
            num_frag_targs=batch["num_frag_targs"] if name=="train" else None,
        )
        if self.binned_targs:
            pred_inten = pred["inten_pred_end_to_end"]["output_binned"]
        else:
            pred_inten = pred["inten_pred_end_to_end"]["output"]
            pred_inten = torch.stack((batch["masses"], pred_inten), dim=-1)
            pred_inten = pred_inten.reshape(pred_inten.shape[0], -1, 2)  # B x (Out * Mass shifts) x 2
        batch_size = len(batch["names"])
        if name != "train":
            loss_fn = functools.partial(self.inten_predictor.loss_fn, use_hun=True)  # use hungarian in val and test
            loss = loss_fn(pred_inten, batch["inten_targs"], parent_mass=batch["precursor_mzs"])
            loss = {k: v.mean() for k, v in loss.items()}        
            self.log(
                f"{name}_loss", loss["loss"].item(), batch_size=batch_size, on_epoch=True
            )            
            return loss
        else:
            if self.binned_targs:
                pred_inten_magma = pred["inten_pred_magma"]["output_binned"]
            else:
                pred_inten_magma = pred["inten_pred_end_to_end"]["output"]
                pred_inten_magma = torch.stack((batch["masses"], pred_inten_magma), dim=-1)
                pred_inten_magma = pred_inten_magma.reshape(pred_inten_magma.shape[0], -1, 2)  # B x (Out * Mass shifts) x 2
            inten_loss_fn = self.inten_predictor.loss_fn
            end_to_end_inten_loss = inten_loss_fn(pred_inten, batch["inten_targs"], parent_mass=batch["precursor_mzs"])["loss"].mean()
            magma_inten_loss = inten_loss_fn(pred_inten_magma, batch["inten_targs"], parent_mass=batch["precursor_mzs"])["loss"].mean()
            breakpoints_pred = pred["breakpoints_pred"]
            if self.enable_aux_loss:
                frag_loss = 0
                for i in range(self.frag_decoder_layers):
                    frag_loss += self.frag_predictor.frag_loss(
                        breakpoints_pred["frag_logits"][i], breakpoints_pred["frag_card_logits"][i], 
                        batch["frag_targs"], batch["num_frag_targs"], batch["num_atoms"], batch["adj_matrices"],
                    ) * 0.9**(self.frag_decoder_layers-1-i)
            else:
                frag_loss = self.frag_predictor.frag_loss(
                    breakpoints_pred["frag_logits"][-1], breakpoints_pred["frag_card_logits"][-1], 
                    batch["frag_targs"], batch["num_frag_targs"], batch["num_atoms"], batch["adj_matrices"],
                )
            self.step += 1
            self.log("train_end_to_end_inten_loss", end_to_end_inten_loss, batch_size=batch_size, on_epoch=True)
            self.log("train_magma_inten_loss", magma_inten_loss, batch_size=batch_size, on_epoch=True)
            self.log("train_frag_loss", frag_loss, batch_size=batch_size, on_epoch=True)
            
            magma_weight = self.magma_weight_scheduler()
            loss = self.inten_weight * (magma_weight * magma_inten_loss + (1-magma_weight)*end_to_end_inten_loss) + self.frag_weight * frag_loss
            self.log("train_loss", loss, batch_size=batch_size, on_epoch=True)
            return {"loss":loss}

    def training_step(self, batch, batch_idx):
        """training_step."""
        return self._common_step(batch, name="train")

    def validation_step(self, batch, batch_idx):
        """validation_step."""
        return self._common_step(batch, name="val")

    def test_step(self, batch, batch_idx):
        """test_step."""
        return self._common_step(batch, name="test")
    
    def magma_weight_scheduler(self):
        if self.step >= self.magma_warmup_steps:
            # Adjust
            step = self.step - self.magma_warmup_steps
            weight = self.magma_decay_rate ** (step // self.magma_decay_steps)
        else:
            weight = 1
        return weight

    def configure_optimizers(self):
        decay_params, no_decay_params = [], []

        def _is_no_decay_param(name: str, param: torch.nn.Parameter) -> bool:
            name_l = name.lower()
            return param.ndim == 1 or name.endswith("bias") or ("norm" in name_l) or ("embed" in name_l)

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if _is_no_decay_param(name, param):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )
        scheduler = nn_utils.build_lr_scheduler(optimizer=optimizer, 
                    lr_decay_rate=self.lr_decay_rate, warmup=self.warmup)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "frequency": 1, "interval": "step"}}

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        # For LambdaLR, just call step() without arguments
        scheduler.step()

    def predict_mol(self, smi, collision_eng, adduct, device, binned_out=False):
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
        graphormer_inputs = [self.frag_predictor.tree_processor.create_graphormer_input(mol=m, multi_hop_max_dist=self.frag_predictor.tree_processor.multi_hop_max_dist) for m in mols]
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
            pred = self.forward(
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
                max_broken_bonds = self.max_broken_bonds
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
    
    def predict_inten(self, graphormer_input, num_atoms, adducts, collision_engs, root_form_vecs, masses, 
                adduct_mass_shifts, atom_form_vecs, adj_matrices, atom_hs, total_hs, binned_out=False):
        predict_obj = self.forward(graphormer_input, 
                        num_atoms, adducts, 
                        collision_engs, 
                        root_form_vecs, 
                        masses, 
                        adduct_mass_shifts, 
                        atom_form_vecs, 
                        adj_matrices, 
                        atom_hs, 
                        total_hs
                    )
        out = predict_obj["inten_pred_end_to_end"]
        num_frag_targs = predict_obj["frags_pred"]["fragment_count"]
        output = out["output"]
        out_preds_binned = out["output_binned"]
        out_preds = [
            pred[:num_frag, :]
            for pred, num_frag in zip(output, num_frag_targs)
        ]

        if binned_out:
            out_dict = {
                "spec": out_preds_binned,
            }
        else:
            out_dict = {
                "spec": out_preds,
            }
        return out_dict
    