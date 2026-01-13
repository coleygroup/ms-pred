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
            inten_weight: float = 10.0,
            frag_weight: float = 1.0,
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
                                            max_frags=self.max_breakpoints, embed_adduct=self.embed_adduct, embed_collision=self.embed_collision, 
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
        inten_pred_magma = self.inten_predictor(None, collision_engs, adducts,
                                masses,
                                adduct_mass_shifts,
                                root_form_vecs,
                                atom_form_vecs,
                                num_atoms,
                                adj_matrices=adj_matrices,
                                frag_targs=frag_targs,
                                num_frag_targs=num_frag_targs,
                                atom_hs=atom_hs,
                                total_hs=total_hs,
                                graphormer_input=graphormer_input
                            ) if training else None
        frag_logits = breakpoints_pred["frag_logits"][-1]
        frag_card_logits = breakpoints_pred["frag_card_logits"][-1]
        with torch.no_grad():
            breakpoints = self.frag_predictor.breakpoint_inference(frag_logits, frag_card_logits, num_atoms)
            fragments, fragment_count = self.frag_predictor.breakpoints_to_patterns(breakpoints, adj_matrices, num_atoms)
            fragments = nn_utils.pack_padded_tensor(fragments, fragment_count).bool()
        inten_pred_end_to_end = self.inten_predictor(None, collision_engs, adducts,
                                masses,
                                adduct_mass_shifts,
                                root_form_vecs,
                                atom_form_vecs,
                                num_atoms,
                                adj_matrices=adj_matrices,
                                frag_targs=fragments,
                                num_frag_targs=fragment_count,
                                atom_hs=atom_hs,
                                total_hs=total_hs,
                                graphormer_input=graphormer_input
                            )
        return {"breakpoints_pred":breakpoints_pred, "inten_pred_end_to_end":inten_pred_end_to_end, "inten_pred_magma":inten_pred_magma}
    
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
            frag_targs=batch["frag_targs"],
            num_frag_targs=batch["num_frag_targs"],
            atom_hs=batch["atom_hs"],
            total_hs=batch["total_hs"],
            training=name=="train"
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
    