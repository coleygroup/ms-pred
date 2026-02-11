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

import ms_pred.nn_utils as nn_utils
from ms_pred.graphormer.graphormer_graph_encoder import GraphormerGraphEncoder
import torch.nn.functional as F
import ms_pred.magma.fragmentation as fragmentation
import ms_pred.common as common
from ms_pred.iceberg_transformer.dataset import TreeProcessor
import numpy as np

import copy


class FragOnlyModel(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int = 512,
        layers: int = 6,
        decoder_layers: int = 3,
        encoder_layers: int = 0,
        dropout: float = 0.1,
        learning_rate: float = 7e-4,
        lr_decay_rate: float = 1.0,
        weight_decay: float = 0.0,
        warmup: int = 1000,
        root_encode: str = "graphormer",  # or "gnn"
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
        gamma: float = 2,
        include_unassigned: bool = False,
        max_broken_bonds: int = 6,
        pe_embed_k: int = 0,
        enable_aux_loss: bool = False,
        enable_decoder_norm: bool = False,
        sk_tau: float = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.layers = layers
        self.decoder_layers = decoder_layers
        self.encoder_layers = encoder_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.root_encode = root_encode
        self.max_frags = max_breakpoints
        self.nhead = 8
        self.embed_adduct = embed_adduct
        self.embed_collision = embed_collision
        self.embed_elem_group = embed_elem_group
        self.encode_forms = encode_forms
        self.linsat_tau = linsat_tau
        self.gamma = gamma
        self.include_unassigned = include_unassigned
        self.max_broken_bonds = max_broken_bonds
        self.output_size = (self.max_broken_bonds) * 2 + 1
        self.pe_embed_k = pe_embed_k
        self.multi_hop_max_dist = multi_hop_max_dist
        self.tree_processor = TreeProcessor(
            pe_embed_k=pe_embed_k,
            root_encode=root_encode,
            embed_elem_group=embed_elem_group,
            multi_hop_max_dist=multi_hop_max_dist,
        )
        self.enable_aux_loss = enable_aux_loss

        adduct_shift = 0
        if self.embed_adduct:
            adduct_types = len(set(common.ion2onehot_pos.values()))
            onehot_types = torch.eye(adduct_types)
            if self.embed_elem_group:
                adduct_modes = len(set([j for i in common.ion_pos2extra_multihot.values() for j in i]))
                multihot_modes = torch.zeros((adduct_types, adduct_modes))
                for i in range(adduct_types):
                    for j in common.ion_pos2extra_multihot[i]:
                        multihot_modes[i, j] = 1
                adduct_embedder = torch.cat((onehot_types, multihot_modes), dim=-1)
                self.adduct_embedder = nn.Parameter(adduct_embedder.float())
                self.adduct_embedder.requires_grad = False
                adduct_shift = adduct_types + adduct_modes
            else:
                self.adduct_embedder = nn.Parameter(onehot_types.float())
                self.adduct_embedder.requires_grad = False
                adduct_shift = adduct_types
                collision_shift = 0

        if self.embed_collision:
            pe_dim = common.COLLISION_PE_DIM
            pe_scalar = common.COLLISION_PE_SCALAR
            pe_power = 2 * torch.arange(pe_dim // 2) / pe_dim
            self.collision_embedder_denominators = nn.Parameter(torch.pow(pe_scalar, pe_power))
            self.collision_embedder_denominators.requires_grad = False
            collision_shift = pe_dim

            self.collision_embed_merged = nn.Parameter(torch.zeros(pe_dim))
            self.collision_embed_merged.requires_grad = False
        
        self.formula_in_dim = 0
        if self.encode_forms:
            self.embedder = nn_utils.get_embedder("abs-sines")
            self.formula_dim = common.NORM_VEC.shape[0]

            # Calculate formula dim
            self.formula_in_dim = self.formula_dim * self.embedder.num_dim
            self.formula_mapper = nn.Linear(self.formula_in_dim+self.hidden_size, self.hidden_size)
        # Root encoder
        if root_encode == "graphormer":
            self.root_module = GraphormerGraphEncoder(
                num_atom_features=node_feats+adduct_shift+collision_shift,
                num_degree=8,
                num_edge_features=edge_feats,
                num_spatial=1025,
                num_edge_dis=num_edge_dis,
                edge_type="multi_hop",
                multi_hop_max_dist=multi_hop_max_dist,
                num_encoder_layers=layers,
                embedding_dim=hidden_size,
                ffn_embedding_dim=4*hidden_size,
                num_attention_heads=self.nhead,
                dropout=dropout,
                attention_dropout=dropout,
                activation_dropout=dropout,
                apply_graphormer_init=True,
            )
        elif root_encode == "gnn":
            self.root_module = nn_utils.MoleculeGNN(
                hidden_size=hidden_size,
                num_step_message_passing=layers,
                set_transform_layers=0,
                mpnn_type="GGNN",
                gnn_node_feats=node_feats+adduct_shift+collision_shift,
                gnn_edge_feats=edge_feats,
                dropout=dropout,
            )
            self.pool = dgl_nn.AvgPooling()
        else:
            raise ValueError(f"Unsupported root_encode: {root_encode}")
        
        # Fragment query tokens and decoder
        self.enable_decoder_norm = enable_decoder_norm
        self.fragment_decoder = nn_utils.SlotDecoder(
            hidden_dim=hidden_size,
            num_slots=max_breakpoints,
            nhead=self.nhead,
            num_layers=self.decoder_layers,
            dropout=dropout,
            enable_norm=self.enable_decoder_norm
        )
        if self.encoder_layers > 0:
            fragment_encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=self.nhead,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.fragment_encoder = nn.TransformerEncoder(
                fragment_encoder_layer,
                num_layers=self.encoder_layers,
            )
        buckets = torch.DoubleTensor(np.linspace(0, 1500, 15000))
        self.inten_buckets = nn.Parameter(buckets)
        self.inten_buckets.requires_grad = False
        
        # self.frag_logit_mapper = nn_utils.MLPBlocks(hidden_size, hidden_size, dropout, 3, use_residuals=True)
        # self.frag_logit_mapper = nn_utils.MultiHeadCrossAttentionLogits(self.hidden_size, self.nhead)
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss(reduction="none")

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
            lr=self.learning_rate,
        )
        scheduler = nn_utils.build_lr_scheduler(optimizer=optimizer, 
                    lr_decay_rate=self.lr_decay_rate, warmup=self.warmup)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "frequency": 1, "interval": "step"}}

    def forward(self, graphormer_input, num_atoms, adducts, collision_engs, root_form_vecs, root_reprs=None) -> Dict[str, torch.Tensor]:
        device = num_atoms.device
        batch_size = num_atoms.shape[0]
        embed_adducts = self.adduct_embedder[adducts.long()]

        if self.root_encode == "graphormer":
            assert graphormer_input is not None, "graphormer_input required for graphormer root_encode"
            original_node_features=node_features=graphormer_input['x']  # [B, max_nodes, num_features]
            max_nodes = node_features.shape[1]
            if self.embed_adduct:
                    # embed_adducts: [B, adduct_dim]
                    # Expand to [B, max_nodes, adduct_dim]
                embed_adducts_expanded = embed_adducts.unsqueeze(1).expand(batch_size, max_nodes, -1)
                node_features = torch.cat([node_features, embed_adducts_expanded], dim=-1)
                
                # Add collision embeddings if enabled
            if self.embed_collision:
                embed_collision = torch.cat(
                    (torch.sin(collision_engs.unsqueeze(1) / self.collision_embedder_denominators.unsqueeze(0)),
                    torch.cos(collision_engs.unsqueeze(1) / self.collision_embedder_denominators.unsqueeze(0))),
                    dim=1
                )
                
                embed_collision = torch.where(  # handle entries without collision energy (== nan)
                    torch.isnan(embed_collision), self.collision_embed_merged.unsqueeze(0), embed_collision
                )   
                # Expand collision embeddings to all nodes in each molecule
                # embed_collision: [B, collision_dim]
                # Expand to [B, max_nodes, collision_dim]
                embed_collision_expanded = embed_collision.unsqueeze(1).expand(batch_size, max_nodes, -1)
                node_features = torch.cat([node_features, embed_collision_expanded], dim=-1)
        
            graphormer_input['x'] = node_features
            inner_states, graph_rep = self.root_module(graphormer_input)
            final_layer_output = inner_states[-1]  # [T, B, H]
            node_embeddings = final_layer_output[1:].transpose(0, 1)  # [B, max_nodes, H]
            root_tokens = graph_rep.unsqueeze(1)  # [B,1,H]
            max_nodes = node_embeddings.shape[1]
            graphormer_input['x']=original_node_features
        else:
            pass

        node_mask = torch.arange(max_nodes, device=device).unsqueeze(0) >= num_atoms.unsqueeze(1)  # [B,max_nodes]
        frag_mask = F.pad(node_mask, (1, 0, 0, 0), mode="constant", value=0).bool()
        if self.encode_forms:
            encoded_form = self.embedder(root_form_vecs)[:, None, :]
            root_tokens = self.formula_mapper(torch.cat((root_tokens, encoded_form), dim=-1))
        frag_vecs = self.fragment_decoder(root_tokens, node_embeddings, memory_key_padding_mask=frag_mask)
        if self.encoder_layers > 0:
            frag_vecs_flatten = frag_vecs.reshape(-1, self.max_frags, self.hidden_size)
            frag_vecs_encoded = self.fragment_encoder(frag_vecs_flatten)
            frag_vecs = frag_vecs_encoded.reshape(self.decoder_layers, batch_size, self.max_frags, self.hidden_size)
        frag_logits = torch.einsum("nbij,bkj->nbik", frag_vecs, node_embeddings)
        frag_logits = frag_logits.masked_fill(node_mask.unsqueeze(1).unsqueeze(0), -99999)
        return {"frag_logits": frag_logits}



    def frag_loss(
        self,
        frags_predicted: torch.Tensor,
        frag_targs: torch.Tensor,
        num_frag_targs: torch.Tensor,
        num_atoms: torch.Tensor,
    ) -> torch.Tensor:
        # frags_predicted: [B, max_frags, max_nodes]
        # frag_targs: packed [sum_frags, max_nodes], num_frag_targs: [B]
        B, max_frags, max_nodes = frags_predicted.shape
        
        # frag_targs_padded = nn_utils.pad_packed_tensor(
        #     frag_targs, num_frag_targs, False
        # )[:, :-1, :]  # [B, max_targs-1, max_nodes]
        frag_targs_padded = nn_utils.pad_packed_tensor(
            frag_targs, num_frag_targs, False
        ).float()

        frags_predicted_expanded = frags_predicted.unsqueeze(2).expand(-1, -1, frag_targs_padded.shape[1], -1)
        frag_targs_expanded = frag_targs_padded.unsqueeze(1).expand(*frags_predicted_expanded.shape)
        cost = self.bce_loss(
            frags_predicted_expanded,
            frag_targs_expanded,
        ).mean(dim=-1)  # [B, max_targs, max_breakpoints]
        
        assign = pygm.hungarian(
            -cost, backend="pytorch", n2=num_frag_targs
        )  # [B, max_targs, max_breakpoints]
        
        frags_predicted_assigned = torch.matmul(frags_predicted.transpose(1, 2), assign).transpose(1, 2)
        loss = torch.sum(self.bce_loss(frags_predicted_assigned, frag_targs_padded), dim=-1)  # [B, max_targs]
        max_targs = frag_targs_padded.shape[1]

        # loss = torch.sum(self.binary_focal_loss(node_rank_assigned, frag_targs_padded, num_atoms), dim=-1)+self.cross_entropy(frag_card_predicted, frag_cards_targs)
        frag_targs_mask = num_frag_targs[:, None] <= torch.arange(max_targs, device=loss.device)[None, :]
        loss = torch.sum(loss.masked_fill(frag_targs_mask, 0), dim=-1)/(num_frag_targs*num_atoms)
        return torch.mean(loss)

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        out = self.forward(batch["graphormer_input"], batch["num_atoms"], batch["adducts"], batch["collision_engs"], batch["root_form_vecs"])
        if self.enable_aux_loss:
            loss = 0
            for i in range(out["frag_logits"].shape[0]):
                frag_preds = self.sigmoid(out["frag_logits"][i])
                loss += self.frag_loss(
                frag_preds, batch["frag_targs"], batch["num_frag_targs"], batch["num_atoms"]
            ) * 0.9**(self.decoder_layers-1-i)
        else:
            frag_preds = self.sigmoid(out["frag_logits"][-1])
            loss = self.frag_loss(
                frag_preds, batch["frag_targs"], batch["num_frag_targs"], batch["num_atoms"]
            )
        self.log(
            "train_loss", loss.item(), prog_bar=True, on_epoch=True, batch_size=len(batch["names"])
        )
        return {"loss": loss}

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        out = self.forward(batch["graphormer_input"], batch["num_atoms"], batch["adducts"], batch["collision_engs"], batch["root_form_vecs"])
        frag_preds = self.sigmoid(out["frag_logits"][-1])

        loss = self.frag_loss(
            frag_preds, batch["frag_targs"], batch["num_frag_targs"], batch["num_atoms"]
        )
        patterns = (frag_preds > 0.1).float()
        
        frag_targs_padded = nn_utils.pad_packed_tensor(batch["frag_targs"], batch["num_frag_targs"], 0)
        patterns_count = torch.full_like(batch["num_frag_targs"], self.max_frags)
        metrics = self.pattern_match_metrics(patterns, frag_targs_padded, batch["num_frag_targs"], patterns_count = patterns_count)
        mz_metrics = self.mz_metrics(patterns, batch["inten_targs"], batch["masses"], batch["atom_hs"], patterns_count, batch["adduct_mass_shifts"])
        recall, precision = metrics["recall"], metrics["precision"]
        mz_recall, mz_precision = mz_metrics["recall"], mz_metrics["precision"]
        self.log(
            "val_loss", loss.item(), prog_bar=True, on_epoch=True, batch_size=len(batch["names"])
        )
        self.log(
            "val_recall", recall.item(), prog_bar=True, on_epoch=True, batch_size=len(batch["names"])
        )
        self.log(
            "val_precision", precision.item(), prog_bar=True, on_epoch=True, batch_size=len(batch["names"])
        )
        self.log(
            "val_mz_recall", mz_recall.item(), prog_bar=True, on_epoch=True, batch_size=len(batch["names"])
        )
        self.log(
            "val_mz_precision", mz_precision.item(), prog_bar=True, on_epoch=True, batch_size=len(batch["names"])
        )
        return {"loss": loss}

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        out = self.forward(batch["graphormer_input"], batch["num_atoms"], batch["adducts"], batch["collision_engs"], batch["root_form_vecs"])
        frag_preds = self.sigmoid(out["frag_logits"][-1])
        loss = self.frag_loss(
            frag_preds, batch["frag_targs"], batch["num_frag_targs"], batch["num_atoms"]
        )
        patterns = (frag_preds > 0.1).float()
        unique_patterns_test = torch.unique(patterns[0], dim=0)
        for p in frag_preds[0]:
            logging.debug(f"Predicted fragment: {p}")
        for pattern in unique_patterns_test:
            logging.debug(f"Predicted pattern: {pattern}")
        assert False
        frag_targs_padded = nn_utils.pad_packed_tensor(batch["frag_targs"], batch["num_frag_targs"], 0)
        patterns_count = torch.full_like(batch["num_frag_targs"], self.max_frags)
        metrics = self.pattern_match_metrics(patterns, frag_targs_padded, batch["num_frag_targs"], patterns_count = patterns_count)
        mz_metrics = self.mz_metrics(patterns, batch["inten_targs"], batch["masses"], batch["atom_hs"], patterns_count, batch["adduct_mass_shifts"])
        recall, precision = metrics["recall"], metrics["precision"]
        mz_recall, mz_precision = mz_metrics["recall"], mz_metrics["precision"]
        self.log(
            "test_loss", loss.item(), prog_bar=True, on_epoch=True, batch_size=len(batch["names"])
        )
        self.log(
            "test_recall", recall.item(), prog_bar=True, on_epoch=True, batch_size=len(batch["names"])
        )
        self.log(
            "test_precision", precision.item(), prog_bar=True, on_epoch=True, batch_size=len(batch["names"])
        )
        self.log(
            "test_mz_recall", mz_recall.item(), prog_bar=True, on_epoch=True, batch_size=len(batch["names"])
        )
        self.log(
            "test_mz_precision", mz_precision.item(), prog_bar=True, on_epoch=True, batch_size=len(batch["names"])
        )
        return {"loss": loss}

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        # For LambdaLR, just call step() without arguments
        scheduler.step()

    def pattern_match_metrics(self, pred_mask, targ_mask, num_targs, patterns_count=None):
        """
        Compute recall and precision using Hungarian matching between predicted and target binary patterns.

        Args:
            pred_mask: (B, P, N) predicted masks (binary / bool)
            targ_mask: (B, T, N) target masks (binary / bool)
            num_targs: (B,) number of valid target masks per batch

        Returns:
            recall: fraction of target masks matched by predicted masks under optimal assignment
            precision: fraction of predicted masks matched by target masks under optimal assignment
        """
        B, P, N = pred_mask.shape
        _, T, _ = targ_mask.shape
        device = pred_mask.device

        # build exact-match similarity matrix: 1 if equal across nodes, else 0
        pred_exp = pred_mask.unsqueeze(1).expand(B, T, P, N)  # (B, T, P, N)
        targ_exp = targ_mask.unsqueeze(2).expand(B, T, P, N)  # (B, T, P, N)
        sim = (pred_exp == targ_exp).all(dim=-1).float()  # (B, T, P)
        valid_targs = torch.arange(T, device=device).unsqueeze(0) < num_targs.unsqueeze(1)  # (B, T)
        sim = sim * valid_targs.unsqueeze(2).float()  # mask out invalid targets
        valid_patterns = torch.arange(P, device=device).unsqueeze(0) < patterns_count.unsqueeze(1)  # (B, P)
        sim = sim * valid_patterns.unsqueeze(1).float()


        # Use Hungarian to find optimal one-to-one matching that maximizes total exact matches
        # pygm.hungarian expects a score matrix (higher better). Provide n1=T (targets), n2=P (preds)
        assign = pygm.hungarian(sim, backend="pytorch", n1=num_targs, n2=patterns_count)
        # assign is shape (B, T, P) with 1s for selected assignments
        matched_score = sim * assign
        matched_targets = (matched_score.sum(dim=2) > 0).float()  # (B, T)

        match_counts = matched_targets.sum(dim=1).float()
        recall = match_counts / num_targs.clamp(min=1).float()
        precision = match_counts/patterns_count
        return {"recall":recall.mean(), "precision":precision.mean()}

    def mz_metrics(self, pred_mask, targ_mz, mass, atom_hs, num_frag_pred, adduct_mass_shifts):
        device = mass.device
        pred_mass_center = torch.sum(pred_mask * mass.unsqueeze(1), dim=-1)
        mol_total_hs = torch.sum(atom_hs, dim=-1)
        frag_hs = torch.sum(pred_mask*atom_hs.unsqueeze(1), dim=-1)
        max_remove = torch.clamp(frag_hs, max=self.max_broken_bonds)
        max_add = torch.clamp(mol_total_hs.unsqueeze(1) - frag_hs, max=self.max_broken_bonds)
        hydrogen_shift = torch.arange(-self.max_broken_bonds, self.max_broken_bonds + 1, device=device) * common.ELEMENT_TO_MASS["H"]
        
        possible_mass = (
            pred_mass_center[:, :, None, None]
            + hydrogen_shift[None, None, None, :]
            + adduct_mass_shifts[:, None, :, None]
        )
        batch_size = pred_mask.shape[0]

        max_inten_shift = (self.output_size - 1) / 2  # Center shift for hydrogen range
        max_break_ar = torch.arange(self.output_size, device=device)[None, None, :].to(device)
        max_breaks_ub = max_add + max_inten_shift  # [B, max_breakpoints]
        max_breaks_lb = -max_remove + max_inten_shift  # [B, max_breakpoints]

        ub_mask = max_break_ar <= max_breaks_ub[:, :, None]  # [B, max_breakpoints, output_size]
        lb_mask = max_break_ar >= max_breaks_lb[:, :, None]  # [B, max_breakpoints, output_size]
        valid_pos = torch.logical_and(ub_mask, lb_mask)
        max_breakpoints = pred_mask.shape[1]
        frag_mask = torch.arange(max_breakpoints, device=device).unsqueeze(0) < num_frag_pred.unsqueeze(1)
        valid_pos = torch.logical_and(valid_pos, frag_mask[:, :, None]).unsqueeze(-2)
        valid_mass = possible_mass.masked_fill(~valid_pos, 0)
        inverse_indices = torch.clamp(torch.bucketize(valid_mass, self.inten_buckets, right=False), max=len(self.inten_buckets) - 1)
        inverse_indices_flatten = inverse_indices.reshape(batch_size, -1)
        potential_peaks = torch.zeros_like(targ_mz)
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)  # (B, 1)
        potential_peaks[batch_idx, inverse_indices_flatten] = 1
        pos_peaks = (targ_mz > 0).float()
        matched_peak = ((potential_peaks * targ_mz) > 0).float()
        recall = torch.sum(matched_peak, dim=-1)/torch.sum(pos_peaks, dim=-1)
        precision = torch.sum(matched_peak, dim=-1)/torch.sum(potential_peaks, dim=-1)
        return {"recall":recall.mean(), "precision":precision.mean()}

