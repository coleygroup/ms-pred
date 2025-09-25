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
import copy


class FragOnlyModel(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int = 512,
        layers: int = 6,
        dropout: float = 0.1,
        learning_rate: float = 7e-4,
        lr_decay_rate: float = 1.0,
        weight_decay: float = 0.0,
        warmup: int = 1000,
        root_encode: str = "graphormer",  # or "gnn"
        node_feats: int = 128,
        edge_feats: int = 12,
        embed_adduct: bool = True,
        embed_collision: bool = True,
        multi_hop_max_dist: int = 5,
        num_edge_dis: int = 10,
        max_frags: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.root_encode = root_encode
        self.max_frags = max_frags
        self.nhead = 8
        # Runtime checks
        self._checked_opt = False
        self._logged_grad_once = False
        self._frag_vec_prev = None
        self._frag_grad_seen = False
        self._frag_update_seen = False

        # Root encoder
        if root_encode == "graphormer":
            self.root_module = GraphormerGraphEncoder(
                num_atom_features=node_feats,
                num_in_degree=8,
                num_out_degree=8,
                num_edge_features=edge_feats,
                num_spatial=1025,
                num_edge_dis=num_edge_dis,
                edge_type="multi_hop",
                multi_hop_max_dist=multi_hop_max_dist,
                num_encoder_layers=layers,
                embedding_dim=hidden_size,
                ffn_embedding_dim=hidden_size * 4,
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
                gnn_node_feats=node_feats,
                gnn_edge_feats=edge_feats,
                dropout=dropout,
            )
            self.pool = dgl_nn.AvgPooling()
        else:
            raise ValueError(f"Unsupported root_encode: {root_encode}")

        # Fragment query tokens and decoder
        self.fragment_vec = nn.Parameter(torch.zeros((max_frags-1, hidden_size), dtype=torch.float32))
        nn.init.xavier_uniform_(self.fragment_vec)
        self.fragment_vec.requires_grad = True

        self.fragment_mapper = nn_utils.MLPBlocks(
            input_size=2*hidden_size,
            hidden_size=2*hidden_size,
            output_size=hidden_size,
            dropout=dropout,
            use_residuals=True,
            num_layers=1,
        )

        frag_encoder_layer = nn.TransformerEncoderLayer(self.hidden_size, nhead=self.nhead, batch_first=True, dim_feedforward=self.hidden_size * 4, dropout=self.dropout)
        self.frag_encoder = nn.TransformerEncoder(frag_encoder_layer, 3)

        self.bce_loss = nn.BCELoss(reduction="none")
        self.sigmoid = nn.Sigmoid()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = nn_utils.build_lr_scheduler(
            optimizer=optimizer, lr_decay_rate=self.lr_decay_rate, warmup=self.warmup
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "frequency": 1, "interval": "step"}}

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        graphormer_input = batch.get("graphormer_input")
        num_atoms = batch["num_atoms"]  # [B]
        device = num_atoms.device
        batch_size = num_atoms.shape[0]

        if self.root_encode == "graphormer":
            assert graphormer_input is not None, "graphormer_input required for graphormer root_encode"
            inner_states, graph_rep = self.root_module(graphormer_input)
            final_layer_output = inner_states[-1]  # [T, B, H]
            node_embeddings = final_layer_output[1:].transpose(0, 1)  # [B, max_nodes, H]
            root_tokens = graph_rep.unsqueeze(1)  # [B,1,H]
            max_nodes = node_embeddings.shape[1]
        else:
            root_graph = batch["root_reprs"]
            with root_graph.local_scope():
                node_embeddings = self.root_module(root_graph)
                root_tokens = self.pool(root_graph, node_embeddings).unsqueeze(1)
            node_embeddings = nn_utils.pad_packed_tensor(node_embeddings, num_atoms, 0)
            max_nodes = node_embeddings.shape[1]

        # Prepare fragment query tokens and run decoder
        frag_decode_root_tokens = root_tokens  # [B,1,H]
        frag_decode_root_tokens_expanded = frag_decode_root_tokens.expand(batch_size, self.max_frags-1, self.hidden_size)

        frag_vec_expanded = self.fragment_vec.unsqueeze(0).expand(batch_size, self.max_frags-1, self.hidden_size)
        frag_token = self.fragment_mapper(
            torch.cat([frag_decode_root_tokens_expanded, frag_vec_expanded], dim=-1)
        )
        atom_mask = torch.arange(max_nodes, device=device).unsqueeze(0) >= num_atoms.unsqueeze(1)  # [B,max_nodes]
        frag_token = self.frag_encoder(frag_token, src_key_padding_mask=None)  
        frag_logits = torch.bmm(frag_token, node_embeddings.transpose(1, 2))# [B, max_frags-1, max_nodes]
        frag_logits = frag_logits.masked_fill(atom_mask.unsqueeze(1), -99999.0)
        frag_predicted = self.sigmoid(frag_logits)
        frag_predicted = F.pad(frag_predicted, (0, 0, 1, 0, 0, 0), "constant", 1)
        return {"frag_predicted": frag_predicted}

    def _frag_loss(
        self,
        frags_predicted: torch.Tensor,
        frag_targs: torch.Tensor,
        num_frag_targs: torch.Tensor,
        num_atoms: torch.Tensor
    ) -> torch.Tensor:
        # frags_predicted: [B, max_frags, max_nodes]
        # frag_targs: packed [sum_frags, max_nodes], num_frag_targs: [B]
        B, max_frags, max_nodes = frags_predicted.shape
        frag_targs_padded = nn_utils.pad_packed_tensor(
            frag_targs, num_frag_targs, False
        ).float()  # [B, max_targs, max_nodes]
        B, max_targs, _ = frag_targs_padded.shape

        scores_exp = frags_predicted.unsqueeze(1).expand(B, max_targs, max_frags, max_nodes)
        targs_exp = frag_targs_padded.unsqueeze(2).expand_as(scores_exp)
        per_pair_loss = self.bce_loss(scores_exp, targs_exp)  # [B, max_targs, max_frags, max_nodes]
        cost = per_pair_loss.sum(dim=-1)  # [B, max_targs, max_frags]

        n2_cols = torch.full_like(num_frag_targs, max_frags)
        assign = pygm.hungarian(
            -cost, backend="pytorch", n1=num_frag_targs, n2=n2_cols
        )  # [B, max_targs, max_frags]

        loss_assigned = (cost * assign).sum(dim=(1, 2))/(num_frag_targs*num_atoms)
        return loss_assigned.mean()

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        out = self.forward(batch)
        loss = self._frag_loss(
            out["frag_predicted"], batch["frag_targs"], batch["num_frag_targs"], batch["num_atoms"]
        )
        self.log(
            "train_loss", loss.item(), prog_bar=True, on_epoch=True, batch_size=len(batch["names"])
        )
        return {"loss": loss}

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        out = self.forward(batch)
        loss = self._frag_loss(
            out["frag_predicted"], batch["frag_targs"], batch["num_frag_targs"], batch["num_atoms"]
        )
        self.log(
            "val_loss", loss.item(), prog_bar=True, on_epoch=True, batch_size=len(batch["names"])
        )
        return {"loss": loss}

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        out = self.forward(batch)
        loss = self._frag_loss(
            out["frag_predicted"], batch["frag_targs"], batch["num_frag_targs"], batch["num_atoms"]
        )
        self.log("test_loss", loss.item(), on_epoch=True, batch_size=len(batch["names"]))
        return {"loss": loss}

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        # For LambdaLR, just call step() without arguments
        scheduler.step()

