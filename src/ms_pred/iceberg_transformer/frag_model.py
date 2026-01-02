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
        multi_hop_max_dist: int = 5,
        num_edge_dis: int = 10,
        max_frags: int = 40,
        embed_adduct: bool = False,
        embed_collision: bool = False,
        embed_elem_group: bool = False,
        encode_forms: bool = False,
        mlp_layers: int = 1,
        sk_tau: float = 0.05,
        linsat_tau: float = 0.01,
        gamma: float = 2,
        include_unassigned: bool = False,
        max_broken_bonds: int = 6,
        pe_embed_k: int = 0,
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
        self.embed_adduct = embed_adduct
        self.embed_collision = embed_collision
        self.embed_elem_group = embed_elem_group
        self.encode_forms = encode_forms
        self.mlp_layers = mlp_layers
        self.sk_tau = sk_tau
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
                gnn_node_feats=node_feats+adduct_shift+collision_shift,
                gnn_edge_feats=edge_feats,
                dropout=dropout,
            )
            self.pool = dgl_nn.AvgPooling()
        else:
            raise ValueError(f"Unsupported root_encode: {root_encode}")

        # Fragment query tokens and decoder
        self.fragment_decoder = nn_utils.SlotDecoder(
            hidden_dim=hidden_size,
            num_slots=max_frags,
            nhead=self.nhead,
            num_layers=3,
            dropout=dropout
        )
        buckets = torch.DoubleTensor(np.linspace(0, 1500, 15000))
        self.inten_buckets = nn.Parameter(buckets)
        self.inten_buckets.requires_grad = False
        
        self.frag_card_mapper = nn.Linear(hidden_size, fragmentation.FRAGMENT_ENGINE_PARAMS['max_tree_depth']+1)
        # self.frag_logit_mapper = nn_utils.MLPBlocks(hidden_size, hidden_size, dropout, 3, use_residuals=True)
        # self.frag_logit_mapper = nn_utils.MultiHeadCrossAttentionLogits(self.hidden_size, self.nhead)
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()
    
    def cross_entropy(self, preds, targets, weights=None, normalized=True):
        if normalized:
            log_preds = torch.log(preds + 1e-9)
        else:
            log_preds = F.log_softmax(preds, dim=-1)
        loss = targets * log_preds
        if weights is not None:
            loss *= weights
        cross_entropy = -torch.sum(loss, dim=-1)
        return cross_entropy

    def cosine_similarity(self, logit1, logit2):
        sim = nn.CosineSimilarity(dim = -1)
        return sim(logit1, logit2)
    
    def node_ranking(self, breakpoint_logit, num_atoms):
        B, N, N_atom = breakpoint_logit.shape
        n_atom_mask = torch.arange(N_atom, device=breakpoint_logit.device).unsqueeze(0) >= num_atoms.unsqueeze(1)  # [B, N_atom]
        dummy_val = -1e9
        breakpoint_logit = breakpoint_logit.masked_fill(n_atom_mask.unsqueeze(1), dummy_val)
        breakpoint_logit = breakpoint_logit.reshape(B * N, N_atom)
        E = torch.ones((1, N_atom), device=breakpoint_logit.device)
        f_1 = torch.ones((1,), device=breakpoint_logit.device)
        f_2 = torch.full_like(f_1, 2)
        f_3 = torch.full_like(f_1, 3)
        output_logit_1 = linsat_layer(breakpoint_logit, E=E, f=f_1, no_warning=True, max_iter=1, tau=self.linsat_tau).reshape(B, N, N_atom)
        output_logit_2 = linsat_layer(breakpoint_logit, E=E, f=f_2, no_warning=True, max_iter=1, tau=self.linsat_tau).reshape(B, N, N_atom)
        output_logit_3 = linsat_layer(breakpoint_logit, E=E, f=f_3, no_warning=True, max_iter=1, tau=self.linsat_tau).reshape(B, N, N_atom)
        logit_0 = torch.zeros_like(output_logit_1)
        logits = torch.stack([logit_0, output_logit_1, output_logit_2, output_logit_3], dim=-1)
        # assert False, (output_logit_1.max(), output_logit_2.max(), output_logit_3.max())
        # breakpoint_preds = torch.sum(breakpoint_card.unsqueeze(2) * logits, dim=-1)
        # loss = torch.sum((breakpoint_preds.unsqueeze(2) - breakpoint_targs.unsqueeze(1)) ** 2, dim=-1)
        # return {"loss":loss, "preds":breakpoint_preds}
        return logits

    def breakpoint_inference(self, breakpoint_logit, breakpoint_card, num_atoms, debug=False):
        """
        Returns a binary tensor of shape [B, N, N_atom] with k positive entries per last axis,
        where k is the predicted cardinality for each pattern (from breakpoint_card).
        Batchified implementation.
        """
        logits = self.node_ranking(breakpoint_logit, num_atoms)  # [B, N, N_atom, 4]
        B, N, N_atom, _ = logits.shape
        k = torch.argmax(breakpoint_card, dim=-1)  # [B, N]
        breakpoint_preds = torch.gather(logits, index=k[:, :, None, None].expand(B, N, N_atom, 1), dim=-1).squeeze(-1)  # [B, N, N_atom]
        # Flatten for batch processing
        flat_preds = breakpoint_preds.reshape(-1, N_atom)  # [(B*N), N_atom]
        flat_k = k.reshape(-1)  # [(B*N)]
        patterns = torch.zeros_like(flat_preds, dtype=torch.bool)  # [(B*N), N_atom]
        if flat_k.max().item() > 0:
            topk_vals, topk_idx = torch.topk(flat_preds, k=flat_k.max().item(), dim=-1)
            arange_k = torch.arange(flat_k.max().item(), device=breakpoint_preds.device).unsqueeze(0)  # [1, max_k]
            valid_mask = arange_k < flat_k.unsqueeze(1)  # [(B*N), max_k]
            batch_idx = torch.arange(flat_preds.size(0), device=breakpoint_preds.device).unsqueeze(1).expand(-1, flat_k.max().item())  # [(B*N), max_k]
            patterns[batch_idx[valid_mask], topk_idx[valid_mask]] = True
        out = patterns.view(B, N, N_atom)
        return out
    
    def gumbel_topk_from_soft(self, x, k):
        g = -torch.log(-torch.log(torch.rand_like(x)))
        y = x + g
        idx = torch.topk(y, k).indices

        hard = torch.zeros_like(x)
        hard.scatter_(0, idx, 1.0)
        return hard
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = nn_utils.build_lr_scheduler(
            optimizer=optimizer, lr_decay_rate=self.lr_decay_rate, warmup=self.warmup
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "frequency": 1, "interval": "step"}}

    def forward(self, graphormer_input, num_atoms, adducts, collision_engs, root_form_vecs, root_reprs=None) -> Dict[str, torch.Tensor]:
        device = num_atoms.device
        batch_size = num_atoms.shape[0]
        embed_adducts = self.adduct_embedder[adducts.long()]
        
        if self.root_encode == "graphormer":
            assert graphormer_input is not None, "graphormer_input required for graphormer root_encode"
            node_features = graphormer_input['x']  # [B, max_nodes, num_features]
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
        else:
            with root_reprs.local_scope():
                if self.embed_adduct:
                    embed_adducts_expand = embed_adducts.repeat_interleave(
                        root_reprs.batch_num_nodes(), 0
                    )
                    ndata = root_reprs.ndata["h"]
                    ndata = torch.cat([ndata, embed_adducts_expand], -1)
                    root_reprs.ndata["h"] = ndata                    
                if self.embed_collision:                    
                    embed_collision = torch.cat(
                        (torch.sin(collision_engs.unsqueeze(1) / self.collision_embedder_denominators.unsqueeze(0)),
                         torch.cos(collision_engs.unsqueeze(1) / self.collision_embedder_denominators.unsqueeze(0))),
                        dim=1
                    )
                    
                    embed_collision = torch.where(  # handle entries without collision energy (== nan)
                        torch.isnan(embed_collision), self.collision_embed_merged.unsqueeze(0), embed_collision
                    )   
                    embed_collision_expand = embed_collision.repeat_interleave(
                        root_reprs.batch_num_nodes(), 0
                    )
                    ndata = root_reprs.ndata["h"]
                    ndata = torch.cat([ndata, embed_collision_expand], -1)
                    root_reprs.ndata["h"] = ndata
                node_embeddings = self.root_module(root_reprs)
                root_tokens = self.pool(root_reprs, node_embeddings).unsqueeze(1)
            node_embeddings = nn_utils.pad_packed_tensor(node_embeddings, num_atoms, 0)
            max_nodes = node_embeddings.shape[1]

        node_mask = torch.arange(max_nodes, device=device).unsqueeze(0) >= num_atoms.unsqueeze(1)  # [B,max_nodes]
        frag_mask = F.pad(node_mask, (1, 0, 0, 0), mode="constant", value=0).bool()
        if self.encode_forms:
            encoded_form = self.embedder(root_form_vecs)[:, None, :]
            root_tokens = self.formula_mapper(torch.cat((root_tokens, encoded_form), dim=-1))
        frag_vecs = self.fragment_decoder(root_tokens, node_embeddings, memory_key_padding_mask=frag_mask)
        frag_card_logits = self.frag_card_mapper(frag_vecs)  # [num_layers, B, max_frags, 4]
        frag_logits = torch.einsum("nbij,bkj->nbik", frag_vecs, node_embeddings)
        return {"frag_logits": frag_logits, "frag_card_logits": frag_card_logits}
    
    def boundary_nodes(self, A: torch.Tensor, subgraph_mask: torch.Tensor) -> torch.Tensor:
        """
        A: (B, N, N) adjacency matrices
        subgraph_mask: (B, M, N) boolean mask for M subgraphs per graph
        
        Returns:
            boundary_mask: (B, M, N) boolean mask of boundary nodes for each subgraph
        """
        B, N, _ = A.shape
        _, M, _ = subgraph_mask.shape
        
        # Reshape adjacency for broadcasting: (B, 1, N, N)
        A_exp = A.unsqueeze(1)  
        
        # (B, M, N) -> (B, M, 1, N)
        subgraph_mask_exp = subgraph_mask.float().unsqueeze(2)
        
        # Matrix multiplication: (B, M, 1, N) x (B, 1, N, N) -> (B, M, 1, N)
        neighbors = torch.matmul(subgraph_mask_exp, A_exp) > 0
        neighbors = neighbors.squeeze(2)  # (B, M, N)
        
        # Remove nodes already in subgraph
        boundary_mask = neighbors & ~subgraph_mask
        # Count boundary nodes per fragment and assert the maximum is reasonable
        max_boundary_nodes = int(boundary_mask.sum(dim=-1).max().item())
        assert max_boundary_nodes <= 3, max_boundary_nodes
        batch_num = torch.arange(B, device=A.device, dtype=torch.float32)[:, None, None].expand(B, M, 1)
        boundary_mask_batch_num = torch.cat([batch_num, boundary_mask], dim=-1).reshape(B*M, -1)
        boundary_mask_batch_num_unique = torch.unique(boundary_mask_batch_num, dim=0, sorted=True)
        batch_num_unique = torch.bincount(boundary_mask_batch_num_unique[:, 0].to(torch.int64), minlength=B)
        boundary_mask = nn_utils.pad_packed_tensor(
            boundary_mask_batch_num_unique[:, 1:], batch_num_unique, 0
        )
        return {"boundary_mask": boundary_mask, "unique_boundary_patterns":batch_num_unique}

    def _frag_loss(
        self,
        frags_predicted: torch.Tensor,
        frag_card_predicted: torch.Tensor,
        frag_targs: torch.Tensor,
        num_frag_targs: torch.Tensor,
        num_atoms: torch.Tensor,
        adj_matrices: torch.Tensor = None,
        debug=False,
    ) -> torch.Tensor:
        # frags_predicted: [B, max_frags, max_nodes]
        # frag_targs: packed [sum_frags, max_nodes], num_frag_targs: [B]
        B, max_frags, max_nodes = frags_predicted.shape
        
        # frag_targs_padded = nn_utils.pad_packed_tensor(
        #     frag_targs, num_frag_targs, False
        # )[:, :-1, :]  # [B, max_targs-1, max_nodes]
        frag_targs_padded_original = nn_utils.pad_packed_tensor(
            frag_targs, num_frag_targs, False
        )[:, :, :]  # [B, max_targs-1, max_nodes]
        boundary_info = self.boundary_nodes(adj_matrices, frag_targs_padded_original)
        frag_targs_padded = boundary_info["boundary_mask"]
        num_frag_targs = boundary_info["unique_boundary_patterns"]

        node_rank = self.node_ranking(frags_predicted, num_atoms)
        frag_cards_targs = F.one_hot(torch.sum(frag_targs_padded, dim=-1).long(), num_classes=fragmentation.FRAGMENT_ENGINE_PARAMS['max_tree_depth']+1)
        rank_paired = torch.sum(node_rank.unsqueeze(2) * frag_cards_targs[:, None, :, None, :], dim=(-1))
        frag_targs_expanded = frag_targs_padded.unsqueeze(1).expand(rank_paired.shape)
        rank_paired_normed = F.normalize(rank_paired, p=1, dim=-1)
        frag_targs_expanded_normed = F.normalize(frag_targs_expanded, p=1, dim=-1)
        # rank_loss = torch.sum((rank_paired-frag_targs_expanded)**2, dim=-1)
        rank_loss = self.cross_entropy(rank_paired_normed, frag_targs_expanded_normed)

        # rank_loss = torch.sum(self.binary_focal_loss(rank_paired, frag_targs_expanded, num_atoms), dim=-1)


        per_pair_cards_cross_entropy = self.cross_entropy(frag_card_predicted.unsqueeze(2), frag_cards_targs.unsqueeze(1), normalized=False)

        B, max_targs, _ = frag_targs_padded.shape
        
        cost = rank_loss+per_pair_cards_cross_entropy
        assign = pygm.hungarian(
            -cost, backend="pytorch", n2=num_frag_targs
        )  # [B, max_targs, max_frags]

        unassigned_prediction = 1-torch.sum(assign, dim=-1)
        unpaired_tensor = torch.tensor([1, 0, 0, 0], device=assign.device)[None, None, :].expand(frag_card_predicted.shape)
        unassigned_loss = self.cross_entropy(frag_card_predicted, unpaired_tensor, normalized=False) * unassigned_prediction
        unassigned_count = torch.clamp(torch.sum(unassigned_prediction, dim=-1), min=1)
        unassigned_loss = torch.sum(unassigned_loss, dim=-1)/unassigned_count

        
        node_rank_reshape = node_rank.reshape(B, max_frags, -1)
        node_rank_assigned = torch.matmul(node_rank_reshape.transpose(1, 2), assign).transpose(1, 2)
        node_rank_assigned = node_rank_assigned.reshape(B, max_targs, max_nodes, -1)
        node_rank_assigned = torch.sum(node_rank_assigned*frag_cards_targs.unsqueeze(-2), dim=-1)
        frag_card_predicted = torch.matmul(frag_card_predicted.transpose(1, 2), assign).transpose(1, 2)
        
        # preds = torch.matmul(preds.transpose(1, 2), assign).transpose(1, 2)
        
        node_rank_assigned_normed = F.normalize(node_rank_assigned, p=1, dim=-1)
        frag_targs_padded_normed = F.normalize(frag_targs_padded, p=1, dim=-1)
        loss = self.cross_entropy(node_rank_assigned_normed, frag_targs_padded_normed)+self.cross_entropy(frag_card_predicted, frag_cards_targs, normalized=False)
        # loss = torch.sum(self.binary_focal_loss(node_rank_assigned, frag_targs_padded, num_atoms), dim=-1)+self.cross_entropy(frag_card_predicted, frag_cards_targs)
        frag_targs_mask = num_frag_targs[:, None] <= torch.arange(max_targs, device=loss.device)[None, :]

        loss = torch.sum(loss.masked_fill(frag_targs_mask, 0), dim=-1)/num_frag_targs
        if self.include_unassigned:
            loss += unassigned_loss
        return torch.mean(loss)

    def binary_focal_loss(self, pred, targ, num_atoms):
        if len(pred.shape) == 3:
            card_targ = torch.sum(targ, dim=-1)
        else:
            card_targ = torch.sum(targ[:, 0, :, :], dim=-1)
        alpha = 1 - card_targ/num_atoms[:, None]
        if len(pred.shape) >= 3:
            alpha = alpha.unsqueeze(1)
        if len(pred.shape) == 4:
            alpha = alpha.unsqueeze(-1)
        bce_loss = self.bce_loss(pred, targ)
        p_t = pred * targ + (1-pred)*(1-targ)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss
        return loss

    def breakpoints_to_patterns(self, mask, A, num_nodes=None):
        B, M, N = mask.shape
        BM = B * M

        # Base adjacency expansion (B*M, N, N)
        A_exp = A.unsqueeze(1).expand(-1, M, -1, -1).reshape(BM, N, N)
        mask_exp = mask.reshape(BM, N)

        # Compute reachability matrix R for each masked adjacency (boolean closure)
        R = self.compute_reachability(A_exp, mask_exp, num_nodes=num_nodes)  # shape (BM, N, N), dtype=bool

        # Create batch index that stays constant across M patterns
        batch_ids = torch.arange(B, device=R.device).repeat_interleave(M)  # (BM,)

        # Expand per node
        batch_idx = batch_ids.unsqueeze(1).expand(-1, N)  # (BM, N)

        # Append batch index as first bit
        row_repr = torch.cat([
            batch_idx.unsqueeze(-1).to(torch.int64),   # (BM, N, 1)
            R.to(torch.int64)                          # (BM, N, N)
        ], dim=-1)  # (BM, N, N+1)

        # Flatten rows to (BM*N, N+1)
        flat_rows = row_repr.view(BM*N, N+1)

        # Unique per batch (batch id is part of the row)
        unique_rows, inv = torch.unique(flat_rows, dim=0, return_inverse=True)
        is_empty = (unique_rows[:, 1:].sum(dim=-1) == 0)
        unique_rows = unique_rows[~is_empty]
        batch_ids_unique = unique_rows[:, 0]
        pattern_counts = torch.bincount(batch_ids_unique, minlength=B)
        padded_pattern = nn_utils.pad_packed_tensor(
            unique_rows[:, 1:], pattern_counts, 0
        )  # (B, max_patterns, N)
        
        return padded_pattern, pattern_counts

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
        max_breaks_ub = max_add + max_inten_shift  # [B, max_frags]
        max_breaks_lb = -max_remove + max_inten_shift  # [B, max_frags]

        ub_mask = max_break_ar <= max_breaks_ub[:, :, None]  # [B, max_frags, output_size]
        lb_mask = max_break_ar >= max_breaks_lb[:, :, None]  # [B, max_frags, output_size]
        valid_pos = torch.logical_and(ub_mask, lb_mask)
        max_frags = pred_mask.shape[1]
        frag_mask = torch.arange(max_frags, device=device).unsqueeze(0) < num_frag_pred.unsqueeze(1)
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

    def compute_reachability(self, A: torch.Tensor, mask: torch.Tensor = None, num_nodes: torch.Tensor = None) -> torch.Tensor:
        """
        Compute transitive closure (reachability matrix) for a batch of adjacency matrices.

        Args:
            A: (B, N, N) adjacency matrices (bool or int)
            mask: (B, N) boolean tensor of nodes to keep (optional)
                If provided, edges touching unkept nodes will be zeroed out.

        Returns:
            R: (B, N, N) boolean reachability matrices,
            where R[b,i,j] == True means node i can reach node j (including itself).
        """
        device = A.device
        B, N, _ = A.shape
        A = A > 0
        # mask out removed nodes (optional)
        if mask is not None:
            row_mask = mask.unsqueeze(-1)
            col_mask = mask.unsqueeze(-2)
            A = A & ~row_mask & ~col_mask  # zero edges touching removed nodes

        # initialize reachability
        R = A.clone()

        # iterative doubling to compute closure efficiently
        # after log2(N) iterations, R will contain all reachabilities
        step = 1
        while step < N:
            # boolean matmul: (R @ R) > 0
            RR = (R.float() @ R.float()) > 0
            R = R | RR
            step *= 2

        # add self-reachability
        # eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
        # R = R | eye

        # if mask is not None:
        #     row_mask = mask.unsqueeze(-1)
        #     col_mask = mask.unsqueeze(-2)
        #     A = A & ~row_mask & ~col_mask  # zero edges touching removed nodes

        if num_nodes is not None:
            # mask out rows/cols beyond num_nodes
            row_idx = torch.arange(N, device=device).unsqueeze(0)  # (1, N)
            col_idx = torch.arange(N, device=device).unsqueeze(0)  # (1, N)
            valid_row = row_idx < num_nodes.unsqueeze(1)  # (B, N)
            valid_col = col_idx < num_nodes.unsqueeze(1)  # (B, N)
            valid_matrix = valid_row.unsqueeze(-1) & valid_col.unsqueeze(-2)  # (B, N, N)
            valid_matrix = torch.repeat_interleave(valid_matrix, repeats=R.shape[0]//valid_matrix.shape[0], dim=0)
            R = R & valid_matrix

        return R
            

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        out = self.forward(
            graphormer_input=batch.get("graphormer_input"),
            num_atoms=batch["num_atoms"],
            adducts=batch["adducts"],
            collision_engs=batch["collision_engs"],
            root_form_vecs=batch["root_form_vecs"]
        )
        loss = 0
        for i in range(out["frag_logits"].shape[0]):
            loss += self._frag_loss(
            out["frag_logits"][i], out["frag_card_logits"][i], batch["frag_targs"], batch["num_frag_targs"], batch["num_atoms"], batch["adj_matrices"],
        )    
        self.log(
            "train_loss", loss.item(), prog_bar=True, on_epoch=True, batch_size=len(batch["names"])
        )
        return {"loss": loss}


    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        out = self.forward(
            graphormer_input=batch.get("graphormer_input"),
            num_atoms=batch["num_atoms"],
            adducts=batch["adducts"],
            collision_engs=batch["collision_engs"],
            root_form_vecs=batch["root_form_vecs"]
        )
        frag_logits = out["frag_logits"][-1]
        frag_card_logits = out["frag_card_logits"][-1]
        loss = self._frag_loss(
            frag_logits, frag_card_logits, batch["frag_targs"], batch["num_frag_targs"], batch["num_atoms"], batch["adj_matrices"],
        )
        breakpoints = self.breakpoint_inference(frag_logits, frag_card_logits, batch["num_atoms"])
        patterns, patterns_count = self.breakpoints_to_patterns(breakpoints, batch["adj_matrices"], batch["num_atoms"])
        
        frag_targs_padded = nn_utils.pad_packed_tensor(batch["frag_targs"], batch["num_frag_targs"], 0)
        metrics = self.pattern_match_metrics(patterns, frag_targs_padded, batch["num_frag_targs"], patterns_count = patterns_count)
        recall, precision = metrics["recall"], metrics["precision"]
        metrics = self.pattern_match_metrics(patterns, frag_targs_padded, batch["num_frag_targs"], patterns_count = patterns_count)
        mz_metrics = self.mz_metrics(patterns, batch["inten_targs"], batch["weights"], batch["atom_hs"], patterns_count, batch["adduct_mass_shifts"])
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

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        out = self.forward(
            graphormer_input=batch.get("graphormer_input"),
            num_atoms=batch["num_atoms"],
            adducts=batch["adducts"],
            collision_engs=batch["collision_engs"],
            root_form_vecs=batch["root_form_vecs"]
        )
        frag_logits = out["frag_logits"][-1]
        frag_card_logits = out["frag_card_logits"][-1]
        loss = self._frag_loss(
            frag_logits, frag_card_logits, batch["frag_targs"], batch["num_frag_targs"], batch["num_atoms"], batch["adj_matrices"],
        )
        breakpoints = self.breakpoint_inference(frag_logits, frag_card_logits, batch["num_atoms"])
        patterns, patterns_count = self.breakpoints_to_patterns(breakpoints, batch["adj_matrices"], batch["num_atoms"])
        
        frag_targs_padded = nn_utils.pad_packed_tensor(batch["frag_targs"], batch["num_frag_targs"], 0)
        metrics = self.pattern_match_metrics(patterns, frag_targs_padded, batch["num_frag_targs"], patterns_count = patterns_count)
        mz_metrics = self.mz_metrics(patterns, batch["inten_targs"], batch["weights"], batch["atom_hs"], patterns_count, batch["adduct_mass_shifts"])
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

    def restore_breakpoint_patterns(self, frag_predicted: torch.Tensor, frag_card_logits: torch.Tensor) -> torch.Tensor:
        """
        Restore breakpoint patterns from model outputs during inference.

        Args:
            frag_predicted: [B, M, N] - predicted distribution over breakpoints for each fragment
            frag_card: [B, M, 4] - predicted cardinality distribution for each fragment (0, 1, 2, 3)

        Returns:
            patterns: [B, M, N] - binary mask for each fragment, top-k breakpoints selected
        """
        # Get predicted cardinality for each fragment (argmax over last dim, values in {0,1,2,3})
        k = torch.argmax(frag_card_logits, dim=-1)  # [B, M]
        B, M, N = frag_predicted.shape
        patterns = torch.zeros_like(frag_predicted, dtype=torch.bool)  # [B, M, N]

        flat_pred = frag_predicted.reshape(-1, N)  # [(B*M), N]
        flat_k = k.reshape(-1)  # [(B*M)]

        max_k = flat_k.max().item()
        if max_k > 0:
            topk_vals, topk_idx = torch.topk(flat_pred, k=max_k, dim=-1)  # [(B*M), max_k]
            arange_k = torch.arange(max_k, device=frag_predicted.device).unsqueeze(0)  # [1, max_k]
            valid_mask = arange_k < flat_k.unsqueeze(1)  # [(B*M), max_k]
            batch_idx = torch.arange(B * M, device=frag_predicted.device).unsqueeze(1).expand(-1, max_k)  # [(B*M), max_k]
            patterns_flat = patterns.view(-1, N)
            patterns_flat[batch_idx[valid_mask], topk_idx[valid_mask]] = True

        return patterns

    def predict(self, root_smi: str, collision_eng: float, adduct: str, device: str = "cpu") -> dict[str, torch.Tensor]:
        root_form = [common.form_from_smi(rsmi) for rsmi in root_smi]
        root_form_vec = torch.FloatTensor(np.array([common.formula_to_dense(rf) for rf in root_form])).to(device)
        adducts = torch.LongTensor([common.ion2onehot_pos[a] if type(a) is str else a for a in adduct]).to(device)
        collision_engs = torch.FloatTensor(collision_eng).to(device)
        mols = [Chem.MolFromSmiles(rsmi) for rsmi in root_smi]
        graphormer_inputs = [self.tree_processor.create_graphormer_input(mol=m, multi_hop_max_dist=self.tree_processor.multi_hop_max_dist) for m in mols]
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
        with torch.no_grad():
            out = self.forward(
                graphormer_input=graphormer_batch,
                num_atoms=num_atoms,
                adducts=adducts,
                collision_engs=collision_engs,
                root_form_vecs=root_form_vec
            )
            frag_logits = out["frag_logits"][-1]
            frag_card_logits = out["frag_card_logits"][-1]
            breakpoints = self.breakpoint_inference(frag_logits, frag_card_logits, num_atoms)
            patterns, patterns_count = self.breakpoints_to_patterns(breakpoints, adj_matrices_batch, num_atoms)
        return {"fragment_patterns": patterns.bool(), "patterns_count": patterns_count}
        


