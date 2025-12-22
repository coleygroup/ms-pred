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
        mlp_layers: int = 1,
        sk_tau: float = 0.05,
        linsat_tau: float = 0.01,
        gamma: float = 2,
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
        self.mlp_layers = mlp_layers
        self.sk_tau = sk_tau
        self.linsat_tau = linsat_tau
        self.gamma = gamma

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

        # Root encoder
        if root_encode == "graphormer":
            self.root_module = GraphormerGraphEncoder(
                num_atom_features=node_feats+adduct_shift+collision_shift,
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
                gnn_node_feats=node_feats+adduct_shift+collision_shift,
                gnn_edge_feats=edge_feats,
                dropout=dropout,
            )
            self.pool = dgl_nn.AvgPooling()
        else:
            raise ValueError(f"Unsupported root_encode: {root_encode}")

        # Fragment query tokens and decoder
        # self.fragment_decoder = nn_utils.SlotDecoder(
        #     hidden_dim=hidden_size,
        #     num_slots=max_frags-1,
        #     nhead=self.nhead,
        #     num_layers=3
        # )
        self.fragment_attention = nn_utils.MultiHeadSlotAttention(
            dim=hidden_size,
            num_slots=max_frags,
            iters=3,
            hidden_dim=hidden_size,
            heads=4,
            dim_head=hidden_size
        )
        
        self.frag_card_mapper = nn.Linear(hidden_size, fragmentation.FRAGMENT_ENGINE_PARAMS['max_tree_depth']+1)
        # self.frag_logit_mapper = nn_utils.MultiHeadCrossAttentionLogits(self.hidden_size, self.nhead)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()
    
    def cross_entropy(self, preds, targets, weights=None):
        log_preds = torch.log(preds + 1e-9)
        loss = targets * log_preds
        if weights is not None:
            loss *= weights
        cross_entropy = -torch.sum(loss, dim=-1)
        return cross_entropy
    
    def node_ranking_loss_simple(self, breakpoint_predicted, breakpoint_targs, num_atoms):
        return torch.sum((breakpoint_predicted.unsqueeze(2) * (1-breakpoint_targs).unsqueeze(1)) - (breakpoint_predicted.unsqueeze(2) * breakpoint_targs.unsqueeze(1)), dim=-1)


    def node_ranking_loss(self, breakpoint_predicted, breakpoint_targs, num_atoms):
        """
        Args:
            breakpoint_predicted: [B, N, N_atom] - predicted probabilities for each atom (N: num_patterns)
            breakpoint_targs: [B, M, N_atom] - binary mask of true breakpoints (M: num_targets)
            num_atoms: [B] - number of atoms per molecule
        Returns:
            target_sum, non_target_sum: [B, N, M] - sum of prob for target and non-target atoms in top-k
        """
        B, N, N_atom = breakpoint_predicted.shape
        _, M, _ = breakpoint_targs.shape
        device = breakpoint_predicted.device
        num_atoms_mask = (torch.arange(N_atom, device=device).unsqueeze(0) < num_atoms.unsqueeze(1))[:, None, :]  # [B, 1, N_atom]
        breakpoint_predicted = breakpoint_predicted.masked_fill(~num_atoms_mask, 1e9)

        # Compute Sinkhorn probability matrix as in your original code
        sorted_breakpoint_predicted = torch.sort(breakpoint_predicted, dim=-1)
        ranking_dist = torch.abs(
            breakpoint_predicted.unsqueeze(-2) - sorted_breakpoint_predicted.values.unsqueeze(-1)
        )  # [B, N, N_atom, N_atom]
        ranking_dist = ranking_dist.reshape(B * N, N_atom, N_atom)
        prob = pygm.sinkhorn(
            -ranking_dist,
            n1=torch.repeat_interleave(num_atoms, repeats=N),
            n2=torch.repeat_interleave(num_atoms, repeats=N),
            tau=self.sk_tau,
            backend='pytorch',
            max_iter=50,
        )
        prob = prob.reshape(B, N, N_atom, N_atom)

        # Mask padded atoms
        atom_mask = torch.arange(N_atom, device=device).unsqueeze(0) < num_atoms.unsqueeze(1)  # [B, N_atom]
        atom_mask_prob = atom_mask[:, None, :, None] & atom_mask[:, None, None, :]  # [B, 1, N_atom, N_atom]
        prob = prob * atom_mask_prob  # zero out padded atoms

        # For each target pattern, get nonzero (target atoms) and zero (non-target atoms) indices
        target_mask = breakpoint_targs.bool()  # [B, M, N_atom]
        non_target_mask = ~target_mask  # [B, M, N_atom]

        # For each (b, n), get k = number of nonzero entries in target_mask
        k = target_mask.sum(dim=-1)  # [B, M]
        k_expand = k[:, None, :]  # [B, 1, M]
        max_k = k.max().item()
        arange_k = torch.arange(N_atom, device=device).view(1, 1, 1, N_atom)  # [1, 1, 1, N_atom]
        topk_mask = arange_k < k_expand.unsqueeze(-1)  # [B, 1, M, N_atom]

        # Expand for broadcasting
        prob_for_targets = prob.unsqueeze(2).expand(B, N, M, N_atom, N_atom)  # [B, N, M, N_atom, N_atom]
        target_mask_expand = target_mask[:, None, :, :].expand(B, N, M, N_atom)  # [B, N, M, N_atom]
        non_target_mask_expand = non_target_mask[:, None, :, :].expand(B, N, M, N_atom)  # [B, N, M, N_atom]
        topk_mask_expand = topk_mask.expand(B, N, M, N_atom)  # [B, N, M, N_atom]

        # For target atoms: sum over i (target atoms), sum over j (top-k)
        target_atom_mask = target_mask_expand.unsqueeze(-1)  # [B, N, M, N_atom, 1]
        topk_col_mask = topk_mask_expand.unsqueeze(-2)  # [B, N, M, 1, N_atom]
        mask_targets = target_atom_mask & topk_col_mask  # [B, N, M, N_atom, N_atom]
        target_sum = (prob_for_targets * mask_targets.float()).sum(dim=(-2, -1))  # [B, N, M]

        # For non-target atoms: sum over i (non-target atoms), sum over j (top-k)
        non_target_atom_mask = non_target_mask_expand.unsqueeze(-1)  # [B, N, M, N_atom, 1]
        mask_non_targets = non_target_atom_mask & topk_col_mask  # [B, N, M, N_atom, N_atom]
        non_target_sum = (prob_for_targets * mask_non_targets.float()).sum(dim=(-2, -1))  # [B, N, M]

        return target_sum, non_target_sum
    
    
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
        output_logit_1 = linsat_layer(breakpoint_logit, E=E, f=f_1, no_warning=True, max_iter=100, tau=self.linsat_tau).reshape(B, N, N_atom)
        output_logit_2 = linsat_layer(breakpoint_logit, E=E, f=f_2, no_warning=True, max_iter=100, tau=self.linsat_tau).reshape(B, N, N_atom)
        output_logit_3 = linsat_layer(breakpoint_logit, E=E, f=f_3, no_warning=True, max_iter=100, tau=self.linsat_tau).reshape(B, N, N_atom)
        logit_0 = torch.zeros_like(output_logit_1)
        logits = torch.stack([logit_0, output_logit_1, output_logit_2, output_logit_3], dim=-1)
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

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        graphormer_input = batch.get("graphormer_input")
        num_atoms = batch["num_atoms"]  # [B]
        device = num_atoms.device
        batch_size = num_atoms.shape[0]
        adducts = batch["adducts"]
        collision_engs = batch["collision_engs"]
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
            root_graph = batch["root_reprs"]
            with root_graph.local_scope():
                if self.embed_adduct:
                    embed_adducts_expand = embed_adducts.repeat_interleave(
                        root_graph.batch_num_nodes(), 0
                    )
                    ndata = root_graph.ndata["h"]
                    ndata = torch.cat([ndata, embed_adducts_expand], -1)
                    root_graph.ndata["h"] = ndata                    
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
                        root_graph.batch_num_nodes(), 0
                    )
                    ndata = root_graph.ndata["h"]
                    ndata = torch.cat([ndata, embed_collision_expand], -1)
                    root_graph.ndata["h"] = ndata
                node_embeddings = self.root_module(root_graph)
                root_tokens = self.pool(root_graph, node_embeddings).unsqueeze(1)
            node_embeddings = nn_utils.pad_packed_tensor(node_embeddings, num_atoms, 0)
            max_nodes = node_embeddings.shape[1]

        # Prepare fragment query tokens and run decoder
        node_mask = torch.arange(max_nodes, device=device).unsqueeze(0) >= num_atoms.unsqueeze(1)  # [B,max_nodes]
        frag_mask = F.pad(node_mask, (1, 0, 0, 0), mode="constant", value=0).bool()

        frag_vec = self.fragment_attention(final_layer_output.transpose(0, 1), key_padding_mask=frag_mask)
        frag_card = self.softmax(self.frag_card_mapper(frag_vec))  # [B, max_frags, 4]
        # frag_logits = self.frag_logit_mapper(frag_vec, node_embeddings, node_mask)
        frag_logits = torch.bmm(frag_vec, node_embeddings.transpose(1, 2))# [B, max_frags, max_nodes]
        
        return {"frag_predicted": frag_logits, "frag_card": frag_card}
    
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
        rank_loss = torch.sum((rank_paired-frag_targs_expanded)**2, dim=-1)
        # rank_loss = torch.sum(self.binary_focal_loss(rank_paired, frag_targs_expanded, num_atoms), dim=-1)


        per_pair_cards_cross_entropy = self.cross_entropy(frag_card_predicted.unsqueeze(2), frag_cards_targs.unsqueeze(1))

        B, max_targs, _ = frag_targs_padded.shape
        
        cost = rank_loss+per_pair_cards_cross_entropy
        assign = pygm.hungarian(
            -cost, backend="pytorch", n2=num_frag_targs
        )  # [B, max_targs, max_frags]
        # assert False, (assign.shape, node_rank.shape, frag_cards_targs.shape)
        node_rank_reshape = node_rank.reshape(B, max_frags, -1)
        node_rank_assigned = torch.matmul(node_rank_reshape.transpose(1, 2), assign).transpose(1, 2)
        node_rank_assigned = node_rank_assigned.reshape(B, max_targs, max_nodes, -1)
        node_rank_assigned = torch.sum(node_rank_assigned*frag_cards_targs.unsqueeze(-2), dim=-1)
        frag_cards_predicted = torch.matmul(frag_card_predicted.transpose(1, 2), assign).transpose(1, 2)
        
        # preds = torch.matmul(preds.transpose(1, 2), assign).transpose(1, 2)
        
        loss = torch.sum((node_rank_assigned-frag_targs_padded)**2, dim=-1)+self.cross_entropy(frag_cards_predicted, frag_cards_targs)
        # loss = torch.sum(self.binary_focal_loss(node_rank_assigned, frag_targs_padded, num_atoms), dim=-1)+self.cross_entropy(frag_cards_predicted, frag_cards_targs)
        frag_targs_mask = num_frag_targs[:, None] <= torch.arange(max_targs, device=loss.device)[None, :]

        loss = loss.masked_fill(frag_targs_mask, 0)
        return torch.mean(torch.sum(loss, dim=-1)/num_frag_targs)

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
        batch_ids_unique = unique_rows[:, 0]
        pattern_counts = torch.bincount(batch_ids_unique, minlength=B)
        padded_pattern = nn_utils.pad_packed_tensor(
            unique_rows[:, 1:], pattern_counts, 0
        )  # (B, max_patterns, N)
        
        return padded_pattern, pattern_counts

    def pattern_match_recall(self, pred_mask, targ_mask, num_targs, patterns_count=None):
        """
        Compute recall using Hungarian matching between predicted and target binary patterns.

        Args:
            pred_mask: (B, P, N) predicted masks (binary / bool)
            targ_mask: (B, T, N) target masks (binary / bool)
            num_targs: (B,) number of valid target masks per batch

        Returns:
            recall: (B,) fraction of target masks matched by predicted masks under optimal assignment
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
        return recall.mean()

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
        out = self.forward(batch)
        loss = self._frag_loss(
            out["frag_predicted"], out["frag_card"], batch["frag_targs"], batch["num_frag_targs"], batch["num_atoms"], batch["adj_matrices"],
        )
        
        self.log(
            "train_loss", loss.item(), prog_bar=True, on_epoch=True, batch_size=len(batch["names"])
        )
        return {"loss": loss}


    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        out = self.forward(batch)
        loss = self._frag_loss(
            out["frag_predicted"], out["frag_card"], batch["frag_targs"], batch["num_frag_targs"], batch["num_atoms"], batch["adj_matrices"],
        )
        breakpoints = self.breakpoint_inference(out["frag_predicted"], out["frag_card"], batch["num_atoms"])
        patterns, patterns_count = self.breakpoints_to_patterns(breakpoints, batch["adj_matrices"], batch["num_atoms"])
        
        frag_targs_padded = nn_utils.pad_packed_tensor(batch["frag_targs"], batch["num_frag_targs"], 0)
        recall = self.pattern_match_recall(patterns, frag_targs_padded, batch["num_frag_targs"], patterns_count = patterns_count)
        self.log(
            "val_loss", loss.item(), prog_bar=True, on_epoch=True, batch_size=len(batch["names"])
        )
        self.log(
            "val_recall", recall.item(), prog_bar=True, on_epoch=True, batch_size=len(batch["names"])
        )

        return {"loss": loss}

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        out = self.forward(batch)
        loss = self._frag_loss(
            out["frag_predicted"], out["frag_card"], batch["frag_targs"], batch["num_frag_targs"], batch["num_atoms"], batch["adj_matrices"], debug=True
        )
        breakpoints = self.breakpoint_inference(out["frag_predicted"], out["frag_card"], batch["num_atoms"], debug=True)

        frag_targs_padded = nn_utils.pad_packed_tensor(batch["frag_targs"], batch["num_frag_targs"], 0)
        
        patterns, patterns_count = self.breakpoints_to_patterns(breakpoints, batch["adj_matrices"], batch["num_atoms"])
        frag_targs_padded = nn_utils.pad_packed_tensor(batch["frag_targs"], batch["num_frag_targs"], 0)
        recall = self.pattern_match_recall(patterns, frag_targs_padded, batch["num_frag_targs"], patterns_count = patterns_count)
        self.log(
            "test_loss", loss.item(), on_epoch=True, batch_size=len(batch["names"])
        )
        self.log(
            "test_recall", recall.item(), prog_bar=True, on_epoch=True, batch_size=len(batch["names"])
        )
        return {"loss": loss}

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        # For LambdaLR, just call step() without arguments
        scheduler.step()

    def restore_breakpoint_patterns(self, frag_predicted: torch.Tensor, frag_card: torch.Tensor) -> torch.Tensor:
        """
        Restore breakpoint patterns from model outputs during inference.

        Args:
            frag_predicted: [B, M, N] - predicted distribution over breakpoints for each fragment
            frag_card: [B, M, 4] - predicted cardinality distribution for each fragment (0, 1, 2, 3)

        Returns:
            patterns: [B, M, N] - binary mask for each fragment, top-k breakpoints selected
        """
        # Get predicted cardinality for each fragment (argmax over last dim, values in {0,1,2,3})
        k = torch.argmax(frag_card, dim=-1)  # [B, M]
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

