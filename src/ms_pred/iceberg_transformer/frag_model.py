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
        mlp_layers: int = 1
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
        self.fragment_attention = nn_utils.SlotAttention(
            dim=hidden_size,
            num_slots=max_frags-1,
            iters=3
        )

        self.frag_card_mapper = nn.Linear(hidden_size, fragmentation.FRAGMENT_ENGINE_PARAMS['max_tree_depth']+1)

        self.softmax = nn.Softmax(dim=-1)
        self.KLDiv = nn.KLDivLoss(reduction='batchmean')
    
    def cross_entropy(self, preds, targets):
        log_preds = torch.log(preds + 1e-9)
        cross_entropy = -torch.sum(targets * log_preds, dim=-1)
        return cross_entropy

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
        frag_mask = F.pad(node_mask, (1, 0, 0, 0), mode="constant", value=0)

        frag_vec = self.fragment_attention(final_layer_output.transpose(0, 1), key_padding_mask=frag_mask)
        frag_logits = torch.bmm(frag_vec, node_embeddings.transpose(1, 2))  # [B, max_frags-1, max_nodes]
        frag_card = self.softmax(self.frag_card_mapper(frag_vec))
        # frag_card = F.pad(frag_card, (1, 0, 0, 0, 0, 0), mode="constant", value=0)

        atom_mask = torch.arange(max_nodes, device=device).unsqueeze(0) >= num_atoms.unsqueeze(1)  # [B,max_nodes]
        frag_logits = torch.bmm(frag_vec, node_embeddings.transpose(1, 2))# [B, max_frags-1, max_nodes]
        frag_logits = frag_logits.masked_fill(atom_mask.unsqueeze(1), -99999)
        frag_predicted = self.softmax(frag_logits)
        return {"frag_predicted": frag_predicted, "frag_card": frag_card}
    
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
        boundary_mask_batch_num = torch.cat([boundary_mask, batch_num], dim=-1).reshape(B*M, -1)
        boundary_mask_batch_num_unique = torch.unique(boundary_mask_batch_num, dim=0, sorted=True)
        batch_num_unique = torch.bincount(boundary_mask_batch_num_unique[:, -1].to(torch.int64), minlength=B)
        boundary_mask = nn_utils.pad_packed_tensor(
            boundary_mask_batch_num_unique[:, :-1], batch_num_unique, 0
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
    ) -> torch.Tensor:
        # frags_predicted: [B, max_frags, max_nodes]
        # frag_targs: packed [sum_frags, max_nodes], num_frag_targs: [B]
        B, max_frags, max_nodes = frags_predicted.shape
        
        # frag_targs_padded = nn_utils.pad_packed_tensor(
        #     frag_targs, num_frag_targs, False
        # )[:, :-1, :]  # [B, max_targs-1, max_nodes]
        frag_targs_padded = nn_utils.pad_packed_tensor(
            frag_targs, num_frag_targs, False
        )[:, :, :]  # [B, max_targs-1, max_nodes]
        boundary_info = self.boundary_nodes(adj_matrices, frag_targs_padded)
        frag_targs_padded = boundary_info["boundary_mask"][:, 1:, :]
        num_frag_targs = boundary_info["unique_boundary_patterns"]-1
        num_frag_targs = torch.where(num_frag_targs > 0, num_frag_targs, torch.ones_like(num_frag_targs))

        frag_cards_targs = F.one_hot(torch.sum(frag_targs_padded, dim=-1).long(), num_classes=fragmentation.FRAGMENT_ENGINE_PARAMS['max_tree_depth']+1)
        per_pair_cards_cross_entropy = self.cross_entropy(frag_card_predicted.unsqueeze(1), frag_cards_targs.unsqueeze(2))
        B, max_targs, _ = frag_targs_padded.shape
        
        frag_targs_padded = F.normalize(frag_targs_padded.float(), p=1, dim=-1)

        scores_exp = frags_predicted.unsqueeze(1).expand(B, max_targs, max_frags, max_nodes)
        targs_exp = frag_targs_padded.unsqueeze(2).expand_as(scores_exp)
        
        per_pair_cross_entropy = self.cross_entropy(scores_exp, targs_exp)  # [B, max_targs, max_frags, max_nodes]
        cost = (per_pair_cross_entropy+per_pair_cards_cross_entropy)
        n2_cols = torch.full_like(num_frag_targs, max_frags)
        assign = pygm.hungarian(
            -cost, backend="pytorch", n1=num_frag_targs, n2=n2_cols
        )  # [B, max_targs, max_frags]

        loss_assigned = (cost * assign).sum(dim=(1, 2))/(num_frag_targs*num_atoms)
        loss_assigned = torch.where(boundary_info["unique_boundary_patterns"] > 1, loss_assigned, torch.zeros_like(loss_assigned))
        return loss_assigned.mean()

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
        
        return padded_pattern[:, 1:, :], pattern_counts
    
    def pattern_match_recall(self, pred_mask, targ_mask, num_targs):
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


        # Use Hungarian to find optimal one-to-one matching that maximizes total exact matches
        # pygm.hungarian expects a score matrix (higher better). Provide n1=T (targets), n2=P (preds)
        assign = pygm.hungarian(sim, backend="pytorch", n1=torch.full((B,), T, dtype=torch.long, device=device), n2=torch.full((B,), P, dtype=torch.long, device=device))
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
        eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
        R = R | eye

        if mask is not None:
            row_mask = mask.unsqueeze(-1)
            col_mask = mask.unsqueeze(-2)
            A = A & ~row_mask & ~col_mask  # zero edges touching removed nodes

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
        breakpoints = self.restore_breakpoint_patterns(out["frag_predicted"], out["frag_card"])
        patterns, patterns_count = self.breakpoints_to_patterns(breakpoints, batch["adj_matrices"], batch["num_atoms"])
        
        frag_targs_padded = nn_utils.pad_packed_tensor(batch["frag_targs"], batch["num_frag_targs"], 0)
        recall = self.pattern_match_recall(patterns, frag_targs_padded, batch["num_frag_targs"])
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
            out["frag_predicted"], out["frag_card"], batch["frag_targs"], batch["num_frag_targs"], batch["num_atoms"], batch["adj_matrices"],
        )
        breakpoints = self.restore_breakpoint_patterns(out["frag_predicted"], out["frag_card"])
        patterns, patterns_count = self.breakpoints_to_patterns(breakpoints, batch["adj_matrices"], batch["num_atoms"])
        patterns = F.pad(patterns, (0, 0, 1, 0, 0, 0), mode="constant", value=1)
        frag_targs_padded = nn_utils.pad_packed_tensor(batch["frag_targs"], batch["num_frag_targs"], 0)
        recall = self.pattern_match_recall(patterns, frag_targs_padded, batch["num_frag_targs"])
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

