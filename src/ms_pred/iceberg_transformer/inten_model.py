"""Inten model """
import numpy as np
from typing import List
import json
import os
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter as ts
import dgl
import dgl.nn as dgl_nn
import copy
import pygmtools as pygm
import functools
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from ms_pred.graphormer.graphormer_graph_encoder import GraphormerGraphEncoder

import ms_pred.common as common
import ms_pred.nn_utils as nn_utils
import ms_pred.magma.fragmentation as fragmentation
import ms_pred.magma.run_magma as magma
from .dataset import TreeProcessor 

class IntenModel(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int,
        layers: int = 2,
        set_layers: int = 2,
        decoder_layers: int = 3,
        encoder_layers: int = 3,
        learning_rate: float = 7e-4,
        lr_decay_rate: float = 1.0,
        weight_decay: float = 0,
        dropout: float = 0,
        mpnn_type: str = "GGNN",
        node_feats: int = common.ELEMENT_DIM + common.MAX_H,
        edge_feats: int = 12,
        pe_embed_k: int = 0,
        max_broken_bonds: int = 6,
        root_encode: str = "gnn",
        warmup: int = 1000,
        embed_adduct=False,
        embed_collision=False,
        embed_elem_group=False,
        encode_forms: bool = False,
        loss_fn: str = "cosine",
        binned_targs:bool = False,
        sk_tau: float = 0.01,
        max_frags: int = 100,
        ppm_tol: float = 20,
        multi_hop_max_dist: int = 5,
        num_edge_dis: int = 10,
        **kwargs,
    ):
        """__init__ _summary_

        Args:
            hidden_size (int): _description_
            layers (int, optional): _description_. Defaults to 2.
            set_layers (int, optional): _description_. Defaults to 2.
            learning_rate (float, optional): _description_. Defaults to 7e-4.
            lr_decay_rate (float, optional): _description_. Defaults to 1.0.
            weight_decay (float, optional): _description_. Defaults to 0.
            dropout (float, optional): _description_. Defaults to 0.
            mpnn_type (str, optional): _description_. Defaults to "GGNN".
            node_feats (int, optional): _description_. Defaults to common.ELEMENT_DIM+common.MAX_H.
            pe_embed_k (int, optional): _description_. Defaults to 0.
            max_broken_bonds (int, optional): _description_. Defaults to 6.
            root_encode (str, optional): _description_. Defaults to "gnn".
            warmup (int, optional): _description_. Defaults to 1000.
            embed_adduct (bool, optional): _description_. Defaults to False.
            embed_collision (bool, optional): _description_. Defaults to False.
            embed_elem_group (bool, optional): _description_. Defaults to False.
            encode_forms (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: _description_
            NotImplementedError: _description_
        """
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.root_encode = root_encode
        self.pe_embed_k = pe_embed_k
        self.embed_adduct = embed_adduct
        self.embed_collision = embed_collision
        self.embed_elem_group = embed_elem_group
        self.encode_forms = encode_forms
        self.decoder_layers = decoder_layers
        self.encoder_layers = encoder_layers
        self.binned_targs = binned_targs
        self.max_frags = max_frags
        self.multi_hop_max_dist = multi_hop_max_dist
        self.num_edge_dis = num_edge_dis


        self.pool = dgl_nn.AvgPooling()

        self.formula_in_dim = 0
        if self.encode_forms:
            self.embedder = nn_utils.get_embedder("abs-sines")
            self.formula_dim = common.NORM_VEC.shape[0]

            # Calculate formula dim
            self.formula_in_dim = self.formula_dim * self.embedder.num_dim

            # Account for diffs
            self.formula_in_dim *= 2

        self.layers = layers
        self.mpnn_type = mpnn_type
        self.set_layers = set_layers

        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.dropout = dropout

        self.max_broken_bonds = max_broken_bonds
        self.sk_tau = sk_tau
        self.ppm_tol = ppm_tol
        self.mass_tol = self.ppm_tol * 1e-6
        self.cos_fn = nn.CosineSimilarity()

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

        # Define network - handle different encoding types
        if self.root_encode == "gnn":
            # Create traditional GNN for gnn encoding
            self.root_module = nn_utils.MoleculeGNN(
                hidden_size=self.hidden_size,
                num_step_message_passing=self.layers,
                set_transform_layers=self.set_layers,
                mpnn_type=self.mpnn_type,
                gnn_node_feats=node_feats + adduct_shift + collision_shift,
                gnn_edge_feats=edge_feats,
                dropout=self.dropout,
            )
        elif self.root_encode == "graphormer":
            # For pure Graphormer encoding, create Graphormer
            
            self.root_module = GraphormerGraphEncoder(
                num_atom_features=node_feats+adduct_shift+collision_shift,
                num_degree=8,  # Sufficient for molecular graphs
                num_edge_features=edge_feats, 
                num_spatial=1025,  # spatial_pos_max + 1 for padding
                num_edge_dis=self.num_edge_dis,  # Edge distance features
                edge_type="multi_hop",  # Use multi-hop edge features
                multi_hop_max_dist=self.multi_hop_max_dist,  # Maximum distance for multi-hop features
                num_encoder_layers=self.layers,  # Use layers parameter
                embedding_dim=self.hidden_size,
                ffn_embedding_dim=self.hidden_size * 4,
                num_attention_heads=8,
                dropout=self.dropout,
                attention_dropout=self.dropout,
                activation_dropout=self.dropout,
                apply_graphormer_init=True,
            )          
            # Also create transformation layer for rich tensor features
        else:
            raise ValueError(f"Unsupported root_encode: {self.root_encode}")

        self.nhead = 8
        assert decoder_layers > 0, "Decoder layers must be greater than 0"
        # self.frag_decoder = EfficientAttentionTransformerDecoder(self.decoder_layers, self.hidden_size, self.nhead, dim_feedforward=self.hidden_size * 4, dropout=self.dropout)
        # if self.encoder_layers > 0:
        #     self.inten_trans_layers = EfficientAttentionTransformerEncoder(self.encoder_layers, self.hidden_size, nhead=8, dim_feedforward=self.hidden_size * 4, dropout=self.dropout)
        frag_decoder_layer = nn.TransformerDecoderLayer(self.hidden_size, nhead=self.nhead, batch_first=True, dim_feedforward=self.hidden_size * 4, dropout=self.dropout)
        self.frag_decoder = nn.TransformerDecoder(frag_decoder_layer, self.decoder_layers)
        if self.encoder_layers > 0:
            inten_trans_layer = nn.TransformerEncoderLayer(
                self.hidden_size,
                nhead=8,
                batch_first=True,
                dim_feedforward=self.hidden_size * 4,
                dropout=self.dropout
            )
            self.inten_trans_layers = nn.TransformerEncoder(inten_trans_layer, self.encoder_layers)
        self.inten_activation = nn.Softmax(dim=-1)
        self.output_size = (self.max_broken_bonds) * 2 + 1
        self.output_map = nn.Linear(self.hidden_size, self.output_size * 2)
        self.isomer_attn_out = copy.deepcopy(self.output_map)

        self.sigmoid = nn.Sigmoid()

        # (Re)init formula embedding pieces (independent of encode_forms flag for downstream usage)
        self.embedder = nn_utils.get_embedder("abs-sines")
        self.formula_dim = common.NORM_VEC.shape[0]
        self.formula_in_dim = self.formula_dim * self.embedder.num_dim

        # Intensity buckets + fragment token mapper
        buckets = torch.DoubleTensor(np.linspace(0, 1500, 15000))
        self.inten_buckets = nn.Parameter(buckets)
        self.inten_buckets.requires_grad = False
        self.token_mapper = nn.Linear(self.hidden_size + 2 * self.formula_in_dim 
                                    + (self.max_broken_bonds + 1), self.hidden_size)

        if loss_fn == "cosine":
            self.loss_fn = self.cos_loss
        elif loss_fn == "entropy":
            self.loss_fn = self.entropy_loss
        elif loss_fn == "weighted_entropy":
            self.loss_fn = functools.partial(self.entropy_loss, weighted=True)
        else:
            raise NotImplementedError()

    def forward(
        self,
        root_repr,
        collision_engs,
        adducts,
        weights,
        adduct_mass_shifts,
        root_form_vecs,
        atom_form_vecs,
        num_atoms,
        adj_matrices=None,
        frag_targs=None,
        num_frag_targs=None,
        atom_hs=None,
        total_hs=None,
        graphormer_input=None,  # New parameter for Graphormer input format
    ):
        """forward _summary_

        Args:
            root_repr (_type_): _description_
            collision_engs (_type_): _description_
            adducts (_type_): _description_
            weights (_type_): _description_
            adduct_mass_shifts (_type_): _description_
            root_form_vecs (_type_): _description_
            atom_form_vecs (_type_): _description_
            adj_matrices (_type_, optional): Adjacency matrices of molecular graphs [B, max_nodes, max_nodes]. Defaults to None.
            frag_targs (_type_, optional): _description_. Defaults to None.
            num_frag_targs (_type_, optional): _description_. Defaults to None.
            atom_hs (_type_, optional): Hydrogen count per atom [B, max_atoms]. Defaults to None.
            total_hs (_type_, optional): Total hydrogen count per molecule [B]. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        batch_size = collision_engs.shape[0]
        embed_adducts = self.adduct_embedder[adducts.long()]
        if self.root_encode == "fp":
            raise NotImplementedError()
        elif self.root_encode == "gnn":
            # Use traditional GNN processing with DGL graphs
            with root_repr.local_scope():
                if self.embed_adduct:
                    embed_adducts_expand = embed_adducts.repeat_interleave(
                        root_repr.batch_num_nodes(), 0
                    )
                    ndata = root_repr.ndata["h"]
                    ndata = torch.cat([ndata, embed_adducts_expand], -1)
                    root_repr.ndata["h"] = ndata                    
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
                        root_repr.batch_num_nodes(), 0
                    )
                    ndata = root_repr.ndata["h"]
                    ndata = torch.cat([ndata, embed_collision_expand], -1)
                    root_repr.ndata["h"] = ndata
                                
                node_embeddings = self.root_module(root_repr)
                root_tokens = self.pool(root_repr, node_embeddings).unsqueeze(1)
                # Prepare frag decode inputs by padding node embeddings to [B, max_nodes, H]
                max_nodes = node_embeddings.shape[1]
        elif self.root_encode == "graphormer":
            # Use Graphormer for encoding
            if graphormer_input is not None:
                # Prepare adduct and collision embeddings to concatenate with graphormer_input
                
                # Start with the existing node features: [B, max_nodes, num_features]
                node_features = graphormer_input['x']  # [B, max_nodes, num_features]
                max_nodes = node_features.shape[1]
                
                # Add adduct embeddings if enabled
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
                
                # Update the modified graphormer input with enriched node features
                graphormer_input['x'] = node_features
                
                # Use Graphormer with the modified input containing adduct and collision embeddings
                inner_states, graph_rep = self.root_module(graphormer_input)
                
                # Extract node-level embeddings from final layer
                final_layer_output = inner_states[-1]  # [T, B, H] where T = n_nodes + 1
                
                # Remove graph token (first position) and transpose to get node embeddings
                node_embeddings = final_layer_output[1:].transpose(0, 1)  # [B, T-1, H]
                root_tokens = graph_rep.unsqueeze(1)

            else:
                raise ValueError("graphormer_input is required when root_encode='graphormer'")     
        else:
            pass
            
        # adj_matrices contains the adjacency matrices for the molecular graphs [B, max_nodes, max_nodes]
        # Currently available but not utilized in the model - can be used for graph-based operations
        # if adj_matrices is not None:
        #     # Future implementation: use adjacency matrices for graph convolutions, attention, etc.
        #     pass
            
        # atom_embeddings is [N_atom, H] where N_atom = total atoms across all molecules
        # frag_targs is [N1, N2] - fragment masks
        # num_frag_targs is [B] - number of fragments per batch entry
        
        # Calculate fragment form vectors by summing atom form vectors
        # atom_form_vecs: [N_total_atoms, form_dim] - flattened atom form vectors across all molecules in batch
        # frag_targs: [N1, N2] - fragment masks (1.0 for atoms in fragment, 0.0 otherwise)
        device = frag_targs.device
        atom_form_vecs_padded = nn_utils.pad_packed_tensor(atom_form_vecs, num_atoms, 0)
        atom_mask = torch.arange(node_embeddings.shape[1], device=device).unsqueeze(0) >= num_atoms.unsqueeze(1)


        # Mask logits for padded atoms so they don't contribute; large negative drives prob->0

        frag_targs_padded = nn_utils.pad_packed_tensor(frag_targs, num_frag_targs, True)

        # Now expand to fragment structure and compute per-fragment quantities in a vectorized way
        # Build fragment->molecule index mapping and expand per-molecule tensors

        # Map each fragment to its molecule index: [N1]
        frag_to_mol = torch.arange(batch_size, device=device).repeat_interleave(num_frag_targs)

        # Expand atom form vectors per fragment: [N1, N2, form_dim]
        expanded_atom_form_vecs = torch.repeat_interleave(atom_form_vecs_padded, num_frag_targs, dim=0)
        
        # Fragment masks (packed): [N1, N2]
        frag_masks = frag_targs.bool()

        # Hydrogen counts per fragment
        mol_atom_hs = atom_hs[frag_to_mol]  # [N1, N2]
        frag_hs = (frag_masks.float() * mol_atom_hs.float()).sum(dim=1)  # [N1]

        # Total H per fragment and max add/remove
        mol_total_hs = total_hs[frag_to_mol].float()  # [N1]
        max_remove = torch.clamp(frag_hs, max=self.max_broken_bonds)  # [N1]
        max_add = torch.clamp(mol_total_hs - frag_hs, max=self.max_broken_bonds)  # [N1]

        # Broken bonds per fragment (edges crossing fragment boundary)
        adj_for_frags = adj_matrices[frag_to_mol]  # [N1, N2, N2]
        non_frag_masks = ~frag_masks  # [N1, N2]
        boundary_mask = frag_masks.unsqueeze(-1) & non_frag_masks.unsqueeze(-2)  # [N1, N2, N2]
        cross_bonds = adj_for_frags * boundary_mask.float()  # [N1, N2, N2]
        num_broken = cross_bonds.sum(dim=(1, 2))  # [N1]
        
         
        num_broken_padded = nn_utils.pad_packed_tensor(num_broken, num_frag_targs, 0)
        max_add_padded = nn_utils.pad_packed_tensor(max_add, num_frag_targs, 0)
        max_remove_padded = nn_utils.pad_packed_tensor(max_remove, num_frag_targs, 0)
        # Apply fragment masks and sum to get fragment form vectors
        # frag_targs: [N1, N2] with 1.0 for atoms in fragment, 0.0 for atoms not in fragment
        masked_form_vecs = expanded_atom_form_vecs * frag_targs.unsqueeze(-1).float()  # [N1, N2, form_dim]
        fragment_form_vecs_flat = torch.sum(masked_form_vecs, dim=1)  # [N1, form_dim]
        
        # Reshape back to batch format [B, max_frags, form_dim]
        max_frags = max(num_frag_targs).item()
        fragment_form_vecs = nn_utils.pad_packed_tensor(fragment_form_vecs_flat, num_frag_targs, 0)
        root_tokens_expanded = root_tokens.expand(batch_size, max_frags, self.hidden_size)
        diffs = root_form_vecs[:, None, :] - fragment_form_vecs
        form_encodings = self.embedder(fragment_form_vecs)
        diff_encodings = self.embedder(diffs)
        # One-hot encode (clamped) broken bond counts: [B, max_frags, (max_broken_bonds+1)]
        num_broken_clamped = torch.clamp(num_broken_padded, max=self.max_broken_bonds).long()
        broken_bonds_embedded = F.one_hot(num_broken_clamped, num_classes=self.max_broken_bonds + 1).float()
        root_token_embedded = self.token_mapper(
            torch.cat([root_tokens_expanded, form_encodings, diff_encodings, broken_bonds_embedded], dim=-1)
        )
        frag_mask = torch.arange(max_frags, device=device).unsqueeze(0) >= num_frag_targs.unsqueeze(-1)  # [B, max_frags]
        frag_targs_padded = torch.repeat_interleave(frag_targs_padded, self.nhead, dim=0)
        hidden = self.frag_decoder(
            tgt=root_token_embedded,
            memory=node_embeddings,
            memory_mask=~frag_targs_padded,
            tgt_key_padding_mask=frag_mask,
        )
        if self.encoder_layers > 0:
            hidden = self.inten_trans_layers(hidden, src_key_padding_mask=frag_mask)
        
        # Hydrogen mass shifts vector
        hydrogen_shift = torch.arange(-self.max_broken_bonds, self.max_broken_bonds + 1, device=device) * common.ELEMENT_TO_MASS["H"]

        # Calculate net fragment weights (vectorized)
        # frag_targs: [N1, N2], weights: [B, N2], num_frag_targs: [B]
        weights_expanded = weights[frag_to_mol]  # [N1, N2]
        frag_targs_f = frag_targs.float()
        net_fragment_weight_flat = (weights_expanded * frag_targs_f).sum(dim=-1)  # [N1]

        # Pad back to [B, max_frags]
        net_fragment_weight = nn_utils.pad_packed_tensor(net_fragment_weight_flat, num_frag_targs, 0)
        fragment_weight = (
            net_fragment_weight[:, :, None, None]
            + hydrogen_shift[None, None, None, :]
            + adduct_mass_shifts[:, None, :, None]
        )
        fragment_weight = torch.where(fragment_weight > 0, fragment_weight, torch.zeros_like(fragment_weight))
        
        # Build mask for valid hydrogen shifts using max_add and max_remove
        # Similar to the logic in inten_model.py
        max_inten_shift = (self.output_size - 1) / 2  # Center shift for hydrogen range
        max_break_ar = torch.arange(self.output_size, device=device)[None, None, :].to(device)
        max_breaks_ub = max_add_padded + max_inten_shift  # [B, max_frags]
        max_breaks_lb = -max_remove_padded + max_inten_shift  # [B, max_frags]

        ub_mask = max_break_ar <= max_breaks_ub[:, :, None]  # [B, max_frags, output_size]
        lb_mask = max_break_ar >= max_breaks_lb[:, :, None]  # [B, max_frags, output_size]

        # B x max_frags x output_size
        valid_pos = torch.logical_and(ub_mask, lb_mask)
        valid_pos = torch.logical_and(valid_pos, ~frag_mask[:, :, None]).unsqueeze(-2)
        valid_pos = valid_pos.expand(batch_size, max_frags, 2, self.output_size).reshape(batch_size, max_frags, -1)
        masses = fragment_weight.reshape(batch_size, max_frags, -1)
    
        # B x L x Output
        output = self.output_map(hidden)
        attn_weights = self.isomer_attn_out(hidden)

        # Mask attn weights
        attn_weights.masked_fill_(~valid_pos, -99999)  # -float("inf"))


        # Calc inverse indices => B x Out x L x 2 x shift
        inverse_indices = torch.clamp(torch.bucketize(masses, self.inten_buckets, right=False), max=len(self.inten_buckets) - 1)
        # B x Out x (L * 2 * Mass shifts)

        # B x Outs x ( L * 2 * mass shifts )
        pool_weights = ts.scatter_softmax(attn_weights, index=inverse_indices, dim=-1)
        weighted_out = pool_weights * output
        weighted_out_reshape = weighted_out.reshape(batch_size, -1)
        inverse_indices_reshaped = inverse_indices.reshape(batch_size, -1)

        # B x Outs x (UNIQUE(L * 2 * mass shifts))
        output_binned = ts.scatter_add(
            weighted_out_reshape,
            index=inverse_indices_reshaped,
            dim=-1,
            dim_size=self.inten_buckets.shape[-1],
        )

        # B x Outs x binned
        valid_pos_reshape = valid_pos.expand(*inverse_indices.shape).reshape(batch_size, -1)
        valid_pos_binned = ts.scatter_max(
            (valid_pos_reshape).long(),
            index=inverse_indices_reshaped,
            dim_size=self.inten_buckets.shape[-1],
            dim=-1,
        )[0].bool()

        # Activate each dim with its respective output activation
        # Helpful for hurdle or probabilistic models
        output_binned = self.inten_activation(output_binned)
        output_binned = output_binned.masked_fill(~valid_pos_binned, 0)


        # Index into output binned using inverse_indices_reshaped
        # Revert the binned output back to frags for attribution
        # B x Out x (L * 2 * Mass shifts)
        output_unbinned = torch.take_along_dim(
            output_binned, inverse_indices_reshaped, dim=-1
        )
        max_frags = masses.shape[1]
        output_unbinned = output_unbinned.reshape(
            batch_size, max_frags, -1
        )
        output_unbinned_alpha = output_unbinned * pool_weights

        return {"output_binned": output_binned, "output": output_unbinned_alpha}

    
    def _common_step(self, batch, name="train"):
        pred_obj = self.forward(
            batch["root_reprs"],
            batch["collision_engs"],
            batch["adducts"],
            batch["weights"],
            batch["adduct_mass_shifts"],
            batch["root_form_vecs"],
            batch["atom_form_vecs"],
            batch["num_atoms"],
            adj_matrices=batch["adj_matrices"],
            frag_targs=batch["frag_targs"],
            num_frag_targs=batch["num_frag_targs"],
            atom_hs=batch["atom_hs"],
            total_hs=batch["total_hs"],
            graphormer_input=batch.get("graphormer_input", None),  # Pass Graphormer input if available
        )

        if self.binned_targs:
            pred_inten = pred_obj["output_binned"]
        else:
            pred_inten = pred_obj["output"]
            pred_inten = torch.stack((batch["masses"], pred_inten), dim=-1)
            pred_inten = pred_inten.reshape(pred_inten.shape[0], -1, 2)  # B x (Out * Mass shifts) x 2
        batch_size = len(batch["names"])

        if name == 'train':
            loss_fn = self.loss_fn
        else:
            loss_fn = functools.partial(self.loss_fn, use_hun=True)  # use hungarian in val and test
        loss = loss_fn(pred_inten, batch["inten_targs"], parent_mass=batch["precursor_mzs"])
        loss = {k: v.mean() for k, v in loss.items()}
        
        self.log(
            f"{name}_loss", loss["loss"].item(), batch_size=batch_size, on_epoch=True
        )
        for k, v in loss.items():
            if k != "loss":
                self.log(f"{name}_aux_{k}", v.item(), batch_size=batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        """training_step."""
        return self._common_step(batch, name="train")

    def validation_step(self, batch, batch_idx):
        """validation_step."""
        return self._common_step(batch, name="val")

    def test_step(self, batch, batch_idx):
        """test_step."""
        return self._common_step(batch, name="test")

    def configure_optimizers(self):
        """configure_optimizers."""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = nn_utils.build_lr_scheduler(
            optimizer=optimizer, lr_decay_rate=self.lr_decay_rate, warmup=self.warmup
        )
        ret = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": "step",
            },
        }
        return ret
    
    def cos_loss(self, pred, targ, parent_mass=None, use_hun=False):
        """cos_loss.

        Args:
            pred:
            targ:
        """
        if not self.binned_targs:
            tol = parent_mass * self.mass_tol
            mask = torch.logical_and(
                torch.abs(pred[:, :, None, 0] - targ[:, None, :, 0]) < tol[:, None, None],
                targ[:, None, :, 0] > 0
            )
            pred_norm = pred[:, :, 1].norm(dim=-1)
            targ_norm = targ[:, :, 1].norm(dim=-1)
            score = pred[:, :, None, 1] * targ[:, None, :, 1] / (pred_norm[:, None, None] * targ_norm[:, None, None])
            score = torch.where(mask, score, torch.zeros_like(score))
            target_nums = torch.sum(targ[:, :, 1] != 0, dim=-1)
            if use_hun:
                assign = pygm.hungarian(score, n2=target_nums, backend='pytorch')
            else:
                _score = torch.where(mask, score, torch.full_like(score, -1e3))
                assign = pygm.sinkhorn(_score, n2=target_nums, tau=self.sk_tau, dummy_row=True, max_iter=20, backend='pytorch')
            loss = 1 - torch.sum(assign * score, dim=(1, 2))
        else:
            loss = 1 - self.cos_fn(pred, targ)
        return {"loss": loss}

    def entropy_loss(self, pred, targ, parent_mass=None, use_hun=False, weighted=False):
        """entropy_loss.

        Args:
            pred:
            targ:
        """
        def norm_peaks(prob):
            return prob / (prob.sum(dim=-1, keepdim=True) + 1e-22)
        def entropy(prob):
            assert torch.all(torch.abs(prob.sum(dim=-1) - 1) < 1e-3), prob.sum(dim=-1)
            return -torch.sum(prob * torch.log(prob + 1e-22), dim=-1) / 1.3862943611198906 # norm by log(4)

        if not self.binned_targs:
            if weighted:
                raise NotImplementedError
            tol = parent_mass * self.mass_tol
            mask = torch.logical_and(
                torch.abs(pred[:, :, None, 0] - targ[:, None, :, 0]) < tol[:, None, None],
                targ[:, None, :, 0] > 0
            )
            score = pred[:, :, None, 1] * targ[:, None, :, 1]
            score = torch.where(mask, score, torch.zeros_like(score))
            target_nums = torch.sum(targ[:, :, 1] != 0, dim=-1)
            if use_hun:
                assign = pygm.hungarian(score, n2=target_nums, backend='pytorch')
            else:
                _score = torch.where(mask, score, torch.full_like(score, -1e3))
                assign = pygm.sinkhorn(_score, n2=target_nums, tau=self.sk_tau, dummy_row=True, max_iter=20, backend='pytorch')
            pred_norm = norm_peaks(pred[:, :, 1])
            targ_norm = norm_peaks(targ[:, :, 1])
            merged_peaks = torch.cat((
                torch.bmm(assign, targ_norm.unsueeze(-1)).squeeze(-1) + pred_norm, # usually n_pred > n_targ
                (1 - assign.sum(dim=1)) * targ_norm,
            ), dim=1)
            entropy_mix = entropy(merged_peaks)
            entropy_pred = entropy(pred_norm)
            entropy_targ = entropy(targ_norm)
            loss = 2 * entropy_mix - entropy_pred - entropy_targ

        else:
            pred_norm = norm_peaks(pred)
            targ_norm = norm_peaks(targ)
            if weighted:
                def reweight_spec(norm_spec):
                    entropy_spec = entropy(norm_spec)
                    weight = torch.where(entropy_spec < 3, 0.25 + 0.25 * entropy_spec, torch.ones_like(entropy_spec))
                    weighted_spec = norm_spec ** weight.unsqueeze(-1)
                    weighted_spec = norm_peaks(weighted_spec)
                    return weighted_spec
                pred_norm = reweight_spec(pred_norm)
                targ_norm = reweight_spec(targ_norm)
            entropy_pred = entropy(pred_norm)
            entropy_targ = entropy(targ_norm)
            entropy_mix = entropy((pred_norm + targ_norm) / 2)
            loss = 2 * entropy_mix - entropy_pred - entropy_targ
        return {"loss": loss}
    
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        # For LambdaLR, just call step() without arguments
        scheduler.step()
    
