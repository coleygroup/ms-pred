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
import torch_scatter as ts
import torch.nn.functional as F
import copy
import ms_pred.common as common
from LinSATNet import linsat_layer, init_constraints
import math
import numpy as np
from ms_pred.iceberg_transformer.dataset import TreeProcessor
import dgl
from rdkit import Chem  # type: ignore
import functools

class JointModel(pl.LightningModule):
    def __init__(self, 
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
            embed_instrument: bool = False, 
            encode_forms: bool = False,
            linsat_tau: float = 0.01,
            max_broken_bonds: int = 6,
            pe_embed_k: int = 0,
            enable_aux_loss: bool = False,
            enable_decoder_norm: bool = False,
            inten_decoder_layers: int = 3,
            inten_encoder_layers: int = 3,
            inten_dropout: float = 0,
            inten_loss_fn: str = "cosine",
            binned_targs:bool = False,
            sk_tau: float = 0.01,
            ppm_tol: float = 20,
            contr_weight: float = 1.0,
            contr_threshold: float = 0.5,
            contr_loss_fn: str = "entropy",
            inten_weight: float = 1,
            frag_weight: float = 0.1,
            graphormer_dropout: float = 0.05,
            graphormer_layers: int = 5,
            hidden_size: int = 512,
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
        self.embed_instrument = embed_instrument
        self.linsat_tau = linsat_tau
        self.max_broken_bonds = max_broken_bonds
        self.pe_embed_k = pe_embed_k
        self.enable_aux_loss = enable_aux_loss
        self.enable_decoder_norm = enable_decoder_norm

        self.inten_decoder_layers = inten_decoder_layers
        self.inten_encoder_layers = inten_encoder_layers
        self.inten_dropout = inten_dropout
        self.inten_loss_fn = inten_loss_fn
        self.binned_targs = binned_targs
        self.sk_tau = sk_tau
        self.ppm_tol = ppm_tol
        self.contr_weight = contr_weight
        self.contr_loss_fn = contr_loss_fn
        self.contr_threshold = contr_threshold
        self.graphormer_dropout = graphormer_dropout
        self.graphormer_layers = graphormer_layers
        self.hidden_size=hidden_size
        self.nhead=8


        self.tree_processor = TreeProcessor(
            pe_embed_k=pe_embed_k,
            root_encode="graphormer",
            embed_elem_group=embed_elem_group,
            multi_hop_max_dist=multi_hop_max_dist,
        )

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

        instrument_shift = 0
        if self.embed_instrument:
            instrument_types = len(set(common.instrument2onehot_pos.values()))
            onehot_types = torch.eye(instrument_types)
            self.instrument_embedder = nn.Parameter(onehot_types.float())
            self.instrument_embedder.requires_grad = False
            instrument_shift = instrument_types
        self.root_module = GraphormerGraphEncoder(
            num_atom_features=node_feats+adduct_shift+collision_shift+instrument_shift,
            num_degree=8,  # Sufficient for molecular graphs
            num_edge_features=edge_feats, 
            num_spatial=1025,  # spatial_pos_max + 1 for padding
            num_edge_dis=self.num_edge_dis,  # Edge distance features
            edge_type="multi_hop",  # Use multi-hop edge features
            multi_hop_max_dist=self.multi_hop_max_dist,  # Maximum distance for multi-hop features
            num_encoder_layers=self.graphormer_layers,  # Use layers parameter
            embedding_dim=self.hidden_size,
            ffn_embedding_dim=4*self.hidden_size,
            num_attention_heads=self.nhead,
            dropout=self.graphormer_dropout,
            attention_dropout=self.graphormer_dropout,
            activation_dropout=self.graphormer_dropout,
            apply_graphormer_init=True,
        )
        if self.encode_forms:
            self.embedder = nn_utils.get_embedder("abs-sines")
            self.formula_dim = common.NORM_VEC.shape[0]

            # Calculate formula dim
            self.formula_in_dim = self.formula_dim * self.embedder.num_dim
            self.formula_mapper = nn.Linear(self.formula_in_dim+self.hidden_size, self.hidden_size)
        buckets = torch.DoubleTensor(np.linspace(0, 1500, 15000))
        self.inten_buckets = nn.Parameter(buckets)
        self.inten_buckets.requires_grad = False
        token_size = self.hidden_size + 2 * self.formula_in_dim + (self.max_broken_bonds + 1)
        self.token_mapper = nn.Linear(token_size, self.hidden_size)
        self.enable_decoder_norm = enable_decoder_norm
        self.fragment_decoder = nn_utils.SlotDecoder(
            hidden_dim=hidden_size,
            num_slots=max_breakpoints,
            nhead=self.nhead,
            num_layers=self.frag_decoder_layers,
            dropout=frag_dropout,
            enable_norm=self.enable_decoder_norm
        )
        if self.frag_encoder_layers > 0:
            fragment_encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=self.nhead,
                dim_feedforward=hidden_size * 4,
                dropout=frag_dropout,
                batch_first=True,
            )
            self.fragment_encoder = nn.TransformerEncoder(
                fragment_encoder_layer,
                num_layers=self.frag_encoder_layers,
            )
        self.tree_processor = TreeProcessor(
            pe_embed_k=pe_embed_k,
            root_encode="graphormer",
            embed_elem_group=embed_elem_group,
            multi_hop_max_dist=multi_hop_max_dist,
        )
        self.frag_card_mapper = nn.Linear(hidden_size, fragmentation.FRAGMENT_ENGINE_PARAMS['max_tree_depth']+1)

        inten_decoder_layer = nn.TransformerDecoderLayer(self.hidden_size, nhead=self.nhead, batch_first=True, dim_feedforward=self.hidden_size * 4, dropout=self.inten_dropout)
        self.inten_decoder = nn.TransformerDecoder(inten_decoder_layer, self.inten_decoder_layers)
        if self.inten_encoder_layers > 0:
            inten_encoder_layer = nn.TransformerEncoderLayer(
                self.hidden_size,
                nhead=self.nhead,
                batch_first=True,
                dim_feedforward=self.hidden_size * 4,
                dropout=self.inten_dropout
            )
            self.inten_encoder = nn.TransformerEncoder(inten_encoder_layer, self.inten_encoder_layers)
        self.inten_activation = nn.Softmax(dim=-1)
        self.output_size = (self.max_broken_bonds) * 2 + 1
        self.output_map = nn.Linear(self.hidden_size, self.output_size * 2)
        self.isomer_attn_out = copy.deepcopy(self.output_map)
        self.cos_fn = nn.CosineSimilarity()
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
        if inten_loss_fn == "cosine":
            self.inten_loss_fn = self.cos_loss
        elif inten_loss_fn == "entropy":
            self.inten_loss_fn = self.entropy_loss
        elif inten_loss_fn == "weighted_entropy":
            self.inten_loss_fn = functools.partial(self.entropy_loss, weighted=True)
        else:
            raise NotImplementedError()

        if contr_loss_fn == "cosine":
            self.contr_loss_fn = self.cos_loss
        elif contr_loss_fn == "entropy":
            self.contr_loss_fn = self.entropy_loss
        elif contr_loss_fn == "weighted_entropy":
            self.contr_loss_fn = functools.partial(self.entropy_loss, weighted=True)
        else:
            raise NotImplementedError()

    def molecular_embedding(self, adducts, collision_engs, graphormer_input=None, instruments=None):
        embed_adducts = self.adduct_embedder[adducts.long()]
        batch_size = collision_engs.shape[0]
        # Use Graphormer for encoding
        if graphormer_input is not None:
            # Prepare adduct and collision embeddings to concatenate with graphormer_input
            
            # Start with the existing node features: [B, max_nodes, num_features]
            original_node_features = node_features = graphormer_input['x']  # [B, max_nodes, num_features]
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
            
            if self.embed_instrument:
                embed_instruments = self.instrument_embedder[instruments.long()]
                embed_instruments_expanded = embed_instruments.unsqueeze(1).expand(batch_size, max_nodes, -1)
                node_features = torch.cat([node_features, embed_instruments_expanded], dim=-1)

            # Update the modified graphormer input with enriched node features
            graphormer_input['x'] = node_features
            
            # Use Graphormer with the modified input containing adduct and collision embeddings
            inner_states, graph_rep = self.root_module(graphormer_input)
            
            # Extract node-level embeddings from final layer
            final_layer_output = inner_states[-1]  # [T, B, H] where T = n_nodes + 1
            
            # Remove graph token (first position) and transpose to get node embeddings
            node_embeddings = final_layer_output[1:].transpose(0, 1)  # [B, T-1, H]
            root_tokens = graph_rep.unsqueeze(1)
            graphormer_input['x'] = original_node_features
        else:
            raise ValueError("graphormer_input is required")     

        return {"root_tokens":root_tokens, "node_embeddings":node_embeddings}
    
    def breakpoint_forward(self, node_embeddings, root_tokens, num_atoms, root_form_vecs):
        batch_size, max_nodes, _ = node_embeddings.shape
        device = node_embeddings.device

        node_mask = torch.arange(max_nodes, device=device).unsqueeze(0) >= num_atoms.unsqueeze(1)  # [B,max_nodes]
        frag_mask = F.pad(node_mask, (1, 0, 0, 0), mode="constant", value=0).bool()
        if self.encode_forms:
            encoded_form = self.embedder(root_form_vecs)[:, None, :]
            root_tokens = self.formula_mapper(torch.cat((root_tokens, encoded_form), dim=-1))
            frag_vecs = self.fragment_decoder(root_tokens, node_embeddings, memory_key_padding_mask=frag_mask)
        if self.frag_encoder_layers > 0:
            frag_vecs_flatten = frag_vecs.reshape(-1, self.max_breakpoints, self.hidden_size)
            frag_vecs_encoded = self.fragment_encoder(frag_vecs_flatten)
            frag_vecs = frag_vecs_encoded.reshape(self.frag_decoder_layers, batch_size, self.max_breakpoints, self.hidden_size)
        frag_card_logits = self.frag_card_mapper(frag_vecs)  # [num_layers, B, max_breakpoints, 4]
        frag_logits = torch.einsum("nbij,bkj->nbik", frag_vecs, node_embeddings)
        return {"frag_logits": frag_logits, "frag_card_logits": frag_card_logits}
    
    def inten_calculation(self, root_tokens, node_embeddings, frag_targs, num_frag_targs, root_form_vecs, atom_form_vecs, num_atoms, atom_hs, 
                          total_hs, adj_matrices, adduct_mass_shifts, masses):
        batch_size = root_tokens.shape[0]
        device = frag_targs.device
        atom_form_vecs_padded = nn_utils.pad_packed_tensor(atom_form_vecs, num_atoms, 0)

        frag_targs_padded = nn_utils.pad_packed_tensor(frag_targs, num_frag_targs, True)

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
        max_frags = frag_targs_padded.shape[1]
        fragment_form_vecs = nn_utils.pad_packed_tensor(fragment_form_vecs_flat, num_frag_targs, 0)
        root_tokens_expanded = root_tokens.expand(batch_size, max_frags, self.hidden_size)
        diffs = root_form_vecs[:, None, :] - fragment_form_vecs
        form_encodings = self.embedder(fragment_form_vecs)
        diff_encodings = self.embedder(diffs)
        # One-hot encode (clamped) broken bond counts: [B, max_frags, (max_broken_bonds+1)]
        num_broken_clamped = torch.clamp(num_broken_padded, max=self.max_broken_bonds).long()
        broken_bonds_embedded = F.one_hot(num_broken_clamped, num_classes=self.max_broken_bonds + 1).float()
        token_list = [root_tokens_expanded, form_encodings, diff_encodings, broken_bonds_embedded]
        root_token_embedded = self.token_mapper(
            torch.cat(token_list, dim=-1)
        )
        frag_mask = torch.arange(max_frags, device=device).unsqueeze(0) >= num_frag_targs.unsqueeze(-1)  # [B, max_frags]
        frag_targs_padded = torch.repeat_interleave(frag_targs_padded, self.nhead, dim=0)
        hidden = self.inten_decoder(
            tgt=root_token_embedded,
            memory=node_embeddings,
            memory_mask=~frag_targs_padded,
            tgt_key_padding_mask=frag_mask,
        )
        if self.inten_encoder_layers > 0:
            hidden = self.inten_encoder(hidden, src_key_padding_mask=frag_mask)
        
        # Hydrogen mass shifts vector
        hydrogen_shift = torch.arange(-self.max_broken_bonds, self.max_broken_bonds + 1, device=device) * common.ELEMENT_TO_MASS["H"]

        # Calculate net fragment masses (vectorized)
        # frag_targs: [N1, N2], masses: [B, N2], num_frag_targs: [B]
        masses_expanded = masses[frag_to_mol]  # [N1, N2]
        frag_targs_f = frag_targs.float()
        net_fragment_mass_flat = (masses_expanded * frag_targs_f).sum(dim=-1)  # [N1]

        # Pad back to [B, max_frags]
        net_fragment_mass = nn_utils.pad_packed_tensor(net_fragment_mass_flat, num_frag_targs, 0)
        fragment_mass = (
            net_fragment_mass[:, :, None, None]
            + hydrogen_shift[None, None, None, :]
            + adduct_mass_shifts[:, None, :, None]
        )
        fragment_mass = torch.where(fragment_mass > 0, fragment_mass, torch.zeros_like(fragment_mass))
        
        # Build mask for valid hydrogen shifts using max_add and max_remove
        max_inten_shift = (self.output_size - 1) / 2  # Center shift for hydrogen range
        max_break_ar = torch.arange(self.output_size, device=device)[None, None, :]
        max_breaks_ub = max_add_padded + max_inten_shift  # [B, max_frags]
        max_breaks_lb = -max_remove_padded + max_inten_shift  # [B, max_frags]

        ub_mask = max_break_ar <= max_breaks_ub[:, :, None]  # [B, max_frags, output_size]
        lb_mask = max_break_ar >= max_breaks_lb[:, :, None]  # [B, max_frags, output_size]

        # B x max_frags x output_size
        valid_pos = torch.logical_and(ub_mask, lb_mask)
        valid_pos = torch.logical_and(valid_pos, ~frag_mask[:, :, None]).unsqueeze(-2)
        valid_pos = valid_pos.expand(batch_size, max_frags, 2, self.output_size).reshape(batch_size, max_frags, -1)
        masses = fragment_mass.reshape(batch_size, max_frags, -1)
    
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
        output_binned = output_binned.masked_fill(~valid_pos_binned, -99999)
        output_binned = self.inten_activation(output_binned)


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
        
        return {"output_binned": output_binned, "output": output_unbinned}

    def forward(self, graphormer_input, num_atoms, adducts, collision_engs, root_form_vecs, masses, 
                adduct_mass_shifts, atom_form_vecs, adj_matrices, atom_hs, total_hs, instruments=None,
                include_magma=False, frag_targs=None, num_frag_targs=None, is_decoy=False):
        mol_embeddings = self.molecular_embedding(adducts, collision_engs, graphormer_input=graphormer_input, instruments=instruments)
        root_tokens = mol_embeddings["root_tokens"]
        node_embeddings = mol_embeddings["node_embeddings"]
        if not is_decoy:
            breakpoints_pred = self.breakpoint_forward(node_embeddings, root_tokens, num_atoms, root_form_vecs)
        else:
            with torch.no_grad():
                breakpoints_pred = self.breakpoint_forward(node_embeddings, root_tokens, num_atoms, root_form_vecs)
        frag_logits = breakpoints_pred["frag_logits"][-1]
        frag_card_logits = breakpoints_pred["frag_card_logits"][-1]
        with torch.no_grad():
            breakpoints = self.breakpoint_inference(frag_logits, frag_card_logits, num_atoms)
            fragments, fragment_count = self.breakpoints_to_patterns(breakpoints, adj_matrices, num_atoms)
            fragments = nn_utils.pack_padded_tensor(fragments, fragment_count).bool()
        if include_magma:
            batch_size = root_tokens.shape[0]
            all_frag_targs = torch.cat([frag_targs, fragments], dim=0)
            root_tokens = root_tokens[None, :, :].expand(2, -1, 1, -1).reshape(-1, 1, root_tokens.shape[2])
            node_embeddings = node_embeddings[None, :, :, :].expand(2, -1, -1, -1).reshape(-1, node_embeddings.shape[1], node_embeddings.shape[2])
            all_num_frag_targs = torch.cat([num_frag_targs, fragment_count], dim=0)
            root_form_vecs = root_form_vecs[None, :, :].expand(2, -1, -1).reshape(-1, root_form_vecs.shape[1])
            atom_form_vecs = atom_form_vecs[None, :, :].expand(2, -1, -1).reshape(-1, atom_form_vecs.shape[1])
            num_atoms = num_atoms[None, :].expand(2, -1).reshape(-1)
            atom_hs = atom_hs[None, :, :].expand(2, -1, -1).reshape(-1, atom_hs.shape[1])
            total_hs = total_hs[None, :].expand(2, -1).reshape(-1)
            adj_matrices = adj_matrices[None, :, :, :].expand(2, -1, -1, -1).reshape(-1, adj_matrices.shape[1], adj_matrices.shape[2])
            adduct_mass_shifts = adduct_mass_shifts[None, :, :].expand(2, -1, -1).reshape(-1, adduct_mass_shifts.shape[1])
            masses = masses[None, :, :].expand(2, -1, -1).reshape(-1, masses.shape[1])
            inten_pred = self.inten_calculation(
                root_tokens,
                node_embeddings,
                all_frag_targs,
                all_num_frag_targs,
                root_form_vecs,
                atom_form_vecs,
                num_atoms, atom_hs,
                total_hs,
                adj_matrices,
                adduct_mass_shifts,
                masses,
            )
            inten_pred_magma = {k: v[:batch_size] for k, v in inten_pred.items()}
            inten_pred_end_to_end = {k: v[batch_size:] for k, v in inten_pred.items()}
        else:
            inten_pred_magma=None
            inten_pred_end_to_end = self.inten_calculation(
                                    root_tokens,
                                    node_embeddings,
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
        max_flat_k = flat_k.max().item()
        if max_flat_k > 0:
            topk_vals, topk_idx = torch.topk(flat_preds, k=max_flat_k, dim=-1)
            arange_k = torch.arange(max_flat_k, device=breakpoint_preds.device).unsqueeze(0)  # [1, max_k]
            valid_mask = arange_k < flat_k.unsqueeze(1)  # [(B*N), max_k]
            batch_idx = torch.arange(flat_preds.shape[0], device=breakpoint_preds.device).unsqueeze(1).expand(-1, max_flat_k)  # [(B*N), max_k]
            patterns[batch_idx[valid_mask], topk_idx[valid_mask]] = True
        out = patterns.view(B, N, N_atom)
        return out
    
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
        debug_mask = torch.arange(mask.shape[-1], device=mask.device)[None, :] < num_nodes[:, None]
        debug_mask = torch.logical_and(debug_mask, pattern_counts[:, None]==0)
        padded_pattern[:, 0, :] = torch.logical_or(padded_pattern[:, 0, :], debug_mask)
        pattern_counts = torch.clamp(pattern_counts, min=1)
        return padded_pattern, pattern_counts
    
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


        # Floyd-Warshall style reachability: A[i,j] = A[i,j] or (A[i,k] and A[k,j]) for all k
        for k in range(N):
            A = A | (A[:, :, k:k+1] & A[:, k:k+1, :])

        if num_nodes is not None:
            # mask out rows/cols beyond num_nodes
            row_idx = torch.arange(N, device=device).unsqueeze(0)  # (1, N)
            col_idx = torch.arange(N, device=device).unsqueeze(0)  # (1, N)
            valid_row = row_idx < num_nodes.unsqueeze(1)  # (B, N)
            valid_col = col_idx < num_nodes.unsqueeze(1)  # (B, N)
            valid_matrix = valid_row.unsqueeze(-1) & valid_col.unsqueeze(-2)  # (B, N, N)
            valid_matrix = torch.repeat_interleave(valid_matrix, repeats=A.shape[0]//valid_matrix.shape[0], dim=0)
            A = A & valid_matrix

        return A

    def _common_step(self, batch, name="train"):
        if "decoy" not in batch:
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
                instruments=batch["instruments"] if self.embed_instrument else None,
                include_magma=name=="train",
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
                loss_fn = functools.partial(self.inten_loss_fn, use_hun=True)  # use hungarian in val and test
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
                inten_loss_fn = self.inten_loss_fn
                end_to_end_inten_loss = inten_loss_fn(pred_inten, batch["inten_targs"], parent_mass=batch["precursor_mzs"])["loss"].mean()
                magma_inten_loss = inten_loss_fn(pred_inten_magma, batch["inten_targs"], parent_mass=batch["precursor_mzs"])["loss"].mean()

                breakpoints_pred = pred["breakpoints_pred"]
                if self.enable_aux_loss:
                    frag_loss = 0
                    for i in range(self.frag_decoder_layers):
                        frag_loss += self.frag_loss(
                            breakpoints_pred["frag_logits"][i], breakpoints_pred["frag_card_logits"][i], 
                            batch["frag_targs"], batch["num_frag_targs"], batch["num_atoms"], batch["adj_matrices"],
                        ) * 0.9**(self.frag_decoder_layers-1-i)
                else:
                    frag_loss = self.frag_loss(
                        breakpoints_pred["frag_logits"][-1], breakpoints_pred["frag_card_logits"][-1], 
                        batch["frag_targs"], batch["num_frag_targs"], batch["num_atoms"], batch["adj_matrices"],
                    )
                self.step += 1
                self.log("train_end_to_end_inten_loss", end_to_end_inten_loss, batch_size=batch_size, on_step=True)
                self.log("train_magma_inten_loss", magma_inten_loss, batch_size=batch_size, on_step=True)
                self.log("train_frag_loss", frag_loss, batch_size=batch_size, on_step=True)
                
                magma_weight = self.magma_weight_scheduler()
                if magma_weight == 1:
                    loss = magma_inten_loss * self.inten_weight + self.frag_weight * frag_loss
                else:
                    loss = self.inten_weight * (magma_weight * magma_inten_loss + (1-magma_weight)*end_to_end_inten_loss) + self.frag_weight * magma_weight * frag_loss
                self.log("train_loss", loss, batch_size=batch_size, on_step=True)
                return {"loss":loss}
        else:
            batch_size = batch["num_decoys_per_entry"].shape[0]
            targ_batch = batch["targ"]
            decoy_batch = batch["decoy"]
            if name == 'train':
                inten_loss_fn = self.inten_loss_fn
                inten_decoy_loss_fn = self.contr_loss_fn
            else:
                inten_loss_fn = functools.partial(self.inten_loss_fn, use_hun=True)  # use hungarian in val and test
                inten_decoy_loss_fn = functools.partial(self.contr_loss_fn, use_hun=True)
            
            targ_pred = self.forward(
                graphormer_input=targ_batch.get("graphormer_input"),
                num_atoms=targ_batch["num_atoms"],
                adducts=targ_batch["adducts"],
                collision_engs=targ_batch["collision_engs"],
                root_form_vecs=targ_batch["root_form_vecs"],
                masses=targ_batch["masses"],
                adduct_mass_shifts=targ_batch["adduct_mass_shifts"],
                atom_form_vecs=targ_batch["atom_form_vecs"],
                adj_matrices=targ_batch["adj_matrices"],
                atom_hs=targ_batch["atom_hs"],
                total_hs=targ_batch["total_hs"],
                instruments=targ_batch["instruments"] if self.embed_instrument else None,
                include_magma=False,
                frag_targs=None,
                num_frag_targs=None,
            )
            decoy_pred = self.forward(
                graphormer_input=decoy_batch.get("graphormer_input"),
                num_atoms=decoy_batch["num_atoms"],
                adducts=decoy_batch["adducts"],
                collision_engs=decoy_batch["collision_engs"],
                root_form_vecs=decoy_batch["root_form_vecs"],
                masses=decoy_batch["masses"],
                adduct_mass_shifts=decoy_batch["adduct_mass_shifts"],
                atom_form_vecs=decoy_batch["atom_form_vecs"],
                adj_matrices=decoy_batch["adj_matrices"],
                atom_hs=decoy_batch["atom_hs"],
                total_hs=decoy_batch["total_hs"],
                instruments=decoy_batch["instruments"] if self.embed_instrument else None,
                include_magma=False,
                is_decoy=True,
                frag_targs=None,
                num_frag_targs=None,
            )
            decoy_inten_targs = targ_batch["inten_targs"].repeat_interleave(batch["num_decoys_per_entry"], dim=0)
            end_to_end_inten_loss = inten_loss_fn(targ_pred["inten_pred_end_to_end"]["output_binned"], targ_batch["inten_targs"], parent_mass=targ_batch["precursor_mzs"])["loss"]
            decoy_spec_loss = inten_decoy_loss_fn(decoy_pred["inten_pred_end_to_end"]["output_binned"], decoy_inten_targs, parent_mass=decoy_batch["precursor_mzs"])["loss"]
            targ_contr_loss = inten_decoy_loss_fn(targ_pred["inten_pred_end_to_end"]["output_binned"], targ_batch["inten_targs"], parent_mass=targ_batch["precursor_mzs"])["loss"]
            split_end = torch.cumsum(batch["num_decoys_per_entry"], dim=0)
            split_start = split_end - batch["num_decoys_per_entry"]
            decoy_spec_loss = [decoy_spec_loss[s:e] for s, e in zip(split_start, split_end)]
            decoy_spec_loss = torch.nn.utils.rnn.pad_sequence(decoy_spec_loss, batch_first=True, padding_value=1) # cos_loss <=1 by definition
            decoy_spec_loss = torch.cat((targ_contr_loss.unsqueeze(1), decoy_spec_loss), dim=1)
            decoy_spec_loss_sorted = torch.sort(decoy_spec_loss, dim=-1).values.detach()
            ranking_dist = torch.abs(decoy_spec_loss[:, :, None] - decoy_spec_loss_sorted[:, None, :])
            top1_prob = pygm.sinkhorn(-ranking_dist, n1=batch["num_decoys_per_entry"]+1, n2=batch["num_decoys_per_entry"]+1, tau=self.sk_tau, backend='pytorch')[:, 0, 0]
            contr_loss = torch.relu(-torch.log(top1_prob + self.contr_threshold))  # shift & cut ce loss for probs > 0.5
            if name != "train":  
                loss = {
                    "spec_loss": end_to_end_inten_loss,
                    "contr_loss": contr_loss,
                    "loss": end_to_end_inten_loss + contr_loss * self.contr_weight,
                }
                loss = {k: v.mean() for k, v in loss.items()}      
                self.log(
                    f"{name}_loss", loss["loss"].item(), batch_size=batch_size, on_epoch=True
                )
                for k, v in loss.items():
                    if k != "loss":
                        self.log(f"{name}_aux_{k}", v.item(), batch_size=batch_size)
                return loss
            else:
                breakpoints_pred = targ_pred["breakpoints_pred"]
                if self.enable_aux_loss:
                    frag_loss = 0
                    for i in range(self.frag_decoder_layers):
                        frag_loss += self.frag_loss(
                            breakpoints_pred["frag_logits"][i], breakpoints_pred["frag_card_logits"][i], 
                            targ_batch["frag_targs"], targ_batch["num_frag_targs"], targ_batch["num_atoms"], targ_batch["adj_matrices"],
                        ) * 0.9**(self.frag_decoder_layers-1-i)
                else:
                    frag_loss = self.frag_loss(
                        breakpoints_pred["frag_logits"][-1], breakpoints_pred["frag_card_logits"][-1], 
                        targ_batch["frag_targs"], targ_batch["num_frag_targs"], targ_batch["num_atoms"], targ_batch["adj_matrices"],
                    )
                self.step += 1
                end_to_end_inten_loss = end_to_end_inten_loss.mean()
                contr_loss = contr_loss.mean()
                self.log("train_end_to_end_inten_loss", end_to_end_inten_loss.item(), batch_size=batch_size, on_epoch=True)
                self.log("train_frag_loss", frag_loss.item(), batch_size=batch_size, on_epoch=True)
                self.log("train_contr_loss", contr_loss.item(), batch_size=batch_size, on_epoch=True)
                magma_weight = self.magma_weight_scheduler()
                loss = self.inten_weight * end_to_end_inten_loss + self.frag_weight * magma_weight * frag_loss + contr_loss * self.contr_weight
                self.log("train_loss", loss.item(), batch_size=batch_size, on_epoch=True)
                return {"loss": loss}

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

    def predict_mol(self, smi, collision_eng, adduct, device, instrument=None, binned_out=False):
        if not getattr(self, "_predict_prepared", False):
            self.eval()
            self.freeze()
            self._predict_prepared = True
        root_smi = smi
        if type(root_smi) is str:
            batched_input = False
            root_smi = [root_smi]
            collision_eng = [collision_eng]
            adduct = [adduct]
            if self.embed_instrument:
                instrument = [instrument]
        else:
            batched_input = True
        batch_size = len(root_smi)
        to_tensor = lambda x: torch.tensor(x, device=device, dtype=torch.float) if x is not None else x
        instruments = to_tensor([common.instrument2onehot_pos[i] for i in instrument]) if self.embed_instrument else None
        adducts = to_tensor([common.ion2onehot_pos[a] for a in adduct])
        collision_engs = to_tensor(collision_eng)
        mols = [Chem.MolFromSmiles(rsmi) for rsmi in root_smi]
        graphormer_inputs = [self.tree_processor.create_graphormer_input(mol=m, multi_hop_max_dist=self.tree_processor.multi_hop_max_dist) for m in mols]
        num_atoms = torch.tensor([gf['num_atoms'] for gf in graphormer_inputs], dtype=torch.long, device=device)
        max_nodes_gf = max(gf['x'].shape[0] for gf in graphormer_inputs)
        max_dist = max(gf['edge_input'].shape[2] for gf in graphormer_inputs)
        node_feat_dim = graphormer_inputs[0]['x'].shape[1]
        edge_feat_dim = graphormer_inputs[0]['attn_edge_type'].shape[2]
        batch_size = len(graphormer_inputs)
        x_batch = torch.zeros([batch_size, max_nodes_gf, node_feat_dim], dtype=torch.float32, device=device)
        attn_bias_batch = torch.full([batch_size, max_nodes_gf + 1, max_nodes_gf + 1], -99999, dtype=torch.float32, device=device)
        attn_edge_type_batch = torch.zeros([batch_size, max_nodes_gf, max_nodes_gf, edge_feat_dim], dtype=torch.float32, device=device)
        spatial_pos_batch = torch.zeros([batch_size, max_nodes_gf, max_nodes_gf], dtype=torch.long, device=device)
        degree_batch = torch.zeros([batch_size, max_nodes_gf], dtype=torch.long, device=device)
        edge_input_batch = torch.zeros([batch_size, max_nodes_gf, max_nodes_gf, max_dist, edge_feat_dim], dtype=torch.float32, device=device)
        adj_matrices = [Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True) for mol in mols]
        adj_matrices = [torch.from_numpy(adj_matrix).float().to(device) for adj_matrix in adj_matrices]
        max_nodes = torch.max(num_atoms).item()
        adj_matrices_batch = torch.zeros([batch_size, max_nodes, max_nodes], dtype=torch.float32, device=device)
        for i, adj in enumerate(adj_matrices):
            adj_matrices_batch[i, :adj.shape[0], :adj.shape[1]] = adj
        for i, gf_input in enumerate(graphormer_inputs):
            num_nodes = gf_input['x'].shape[0]
            edge_dist = gf_input['edge_input'].shape[2]
            x_batch[i, :num_nodes] = gf_input['x'].to(device)
            attn_bias_batch[i, :num_nodes+1, :num_nodes+1] = gf_input['attn_bias'].to(device)
            attn_edge_type_batch[i, :num_nodes, :num_nodes] = gf_input['attn_edge_type'].to(device)
            spatial_pos_batch[i, :num_nodes, :num_nodes] = gf_input['spatial_pos'].to(device)
            degree_batch[i, :num_nodes] = gf_input['degree'].to(device)
            edge_input_batch[i, :num_nodes, :num_nodes, :edge_dist] = gf_input['edge_input'].to(device)

        graphormer_batch = {
            'x': x_batch,
            'attn_bias': attn_bias_batch,
            'attn_edge_type': attn_edge_type_batch,
            'spatial_pos': spatial_pos_batch,
            'degree': degree_batch,
            'edge_input': edge_input_batch,
        }
        adduct_mass_shift = torch.tensor([[
            common.ion2mass[mol_adduct],
            -common.ELECTRON_MASS if common.is_positive_adduct(mol_adduct) else common.ELECTRON_MASS,
        ] for mol_adduct in adduct], device=device)
        engines = [fragmentation.FragmentEngine(mol_str=rsmi, mol_str_type="smiles", mol_str_canonicalized=True) for rsmi in root_smi]
        total_atom_masses = [torch.from_numpy(engine.atom_weights_h).to(device) for engine in engines]
        masses_padded = torch.nn.utils.rnn.pad_sequence(total_atom_masses, batch_first=True)
        root_forms = [common.form_from_smi(rsmi) for rsmi in root_smi]
        root_form_vecs = torch.stack([torch.from_numpy(common.formula_to_dense(root_form)) for root_form in root_forms]).to(device, non_blocking=True)
        atom_hs_list = [torch.tensor(engine.atom_hs, device=device) for engine in engines]
        atom_hs_padded = torch.nn.utils.rnn.pad_sequence(atom_hs_list, batch_first=True)
        total_hs = torch.tensor([engine.total_hs for engine in engines], device=device, dtype=torch.long)

        atom_symbols_batch = [engine.atom_symbols for engine in engines]
        atom_form_vecs_np = [[common.formula_to_dense(f"{s}H{h}") for s, h in zip(atom_symbols, num_hs)] for atom_symbols, num_hs in zip(atom_symbols_batch, atom_hs_list)]
        atom_form_vecs_padded = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(np.stack(atom_form_vec_np, axis=0)).to(device) for atom_form_vec_np in atom_form_vecs_np], batch_first=True)
        atom_form_vecs = nn_utils.pack_padded_tensor(atom_form_vecs_padded, lengths=num_atoms)
        with torch.inference_mode():
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
                instruments=instruments if self.embed_instrument else None,
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
                adduct_mass_shifts, atom_form_vecs, adj_matrices, atom_hs, total_hs, instruments=None, binned_out=False):
        predict_obj = self.forward(graphormer_input, 
                        num_atoms, adducts, 
                        collision_engs, 
                        root_form_vecs, 
                        masses, 
                        adduct_mass_shifts, 
                        atom_form_vecs, 
                        adj_matrices, 
                        atom_hs, 
                        total_hs,
                        instruments=instruments if self.embed_instrument else None,
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
        max_flat_k = flat_k.max().item()
        if max_flat_k > 0:
            topk_vals, topk_idx = torch.topk(flat_preds, k=max_flat_k, dim=-1)
            arange_k = torch.arange(max_flat_k, device=breakpoint_preds.device).unsqueeze(0)  # [1, max_k]
            valid_mask = arange_k < flat_k.unsqueeze(1)  # [(B*N), max_k]
            batch_idx = torch.arange(flat_preds.shape[0], device=breakpoint_preds.device).unsqueeze(1).expand(-1, max_flat_k)  # [(B*N), max_k]
            patterns[batch_idx[valid_mask], topk_idx[valid_mask]] = True
        out = patterns.view(B, N, N_atom)
        return out

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
    
    def boundary_nodes(self, A: torch.Tensor, subgraph_mask: torch.Tensor):
        """
        A: (B, N, N) bool
        subgraph_mask: (B, M, N) bool
        """

        B, N, _ = A.shape
        _, M, _ = subgraph_mask.shape
        device = A.device

        neighbors = (subgraph_mask.unsqueeze(-1) & A.unsqueeze(1)).any(dim=2)
        boundary_mask = neighbors & ~subgraph_mask  # (B, M, N)

        chunk_size = 64
        num_chunks = (N + chunk_size - 1) // chunk_size

        padded_N = num_chunks * chunk_size
        pad_width = padded_N - N

        if pad_width > 0:
            boundary_mask = torch.nn.functional.pad(boundary_mask, (0, pad_width))

        boundary_mask = boundary_mask.view(B, M, num_chunks, chunk_size)

        powers = (1 << torch.arange(chunk_size, device=device, dtype=torch.int64))
        chunks = (boundary_mask.to(torch.int64) * powers).sum(dim=-1)  # (B, M, num_chunks)

        batch_ids = torch.arange(B, device=device).view(B, 1, 1).expand(B, M, 1)
        combined = torch.cat([batch_ids, chunks], dim=-1)  # (B, M, 1+num_chunks)

        combined_flat = combined.view(-1, 1 + num_chunks)

        unique_combined = torch.unique(combined_flat, dim=0)

        unique_batch_ids = unique_combined[:, 0]
        unique_chunks = unique_combined[:, 1:]

        unique_boundary_patterns = torch.bincount(
            unique_batch_ids,
            minlength=B
        )

        bits = (unique_chunks.unsqueeze(-1) & powers) > 0
        recovered = bits.view(-1, padded_N)[..., :N]  # remove padding

        max_unique = unique_boundary_patterns.max().item()

        boundary_mask_out = torch.zeros(
            B, max_unique, N,
            dtype=torch.float,
            device=device
        )

        counts = unique_boundary_patterns
        offsets = torch.cumsum(counts, dim=0)
        starts = offsets - counts

        row_indices = (
            torch.arange(unique_batch_ids.shape[0], device=device)
            - starts[unique_batch_ids]
        )

        boundary_mask_out[unique_batch_ids, row_indices] = recovered.float()

        return {
            "boundary_mask": boundary_mask_out,
            "unique_boundary_patterns": unique_boundary_patterns
        }
    def frag_loss(
        self,
        frags_predicted: torch.Tensor,
        frag_card_predicted: torch.Tensor,
        frag_targs: torch.Tensor,
        num_frag_targs: torch.Tensor,
        num_atoms: torch.Tensor,
        adj_matrices: torch.Tensor = None,
    ) -> torch.Tensor:
        # frags_predicted: [B, max_breakpoints, max_nodes]
        # frag_targs: packed [sum_frags, max_nodes], num_frag_targs: [B]
        B, max_breakpoints, max_nodes = frags_predicted.shape
        
        # frag_targs_padded = nn_utils.pad_packed_tensor(
        #     frag_targs, num_frag_targs, False
        # )[:, :-1, :]  # [B, max_targs-1, max_nodes]
        frag_targs_padded_original = nn_utils.pad_packed_tensor(
            frag_targs, num_frag_targs, False
        )[:, :, :]  # [B, max_targs-1, max_nodes]
        boundary_info = self.boundary_nodes(adj_matrices>0, frag_targs_padded_original)
        frag_targs_padded = boundary_info["boundary_mask"]
        num_frag_targs = boundary_info["unique_boundary_patterns"]

        node_rank = self.node_ranking(frags_predicted, num_atoms)
        frag_cards_targs = F.one_hot(torch.sum(frag_targs_padded, dim=-1).long(), num_classes=fragmentation.FRAGMENT_ENGINE_PARAMS['max_tree_depth']+1)
        rank_paired = torch.sum(node_rank.unsqueeze(2) * frag_cards_targs[:, None, :, None, :], dim=(-1))
        frag_targs_expanded = frag_targs_padded.unsqueeze(1).expand(rank_paired.shape)
        rank_paired_normed = F.normalize(rank_paired, p=1, dim=-1)
        frag_targs_expanded_normed = F.normalize(frag_targs_expanded, p=1, dim=-1)
        rank_loss = self.cross_entropy(rank_paired_normed, frag_targs_expanded_normed)

        per_pair_cards_cross_entropy = self.cross_entropy(frag_card_predicted.unsqueeze(2), frag_cards_targs.unsqueeze(1), normalized=False)

        B, max_targs, _ = frag_targs_padded.shape
        
        cost = rank_loss+per_pair_cards_cross_entropy
        assign = pygm.hungarian(
            -cost, backend="pytorch", n2=num_frag_targs
        )  # [B, max_targs, max_breakpoints]
        
        node_rank_reshape = node_rank.reshape(B, max_breakpoints, -1)
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
        return torch.mean(loss)
    
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