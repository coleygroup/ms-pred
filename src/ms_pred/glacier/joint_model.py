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
from ms_pred.glacier.dataset import TreeProcessor
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
            num_bins: int = 15000,
            upper_limit: int = 1500,
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
        self.sk_tau = sk_tau
        self.ppm_tol = ppm_tol
        self.contr_weight = contr_weight
        self.contr_loss_fn = contr_loss_fn
        self.contr_threshold = contr_threshold
        self.graphormer_dropout = graphormer_dropout
        self.graphormer_layers = graphormer_layers
        self.hidden_size=hidden_size
        self.nhead=8
        self.num_bins = num_bins
        self.upper_limit = upper_limit


        self.tree_processor = TreeProcessor(
            pe_embed_k=pe_embed_k,
            root_encode="graphormer",
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
        self.formula_in_dim = 0
        if self.encode_forms:
            self.embedder = nn_utils.get_embedder("abs-sines")
            self.formula_dim = common.NORM_VEC.shape[0]

            # Calculate formula dim
            self.formula_in_dim = self.formula_dim * self.embedder.num_dim
            self.formula_mapper = nn.Linear(self.formula_in_dim+self.hidden_size, self.hidden_size)
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
        self.inten_activation = nn.Sigmoid()
        self.output_size = (self.max_broken_bonds) * 2 + 1
        self.output_map = nn.Linear(self.hidden_size, self.output_size * 2)
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
            self.inten_loss_fn = self.cos_loss_sparse
        elif inten_loss_fn == "entropy":
            self.inten_loss_fn = self.entropy_loss_sparse
        else:
            raise NotImplementedError()

        if contr_loss_fn == "cosine":
            self.contr_loss_fn = self.cos_loss_sparse
        elif contr_loss_fn == "entropy":
            self.contr_loss_fn = self.entropy_loss_sparse
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

        # B x ( L * 2 * mass shifts )
        output_unbinned = self.inten_activation(output)
        output_unbinned = output_unbinned.masked_fill(~valid_pos, 0)
        output_unbinned = output_unbinned.reshape(batch_size, -1)
        masses = masses.reshape(batch_size, -1)
        # Keep [mass, intensity] ordering to match sparse loss binning.
        inten_output = torch.stack((masses, output_unbinned), dim=-1)
        
        return {"output": inten_output}

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
            pred_inten = pred["inten_pred_end_to_end"]["output"]
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
                pred_inten_magma = pred["inten_pred_magma"]["output"]
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
            end_to_end_inten_loss = inten_loss_fn(targ_pred["inten_pred_end_to_end"]["output"], targ_batch["inten_targs"], parent_mass=targ_batch["precursor_mzs"])["loss"]
            decoy_spec_loss = inten_decoy_loss_fn(decoy_pred["inten_pred_end_to_end"]["output"], decoy_inten_targs, parent_mass=decoy_batch["precursor_mzs"])["loss"]
            targ_contr_loss = inten_decoy_loss_fn(targ_pred["inten_pred_end_to_end"]["output"], targ_batch["inten_targs"], parent_mass=targ_batch["precursor_mzs"])["loss"]
            split_end = torch.cumsum(batch["num_decoys_per_entry"], dim=0)
            split_start = split_end - batch["num_decoys_per_entry"]
            decoy_spec_loss = [decoy_spec_loss[s:e] for s, e in zip(split_start, split_end)]
            decoy_spec_loss = torch.nn.utils.rnn.pad_sequence(decoy_spec_loss, batch_first=True, padding_value=1) # cos_loss <=1 by definition
            decoy_spec_loss = torch.cat((targ_contr_loss.unsqueeze(1), decoy_spec_loss), dim=1)
            decoy_spec_loss_sorted = torch.sort(decoy_spec_loss, dim=-1).values.detach()
            ranking_dist = torch.abs(decoy_spec_loss[:, :, None] - decoy_spec_loss_sorted[:, None, :])
            top1_prob = pygm.sinkhorn(-ranking_dist, n1=batch["num_decoys_per_entry"]+1, n2=batch["num_decoys_per_entry"]+1, tau=self.sk_tau, backend='pytorch')[:, 0, 0]
            contr_loss = torch.relu(-torch.log(top1_prob + self.contr_threshold))  # shift & cut ce loss for probs > contr_threshold
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

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric=None):  # fix lightning API mismatch for torch>=2.0
        # For LambdaLR, just call step() without arguments
        scheduler.step()

    def predict_mol(self, smi, collision_eng, adduct, device, instrument=None):
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
            out = self._assemble_spec_frag(pred, adduct_mass_shift)

            if batched_input:
                return out
            else:
                return {k: v[0] for k, v in out.items()}

    def _assemble_spec_frag(self, pred, adduct_mass_shifts):
        """Turn a raw forward() output into per-molecule sparse spec rows + frag masks.

        Shared by ``predict_mol`` (raw-SMILES path) and ``predict_inten_frag_batch``
        (DataLoader path). Returns ``{"spec": [Tensor[num_rows, 2], ...],
        "frag": [Tensor[num_rows, num_atoms], ...]}`` with one entry per molecule.
        """
        inten_out = pred["inten_pred_end_to_end"]
        frags_pred = pred["frags_pred"]
        output = inten_out["output"]
        num_shifts = adduct_mass_shifts.shape[1] * self.output_size

        out = {"spec": [], "frag": []}
        fragments = nn_utils.pad_packed_tensor(frags_pred["fragments"], frags_pred["fragment_count"], 0)

        for i, n in enumerate(frags_pred["fragment_count"]):
            num_rows = int(n.item()) * num_shifts
            spec_pred = output[i, :num_rows, :]
            out_frag = fragments[i, :n].repeat_interleave(num_shifts, dim=0)
            out["spec"].append(spec_pred)
            out["frag"].append(out_frag)
        return out

    def predict_inten_frag_batch(self, batch):
        """Run forward + spec/frag assembly on an already-collated, on-device batch.

        ``batch`` is the dict produced by ``IntenDataset.collate_fn`` (the same keys
        ``forward`` consumes). Featurization (incl. ``create_graphormer_input``) has
        already happened in the DataLoader workers, so this only does GPU work.
        Returns the same ``{"spec": [...], "frag": [...]}`` structure as
        ``predict_mol`` with ``batched_input=True``.
        """
        if not getattr(self, "_predict_prepared", False):
            self.eval()
            self.freeze()
            self._predict_prepared = True
        with torch.inference_mode():
            pred = self.forward(
                batch["graphormer_input"],
                batch["num_atoms"],
                batch["adducts"],
                batch["collision_engs"],
                root_form_vecs=batch["root_form_vecs"],
                masses=batch["masses"],
                adduct_mass_shifts=batch["adduct_mass_shifts"],
                atom_form_vecs=batch["atom_form_vecs"],
                adj_matrices=batch["adj_matrices"],
                atom_hs=batch["atom_hs"],
                total_hs=batch["total_hs"],
                instruments=batch["instruments"] if self.embed_instrument else None,
            )
            return self._assemble_spec_frag(pred, batch["adduct_mass_shifts"])

    def predict_inten_frag_batch_sparse(self, batch, sparse_k):
        """Forward + vectorized sparse top-k selection for a collated, on-device batch.

        Equivalent to ``predict_inten_frag_batch`` followed by per-molecule top-k
        selection, but done as a few batched GPU ops + 3 host transfers instead of a
        Python loop with one ``topk``/``.cpu()`` per molecule (which serializes ~2*B
        GPU syncs per batch and starves the GPU). Returns CPU numpy arrays:
            masses [B, K], intens [B, K], frags [B, K, N] (uint8), counts [B]
        where for molecule i only the first ``counts[i]`` rows are valid (matching the
        old behavior of taking ``min(sparse_k, n_rows)`` rows).
        """
        if not getattr(self, "_predict_prepared", False):
            self.eval()
            self.freeze()
            self._predict_prepared = True
        with torch.inference_mode():
            pred = self.forward(
                batch["graphormer_input"],
                batch["num_atoms"],
                batch["adducts"],
                batch["collision_engs"],
                root_form_vecs=batch["root_form_vecs"],
                masses=batch["masses"],
                adduct_mass_shifts=batch["adduct_mass_shifts"],
                atom_form_vecs=batch["atom_form_vecs"],
                adj_matrices=batch["adj_matrices"],
                atom_hs=batch["atom_hs"],
                total_hs=batch["total_hs"],
                instruments=batch["instruments"] if self.embed_instrument else None,
            )
            output = pred["inten_pred_end_to_end"]["output"]  # [B, R, 2]
            frag_count = pred["frags_pred"]["fragment_count"]  # [B]
            fragments = nn_utils.pad_packed_tensor(
                pred["frags_pred"]["fragments"], frag_count, 0
            )  # [B, max_frag, N]

            B, R, _ = output.shape
            device = output.device
            num_shifts = batch["adduct_mass_shifts"].shape[1] * self.output_size
            num_rows = frag_count * num_shifts  # [B] valid spec rows per molecule

            # Mask padding rows out of the top-k by intensity.
            row_ar = torch.arange(R, device=device).unsqueeze(0)  # [1, R]
            valid = row_ar < num_rows.unsqueeze(1)  # [B, R]
            masses = output[:, :, 0]
            intens = output[:, :, 1].masked_fill(~valid, float("-inf"))

            k = min(sparse_k, R)
            top_int, top_idx = torch.topk(intens, k=k, dim=1)  # [B, k]
            sel_mass = torch.gather(masses, 1, top_idx)  # [B, k]
            sel_int = torch.where(torch.isfinite(top_int), top_int, torch.zeros_like(top_int))

            # Each selected spec row j maps to fragment j // num_shifts.
            frag_idx = torch.div(top_idx, num_shifts, rounding_mode="floor")  # [B, k]
            N = fragments.shape[2]
            sel_frag = torch.gather(fragments, 1, frag_idx.unsqueeze(-1).expand(B, k, N))  # [B, k, N]
            counts = torch.clamp(num_rows, max=k)  # [B]

            return {
                "masses": sel_mass.cpu().numpy(),
                "intens": sel_int.cpu().numpy(),
                "frags": sel_frag.bool().cpu().numpy(),
                "counts": counts.cpu().numpy(),
            }
    
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
        if binned_out:
            out_binned = self._bin_unbinned_spectra_batch(output)
            return {"spec": out_binned}

        out_preds = [
            pred[:num_frag, :]
            for pred, num_frag in zip(output, num_frag_targs)
        ]
        
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

    def cos_loss_sparse(self, pred, targ, parent_mass=None, use_hun=False):
        """Sparse cosine distance matching retrieval-time sparse scoring.

        Expects unbinned spectra with shape [B, N, 2],
        where channel 0 is mass and channel 1 is intensity.
        """
        if pred.ndim != 3 or targ.ndim != 3 or pred.shape[-1] != 2 or targ.shape[-1] != 2:
            raise ValueError(
                f"cos_loss_sparse expects pred/targ shaped [B, N, 2], got {tuple(pred.shape)} and {tuple(targ.shape)}"
            )
        eps = 1e-22
        compute_dtype = torch.float32
        batch_size = pred.shape[0]

        pred_batch, pred_bin, pred_val = self._coalesce_unbinned_spectra_sparse(pred, compute_dtype)
        targ_batch, targ_bin, targ_val = self._coalesce_unbinned_spectra_sparse(targ, compute_dtype)

        pred_norm_sq = torch.zeros(batch_size, dtype=compute_dtype, device=pred.device)
        targ_norm_sq = torch.zeros(batch_size, dtype=compute_dtype, device=pred.device)
        if pred_val.numel() > 0:
            pred_norm_sq.scatter_add_(0, pred_batch, pred_val * pred_val)
        if targ_val.numel() > 0:
            targ_norm_sq.scatter_add_(0, targ_batch, targ_val * targ_val)

        dot = torch.zeros(batch_size, dtype=compute_dtype, device=pred.device)
        if pred_val.numel() > 0 and targ_val.numel() > 0:
            pred_key = pred_batch * self.num_bins + pred_bin
            targ_key = targ_batch * self.num_bins + targ_bin
            targ_sort = torch.argsort(targ_key)
            targ_key_sorted = targ_key[targ_sort]
            targ_val_sorted = targ_val[targ_sort]

            pos = torch.searchsorted(targ_key_sorted, pred_key)
            in_bounds = pos < targ_key_sorted.numel()
            matched = in_bounds & (targ_key_sorted[pos.clamp_max(targ_key_sorted.numel() - 1)] == pred_key)
            if torch.any(matched):
                matched_pos = pos[matched]
                dot_terms = pred_val[matched] * targ_val_sorted[matched_pos]
                dot.scatter_add_(0, pred_batch[matched], dot_terms)

        pred_norm = torch.sqrt(pred_norm_sq)
        targ_norm = torch.sqrt(targ_norm_sq)
        valid = (pred_norm > 0) & (targ_norm > 0)
        denom = pred_norm * targ_norm + eps
        loss_val = 1 - (dot / denom)
        loss = torch.where(valid, loss_val, torch.ones_like(loss_val))
        return {"loss": loss}

    def _entropy_from_probs_torch(self, vals: torch.Tensor) -> torch.Tensor:
        vals = vals[vals > 0]
        if vals.numel() == 0:
            return torch.zeros((), dtype=vals.dtype, device=vals.device)
        return -torch.sum(vals * torch.log(vals + 1e-22))

    def _coalesce_unbinned_spectra_sparse(self, specs: torch.Tensor, dtype: torch.dtype):
        """Coalesce unbinned [B, N, 2] spectra into sparse (batch, bin, value) tuples."""
        if specs.ndim != 3 or specs.shape[-1] != 2:
            raise ValueError(f"Expected spectra shaped [B, N, 2], got {tuple(specs.shape)}")

        batch_size = specs.shape[0]
        mz = specs[:, :, 0]
        inten = specs[:, :, 1].to(dtype)

        scale = (self.num_bins - 1) / float(self.upper_limit)
        bin_idx = torch.floor(mz * scale).long() + 1
        valid = (inten > 0) & (bin_idx >= 0) & (bin_idx < self.num_bins)

        if not torch.any(valid):
            empty_long = torch.empty(0, dtype=torch.long, device=specs.device)
            empty_float = torch.empty(0, dtype=dtype, device=specs.device)
            return empty_long, empty_long, empty_float

        batch_idx = torch.arange(batch_size, device=specs.device).unsqueeze(1).expand_as(bin_idx)
        batch_flat = batch_idx[valid]
        bin_flat = bin_idx[valid]
        inten_flat = inten[valid]

        key = batch_flat * self.num_bins + bin_flat
        uniq_key, inv = torch.unique(key, sorted=True, return_inverse=True)
        pooled, _ = ts.scatter_max(inten_flat, inv, dim=0, dim_size=uniq_key.numel())
        pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))

        out_batch = torch.div(uniq_key, self.num_bins, rounding_mode="floor")
        out_bin = uniq_key % self.num_bins
        return out_batch, out_bin, pooled

    def _bin_unbinned_spectra_batch(self, specs: torch.Tensor) -> torch.Tensor:
        """Bin a batch of unbinned spectra shaped [B, N, 2] to [B, num_bins]."""
        if specs.ndim != 3 or specs.shape[-1] != 2:
            raise ValueError(f"Expected spectra shaped [B, N, 2], got {tuple(specs.shape)}")

        batch_size = specs.shape[0]
        mz = specs[:, :, 0]
        inten = specs[:, :, 1]

        scale = (self.num_bins - 1) / float(self.upper_limit)
        bin_idx = torch.floor(mz * scale).long() + 1
        valid = (inten > 0) & (bin_idx >= 0) & (bin_idx < self.num_bins)

        if not torch.any(valid):
            return torch.zeros((batch_size, self.num_bins), dtype=specs.dtype, device=specs.device)

        offsets = (torch.arange(batch_size, device=specs.device).unsqueeze(1) * self.num_bins)
        global_idx = (bin_idx + offsets).reshape(-1)
        valid_flat = valid.reshape(-1)
        global_idx = global_idx[valid_flat]
        inten_flat = inten.reshape(-1)[valid_flat]

        pooled_flat, _ = ts.scatter_max(inten_flat, global_idx, dim=0, dim_size=batch_size * self.num_bins)
        pooled_flat = torch.where(torch.isfinite(pooled_flat), pooled_flat, torch.zeros_like(pooled_flat))
        return pooled_flat.view(batch_size, self.num_bins)

    def entropy_loss_sparse(self, pred, targ, parent_mass=None, use_hun=False):
        """Sparse entropy distance matching retrieval-time sparse scoring.

        Expects unbinned spectra with shape [B, N, 2],
        where channel 0 is mass and channel 1 is intensity.
        """
        if pred.ndim != 3 or targ.ndim != 3 or pred.shape[-1] != 2 or targ.shape[-1] != 2:
            raise ValueError(
                f"entropy_loss_sparse expects pred/targ shaped [B, N, 2], got {tuple(pred.shape)} and {tuple(targ.shape)}"
            )
        eps = 1e-22
        compute_dtype = torch.float32
        log4 = torch.log(torch.tensor(4.0, dtype=compute_dtype, device=pred.device))
        batch_size = pred.shape[0]

        pred_batch, pred_bin, pred_val = self._coalesce_unbinned_spectra_sparse(pred, compute_dtype)
        targ_batch, targ_bin, targ_val = self._coalesce_unbinned_spectra_sparse(targ, compute_dtype)

        pred_sum = torch.zeros(batch_size, dtype=compute_dtype, device=pred.device)
        targ_sum = torch.zeros(batch_size, dtype=compute_dtype, device=pred.device)
        if pred_val.numel() > 0:
            pred_sum.scatter_add_(0, pred_batch, pred_val)
        if targ_val.numel() > 0:
            targ_sum.scatter_add_(0, targ_batch, targ_val)

        valid = (pred_sum > 0) & (targ_sum > 0)

        h_pred = torch.zeros(batch_size, dtype=compute_dtype, device=pred.device)
        if pred_val.numel() > 0:
            pred_probs = pred_val / (pred_sum[pred_batch] + eps)
            pred_h_terms = -(pred_probs * torch.log(pred_probs + eps))
            h_pred.scatter_add_(0, pred_batch, pred_h_terms)

        h_true = torch.zeros(batch_size, dtype=compute_dtype, device=pred.device)
        if targ_val.numel() > 0:
            targ_probs = targ_val / (targ_sum[targ_batch] + eps)
            targ_h_terms = -(targ_probs * torch.log(targ_probs + eps))
            h_true.scatter_add_(0, targ_batch, targ_h_terms)

        h_mix = torch.zeros(batch_size, dtype=compute_dtype, device=pred.device)
        if pred_val.numel() > 0 or targ_val.numel() > 0:
            pred_key = pred_batch * self.num_bins + pred_bin
            targ_key = targ_batch * self.num_bins + targ_bin
            pred_probs = pred_val / (pred_sum[pred_batch] + eps) if pred_val.numel() > 0 else torch.empty(0, dtype=compute_dtype, device=pred.device)
            targ_probs = targ_val / (targ_sum[targ_batch] + eps) if targ_val.numel() > 0 else torch.empty(0, dtype=compute_dtype, device=pred.device)

            mix_key = torch.cat((pred_key, targ_key), dim=0)
            mix_comp = torch.cat((0.5 * pred_probs, 0.5 * targ_probs), dim=0)
            if mix_key.numel() > 0:
                uniq_mix_key, inv = torch.unique(mix_key, sorted=True, return_inverse=True)
                mix_probs = torch.zeros(uniq_mix_key.numel(), dtype=compute_dtype, device=pred.device)
                mix_probs.scatter_add_(0, inv, mix_comp)
                mix_h_terms = -(mix_probs * torch.log(mix_probs + eps))
                mix_batch = torch.div(uniq_mix_key, self.num_bins, rounding_mode="floor")
                h_mix.scatter_add_(0, mix_batch, mix_h_terms)

        loss_val = (2 * h_mix - h_pred - h_true) / log4
        loss = torch.where(valid, loss_val, torch.ones_like(loss_val))
        return {"loss": loss}
