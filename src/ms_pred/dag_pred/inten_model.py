"""DAG intensity prediction model."""
import numpy as np
import copy
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch_scatter as ts
import dgl.nn as dgl_nn
import pygmtools as pygm
import functools


import ms_pred.common as common
import ms_pred.dag_pred.dag_data as dag_data
import ms_pred.nn_utils as nn_utils
import ms_pred.magma.fragmentation as fragmentation
import ms_pred.magma.run_magma as run_magma


class IntenGNN(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int,
        gnn_layers: int = 2,
        mlp_layers: int = 0,
        set_layers: int = 2,
        learning_rate: float = 7e-4,
        lr_decay_rate: float = 1.0,
        weight_decay: float = 0,
        dropout: float = 0,
        mpnn_type: str = "PNA",
        pool_op: str = "avg",
        node_feats: int = common.ELEMENT_DIM + common.MAX_H,
        pe_embed_k: int = 0,
        max_broken: int = run_magma.FRAGMENT_ENGINE_PARAMS["max_broken_bonds"],
        frag_set_layers: int = 0,
        loss_fn: str = "cosine",
        root_encode: str = "gnn",
        inject_early: bool = False,
        warmup: int = 1000,
        embed_adduct: bool = False,
        embed_collision: bool = False,
        embed_elem_group: bool = False,
        include_unshifted_mz: bool = False,
        binned_targs: bool = True,
        encode_forms: bool = False,
        add_hs: bool = False,
        sk_tau: float = 0.01,
        contr_weight: float = 1.,
        ppm_tol: float = 20,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.pe_embed_k = pe_embed_k
        self.root_encode = root_encode
        self.pool_op = pool_op
        self.inject_early = inject_early
        self.embed_adduct = embed_adduct
        self.embed_collision = embed_collision
        self.embed_elem_group = embed_elem_group
        self.include_unshifted_mz = include_unshifted_mz
        self.binned_targs = binned_targs
        self.encode_forms = encode_forms
        self.add_hs = add_hs
        self.sk_tau = sk_tau
        self.contr_weight = contr_weight

        self.tree_processor = dag_data.TreeProcessor(
            root_encode=root_encode, pe_embed_k=pe_embed_k, add_hs=add_hs, embed_elem_group=self.embed_elem_group,
        )

        self.formula_in_dim = 0
        if self.encode_forms:
            self.embedder = nn_utils.get_embedder("abs-sines")
            self.formula_dim = common.NORM_VEC.shape[0]

            # Calculate formula dim
            self.formula_in_dim = self.formula_dim * self.embedder.num_dim

            # Account for diffs
            self.formula_in_dim *= 2

        self.gnn_layers = gnn_layers
        self.set_layers = set_layers
        self.frag_set_layers = frag_set_layers
        self.mpnn_type = mpnn_type
        self.mlp_layers = mlp_layers

        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.warmup = warmup

        self.weight_decay = weight_decay
        self.dropout = dropout

        self.max_broken = max_broken + 1
        self.broken_onehot = torch.nn.Parameter(torch.eye(self.max_broken))
        self.broken_onehot.requires_grad = False
        self.broken_clamp = max_broken

        edge_feats = fragmentation.MAX_BONDS

        orig_node_feats = node_feats
        if self.inject_early:
            node_feats = node_feats + self.hidden_size

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
            pe_power =  2 * torch.arange(pe_dim // 2) / pe_dim
            self.collision_embedder_denominators = nn.Parameter(torch.pow(pe_scalar, pe_power))
            self.collision_embedder_denominators.requires_grad = False
            collision_shift = pe_dim

            # Not used: Compute the merged collision embedding as the mean of all energies 0 - 100 eV
            # collision_eng_steps = torch.arange(0, 100, 0.01)
            # self.collision_embed_merged = nn.Parameter(torch.cat(
            #     (torch.sin(collision_eng_steps.unsqueeze(1) / self.collision_embedder_denominators.unsqueeze(0)),
            #      torch.cos(collision_eng_steps.unsqueeze(1) / self.collision_embedder_denominators.unsqueeze(0))),
            #     dim=1
            # ).mean(dim=0))
            # self.collision_embed_merged.requires_grad = False

            # All-zero for collision == nan
            self.collision_embed_merged = nn.Parameter(torch.zeros(pe_dim))
            self.collision_embed_merged.requires_grad = False

        # Define network
        self.gnn = nn_utils.MoleculeGNN(
            hidden_size=self.hidden_size,
            num_step_message_passing=self.gnn_layers,
            set_transform_layers=self.set_layers,
            mpnn_type=self.mpnn_type,
            gnn_node_feats=node_feats + adduct_shift + collision_shift,
            gnn_edge_feats=edge_feats,
            dropout=self.dropout,
        )

        if self.root_encode == "gnn":
            self.root_module = self.gnn

            # if inject early, need separate root and child GNN's
            if self.inject_early:
                self.root_module = nn_utils.MoleculeGNN(
                    hidden_size=self.hidden_size,
                    num_step_message_passing=self.gnn_layers,
                    set_transform_layers=self.set_layers,
                    mpnn_type=self.mpnn_type,
                    gnn_node_feats=node_feats + adduct_shift,
                    gnn_edge_feats=edge_feats,
                    dropout=self.dropout,
                )
        elif self.root_encode == "fp":
            self.root_module = nn_utils.MLPBlocks(
                input_size=2048,
                hidden_size=self.hidden_size,
                output_size=None,
                dropout=self.dropout,
                num_layers=1,
                use_residuals=True,
            )
        else:
            raise ValueError()

        # MLP layer to take representations from the pooling layer
        # And predict a single scalar value at each of them
        # I.e., Go from size B x 2h -> B x 1
        self.intermediate_out = nn_utils.MLPBlocks(
            input_size=self.hidden_size * 3 + self.max_broken + self.formula_in_dim,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            num_layers=self.mlp_layers,
            use_residuals=True,
        )

        trans_layer = nn_utils.TransformerEncoderLayer(
            self.hidden_size,
            nhead=8,
            batch_first=True,
            norm_first=False,
            dim_feedforward=self.hidden_size * 4,
        )
        self.trans_layers = nn_utils.get_clones(trans_layer, self.frag_set_layers)

        self.ppm_tol = ppm_tol
        self.mass_tol = self.ppm_tol * 1e-6
        self.loss_fn_name = loss_fn
        self.cos_fn = nn.CosineSimilarity()
        if loss_fn == "cosine":
            self.loss_fn = self.cos_loss
            self.output_activations = [nn.Sigmoid()]
        elif loss_fn == "entropy":
            self.loss_fn = self.entropy_loss
            self.output_activations = [nn.Sigmoid()]
        elif loss_fn == "weighted_entropy":
            self.loss_fn = functools.partial(self.entropy_loss, weighted=True)
            self.output_activations = [nn.Sigmoid()]
        else:
            raise NotImplementedError()

        self.num_outputs = len(self.output_activations)
        self.output_size = run_magma.FRAGMENT_ENGINE_PARAMS["max_broken_bonds"] * 2 + 1
        if self.include_unshifted_mz:
            self.output_map = nn.Linear(
                self.hidden_size, self.num_outputs * self.output_size * 2
                # output_size dim for adduct mass shift & output_size dim for no mass shift
            )
        else:
            self.output_map = nn.Linear(
                self.hidden_size, self.num_outputs * self.output_size
            )

        # Define map from output layer to attn
        self.isomer_attn_out = copy.deepcopy(self.output_map)

        # Define buckets
        buckets = torch.DoubleTensor(np.linspace(0, 1500, 15000))
        self.inten_buckets = nn.Parameter(buckets)
        self.inten_buckets.requires_grad = False

        if self.pool_op == "avg":
            self.pool = dgl_nn.AvgPooling()
        elif self.pool_op == "attn":
            self.pool = dgl_nn.GlobalAttentionPooling(nn.Linear(hidden_size, 1))
        else:
            raise NotImplementedError()

        self.sigmoid = nn.Sigmoid()

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

    def predict(
        self,
        graphs,
        root_reprs,
        ind_maps,
        num_frags,
        max_breaks,
        adducts,
        collision_engs,
        precursor_mzs,
        max_add_hs=None,
        max_remove_hs=None,
        masses=None,
        root_forms=None,
        frag_forms=None,
        binned_out=False,
    ) -> dict:
        """predict _summary_

        Args:
            graphs (_type_): _description_
            root_reprs (_type_): _description_
            ind_maps (_type_): _description_
            num_frags (_type_): _description_
            max_breaks (_type_): _description_
            adducts (_type_): _description_
            collision_engs (_type_): _description_
            precursor_mzs (_type_): _description_
            max_add_hs (_type_, optional): _description_. Defaults to None.
            max_remove_hs (_type_, optional): _description_. Defaults to None.
            masses (_type_, optional): _description_. Defaults to None.
            root_forms (_type_, optional): _description_. Defaults to None.
            frag_forms (_type_, optional): _description_. Defaults to None.
            binned_out (bool, optional): _description_. Defaults to False.

        Raises:
            NotImplementedError: _description_

        Returns:
            dict: _description_
        """
        # B x nodes x num outputs x inten items
        out = self.forward(
            graphs,
            root_reprs,
            ind_maps,
            num_frags,
            adducts=adducts,
            collision_engs=collision_engs,
            precursor_mzs=precursor_mzs,
            broken=max_breaks,
            max_add_hs=max_add_hs,
            max_remove_hs=max_remove_hs,
            masses=masses,
            root_forms=root_forms,
            frag_forms=frag_forms,
        )

        if self.loss_fn_name not in ["cosine", "entropy", "weighted_entropy"]:
            raise NotImplementedError()

        output = out["output"][
            :,
            :,
            0,
        ]
        output_binned = out["output_binned"][:, 0, :]
        out_preds_binned = [i.cpu().detach().numpy() for i in output_binned]
        out_preds = [
            pred[:num_frag, :].cpu().detach().numpy()
            for pred, num_frag in zip(output, num_frags)
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

    def forward(
        self,
        graphs,
        root_repr,
        ind_maps,
        num_frags,
        broken,
        collision_engs,
        precursor_mzs,
        adducts,
        max_add_hs=None,
        max_remove_hs=None,
        masses=None,
        root_forms=None,
        frag_forms=None,
    ):
        """forward _summary_

        Args:
            graphs (_type_): _description_
            root_repr (_type_): _description_
            ind_maps (_type_): _description_
            num_frags (_type_): _description_
            broken (_type_): _description_
            adducts (_type_): _description_
            collision_engs (_type_): _description_
            precursor_mzs (_type_): _description_
            max_add_hs (_type_, optional): _description_. Defaults to None.
            max_remove_hs (_type_, optional): _description_. Defaults to None.
            masses (_type_, optional): _description_. Defaults to None.
            root_forms (_type_, optional): _description_. Defaults to None.
            frag_forms (_type_, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        device = num_frags.device

        if not self.include_unshifted_mz:
            masses = masses[:, :, :1, :].contiguous()  # only keep m/z with adduct shift

        # if root fingerprints:
        embed_adducts = self.adduct_embedder[adducts.long()]
        if self.root_encode == "fp":
            root_embeddings = self.root_module(root_repr)
            raise NotImplementedError()
        elif self.root_encode == "gnn":
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
                root_embeddings = self.root_module(root_repr)
                root_embeddings = self.pool(root_repr, root_embeddings)
        else:
            pass

        # Line up the features to be parallel between fragment avgs and root
        # graphs
        ext_root = root_embeddings[ind_maps]
        # Extend the root further to cover each individual atom
        ext_root_atoms = torch.repeat_interleave(
            ext_root, graphs.batch_num_nodes(), dim=0
        )
        concat_list = [graphs.ndata["h"]]

        if self.inject_early:
            concat_list.append(ext_root_atoms)

        if self.embed_adduct:
            adducts_mapped = embed_adducts[ind_maps]
            adducts_exp = torch.repeat_interleave(
                adducts_mapped, graphs.batch_num_nodes(), dim=0
            )
            concat_list.append(adducts_exp)

        if self.embed_collision:
            collision_mapped = embed_collision[ind_maps]
            collision_exp = torch.repeat_interleave(
                collision_mapped, graphs.batch_num_nodes(), dim=0
            )
            concat_list.append(collision_exp)

        with graphs.local_scope():
            graphs.ndata["h"] = torch.cat(concat_list, -1).float()

            frag_embeddings = self.gnn(graphs)

            # Average embed the full root molecules and fragments
            avg_frags = self.pool(graphs, frag_embeddings)

        # expand broken and map it to each fragment
        broken_arange = torch.arange(broken.shape[-1]).to(device)
        broken_mask = broken_arange[None, :] < num_frags[:, None]

        broken = torch.clamp(broken[broken_mask], max=self.broken_clamp)
        broken_onehots = self.broken_onehot[broken.long()]

        ### Build hidden with forms
        mlp_cat_list = [ext_root, ext_root - avg_frags, avg_frags, broken_onehots]

        hidden = torch.cat(mlp_cat_list, dim=1)

        # Pack so we can use interpeak attn
        padded_hidden = nn_utils.pad_packed_tensor(hidden, num_frags, 0)

        if self.encode_forms:
            diffs = root_forms[:, None, :] - frag_forms
            form_encodings = self.embedder(frag_forms)
            diff_encodings = self.embedder(diffs)
            new_hidden = torch.cat(
                [padded_hidden, form_encodings, diff_encodings], dim=-1
            )
            padded_hidden = new_hidden

        padded_hidden = self.intermediate_out(padded_hidden)
        batch_size, max_frags, hidden_dim = padded_hidden.shape

        # Build up a mask
        arange_frags = torch.arange(padded_hidden.shape[1]).to(device)
        attn_mask = ~(arange_frags[None, :] < num_frags[:, None])

        hidden = padded_hidden
        for trans_layer in self.trans_layers:
            hidden, _ = trans_layer(hidden, src_key_padding_mask=attn_mask)

        # hidden: B x L x h
        # attn_mask: B x L

        # Build mask
        max_inten_shift = (self.output_size - 1) / 2
        max_break_ar = torch.arange(self.output_size, device=device)[None, None, :].to(
            device
        )
        max_breaks_ub = max_add_hs + max_inten_shift
        max_breaks_lb = -max_remove_hs + max_inten_shift

        ub_mask = max_break_ar <= max_breaks_ub[:, :, None]
        lb_mask = max_break_ar >= max_breaks_lb[:, :, None]

        # B x Length x Mass shifts
        valid_pos = torch.logical_and(ub_mask, lb_mask)
        valid_pos = torch.logical_and(valid_pos, ~attn_mask[:, :, None])

        if self.include_unshifted_mz:
            mz_groups = 2
        else:
            mz_groups = 1

        # B x Length x outputs x 2 x Mass shift
        valid_pos = valid_pos[:, :, None, None, :].expand(
            batch_size, max_frags, self.num_outputs, mz_groups, self.output_size
        )
        # B x L x Output
        output = self.output_map(hidden)
        attn_weights = self.isomer_attn_out(hidden)

        # B x L x Out x 2 x Mass shifts
        output = output.reshape(batch_size, max_frags, self.num_outputs, mz_groups, -1)
        attn_weights = attn_weights.reshape(batch_size, max_frags, self.num_outputs, mz_groups, -1)

        # Mask attn weights
        attn_weights.masked_fill_(~valid_pos, -99999)  # -float("inf"))

        # B x Out x L x 2 x Mass shifts
        output = output.transpose(1, 2)
        attn_weights = attn_weights.transpose(1, 2)
        valid_pos_binned = valid_pos.transpose(1, 2)

        # Calc inverse indices => B x Out x L x 2 x shift
        inverse_indices = torch.clamp(torch.bucketize(masses, self.inten_buckets, right=False), max=len(self.inten_buckets) - 1)
        inverse_indices = inverse_indices[:, None, :, :].expand(attn_weights.shape)

        # B x Out x (L * 2 * Mass shifts)
        attn_weights = attn_weights.reshape(batch_size, self.num_outputs, -1)
        output = output.reshape(batch_size, self.num_outputs, -1)
        inverse_indices = inverse_indices.reshape(batch_size, self.num_outputs, -1)
        valid_pos_binned = valid_pos.reshape(batch_size, self.num_outputs, -1)

        # B x Outs x ( L * 2 * mass shifts )
        pool_weights = ts.scatter_softmax(attn_weights, index=inverse_indices, dim=-1)
        weighted_out = pool_weights * output

        # B x Outs x (UNIQUE(L * 2 * mass shifts))
        output_binned = ts.scatter_add(
            weighted_out,
            index=inverse_indices,
            dim=-1,
            dim_size=self.inten_buckets.shape[-1],
        )

        # B x L x Outs x (2 * mass shifts)
        pool_weights_reshaped = pool_weights.reshape(
            batch_size, self.num_outputs, max_frags, -1
        ).transpose(1, 2)
        inverse_indices_reshaped = inverse_indices.reshape(
            batch_size, self.num_outputs, max_frags, -1
        ).transpose(1, 2)

        # B x Outs x binned
        valid_pos_binned = ts.scatter_max(
            (valid_pos_binned).long(),
            index=inverse_indices,
            dim_size=self.inten_buckets.shape[-1],
            dim=-1,
        )[0].bool()

        # Activate each dim with its respective output activation
        # Helpful for hurdle or probabilistic models
        new_outputs_binned = []
        for output_ind, act in enumerate(self.output_activations):
            new_outputs_binned.append(
                act(output_binned[:, output_ind : output_ind + 1, :])
            )
        output_binned = torch.cat(new_outputs_binned, -2)
        output_binned.masked_fill_(~valid_pos_binned, 0)

        # Index into output binned using inverse_indices_reshaped
        # Revert the binned output back to frags for attribution
        # B x Out x (L * 2 * Mass shifts)
        inverse_indices_reshaped_temp = inverse_indices_reshaped.transpose(
            1, 2
        ).reshape(batch_size, self.num_outputs, -1)
        output_unbinned = torch.take_along_dim(
            output_binned, inverse_indices_reshaped_temp, dim=-1
        )
        output_unbinned = output_unbinned.reshape(
            batch_size, self.num_outputs, max_frags, -1
        ).transpose(1, 2)
        output_unbinned_alpha = output_unbinned * pool_weights_reshaped

        return {"output_binned": output_binned, "output": output_unbinned_alpha}

    def _common_step(self, batch, name="train"):
        pred_obj = self.forward(
            batch["frag_graphs"],
            batch["root_reprs"],
            batch["inds"],
            batch["num_frags"],
            broken=batch["broken_bonds"],
            adducts=batch["adducts"],
            collision_engs=batch["collision_engs"],
            precursor_mzs=batch["precursor_mzs"],
            max_remove_hs=batch["max_remove_hs"],
            max_add_hs=batch["max_add_hs"],
            masses=batch["masses"],
            root_forms=batch["root_form_vecs"],
            frag_forms=batch["frag_form_vecs"],
        )
        if self.binned_targs:
            pred_inten = pred_obj["output_binned"]
            pred_inten = pred_inten[:, 0, :]
        else:
            pred_inten = pred_obj["output"]
            pred_inten = pred_inten[:, :, 0, :]
            pred_inten = torch.stack((batch["masses"], pred_inten), dim=-1)
            pred_inten = pred_inten.reshape(pred_inten.shape[0], -1, 2)  # B x (Out * Mass shifts) x 2
        batch_size = len(batch["names"])

        if name == 'train':
            loss_fn = self.loss_fn
            decoy_loss_fn = self.entropy_loss
        else:
            loss_fn = functools.partial(self.loss_fn, use_hun=True)  # use hungarian in val and test
            decoy_loss_fn = functools.partial(self.entropy_loss, use_hun=True)

        if 'is_decoy' in batch and 'mol_num' in batch:  # data with decoys
            true_data_inds = batch["is_decoy"] == 0
            num_true_data = torch.sum(true_data_inds)

            # the real spectrum loss (cosine or entropy)
            spec_loss = loss_fn(pred_inten[true_data_inds], batch["inten_targs"][true_data_inds],
                               parent_mass=batch["precursor_mzs"][true_data_inds]
            )['loss']

            # contrastive ranking loss cosine loss to decoys
            decoy_spec_loss = decoy_loss_fn(pred_inten,
                                            batch["inten_targs"][true_data_inds].repeat_interleave(batch['mol_num'], dim=0),
                                            parent_mass=batch["precursor_mzs"]
                                            )['loss']
            split_end = torch.cumsum(batch['mol_num'], dim=0)
            split_start = split_end - batch['mol_num']
            decoy_spec_loss = [decoy_spec_loss[s:e] for s, e in zip(split_start, split_end)]
            decoy_spec_loss = torch.nn.utils.rnn.pad_sequence(decoy_spec_loss, batch_first=True, padding_value=1) # cos_loss <=1 by definition
            decoy_spec_loss_sorted = torch.sort(decoy_spec_loss, dim=-1).values.detach()
            ranking_dist = torch.abs(decoy_spec_loss[:, :, None] - decoy_spec_loss_sorted[:, None, :])
            top1_prob = pygm.sinkhorn(-ranking_dist, n1=batch["mol_num"], n2=batch["mol_num"], tau=self.sk_tau, backend='pytorch')[:, 0, 0]
            contr_loss = torch.relu(-torch.log(top1_prob + 0.5))  # shift & cut ce loss for probs > 0.5

            loss = {
                "spec_loss": spec_loss,
                "contr_loss": contr_loss,
                "loss": spec_loss + contr_loss * self.contr_weight,
            }
        else:
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
