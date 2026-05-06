import math

import ipdb
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter as ts
import dgl.nn as dgl_nn
import numpy as np
import einops

from rdkit import Chem

import dgl

import ms_pred.common as common
import ms_pred.nn_utils as nn_utils

import ms_pred.massformer_pred._massformer_graph_featurizer as mformer

import ms_pred.massformer_pred.massformer_code.gf_model as gf_model


class ScarfNet(pl.LightningModule):
    """ScarfNet."""

    def __init__(
        self,
        formula_dim: int,
        hidden_size: int,
        gnn_layers: int = 2,
        mlp_layers: int = 2,
        set_layers: int = 2,
        dropout: float = 0.0,
        learning_rate: float = 7e-4,
        lr_decay_rate: float = 1.0,
        weight_decay: float = 0,
        mpnn_type: str = "GGNN",
        atom_feats: list = (
            "a_onehot",
            "a_degree",
            "a_hybrid",
            "a_formal",
            "a_radical",
            "a_ring",
            "a_mass",
            "a_chiral",
        ),
        bond_feats: list = ("b_degree",),
        pool_op: str = "avg",
        pe_embed_k: int = 0,
        num_atom_feats: int = 86,
        num_bond_feats: int = 5,
        loss_fn: str = "mse",
        warmup: int = 1000,
        use_reverse: bool = False,
        no_forward: bool = False,
        embedder="abs-sines",
        use_tbc=True,
        root_embedder="gnn",
        info_join: str = "concat",
        embed_adduct: bool = False,
        **kwargs,
    ):
        """__init__ _summary_

        Args:
            formula_dim (int): _description_
            hidden_size (int): _description_
            gnn_layers (int, optional): _description_. Defaults to 2.
            mlp_layers (int, optional): _description_. Defaults to 2.
            set_layers (int, optional): _description_. Defaults to 2.
            dropout (float, optional): _description_. Defaults to 0.0.
            learning_rate (float, optional): _description_. Defaults to 7e-4.
            lr_decay_rate (float, optional): _description_. Defaults to 1.0.
            weight_decay (float, optional): _description_. Defaults to 0.
            mpnn_type (str, optional): _description_. Defaults to "GGNN".
            atom_feats (list, optional): _description_. Defaults to ( "a_onehot", "a_degree", "a_hybrid", "a_formal", "a_radical", "a_ring", "a_mass", "a_chiral", ).
            bond_feats (list, optional): _description_. Defaults to ("b_degree",).
            pool_op (str, optional): _description_. Defaults to "avg".
            pe_embed_k (int, optional): _description_. Defaults to 0.
            num_atom_feats (int, optional): _description_. Defaults to 86.
            num_bond_feats (int, optional): _description_. Defaults to 5.
            loss_fn (str, optional): _description_. Defaults to "mse".
            warmup (int, optional): _description_. Defaults to 1000.
            use_reverse (bool, optional): _description_. Defaults to False.
            no_forward (bool, optional): _description_. Defaults to False.
            embedder (str, optional): _description_. Defaults to "abs-sines".
            use_tbc (bool, optional): _description_. Defaults to True.
            root_embedder (str, optional): _description_. Defaults to "gnn".
            info_join (str, optional): _description_. Defaults to "concat".
            embed_adduct (bool, optional): _description_. Defaults to False.

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_
            NotImplementedError: _description_
            NotImplementedError: _description_
        """

        super().__init__()
        self.save_hyperparameters()

        # Embedding method
        self.embedder = nn_utils.get_embedder(embedder)

        # Model params
        self.hidden_size = hidden_size
        self.info_join = info_join
        self.embed_adduct = embed_adduct

        self.atom_feats = atom_feats
        self.bond_feats = bond_feats
        self.num_atom_feats = num_atom_feats
        self.num_bond_feats = num_bond_feats
        self.pe_embed_k = pe_embed_k
        self.pool_op = pool_op
        self.use_reverse = use_reverse
        self.no_forward = no_forward

        self.mlp_layers = mlp_layers
        self.gnn_layers = gnn_layers
        self.set_layers = set_layers
        self.mpnn_type = mpnn_type
        self.weight_decay = weight_decay

        self.formula_dim = formula_dim
        self.formula_in_dim = formula_dim * self.embedder.num_dim

        self.dropout = dropout
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.warmup = warmup

        self.formula_onehot = nn.Parameter(torch.eye(self.formula_dim)).float()
        self.formula_onehot.requires_grad = False

        adduct_shift = 0
        if self.embed_adduct:
            adduct_types = len(common.ion2onehot_pos)
            onehot = torch.eye(adduct_types)
            self.adduct_embedder = nn.Parameter(onehot.float())
            self.adduct_embedder.requires_grad = False
            adduct_shift = adduct_types

        self.root_embedder = root_embedder
        if self.root_embedder == "gnn":
            self.gnn = nn_utils.MoleculeGNN(
                hidden_size=self.hidden_size,
                num_step_message_passing=self.gnn_layers,
                set_transform_layers=self.set_layers,
                mpnn_type=self.mpnn_type,
                gnn_node_feats=num_atom_feats + adduct_shift,
                gnn_edge_feats=num_bond_feats,
                dropout=dropout,
            )
            if self.pool_op == "avg":
                self.pool = dgl_nn.AvgPooling()
            elif self.pool_op == "attn":
                self.pool = dgl_nn.GlobalAttentionPooling(nn.Linear(hidden_size, 1))
            else:
                raise NotImplementedError()
        elif self.root_embedder == "fp":
            self.root_embed_module = nn_utils.MLPBlocks(
                input_size=2048,
                hidden_size=self.hidden_size,
                num_layers=self.gnn_layers,
                dropout=self.dropout,
                output_size=self.hidden_size,
                use_residuals=True,
            )
        elif self.root_embedder == "graphormer":
            # Build gfv2 embedder with default args
            self.root_embed_module = gf_model.GFv2Embedder(
                gf_model_name="graphormer_base",
                gf_pretrain_name="pcqm4mv2_graphormer_base",
                fix_num_pt_layers=0,
                reinit_num_pt_layers=-1,
                reinit_layernorm=True,
            )
            embed_dim = self.root_embed_module.get_embed_dim()
            self.embed_to_hidden = nn.Linear(embed_dim + adduct_shift, self.hidden_size)

        else:
            raise NotImplementedError()

        # Define loss fn
        self.loss_fn_name = loss_fn
        if loss_fn == "bce":
            self.loss_fn = self.bce_loss
            self.output_activations = [nn.Sigmoid()]
        elif loss_fn == "bce_softmax":
            self.loss_fn = self.bce_loss_softmax
            self.output_activations = [nn.Sigmoid()]
        else:
            raise NotImplementedError()
        self.num_outputs = len(self.output_activations)

        # Hidden size, diff formula, option formula, atom index being predicted
        self.max_atom_out = common.MAX_ATOM_CT
        self.use_reverse_mult = max(self.use_reverse * 3, 1)
        output_size = (
            self.num_outputs
            * self.formula_dim
            * self.max_atom_out
            * self.use_reverse_mult
        )

        if self.info_join == "concat":
            # Mix with concat of formula encoding, mol encoding, form diff, and
            # opt- encoding
            # Output 1 value for each position in autoregressive atom decoding
            self.input_dim_cat = (
                self.hidden_size + self.formula_in_dim * 2 + formula_dim
            )
            self.output_layers = nn_utils.MLPBlocks(
                input_size=self.input_dim_cat,
                hidden_size=self.hidden_size,
                num_layers=self.mlp_layers,
                dropout=self.dropout,
                output_size=output_size,
                use_residuals=True,
            )
        elif self.info_join == "attn":
            self.input_dim_cat = self.formula_in_dim * 2 + formula_dim
            self.form_layers = nn_utils.MLPBlocks(
                input_size=self.input_dim_cat,
                hidden_size=self.hidden_size,
                num_layers=self.mlp_layers,
                dropout=self.dropout,
                use_residuals=True,
            )
            self.output_transform = nn.Linear(self.hidden_size, output_size)
        else:
            raise NotImplementedError()

        # whether to replace options/diffs past current atom index with to be confirmed token (i.e., distinguish zero
        # from yet to be defined in the autoregressive model.
        self.use_tbc_token = use_tbc

    def bce_loss(self, pred, targ, optionlens, scalar_mult=4):
        """bce_loss.

        Args:
            pred:
            targ:
            optionlens:
            scalar_mult:
        """
        pred = pred[:, :, 0]
        binary_targ = targ > 0

        # Create mask over dead prefixes
        option_shape = pred.shape[-2]
        aranged = torch.arange(option_shape, device=pred.device)
        len_mask = aranged[None, :] <= optionlens[:, None]

        # Weight by intensities but take softmax so zero weights still get some
        # vals
        # Mult by scalar
        bce_loss = F.binary_cross_entropy(pred, binary_targ.float(), reduction="none")

        # Don't need because weight is 0 above
        bce_loss = bce_loss.masked_fill(~len_mask[:, :, None], 0)
        bce_loss = bce_loss.sum(-1)
        bce_loss = bce_loss.sum(-1) / optionlens
        bce_loss_mean = bce_loss.mean()
        return {"loss": bce_loss_mean}

    def forward(
        self,
        graphs: dgl.graph,
        full_formula: torch.FloatTensor,
        options: torch.FloatTensor,
        diffs: torch.FloatTensor,
        option_len: torch.LongTensor,
        mol_inds: torch.LongTensor,
        atom_inds: torch.LongTensor,
        adducts: torch.LongTensor,
    ):
        """forward.

        Notation:
        b = batch size Note batches are at a given prefix tree level. So different batches could represent different
         formulae (i.e., molecules) or expansions at different levels of the prefix tree (or both!).
        F = number of unique complete formulae (i.e. how many molecules are we trying to compute the spectra for). Note
            that this does not _necessarily_ equal batch size (see above).
        e = number of possible elements; i.e., the maximum number of  possible elements we may see in all formulae and
            the dimension of "dense formulae" representation.
        p = prefix size: the maximum number of prefixes which we wish to expand at a given level of a tree
            (maximum across all members of batch). Should be padded along this dimension when forming batches.


        Args:
            graphs (dgl.Graph): graphs with a batch size equal to F
            full_formula (torch.FloatTensor): full_formula  [F, e]
            options (torch.FloatTensor): options  [b, p, e]
            diffs (torch.FloatTensor): diffs [b, p, e]
            option_len (torch.LongTensor): option_len [b] how much of the second dimension within options and diffs
                                            is valid (the rest is just padding to allow the batches to be concatenated
                                             together).
                                            maps from [0, b-1] to [1, p].
            mol_inds (torch.LongTensor): mol_inds [b] maps from [0, b-1] into full formula size [0,F-1].
            atom_inds (torch.LongTensor): atom_inds [b] maps from [0, b-1] to which atom [0, e-1] (i.e., what level of the
                        prefix tree is being predicted in that batch). Note that due to the fact that most atom counts
                        do not need to be predicted (because they are zero in the complete formula and therefore must be
                        zero in all formula) only certain atom indcs are likely to be predicted.
        """

        #

        # Verify types
        options = options.float()
        full_formula = full_formula.float()

        # add to be confirmed token to options/diffs which are yet to be created (i.e. distinguish these from zero).
        # at a high level we will do this by creating a mask of which option diffs are in this position and then
        # set these inputs to a number which represents learned embeddings in the tokenizer.
        if self.use_tbc_token:
            _, pad_size, formula_size = options.shape
            mask_atom_indx_tbc = (
                atom_inds[:, None]
                <= torch.arange(formula_size, device=options.device)[None, :]
            )
            # ^ shape: [b, e]. For each batch, which formulae elements are yet to be confirmed.

            mask_not_in_padding_zone = (
                option_len[:, None] - 1
                >= torch.arange(pad_size, device=options.device)[None, :]
            )
            # ^ shape: [b, p]. For each batch, which formulae are not in the padding zone.

            # we will reset those that have yet to be confirmed and are not part of the padding zone:
            mask_ = torch.logical_and(
                mask_atom_indx_tbc[:, None, :], mask_not_in_padding_zone[:, :, None]
            )

            # set the integers at this position to the index where the learned embeddings start:
            options[mask_] = self.embedder.MAX_COUNT_INT
            diffs[mask_] = self.embedder.MAX_COUNT_INT

        # Convert options and full formula to embedded representation (e.g., binary).
        options = self.embedder(options)
        diffs = self.embedder(diffs)

        # [Mols x hidden]
        if self.root_embedder == "gnn":

            with graphs.local_scope():
                if self.embed_adduct:
                    embed_adducts = self.adduct_embedder[adducts.long()]
                    ndata = graphs.ndata["h"]
                    embed_adducts_expand = embed_adducts.repeat_interleave(
                        graphs.batch_num_nodes(), 0
                    )
                    ndata_new = torch.cat([ndata, embed_adducts_expand], -1)

                    graphs.ndata["h"] = ndata_new

                mol_outputs = self.gnn(graphs)
                mol_embeds = self.pool(graphs, mol_outputs)
        elif self.root_embedder == "fp":
            mol_embeds = self.root_embed_module(graphs)
        elif self.root_embedder == "graphormer":
            mol_embeds = self.root_embed_module({"gf_v2_data": graphs})
            if self.embed_adduct:
                embed_adducts = self.adduct_embedder[adducts.long()]
                mol_embeds = torch.cat([mol_embeds, embed_adducts], -1)
            mol_embeds = self.embed_to_hidden(mol_embeds)

        else:
            raise NotImplementedError()

        # [(Mols * atom type) x hidden]
        mol_embeds_paired = mol_embeds[mol_inds.long()]

        mol_dim, hidden_dim = mol_embeds_paired.shape
        option_dim = options.shape[1]
        device = atom_inds.device

        # [(Mols * atom type) x options x formula_dim]
        option_diffs = diffs

        # Normalize options and option diffs
        # Don't normalize now that we use binarized versions of inputs

        # [(atom types), formula_dim]
        type_vec = self.formula_onehot[atom_inds.long()]

        # [(atom types), options, formula_dim]
        type_vec = einops.repeat(type_vec, "m n -> m k n", k=option_dim)
        mol_embeds_paired = einops.repeat(
            mol_embeds_paired, "m n -> m k n", k=option_dim
        )

        cat_list = []
        if self.info_join == "concat":
            cat_list.extend([mol_embeds_paired, option_diffs, options, type_vec])
            cat_tensor = torch.cat(cat_list, -1)

            # [(mols * atom type) x options x (num_outputs *formula_dim)]
            output = self.output_layers(cat_tensor)
        elif self.info_join == "attn":
            cat_list.extend([option_diffs, options, type_vec])
            cat_tensor = torch.cat(cat_list, -1)

            # (B *  atom_type) x atom type x hidden
            form_embeds = self.form_layers(cat_tensor)

            #  B x L x H
            pad_token = -9999
            atom_embeds = nn_utils.pad_packed_tensor(
                mol_outputs, graphs.batch_num_nodes(), pad_token
            )
            # B x L x H
            embed_mask = atom_embeds == pad_token

            # B x 1 x L
            embed_mask = torch.any(embed_mask, -1)[:, None, :]

            # Need to expand B out to atom type
            # (B * atom_type) x L x H
            atom_embeds = atom_embeds[mol_inds]
            embed_mask = embed_mask[mol_inds]

            # Atom pooling dims
            attn_weights = torch.einsum("blh,bah->bal", atom_embeds, form_embeds)
            attn_weights.masked_fill_(embed_mask, pad_token)

            # Softmax on length dimension
            # B x F x L
            attn_weights = torch.softmax(attn_weights, -1)
            output = torch.einsum("bal,blh->bah", attn_weights, atom_embeds)
            output = self.output_transform(output)
        else:
            raise NotImplementedError()

        # Reshape
        # [(mols*atom type) x options x num_outputs x formula_dim x maxatom ct
        # x use reverse mult]
        output = output.reshape(
            mol_dim,
            option_dim,
            self.num_outputs,
            self.formula_dim,
            self.max_atom_out,
            self.use_reverse_mult,
        )

        #  [(mols * atom type) x options x num_outputs x use reverse mult]
        output = output[torch.arange(mol_dim).to(device), :, :, atom_inds.long(), :]
        # even with TBC tokenization it seems better to have a unique head for each formulae.

        # Activate everything
        # Note: this assumes that everything is a sigmoid!
        new_outputs = []
        for output_ind, act in enumerate(self.output_activations):
            new_outputs.append(act(output[:, :, output_ind : output_ind + 1]))
        output = torch.cat(new_outputs, 2)

        max_pos_num = full_formula[mol_inds.long(), atom_inds.long()]
        mask_ct = torch.arange(self.max_atom_out).to(device)
        mask = mask_ct[None, :] <= max_pos_num[:, None]
        mask = mask[:, None, None, :]
        if self.use_reverse:
            forward, gates, reverse = output[..., 0], output[..., 1], output[..., 2]
            ind_adjusts = max_pos_num[:, None] - mask_ct[None, :]
            ind_adjusts = ind_adjusts[:, None, None, :].expand(reverse.shape)
            ind_adjusts = ind_adjusts % self.max_atom_out
            reverse = torch.gather(reverse, dim=-1, index=ind_adjusts.long())
            if self.no_forward:
                output = reverse
            else:
                output = reverse * (1 - gates) + gates * forward

        else:
            output = output[..., 0]
        # Mask final output
        output = mask * output
        return output

    def make_prediction(
        self, formula_tensors, mol_graphs, max_nodes=500, threshold=1e-20, adducts=None
    ):
        """make_prediction."""

        device = formula_tensors.device
        with torch.no_grad():
            batch_dim, atom_dim = formula_tensors.shape
            full_formula = formula_tensors

            ident = torch.eye(atom_dim, device=device)

            # [B x 1 opt x Atom dim]
            prev_opts = torch.zeros_like(full_formula, device=device)[:, None, :]
            prev_outputs = torch.zeros(prev_opts.shape[:-1], device=device)

            # Mol inds
            mol_inds_base = torch.arange(batch_dim, device=device)

            # Define size of each beam
            prev_option_len = torch.ones(batch_dim, device=device).long()

            # Iterate over all atom types
            for atom_ind in range(atom_dim):

                # Only go investigate an atom type if it's got som enonzero
                # examples in full formula
                num_atoms = full_formula[:, atom_ind]

                # Certain rows may not need to be expanded
                # rows_to_expand = torch.where(full_formula[:, atom_ind] > 0)[0]
                if torch.all(num_atoms == 0):
                    continue

                # Define mol inds; should be batch len
                # We only compute one atom type per batch, so each batch item will
                # only have 1 correpsonding option vector (i.e, batch dim on
                # options and mols are the same)
                mol_inds = mol_inds_base
                atom_inds = torch.ones_like(mol_inds, device=device) * atom_ind

                prefixes = prev_opts
                diffs = full_formula[:, None, :] - prefixes
                alive_prefix_len = prev_option_len

                # graphs  (dgl.graph): Mol graphs
                # full_formula (torch.FloatTensor): full formula
                # options (torch.FloatTensor): options tensor
                # option_len (torch.LongTensor): length of each option tensor
                # mol_inds (torch.LongTensor): Mapping from mols to options
                # atom_inds (torch.LongTensor): Which atom type is being expanded
                output = self.forward(
                    graphs=mol_graphs,
                    full_formula=full_formula,
                    options=prefixes,
                    diffs=diffs,
                    option_len=alive_prefix_len,
                    mol_inds=mol_inds,
                    atom_inds=atom_inds,
                    adducts=adducts,
                )
                if self.loss_fn_name in {
                    "cosine",
                    "bce",
                    "bce_softmax",
                    "cosine_weight",
                    "cosine_div",
                }:
                    output = torch.log(output[:, :, 0])
                else:
                    raise NotImplementedError()

                # Sum log probabilities across beams
                output = output + prev_outputs[:, :, None]

                # Construct proposed new prefix tensor
                onehot = ident[atom_ind]
                suffixes = torch.arange(0, self.max_atom_out, device=device)

                # new_opts: [max options x atom dim]
                suffixes_onehot = onehot[None, :] * suffixes[:, None]

                # Need to cross with prev opts
                # prev_opts: [batch x max options x atom dim]
                # [Batch x new  opts x prev_opts x atom dim]

                # I.e., if prev opts has 5 examples that don't matter, these
                # should _all_ be at the bottom for new opts
                new_prefixes = (
                    prev_opts[:, :, None, :] + suffixes_onehot[None, None, :, :]
                )

                # Get top 50 options for each example first
                # This gets the acutal outputs
                output_reshaped = output.view(batch_dim, -1)
                new_prefixes_reshaped = new_prefixes.view(
                    batch_dim, -1, new_prefixes.shape[-1]
                )

                keep_inds = torch.argsort(output_reshaped, -1, descending=True)[
                    :, :max_nodes
                ]
                output = torch.gather(output_reshaped, dim=-1, index=keep_inds)
                # Expand to fit shape of options
                keep_inds_expanded = einops.repeat(
                    keep_inds, "m n -> m n k", k=atom_dim
                )
                new_prefixes_reshaped = torch.gather(
                    new_prefixes_reshaped, dim=-2, index=keep_inds_expanded
                )

                # Update prev options
                prev_opts = new_prefixes_reshaped
                prev_option_len = torch.sum(output > math.log(threshold + 1e-22), -1)

                new_middle_dim = prev_option_len.max()
                prev_opts = prev_opts[:, :new_middle_dim, :]
                prev_outputs = output[:, :new_middle_dim]

                # TODO: Mask based upon prev_opts_len
                arange_middle = torch.arange(new_middle_dim, device=device)
                dead_mask = arange_middle[None, :] >= prev_option_len[:, None]
                prev_outputs.masked_fill_(dead_mask, -float("inf"))

                # Mask
                # assert(torch.any((prev_opts - full_formula[:, None, :]) > 0))
                # output = output.masked_fill(invalid_mask, 0)

        # Leaving loop, compute full output
        formula_outputs, inten_outputs = [], []
        for idx in range(batch_dim):
            ex_kept = prev_option_len[idx].item()

            # Break if none kept
            if ex_kept == 0:
                inten_outputs.append([])
                formula_outputs.append([])
                continue

            ex_output = prev_outputs[idx][:ex_kept].detach().cpu().numpy()
            ex_options = prev_opts[idx][:ex_kept].detach().cpu().numpy()

            # RDBE filter
            filter_inds = common.rdbe_filter(ex_options)
            ex_output = ex_output[filter_inds]
            ex_options = ex_options[filter_inds]

            keep_formulas = [
                common.vec_to_formula(ex_option) for ex_option in ex_options
            ]
            keep_intens = ex_output
            joint_output = [
                (i, j) for i, j in zip(keep_formulas, keep_intens) if i != ""
            ]
            keep_formulas, keep_intens = zip(*joint_output)
            keep_intens = np.exp(np.array(keep_intens))

            inten_outputs.append(keep_intens)
            formula_outputs.append(keep_formulas)

        return formula_outputs, inten_outputs

    def predict_mol(
        self,
        smi,
        graph_featurizer,
        device="cpu",
        max_nodes=500,
        threshold=1e-20,
        adduct=None,
    ):
        """predict_mol.

        Args:
            smiles:
            max_nodes:
            threshold:
        """
        mol = Chem.MolFromSmiles(smi)

        if self.root_embedder == "gnn":
            root_encode_fn = graph_featurizer.get_dgl_graph
        elif self.root_embedder == "fp":
            root_encode_fn = lambda x: torch.FloatTensor(
                common.get_morgan_fp(x)[None, :]
            )
        else:
            raise NotImplementedError()

        mol_graph = root_encode_fn(mol)
        base_form = common.form_from_smi(smi)
        full_form_vec = common.formula_to_dense(base_form)

        # 1 x Atom
        full_form_vec = torch.tensor(full_form_vec).to(device)[None, :]
        mol_graph = mol_graph.to(device)

        adducts = torch.LongTensor([common.ion2onehot_pos[adduct]])

        # Define form list outptut
        outputs = self.make_prediction(
            full_form_vec,
            mol_graph,
            max_nodes=max_nodes,
            threshold=threshold,
            adducts=adducts,
        )
        return outputs

    def _common_step(self, batch, name="train"):
        outputs = self.forward(
            batch["graphs"],
            batch["formula_tensors"].float(),
            batch["options"].float(),
            batch["diffs"].float(),
            batch["option_len"],
            batch["mol_inds"].long(),
            batch["atom_inds"].float(),
            adducts=batch["adducts"],
        )

        # dict_keys(['names', 'fps', 'formula_tensors', 'options', 'option_len',
        #           'targ_inten', 'atom_inds', 'num_atom_types', 'mol_inds'])
        loss_dict = self.loss_fn(outputs, batch["targ_inten"], batch["option_len"])
        self.log(f"{name}_loss", loss_dict.get("loss"), on_epoch=True, logger=True)
        return loss_dict

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, name="train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, name="val")

    def test_step(self, batch, batch_idx):
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

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric=None):
        scheduler.step()


class ScarfIntenNet(pl.LightningModule):
    """ScarfIntenNet."""

    def __init__(
        self,
        formula_dim: int,
        hidden_size: int,
        gnn_layers: int = 2,
        mlp_layers: int = 2,
        set_layers: int = 2,
        form_set_layers: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 7e-4,
        lr_decay_rate: float = 1.0,
        weight_decay: float = 0,
        mpnn_type: str = "GGNN",
        atom_feats: list = (
            "a_onehot",
            "a_degree",
            "a_hybrid",
            "a_formal",
            "a_radical",
            "a_ring",
            "a_mass",
            "a_chiral",
        ),
        bond_feats: list = ("b_degree",),
        pool_op: str = "avg",
        pe_embed_k: int = 0,
        num_atom_feats: int = 86,
        num_bond_feats: int = 5,
        loss_fn: str = "cos",
        warmup: int = 1000,
        use_reverse: bool = False,
        embedder="abs-sines",
        info_join: str = "concat",
        root_embedder="gnn",
        binned_targs: bool = False,
        embed_adduct: bool = False,
        **kwargs,
    ):
        """__init__.

        Args:
            formula_dim (int): formula_dim
            hidden_size (int): hidden_size
            layers (int): layers
            dropout (float): dropout
            learning_rate (float): learning_rate
            lr_decay_rate (float): lr_decay_rate
            weight_decay (float): weight_decay
            loss_fn (str): loss_fn
            mpnn_type (str): mpnn_type
            set_layers (int): set_layers
            atom_feats (list): atom_feats
            bond_feats (list): bond_feats
            pool_op (str): pool_op
            pe_embed_k (int): pe_embed_k
            num_atom_feats (int): num_atom_feats
            num_bond_feats (int): num_bond_feats
            info_join
            kwargs:
        """
        super().__init__()
        self.save_hyperparameters()
        # Embedding method

        self.embedder = nn_utils.get_embedder(embedder)

        # Model params
        self.hidden_size = hidden_size

        self.info_join = info_join

        self.atom_feats = atom_feats
        self.bond_feats = bond_feats
        self.num_atom_feats = num_atom_feats
        self.num_bond_feats = num_bond_feats
        self.pe_embed_k = pe_embed_k
        self.pool_op = pool_op

        self.mlp_layers = mlp_layers
        self.gnn_layers = gnn_layers
        self.set_layers = set_layers
        self.form_set_layers = form_set_layers
        self.mpnn_type = mpnn_type
        self.weight_decay = weight_decay

        self.formula_dim = formula_dim
        self.formula_in_dim = formula_dim * self.embedder.num_dim

        self.embed_adduct = embed_adduct

        adduct_shift = 0
        if self.embed_adduct:
            adduct_types = len(common.ion2onehot_pos)
            onehot = torch.eye(adduct_types)
            self.adduct_embedder = nn.Parameter(onehot.float())
            self.adduct_embedder.requires_grad = False
            adduct_shift = adduct_types

        self.dropout = dropout
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.warmup = warmup
        self.binned_targs = binned_targs

        self.root_embedder = root_embedder
        if self.root_embedder == "gnn":
            self.gnn = nn_utils.MoleculeGNN(
                hidden_size=self.hidden_size,
                num_step_message_passing=self.gnn_layers,
                set_transform_layers=self.set_layers,
                mpnn_type=self.mpnn_type,
                gnn_node_feats=num_atom_feats + adduct_shift,
                gnn_edge_feats=num_bond_feats,
                dropout=dropout,
            )
            if self.pool_op == "avg":
                self.pool = dgl_nn.AvgPooling()
            elif self.pool_op == "attn":
                self.pool = dgl_nn.GlobalAttentionPooling(nn.Linear(hidden_size, 1))
            else:
                raise NotImplementedError()

        elif self.root_embedder == "fp":
            self.root_embed_module = nn_utils.MLPBlocks(
                input_size=2048,
                hidden_size=self.hidden_size,
                num_layers=self.gnn_layers,
                dropout=self.dropout,
                output_size=self.hidden_size,
                use_residuals=True,
            )
        elif self.root_embedder == "graphormer":
            # Build gfv2 embedder with default args
            self.root_embed_module = gf_model.GFv2Embedder(
                gf_model_name="graphormer_base",
                gf_pretrain_name="pcqm4mv2_graphormer_base",
                fix_num_pt_layers=0,
                # reinit_num_pt_layers=-1,
                reinit_num_pt_layers=-1,
                reinit_layernorm=True,
            )
            embed_dim = self.root_embed_module.get_embed_dim()
            self.embed_to_hidden = nn.Linear(embed_dim + adduct_shift, self.hidden_size)
        else:
            raise NotImplementedError()

        # Define loss fn
        self.loss_fn_name = loss_fn
        if loss_fn == "cosine":
            self.loss_fn = self.cos_loss
            self.cos_fn = nn.CosineSimilarity()
            self.output_activations = [nn.Sigmoid()]
        else:
            raise NotImplementedError()

        self.num_outputs = len(self.output_activations)

        # Mix with concat of formula encoding, mol encoding, form diff, and
        # opt- encoding
        # Output 1 value for each position in autoregressive atom decoding
        formula_in_dim = self.formula_in_dim * 2
        if self.info_join == "concat":
            # Hidden size, diff formula, option formula, atom index being predicted
            self.input_dim_cat = self.hidden_size + formula_in_dim
            self.mlp_layers = nn_utils.MLPBlocks(
                input_size=self.input_dim_cat,
                hidden_size=self.hidden_size,
                num_layers=self.mlp_layers,
                dropout=self.dropout,
                output_size=self.hidden_size,
                use_residuals=True,
            )
        elif self.info_join == "attn":
            self.formula_mlp_in = formula_in_dim
            self.form_mlp = nn_utils.MLPBlocks(
                input_size=self.formula_mlp_in,
                hidden_size=self.hidden_size,
                num_layers=self.mlp_layers,
                dropout=self.dropout,
                output_size=self.hidden_size,
                use_residuals=True,
            )
        else:
            raise NotImplementedError()

        trans_layer = nn_utils.TransformerEncoderLayer(
            self.hidden_size,
            nhead=8,
            batch_first=True,
            norm_first=False,
            dim_feedforward=self.hidden_size * 4,
        )
        self.trans_layers = nn_utils.get_clones(trans_layer, self.form_set_layers)
        self.output_layer = nn.Linear(self.hidden_size, self.num_outputs)

        # Define masss vector
        self.formula_mass_vector = torch.DoubleTensor(common.VALID_MONO_MASSES)
        self.formula_mass_vector = nn.Parameter(self.formula_mass_vector)
        self.formula_mass_vector.requires_grad = False

        buckets = torch.DoubleTensor(np.linspace(0, 1500, 15000))
        self.inten_buckets = nn.Parameter(buckets)
        self.inten_buckets.requires_grad = False

        # Define mapping to attn weights
        self.attn_output = nn.Linear(self.hidden_size, self.num_outputs)

    def cos_loss(self, pred, targ, optionlens):
        """cos_loss.

        Args:
            pred:
            targ:
            optionlens:
        """
        pred = pred[:, :, 0]
        option_shape = pred.shape[-1]
        if not self.binned_targs:
            aranged = torch.arange(option_shape, device=pred.device)
            pad_mask = aranged[None, :] > optionlens[:, None]
            targ = targ.masked_fill(pad_mask, 0)
            pred = pred.masked_fill(pad_mask, 0)
        loss = 1 - self.cos_fn(pred, targ)
        loss = loss.mean()
        return {"loss": loss}

    def _forward_no_bin(
        self,
        graphs: dgl.graph,
        formulae: torch.FloatTensor,
        diffs: torch.FloatTensor,
        num_forms: torch.LongTensor,
        adducts: torch.LongTensor,
    ) -> dict:
        """_forward_no_bin.

        Predict outputs without going all the way to bins

        Args:
            graphs (dgl.graph): graphs
            formulae (torch.FloatTensor): formulae
            diffs (torch.FloatTensor): diffs
            num_forms (torch.LongTensor): num_forms
        """
        device = formulae.device
        batch_dim, form_dim, atom_dim = formulae.shape

        # Convert formulae and diffs to binary
        formulae = self.embedder(formulae)
        diffs = self.embedder(diffs)

        # [Mols x hidden]
        if self.root_embedder == "gnn":
            with graphs.local_scope():
                if self.embed_adduct:
                    embed_adducts = self.adduct_embedder[adducts.long()]
                    ndata = graphs.ndata["h"]
                    embed_adducts_expand = embed_adducts.repeat_interleave(
                        graphs.batch_num_nodes(), 0
                    )
                    ndata = torch.cat([ndata, embed_adducts_expand], -1)
                    graphs.ndata["h"] = ndata
                mol_outputs = self.gnn(graphs)
                mol_embeds = self.pool(graphs, mol_outputs)
        elif self.root_embedder == "fp":
            mol_embeds = self.root_embed_module(graphs)
            raise NotImplementedError()
        elif self.root_embedder == "graphormer":
            mol_embeds = self.root_embed_module({"gf_v2_data": graphs})
            if self.embed_adduct:
                embed_adducts = self.adduct_embedder[adducts.long()]
                mol_embeds = torch.cat([mol_embeds, embed_adducts], -1)
            mol_embeds = self.embed_to_hidden(mol_embeds)
        else:
            raise NotImplementedError()

        # Mols x hidden => mols x forms x hidden
        mol_embeds_expand = mol_embeds[:, None, :].expand(-1, form_dim, -1)

        form_cat = [diffs, formulae]
        if self.info_join == "concat":
            cat_list = [mol_embeds_expand, *form_cat]
            cat_tensor = torch.cat(cat_list, -1)

            # Batch x max len x input => batch x max len x hidden
            hidden = self.mlp_layers(cat_tensor)
        elif self.info_join == "attn":
            cat_list = form_cat
            cat_tensor = torch.cat(cat_list, -1)

            # B x Forms x Hidden
            form_embeds = self.form_mlp(cat_tensor)

            #  B x L x H
            pad_token = -9999
            atom_embeds = nn_utils.pad_packed_tensor(
                mol_outputs, graphs.batch_num_nodes(), pad_token
            )
            # B x L x H
            embed_mask = atom_embeds == pad_token
            # B x 1 x L
            embed_mask = torch.any(embed_mask, -1)[:, None, :]

            # Atom pooling dims
            attn_weights = torch.einsum("blh,bfh->bfl", atom_embeds, form_embeds)
            attn_weights.masked_fill_(embed_mask, pad_token)

            # Softmax on length dimension
            # B x F x L
            attn_weights = torch.softmax(attn_weights, -1)
            hidden = torch.einsum("bfl,blh->bfh", attn_weights, atom_embeds)
        else:
            raise NotImplementedError()

        # Build up a mask
        arange_frags = torch.arange(form_dim).to(device)
        attn_mask = ~(arange_frags[None, :] < num_forms[:, None])

        for trans_layer in self.trans_layers:
            hidden, _ = trans_layer(hidden, src_key_padding_mask=attn_mask)

        output = self.output_layer(hidden)
        return {"output": output, "hidden": hidden, "attn_mask": attn_mask}

    def bin_form_preds(self, formula_in, hidden, output, attn_mask):
        """_bin_form_preds.

        Args:
            formula_in:
            hidden:
            output:
            attn_mask:
        """
        form_masses = (formula_in * self.formula_mass_vector).sum(-1).float()

        # B x forms x outputs
        form_attns = self.attn_output(hidden)
        form_attns.masked_fill_(attn_mask[:, :, None], -99999)

        # B x outputs x forms
        form_attns = form_attns.transpose(2, 1)

        inverse_indices = torch.bucketize(form_masses, self.inten_buckets, right=False)
        inverse_indices = inverse_indices[:, None, :].expand(form_attns.shape)

        # B x outputs x forms
        if self.binned_targs:
            form_attns = ts.scatter_softmax(form_attns, index=inverse_indices, dim=-1)
            output = output.transpose(2, 1)
            output = form_attns * output

            # B x Outs x Unique inds
            output = ts.scatter_add(
                output,
                index=inverse_indices,
                dim_size=self.inten_buckets.shape[-1],
                dim=-1,
            )
        else:
            # If not self.binned_targs, just take a max because it will be
            # untrained pooling
            # B x Outs x Unique inds
            output = output.transpose(2, 1)
            output = ts.scatter_max(
                output,
                index=inverse_indices,
                dim_size=self.inten_buckets.shape[-1],
                dim=-1,
            )

        # B x Outs x Unique inds
        inv_attn_mask = ts.scatter_max(
            (~attn_mask).long(),
            index=inverse_indices[:, 0, :],
            dim_size=self.inten_buckets.shape[-1],
            dim=-1,
        )[0].bool()
        attn_mask = ~inv_attn_mask

        # B x forms x out
        output = output.transpose(1, 2)
        return {"output": output, "attn_mask": attn_mask}

    def forward(
        self,
        graphs: dgl.graph,
        formulae: torch.FloatTensor,
        diffs: torch.FloatTensor,
        num_forms: torch.LongTensor,
        adducts: torch.LongTensor,
        binned_targs: bool = False,
    ):
        """forward.

        Args:
            graphs (dgl.graph): graphs
            formulae (torch.FloatTensor): formulae
            diffs (torch.FloatTensor): diffs
            num_forms (torch.LongTensor): num_forms
            adducts: (torch.LongTensor): adducts
            binned_targs (bool): Binned targs 
        """
        if binned_targs:
            return self.forward_binned(graphs, formulae, diffs, num_forms, adducts)
        else:
            return self.forward_unbinned(graphs, formulae, diffs, num_forms, adducts)


    def forward_unbinned(
        self,
        graphs: dgl.graph,
        formulae: torch.FloatTensor,
        diffs: torch.FloatTensor,
        num_forms: torch.LongTensor,
        adducts: torch.LongTensor,
    ):
        output_dict = self._forward_no_bin(graphs, formulae, diffs, num_forms, adducts)
        # B x Forms x Outs
        # B x Forms (mask where true)
        output = output_dict["output"]
        attn_mask = output_dict["attn_mask"]

        # Activate each dim with its respective output activation
        # Helpful for hurdle or probabilistic models
        new_outputs = []
        for output_ind, act in enumerate(self.output_activations):
            new_outputs.append(act(output[:, :, output_ind : output_ind + 1]))
        output = torch.cat(new_outputs, -1)

        # Zero mask all outputs
        output.masked_fill_(attn_mask[:, :, None], 0)
        return output

    def forward_binned(
        self,
        graphs: dgl.graph,
        formulae: torch.FloatTensor,
        diffs: torch.FloatTensor,
        num_forms: torch.LongTensor,
        adducts: torch.LongTensor,
    ):
        formula_in = formulae
        output_dict = self._forward_no_bin(graphs, formulae, diffs, num_forms, adducts)

        # B x Forms x Outs
        # B x Forms x D
        # B x Forms (mask where true)
        output = output_dict["output"]
        hidden = output_dict["hidden"]
        attn_mask = output_dict["attn_mask"]
        binned_outs = self.bin_form_preds(
            formula_in=formula_in, hidden=hidden, output=output, attn_mask=attn_mask
        )
        attn_mask = binned_outs["attn_mask"]
        output = binned_outs["output"]

        # Activate each dim with its respective output activation
        # Helpful for hurdle or probabilistic models
        new_outputs = []
        for output_ind, act in enumerate(self.output_activations):
            new_outputs.append(act(output[:, :, output_ind : output_ind + 1]))
        output = torch.cat(new_outputs, -1)

        # Zero mask all outputs
        output.masked_fill_(attn_mask[:, :, None], 0)
        return output

    def predict(self, graphs, full_formula, diffs, num_forms, adducts, binned_out=True):
        """predict."""
        out = self.forward(
            graphs=graphs,
            formulae=full_formula,
            diffs=diffs,
            num_forms=num_forms,
            adducts=adducts,
            binned_targs=binned_out,
        )
        if binned_out:
            num_frag = self.inten_buckets.shape[-1]
            num_forms = torch.ones_like(num_forms) * num_frag

        if self.loss_fn_name in ["mse", "cosine"]:
            preds = out[
                :,
                :,
                0,
            ]

            # Make sure list length is correct
            out_preds = [
                pred[:num_frag].cpu().numpy()
                for pred, num_frag in zip(preds, num_forms)
            ]
            out_dict = {
                "spec": out_preds,
            }
        else:
            raise NotImplementedError()
        return out_dict

    def _common_step(self, batch, name="train"):
        outputs = self.forward(
            graphs=batch["graphs"],
            formulae=batch["formulae"],
            diffs=batch["diffs"],
            num_forms=batch["num_forms"],
            adducts=batch["adducts"],
            binned_targs=self.binned_targs,
        )
        loss_dict = self.loss_fn(outputs, batch["intens"], batch["num_forms"])
        self.log(f"{name}_loss", loss_dict.get("loss"), on_epoch=True, logger=True)
        for k, v in loss_dict.items():
            if k != "loss":
                self.log(f"{name}_aux_{k}", v.item())
        return loss_dict

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, name="train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, name="val")

    def test_step(self, batch, batch_idx):
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

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric=None):
        scheduler.step()


class JointModel(pl.LightningModule):
    def __init__(self, gen_model_obj: ScarfNet, inten_model_obj: ScarfIntenNet):
        """__init__.

        Args:
            gen_model_obj (ScarfNet): gen_model_obj
            inten_model_obj (ScarfIntenNet): inten_model_obj
        """
        super().__init__()
        self.gen_model_obj = gen_model_obj
        self.inten_model_obj = inten_model_obj
        self.gen_featurizer = nn_utils.MolDGLGraph(
            atom_feats=self.gen_model_obj.atom_feats,
            bond_feats=self.gen_model_obj.bond_feats,
            pe_embed_k=self.gen_model_obj.pe_embed_k,
        )
        self.inten_featurizer = nn_utils.MolDGLGraph(
            atom_feats=self.inten_model_obj.atom_feats,
            bond_feats=self.inten_model_obj.bond_feats,
            pe_embed_k=self.inten_model_obj.pe_embed_k,
        )

        if self.inten_model_obj.root_embedder == "gnn":
            self.root_encode_fn = self.inten_featurizer.get_dgl_graph
        elif self.inten_model_obj.root_embedder == "fp":
            self.root_encode_fn = lambda x: torch.FloatTensor(
                common.get_morgan_fp(x)[None, :]
            )
        else:
            raise ValueError()

    def predict_mol(
        self,
        smi: str,
        threshold: float,
        device: str,
        max_nodes: int,
        adduct,
        binned_out=False,
    ) -> dict:
        """predict_mol.

        Args:
            smi (str): smi
            threshold (float): threshold
            device (str): device
            max_nodes (int): max_nodes
            adduct:
            binned_out:

        Returns:
            dict:
        """
        # Gen objs
        gen_prefixes, probs = self.gen_model_obj.predict_mol(
            smi=smi,
            graph_featurizer=self.gen_featurizer,
            threshold=threshold,
            device=device,
            max_nodes=max_nodes,
            adduct=adduct,
        )
        log_probs = np.log(probs[0])
        gen_prefixes = gen_prefixes[0]

        # Can replace these repeated calculations with gen model
        mol = Chem.MolFromSmiles(smi)

        mol_graph = self.root_encode_fn(mol)
        adducts = torch.LongTensor([common.ion2onehot_pos[adduct]])

        base_form = common.form_from_smi(smi)
        full_form_vec = common.formula_to_dense(base_form)
        forms = np.vstack([common.formula_to_dense(i) for i in gen_prefixes])
        diffs = full_form_vec[None, :] - forms
        num_forms = [len(forms)]

        # Convert all to tensors
        mol_graph = mol_graph.to(device)
        forms = torch.tensor(forms).to(device)[None, :, :]
        diffs = torch.tensor(diffs).to(device)[None, :, :]
        num_forms = torch.tensor(num_forms).to(device)

        # Make binned out predictions if happening
        inten_outs = self.inten_model_obj.predict(
            mol_graph, forms, diffs, num_forms, adducts=adducts, binned_out=binned_out
        )

        # Define outputs that go in model above
        masses = [common.formula_mass(i) for i in gen_prefixes]

        # Out flattented
        out = {k: v[0].flatten() for k, v in inten_outs.items()}
        forms = gen_prefixes
        out["masses"] = masses
        out["forms"] = forms
        return out
