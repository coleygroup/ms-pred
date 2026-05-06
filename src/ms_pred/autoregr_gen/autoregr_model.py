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


class AutoregrNet(pl.LightningModule):
    """AutoregrNet."""

    def __init__(
        self,
        formula_dim: int,
        hidden_size: int,
        gnn_layers: int = 2,
        set_layers: int = 2,
        rnn_layers: int = 2,
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
        warmup: int = 1000,
        use_reverse: bool = False,
        root_embedder="gnn",
        embed_adduct: bool = False,
        embedder="abs-sines",
        # **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Embedding method
        self.embedder = nn_utils.get_embedder(embedder)
        self.formula_dim = formula_dim + 1
        self.formula_in_dim = formula_dim * self.embedder.num_dim

        # Model params
        self.hidden_size = hidden_size
        self.embed_adduct = embed_adduct

        self.atom_feats = atom_feats
        self.bond_feats = bond_feats
        self.num_atom_feats = num_atom_feats
        self.num_bond_feats = num_bond_feats
        self.pe_embed_k = pe_embed_k
        self.pool_op = pool_op
        self.use_reverse = use_reverse

        self.gnn_layers = gnn_layers
        self.rnn_layers = rnn_layers
        self.set_layers = set_layers
        self.mpnn_type = mpnn_type
        self.weight_decay = weight_decay

        self.dropout = dropout
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.warmup = warmup

        # Define input to RNN
        self.formula_onehot = nn.Parameter(torch.eye(self.formula_dim + 1)).float()
        self.formula_onehot.requires_grad = False

        # Create start token
        self.start_token = self.formula_dim

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

        # Define context layer
        self.context_layer = nn.Linear(
            self.hidden_size + self.formula_in_dim, self.hidden_size
        )

        # Hidden size, diff formula, option formula, atom index being predicted
        self.max_atom_out = common.MAX_ATOM_CT
        self.use_reverse_mult = max(self.use_reverse * 3, 1)
        output_size = self.max_atom_out * self.use_reverse_mult
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_size, output_size),
        )

        self.atom_ct_onehot = nn.Parameter(torch.eye(self.max_atom_out)).float()
        self.atom_ct_onehot.requires_grad = False
        rnn_input_dim = self.formula_dim + self.max_atom_out + 1

        # Define rnn using rnn layers
        self.rnn = nn.LSTM(
            input_size=rnn_input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.rnn_layers,
            dropout=self.dropout,
            batch_first=True,
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def get_hidden(self, graphs, full_formula, adducts):
        # [Mols x hidden]
        form_embedding = self.embedder(full_formula)
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

        # Define full context input
        h0 = self.context_layer(torch.cat([mol_embeds, form_embedding], -1))
        h0 = h0[None, :, :].repeat(self.rnn_layers, 1, 1).contiguous()
        c0 = torch.zeros_like(h0)
        return h0, c0

    def forward(
        self,
        graphs: dgl.graph,
        full_formula: torch.FloatTensor,
        atom_inds: torch.FloatTensor,
        adducts: torch.LongTensor = None,
        atom_ct_inputs: torch.FloatTensor = None,
        hidden_state: torch.FloatTensor = None,
    ):
        full_formula = full_formula.float()
        device = full_formula.device
        batch = full_formula.shape[0]

        if hidden_state is None:
            h0, c0 = self.get_hidden(
                graphs=graphs, full_formula=full_formula, adducts=adducts
            )
        else:
            h0, c0 = hidden_state

        # Three inputs:
        # 1. Context vector (added)
        # 2. The previous token count (including start tokens) --> To make onehot
        # 3. The chem formula index onehot of the next element to be predicted
        atom_cts_vec = self.atom_ct_onehot[atom_ct_inputs.long()]
        atom_types_vec = self.formula_onehot[atom_inds.long()]

        rnn_input_vec = torch.cat([atom_cts_vec, atom_types_vec], -1)

        # Apply rnn
        rnn_output, out_hidden = self.rnn(rnn_input_vec, (h0, c0))

        # Correct this
        outputs = self.output_layer(rnn_output)

        # Use reverse gating
        max_pos_num = full_formula.take_along_dim(atom_inds.long(), -1)
        mask_ct = torch.arange(self.max_atom_out).to(device)
        mask = mask_ct[None, None, :] <= max_pos_num[:, :, None]
        if self.use_reverse:
            forward, gates, reverse = torch.chunk(outputs, 3, -1)
            gates = torch.sigmoid(gates)

            ind_adjusts = max_pos_num[:, :, None] - mask_ct[None, None, :]
            ind_adjusts = ind_adjusts % self.max_atom_out
            reverse = torch.gather(reverse, dim=-1, index=ind_adjusts.long())
            output = reverse * (1 - gates) + gates * forward
        else:
            output = outputs

        # Mask final output and return logits
        output = (~mask * -1e12) + output
        return output, out_hidden

    def make_prediction(self, formula_tensors, mol_graphs, max_nodes=500, adducts=None):
        """make_prediction."""

        device = formula_tensors.device
        with torch.no_grad():
            full_formula = formula_tensors

            nonzero_inds = [torch.where(i > 0)[0] for i in full_formula]
            atom_inds = [torch.tile(i, (max_nodes,)) for i in nonzero_inds]
            lens = [len(i) for i in atom_inds]

            # pad and pack
            atom_inds = torch.nn.utils.rnn.pad_sequence(
                atom_inds, batch_first=True, padding_value=0
            )
            prev_tokens = torch.zeros_like(atom_inds[:, 0])[:, None]

            # Now generate...
            cur_hidden = self.get_hidden(
                graphs=mol_graphs, full_formula=full_formula, adducts=adducts
            )
            output_list = []

            for seq_pos in range(atom_inds.shape[1]):
                atom_ind = atom_inds[:, seq_pos : seq_pos + 1]
                new_out, cur_hidden = self.forward(
                    graphs=mol_graphs,
                    full_formula=full_formula,
                    atom_inds=atom_ind,
                    adducts=adducts,
                    hidden_state=cur_hidden,
                    atom_ct_inputs=prev_tokens,
                )
                new_tokens = torch.argmax(new_out, -1)
                prev_tokens = new_tokens
                output_list.append(new_tokens.detach().cpu().numpy())

        inten_outputs, formula_outputs = [], []
        all_outs = np.array(output_list).squeeze(-1).transpose(1, 0)
        for output_list, ex_len, ex_atom_inds, ex_nonzero in zip(
            all_outs, lens, atom_inds, nonzero_inds
        ):
            # Counts vec
            output_list = output_list[:ex_len]

            # Which positions
            ex_atom_inds = ex_atom_inds[:ex_len].cpu().numpy()
            ex_nonzero = ex_nonzero.cpu().numpy()
            num_pos = len(ex_nonzero)

            # Index into which new formula
            new_form_inds = np.arange(max_nodes).repeat(len(ex_nonzero))

            # Place to store these
            new_form_vecs = np.zeros((max_nodes, self.formula_dim))

            new_form_vecs[new_form_inds, ex_atom_inds] = output_list

            keep_formulas = [
                common.vec_to_formula(ex_option) for ex_option in new_form_vecs
            ]

            # Arbitrarily assign descending intensities
            keep_intens = np.linspace(1, 0.1, len(keep_formulas))

            inten_outputs.append(keep_intens)
            formula_outputs.append(keep_formulas)

        return formula_outputs, inten_outputs

    def _common_step(self, batch, name="train"):
        targets = batch["targ_vectors"]
        zero_tokens = torch.zeros_like(targets[:, 0])
        targ_lens = batch["targ_lens"]

        # Add start token and offset by 1
        atom_ct_inputs = torch.cat([zero_tokens[:, None], targets], -1)[:, :-1]

        outputs, _ = self.forward(
            batch["graphs"],
            batch["formula_tensors"].float(),
            batch["atom_inds"].float(),
            adducts=batch["adducts"],
            atom_ct_inputs=atom_ct_inputs,
        )

        loss_output = self.loss_fn(outputs.transpose(1, 2), targets.long())

        # Zero out loss for invalid targets
        ar_vals = torch.arange(loss_output.shape[1], device=loss_output.device)
        mask = ar_vals[None, :] >= targ_lens[:, None]
        loss_output.masked_fill_(mask, 0)
        loss_ag_inds = loss_output.sum(-1) / targ_lens.float()
        loss_ag_batch = loss_ag_inds.mean()

        loss_dict = {"loss": loss_ag_batch}

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
