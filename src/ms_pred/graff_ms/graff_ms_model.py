""" GRAFF MS model. """
import torch
import pytorch_lightning as pl
import numpy as np

import torch.nn as nn
import dgl.nn as dgl_nn

import torch_scatter as ts
import ms_pred.nn_utils as nn_utils
import ms_pred.common as common


class GraffGNN(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int,
        layers: int = 2,
        dropout: float = 0.0,
        learning_rate: float = 7e-4,
        lr_decay_rate: float = 1.0,
        output_dim: int = 1000,
        upper_limit: int = 1500,
        weight_decay: float = 0,
        loss_fn: str = "mse",
        mpnn_type: str = "GGNN",
        set_layers: int = 2,
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
        embed_adduct: bool = False,
        embed_collision_energy: bool = False,
        warmup: int = 1000,
        num_fixed_forms: int = 10000,
        **kwargs,
    ):
        """__init__ _summary_

        Args:
            hidden_size (int): _description_
            layers (int, optional): _description_. Defaults to 2.
            dropout (float, optional): _description_. Defaults to 0.0.
            learning_rate (float, optional): _description_. Defaults to 7e-4.
            lr_decay_rate (float, optional): _description_. Defaults to 1.0.
            output_dim (int, optional): _description_. Defaults to 1000.
            upper_limit (int, optional): _description_. Defaults to 1500.
            weight_decay (float, optional): _description_. Defaults to 0.
            loss_fn (str, optional): _description_. Defaults to "mse".
            mpnn_type (str, optional): _description_. Defaults to "GGNN".
            set_layers (int, optional): _description_. Defaults to 2.
            atom_feats (list, optional): _description_. Defaults to ( "a_onehot", "a_degree", "a_hybrid", "a_formal", "a_radical", "a_ring", "a_mass", "a_chiral", ).
            bond_feats (list, optional): _description_. Defaults to ("b_degree",).
            pool_op (str, optional): _description_. Defaults to "avg".
            pe_embed_k (int, optional): _description_. Defaults to 0.
            num_atom_feats (int, optional): _description_. Defaults to 86.
            num_bond_feats (int, optional): _description_. Defaults to 5.
            embed_adduct (bool, optional): _description_. Defaults to False.
            embed_collision_energy (bool, optional): _description_. Defaults to False.
            warmup (int, optional): _description_. Defaults to 1000.
            num_fixed_forms (int, optional): _description_. Defaults to 10000.

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_
        """

        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size

        self.atom_feats = atom_feats
        self.bond_feats = bond_feats
        self.num_atom_feats = num_atom_feats
        self.num_bond_feats = num_bond_feats
        self.pe_embed_k = pe_embed_k
        self.pool_op = pool_op
        self.warmup = warmup
        self.num_fixed_forms = num_fixed_forms

        self.layers = layers
        self.set_layers = set_layers
        self.mpnn_type = mpnn_type
        self.output_dim = output_dim
        self.upper_limit = upper_limit
        self.weight_decay = weight_decay
        self.embed_adduct = embed_adduct
        self.embed_collision_energy = embed_collision_energy

        adduct_shift = 0
        if self.embed_adduct:
            adduct_types = len(common.ion2onehot_pos)
            onehot = torch.eye(adduct_types)
            self.adduct_embedder = nn.Parameter(onehot.float())
            self.adduct_embedder.requires_grad = False
            adduct_shift = adduct_types

        ce_shift = 0
        if self.embed_collision_energy:
            ce_shift = 1

        # Get bin masses
        buckets = torch.DoubleTensor(np.linspace(0, 1500, 15000))
        self.inten_buckets = nn.Parameter(buckets)
        self.inten_buckets.requires_grad = False

        # Define output formulae and bins
        self.fixed_forms = torch.zeros(self.num_fixed_forms, common.CHEM_ELEMENT_NUM)
        self.fixed_forms = nn.Parameter(self.fixed_forms)
        self.fixed_forms.requires_grad = False

        self.is_loss = torch.zeros(self.num_fixed_forms)
        self.is_loss = nn.Parameter(self.is_loss)
        self.is_loss.requires_grad = False

        self.weight_mult = torch.from_numpy(common.VALID_MONO_MASSES).float()
        self.weight_mult = nn.Parameter(self.weight_mult)
        self.weight_mult.requires_grad = False

        self.dropout = dropout
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate

        # Define network
        self.gnn = nn_utils.MoleculeGNN(
            hidden_size=self.hidden_size,
            num_step_message_passing=self.layers,
            set_transform_layers=self.set_layers,
            mpnn_type=self.mpnn_type,
            gnn_node_feats=num_atom_feats + adduct_shift + ce_shift,
            gnn_edge_feats=num_bond_feats,
            dropout=dropout,
        )
        if self.pool_op == "avg":
            self.pool = dgl_nn.AvgPooling()
        elif self.pool_op == "attn":
            self.pool = dgl_nn.GlobalAttentionPooling(nn.Linear(hidden_size, 1))
        else:
            raise NotImplementedError()

        self.loss_fn_name = loss_fn
        if loss_fn == "mse":
            self.loss_fn = self.mse_loss
            self.num_outputs = 1
            self.output_activations = [nn.ReLU()]
        elif loss_fn == "cosine":
            self.loss_fn = self.cos_loss
            self.cos_fn = nn.CosineSimilarity()
            self.num_outputs = 1
            self.output_activations = [nn.Sigmoid()]
        elif loss_fn == 'entropy':
            self.loss_fn = self.entropy_loss
            self.num_outputs = 1
            self.output_activations = [nn.Sigmoid()]
        else:
            raise NotImplementedError()

        # Gates, reverse, forward
        self.output_layer = nn.Linear(self.hidden_size, self.num_fixed_forms)

        self.attn_layer = nn.Linear(self.hidden_size, self.num_fixed_forms)

    def set_fixed_forms(self, new_fixed):
        new_val = torch.from_numpy(new_fixed)
        assert self.fixed_forms.data.shape == new_val.shape
        self.fixed_forms.data = new_val.float()
        bool_1 = torch.any(new_val < 0, -1)

        # If we want to include the ability to predict the precursor
        bool_2 = torch.all(new_val <= 0, -1)
        self.is_loss.data = torch.logical_or(bool_1, bool_2).float()

        self.is_loss.requires_grad = False
        self.fixed_forms.data.requires_grad = False

    def cos_loss(self, pred, targ):
        """loss_fn."""
        pred = pred[:, 0, :]
        loss = 1 - self.cos_fn(pred, targ)
        loss = loss.mean()
        return {"loss": loss}

    def mse_loss(self, pred, targ):
        """loss_fn."""
        # Select intens
        pred = pred[:, 0, :]
        mse = torch.sum((pred - targ) ** 2, -1)
        mse = mse.mean()
        return {"loss": mse}

    def entropy_loss(self, pred, targ):
        """entropy_loss.

        Args:
            pred:
            targ:
        """
        pred = pred[:, 0, :]

        def norm_peaks(prob):
            return prob / (prob.sum(dim=-1, keepdim=True) + 1e-22)
        def entropy(prob):
            assert torch.all(torch.abs(prob.sum(dim=-1) - 1) < 1e-3), prob.sum(dim=-1)
            return -torch.sum(prob * torch.log(prob + 1e-22), dim=-1) / 1.3862943611198906 # norm by log(4)

        pred_norm = norm_peaks(pred)
        targ_norm = norm_peaks(targ)
        entropy_pred = entropy(pred_norm)
        entropy_targ = entropy(targ_norm)
        entropy_mix = entropy((pred_norm + targ_norm) / 2)
        loss = 2 * entropy_mix - entropy_pred - entropy_targ
        loss = loss.mean()
        return {"loss": loss}

    def predict(self, graphs, full_forms, adducts=None, nce=None) -> dict:
        """predict."""
        out = self.forward(graphs, full_forms, adducts, nce)
        if self.loss_fn_name in ["mse", "cosine"]:
            out_dict = {"spec": out[:, 0, :]}
        else:
            raise NotImplementedError()
        return out_dict

    def forward(self, graphs, full_forms, adducts, nces):
        """predict spec"""

        if self.embed_adduct:
            embed_adducts = self.adduct_embedder[adducts.long()]
            ndata = graphs.ndata["h"]
            embed_adducts_expand = embed_adducts.repeat_interleave(
                graphs.batch_num_nodes(), 0
            )
            ndata = torch.cat([ndata, embed_adducts_expand], -1)
            graphs.ndata["h"] = ndata

        if self.embed_collision_energy:
            ndata = graphs.ndata["h"]
            ndata = torch.cat([ndata, nces.repeat_interleave(graphs.batch_num_nodes(), 0).unsqueeze(-1)], -1)
            graphs.ndata["h"] = ndata

        output = self.gnn(graphs)
        hidden = self.pool(graphs, output)
        batch_size = output.shape[0]

        # Predict intens at all formulae
        # output = torch.sigmoid(self.output_layer(output))

        output = self.output_layer(hidden)
        attn_weights = self.attn_layer(hidden)

        # Determine which formulae are valid
        device = output.device
        batch_size, form_dim = full_forms.shape

        is_loss_bool = self.is_loss.bool()
        out_frags = torch.zeros(
            self.num_fixed_forms, batch_size, form_dim, device=device
        )
        out_frags[is_loss_bool] = (
            full_forms[None, :, :] + self.fixed_forms[is_loss_bool, None, :]
        )
        out_frags[~is_loss_bool] = self.fixed_forms[~is_loss_bool, None, :].repeat(
            1, batch_size, 1
        )

        # Check that resulting frags are > 0 and < root
        lb = torch.all(out_frags >= 0, -1)
        ub = torch.all((full_forms[None, :, :] - out_frags) >= 0, -1)
        is_valid = torch.logical_and(lb, ub)

        # Num frags x batch ... => batch x num frags ...
        is_valid = is_valid.transpose(0, 1)
        out_frags = out_frags.transpose(0, 1)

        # Get the mass at each new fragment and identify its binned bucket
        resulting_mass = torch.einsum("ijk, k -> ij", out_frags, self.weight_mult)
        inverse_indices = torch.bucketize(
            resulting_mass, self.inten_buckets, right=False
        )

        # Compute scatter max into each bucket
        # Mask invalid
        output = torch.sigmoid(self.output_layer(hidden))
        output = output * is_valid

        # Scatter max the result into each bucket
        # Can consider scatter mean or attentive poling as alternative
        output, ind_maxes = ts.scatter_max(
            output,
            index=inverse_indices,
            dim_size=self.inten_buckets.shape[-1],
            dim=-1,
        )
        assert self.num_outputs == 1
        output = output.reshape(batch_size, 1, -1)
        return output

    def _common_step(self, batch, name="train"):
        pred_spec = self.forward(batch["graphs"], batch["full_forms"], batch["adducts"], batch["nces"])
        loss_dict = self.loss_fn(pred_spec, batch["spectra"])
        self.log(f"{name}_loss", loss_dict.get("loss"))
        if name == 'test':
            loss_dict.update({
                'cos_loss': self.cos_loss(pred_spec, batch["spectra"])['loss'],
                'entr_loss': self.entropy_loss(pred_spec, batch["spectra"])['loss'],
            })
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
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
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
