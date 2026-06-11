""" gnn_model. """
import torch
import pytorch_lightning as pl
import numpy as np

import torch.nn as nn
import dgl.nn as dgl_nn

import ms_pred.nn_utils as nn_utils
import ms_pred.common as common


class ForwardGNN(pl.LightningModule):
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
        use_reverse: bool = True,
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
        warmup: int = 1000,
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
            use_reverse (bool, optional): _description_. Defaults to True.
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
            warmup (int, optional): _description_. Defaults to 1000.

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

        self.layers = layers
        self.set_layers = set_layers
        self.mpnn_type = mpnn_type
        self.output_dim = output_dim
        self.upper_limit = upper_limit
        self.use_reverse = use_reverse
        self.weight_decay = weight_decay
        self.embed_adduct = embed_adduct
        adduct_shift = 0
        if self.embed_adduct:
            adduct_types = len(common.ion2onehot_pos)
            onehot = torch.eye(adduct_types)
            self.adduct_embedder = nn.Parameter(onehot.float())
            self.adduct_embedder.requires_grad = False
            adduct_shift = adduct_types

        # Get bin masses
        self.bin_masses = torch.from_numpy(np.linspace(0, upper_limit, output_dim))
        self.bin_masses = nn.Parameter(self.bin_masses)
        self.bin_masses.requires_grad = False

        self.dropout = dropout
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate

        # Define network
        self.gnn = nn_utils.MoleculeGNN(
            hidden_size=self.hidden_size,
            num_step_message_passing=self.layers,
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
        else:
            raise NotImplementedError()

        # Gates, reverse, forward
        reverse_mult = 3 if self.use_reverse else 1
        self.output_layer = nn.Linear(
            self.hidden_size, self.num_outputs * self.output_dim * reverse_mult
        )

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

    def predict(self, graphs, full_weight=None, adducts=None) -> dict:
        """predict."""
        out = self.forward(graphs, full_weight, adducts)
        if self.loss_fn_name in ["mse", "cosine"]:
            out_dict = {"spec": out[:, 0, :]}
        else:
            raise NotImplementedError()
        return out_dict

    def forward(self, graphs, full_weight, adducts):
        """predict spec"""

        if self.embed_adduct:
            embed_adducts = self.adduct_embedder[adducts.long()]
            ndata = graphs.ndata["h"]
            embed_adducts_expand = embed_adducts.repeat_interleave(
                graphs.batch_num_nodes(), 0
            )
            ndata = torch.cat([ndata, embed_adducts_expand], -1)
            graphs.ndata["h"] = ndata

        output = self.gnn(graphs)
        output = self.pool(graphs, output)
        batch_size = output.shape[0]

        # Convert full weight into bin index
        # Find first index at which it's true
        full_mass_bin = (full_weight[:, None] < self.bin_masses).int().argmax(-1)

        output = self.output_layer(output)
        output = output.reshape(batch_size, self.num_outputs, -1)

        # Get indices where it makes sense to predict a mass
        full_arange = torch.arange(self.output_dim, device=output.device)
        is_valid = full_arange[None, :] <= full_mass_bin[:, None]
        if self.use_reverse:
            forward_preds, rev_preds, gates = torch.chunk(output, 3, -1)
            gates_temp = torch.sigmoid(gates)
            ind_adjusts = full_mass_bin[:, None] - full_arange[None, :]
            ind_adjusts = ind_adjusts[:, None].expand(rev_preds.shape)
            ind_adjusts = ind_adjusts % self.output_dim
            rev_preds_temp = torch.gather(rev_preds, dim=-1, index=ind_adjusts.long())
            output = rev_preds_temp * (1 - gates_temp) + gates_temp * forward_preds

        # Activate each dim with its respective output activation
        # Helpful for hurdle or probabilistic models
        new_outputs = []
        for output_ind, act in enumerate(self.output_activations):
            new_outputs.append(act(output[:, output_ind : output_ind + 1, :]))
        new_out = torch.cat(new_outputs, 1)

        # Mask everything
        output_mask = new_out * is_valid[:, None, :]
        return output_mask

    def _common_step(self, batch, name="train"):
        pred_spec = self.forward(
            batch["graphs"], batch["full_weight"], batch["adducts"]
        )
        loss_dict = self.loss_fn(pred_spec, batch["spectra"])
        self.log(f"{name}_loss", loss_dict.get("loss"))
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

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric=None):
        scheduler.step()
