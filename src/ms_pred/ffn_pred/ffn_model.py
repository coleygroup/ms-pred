""" ffn_model. """
import torch
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np

import ms_pred.nn_utils as nn_utils
import ms_pred.common as common


class ForwardFFN(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int,
        layers: int = 2,
        dropout: float = 0.0,
        learning_rate: float = 7e-4,
        input_dim: int = 2048,
        output_dim: int = 15000,
        upper_limit: int = 1500,
        weight_decay: float = 0,
        use_reverse: bool = True,
        loss_fn: str = "mse",
        lr_decay_rate: float = 1.0,
        embed_adduct: bool = False,
        warmup: int = 1000,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size

        self.layers = layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.upper_limit = upper_limit
        self.use_reverse = use_reverse
        self.weight_decay = weight_decay
        self.lr_decay_rate = lr_decay_rate
        self.embed_adduct = embed_adduct
        self.warmup = warmup

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

        # Define network
        self.activation = nn.ReLU()
        self.init_layer = nn.Linear(self.input_dim + adduct_shift, self.hidden_size)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        middle_layer = nn.Linear(hidden_size, hidden_size)
        self.layers = nn_utils.get_clones(middle_layer, self.layers - 1)

        self.loss_fn_name = loss_fn
        if loss_fn == "mse":
            self.loss_fn = self.mse_loss
            self.output_activations = [nn.ReLU()]
        elif loss_fn == "cosine":
            self.loss_fn = self.cos_loss
            self.cos_fn = nn.CosineSimilarity()
            self.output_activations = [nn.Sigmoid()]
        else:
            raise NotImplementedError()

        self.num_outputs = len(self.output_activations)

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

    def predict(self, fps, full_weight, adducts) -> dict:
        """predict."""
        out = self.forward(fps, full_weight, adducts)
        if self.loss_fn_name in ["mse", "cosine", "cosine_weight", "cosine_div"]:
            out_dict = {"spec": out[:, 0, :]}
        else:
            raise NotImplementedError()
        return out_dict

    def forward(self, fps, full_weight, adducts):
        """predict spec"""

        fps = fps.float()
        if self.embed_adduct:
            embed_adducts = self.adduct_embedder[adducts.long()]
            fps = torch.cat([fps, embed_adducts], -1)

        output = self.init_layer(fps)
        output = self.activation(output)
        output = self.dropout_layer(output)

        # Convert full weight into bin index
        # Find first index at which it's true
        full_mass_bin = (full_weight[:, None] < self.bin_masses).int().argmax(-1)

        for layer_index, layer in enumerate(self.layers):
            output = layer(output)
            output = self.dropout_layer(output)
            output = self.activation(output)

        hidden = output
        output = self.output_layer(hidden)
        batch_size = fps.shape[0]
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
        pred_spec = self.forward(batch["fps"], batch["full_weight"], batch["adducts"])
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
