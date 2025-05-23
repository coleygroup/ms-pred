import torch
import pytorch_lightning as pl
import numpy as np

import torch.nn as nn

import ms_pred.nn_utils as nn_utils
import ms_pred.common as common

from .massformer_code import gf_model
from .massformer_code import model_extract


class MassFormer(pl.LightningModule):
    """
    Implementation of Massformer.

    Note only copy the embedder/immediate ffn layers of Massformer.
    The other parts, e.g., spectral attention, l1 normalization etc have not been
    copied to ensure consistency with other models.
    """

    def __init__(
        self,
        # MF FFN Args:
        mf_num_ff_num_layers,
        mf_ff_h_dim,
        mf_ff_skip,
        mf_layer_type,
        mf_dropout,
        # MF graphformer embedder arguments
        gf_model_name,
        gf_pretrain_name,
        gf_fix_num_pt_layers,
        gf_reinit_num_pt_layers,
        gf_reinit_layernorm,
        learning_rate: float = 7e-4,
        lr_decay_rate: float = 1.0,
        output_dim: int = 1000,
        upper_limit: int = 1500,
        weight_decay: float = 0,
        use_reverse: bool = True,
        loss_fn: str = "mse",
        warmup: int = 1000,
        embed_adduct: bool = True,
        embed_collision_energy: bool = True,
        **kwargs,
    ):
        """ """
        super().__init__()
        self.save_hyperparameters()

        # regularization
        self.mf_dropout = mf_dropout
        self.weight_decay = weight_decay

        self.use_reverse = use_reverse
        self.warmup = warmup
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

        # Embedder
        self.graphormer_embedder = gf_model.GFv2Embedder(
            gf_model_name=gf_model_name,
            gf_pretrain_name=gf_pretrain_name,
            fix_num_pt_layers=gf_fix_num_pt_layers,
            reinit_num_pt_layers=gf_reinit_num_pt_layers,
            reinit_layernorm=gf_reinit_layernorm,
        )

        # MLP on top
        self.mf_ff_skip = mf_ff_skip
        embed_dim = self.graphormer_embedder.get_embed_dim()
        self.ff_layers = nn.ModuleList([])
        if mf_layer_type == "standard":
            ff_layer = model_extract.LinearBlock
        else:
            assert mf_layer_type == "neims", mf_layer_type
            ff_layer = model_extract.NeimsBlock
        self.ff_layers.append(nn.Linear(embed_dim + adduct_shift + ce_shift, mf_ff_h_dim))
        for i in range(mf_num_ff_num_layers):
            self.ff_layers.append(ff_layer(mf_ff_h_dim, mf_ff_h_dim, self.mf_dropout))

        # Get bin masses
        self.upper_limit = upper_limit
        self.bin_masses = torch.from_numpy(np.linspace(0, upper_limit, output_dim))
        self.bin_masses = nn.Parameter(self.bin_masses)
        self.bin_masses.requires_grad = False

        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate

        # Losses and stuff
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
        self.output_dim = output_dim
        reverse_mult = 3 if self.use_reverse else 1
        self.output_layer = nn.Linear(
            mf_ff_h_dim, self.num_outputs * self.output_dim * reverse_mult
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

    def predict(self, graphs, full_weight=None, adducts=None, norm_collision_energy=None) -> dict:
        """predict."""
        out = self.forward(graphs, full_weight, adducts, norm_collision_energy)
        if self.loss_fn_name in ["mse", "cosine"]:
            out_dict = {"spec": out[:, 0, :]}
        else:
            raise NotImplementedError()
        return out_dict

    def forward(self, gf_v2_data, full_weight, adducts, norm_collision_energy):
        """predict spec"""
        # Graphormer
        embeds = self.graphormer_embedder({"gf_v2_data": gf_v2_data})

        if self.embed_adduct:
            embed_adducts = self.adduct_embedder[adducts.long()]
            new_embeds = torch.cat([embeds, embed_adducts], -1)
            embeds = new_embeds

        if self.embed_collision_energy:
            new_embeds = torch.cat([embeds, norm_collision_energy.unsqueeze(-1)], -1)
            embeds = new_embeds

        # apply feedforward layers
        fh = self.ff_layers[0](embeds)
        for ff_layer in self.ff_layers[1:]:
            if self.mf_ff_skip:
                fh = fh + ff_layer(fh)
            else:
                fh = ff_layer(fh)
        output = fh
        batch_size, *_ = fh.shape

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
            batch["gf_v2_data"], batch["full_weight"], batch["adducts"], batch["norm_collision_energies"]
        )
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
