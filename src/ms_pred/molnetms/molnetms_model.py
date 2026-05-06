""" Molnet model. """
from typing import Tuple
import pytorch_lightning as pl

import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

import ms_pred.nn_utils as nn_utils
import ms_pred.common as common
import ms_pred.molnetms.molnetms_data as molnetms_data


class FCResBlock(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, dropout: float = 0.0
    ) -> torch.Tensor:
        super(FCResBlock, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        # Make uniform hidden size
        # hid_dim = int(in_dim / 4)
        self.linear1 = nn.Linear(input_size, hidden_size, bias=False)
        self.bn1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bn2 = nn.LayerNorm(hidden_size)

        self.linear3 = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.bn3 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.LayerNorm(hidden_size)

        self.dp = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
                )

            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x

        x = self.bn1(self.linear1(x))
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn2(self.linear2(x))
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn3(self.linear3(x))

        x = x + F.interpolate(identity.unsqueeze(1), size=x.size()[1]).squeeze()

        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dp(x)
        return x

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.input_size)
            + " -> "
            + str(self.hidden_size)
            + ")"
        )


class MSDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, layers, out_dim, dropout):
        """_summary_

        Args:
            input_size (_type_): _description_
            hidden_size (_type_): _description_
            layers (_type_): _description_
            out_dim (_type_): _description_
            dropout (_type_): _description_
        """
        super(MSDecoder, self).__init__()

        last_size = input_size
        self.blocks = nn.ModuleList([])
        for i in range(layers):
            if i == 0:
                self.blocks.append(
                    FCResBlock(
                        hidden_size=hidden_size, input_size=input_size, dropout=dropout
                    )
                )
            else:
                self.blocks.append(
                    FCResBlock(
                        hidden_size=hidden_size, input_size=hidden_size, dropout=dropout
                    )
                )
            last_size = hidden_size

        self.fc = nn.Linear(last_size, out_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_normal_(
            self.fc.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.fc(x)


class MolConv(nn.Module):
    """MolConv.

    Taken from the 3DMolMS repository

    https://github.com/JosieHong/3DMolMS/blob/main/models/molnet.py

    """

    def __init__(self, in_dim, out_dim, k, remove_xyz=False):
        super(MolConv, self).__init__()
        self.k = k
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.remove_xyz = remove_xyz

        self.dist_ff = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid()
        )
        self.gm2m_ff = nn.Sequential(
            nn.Conv2d(k, 1, kernel_size=1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid()
        )

        if remove_xyz:
            self.update_ff = nn.Sequential(
                nn.Conv2d(in_dim - 3, out_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(negative_slope=0.02),
            )
        else:
            self.update_ff = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(negative_slope=0.02),
            )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
                )

            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, idx_base: torch.Tensor) -> torch.Tensor:

        # dist: torch.Size([batch_size, 1, point_num, k])
        # gm2: torch.Size([batch_size, k, point_num, k])
        # feat_n: torch.Size([batch_size, in_dim, point_num, k])
        # feat_c: torch.Size([batch_size, in_dim, point_num, k])
        dist, gm2, feat_c, feat_n = self._generate_feat(
            x, idx_base, k=self.k, remove_xyz=self.remove_xyz
        )

        w1 = self.dist_ff(dist)
        w2 = self.gm2m_ff(gm2)

        feat = torch.mul(w1, w2) * feat_n + (1 - torch.mul(w1, w2)) * feat_c
        feat = self.update_ff(feat)
        feat = feat.mean(dim=-1, keepdim=False)
        return feat

    def _generate_feat(
        self, x: torch.Tensor, idx_base: torch.Tensor, k: int, remove_xyz: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_dims, num_points = x.size()
        # local graph (knn)
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        dist, idx = pairwise_distance.topk(k=k, dim=2)  # (batch_size, num_points, k)
        dist = -dist

        idx = idx + idx_base
        idx = idx.view(-1)

        # (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
        x = x.transpose(2, 1).contiguous()
        graph_feat = x.view(batch_size * num_points, -1)[idx, :]
        graph_feat = graph_feat.view(batch_size, num_points, k, num_dims)

        # gram matrix
        gm_matrix = torch.matmul(graph_feat, graph_feat.permute(0, 1, 3, 2))

        # double gram matrix
        sub_feat = gm_matrix[:, :, :, 0].unsqueeze(3)
        sub_gm_matrix = torch.matmul(sub_feat, sub_feat.permute(0, 1, 3, 2))
        sub_gm_matrix = F.normalize(sub_gm_matrix, dim=1)

        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        if remove_xyz:
            return (
                dist.unsqueeze(3).permute(0, 3, 1, 2).contiguous(),
                sub_gm_matrix.permute(0, 3, 1, 2).contiguous(),
                x[:, :, :, 3:].permute(0, 3, 1, 2).contiguous(),
                graph_feat[:, :, :, 3:].permute(0, 3, 1, 2).contiguous(),
            )
        else:
            return (
                dist.unsqueeze(3).permute(0, 3, 1, 2).contiguous(),
                sub_gm_matrix.permute(0, 3, 1, 2).contiguous(),
                x.permute(0, 3, 1, 2).contiguous(),
                graph_feat.permute(0, 3, 1, 2).contiguous(),
            )

    def __repr__(self):
        return (
            self.__class__.__name__
            + " k = "
            + str(self.k)
            + " ("
            + str(self.in_dim)
            + " -> "
            + str(self.out_dim)
            + ")"
        )


class Encoder(nn.Module):
    def __init__(self, in_dim, layers, hidden_size, k):
        """Encoder for graph embedding
        Args:
            in_dim (int): input dimension
            layers (list): list of hidden layer dimensions
            hidden_size (int): embedding dimension
            k (int): number of nearest neighbors
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = nn.ModuleList(
            [MolConv(in_dim=in_dim, out_dim=hidden_size, k=k, remove_xyz=True)]
        )
        for i in range(layers - 1):
            self.hidden_layers.append(
                MolConv(in_dim=hidden_size, out_dim=hidden_size, k=k, remove_xyz=False)
            )

        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size * layers, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.merge = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.merge:
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
                )

            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, idx_base: torch.Tensor) -> torch.Tensor:
        """
        x:      set of points, torch.Size([32, 21, 300])
        """
        xs, tmp_x = [], x
        for i, hidden_layer in enumerate(self.hidden_layers):
            tmp_x = hidden_layer(tmp_x, idx_base)
            xs.append(tmp_x)

        x = torch.cat(xs, dim=1)
        x = self.conv(x)
        p1 = F.adaptive_max_pool1d(x, 1).squeeze().view(-1, self.hidden_size)
        p2 = F.adaptive_avg_pool1d(x, 1).squeeze().view(-1, self.hidden_size)

        x = torch.cat((p1, p2), 1)
        x = self.merge(x)
        return x


class MolNetMS(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int,
        top_layers: int,
        neighbors: int,
        layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 7e-4,
        lr_decay_rate: float = 1.0,
        output_dim: int = 1000,
        upper_limit: int = 1500,
        weight_decay: float = 0,
        use_reverse: bool = False,
        loss_fn: str = "mse",
        embed_adduct: bool = False,
        warmup: int = 1000,
        **kwargs,
    ):
        """_summary_

        Args:
            hidden_size (int): _description_
            top_layers (int): _description_
            neighbors (int): _description_
            layers (int, optional): _description_. Defaults to 2.
            dropout (float, optional): _description_. Defaults to 0.1.
            learning_rate (float, optional): _description_. Defaults to 7e-4.
            lr_decay_rate (float, optional): _description_. Defaults to 1.0.
            output_dim (int, optional): _description_. Defaults to 1000.
            upper_limit (int, optional): _description_. Defaults to 1500.
            weight_decay (float, optional): _description_. Defaults to 0.
            use_reverse (bool, optional): _description_. Defaults to True.
            loss_fn (str, optional): _description_. Defaults to "mse".
            embed_adduct (bool, optional): _description_. Defaults to False.
            warmup (int, optional): _description_. Defaults to 1000.

        """
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size

        self.warmup = warmup
        self.layers = layers
        self.dropout = dropout
        self.output_dim = output_dim
        self.upper_limit = upper_limit
        self.use_reverse = use_reverse
        self.weight_decay = weight_decay
        self.embed_adduct = embed_adduct
        self.neighbors = neighbors
        self.top_layers = top_layers

        self.atom_feats = molnetms_data.MolMSFeaturizer.atom_feats()

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

        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate

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
        self.encoder = Encoder(
            in_dim=self.atom_feats,
            layers=self.layers,
            hidden_size=self.hidden_size,
            k=self.neighbors,
        )
        self.decoder = MSDecoder(
            input_size=self.hidden_size + adduct_shift,
            hidden_size=self.hidden_size,
            layers=self.top_layers,
            out_dim=self.output_dim * self.num_outputs * reverse_mult,
            dropout=self.dropout,
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
        batch_size, node_dim, num_points = graphs.shape
        device = graphs.device

        # Encode
        idx_base = (
            torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        )

        # Batch x num points x hidden => batch x hidden x num poionts
        graphs = graphs.transpose(1, 2)

        output = self.encoder(graphs, idx_base)

        # output = self.gnn(graphs)
        # output = self.pool(graphs, output)
        batch_size = output.shape[0]

        # Convert full weight into bin index
        # Find first index at which it's true
        full_mass_bin = (full_weight[:, None] < self.bin_masses).int().argmax(-1)

        if self.embed_adduct:
            embed_adducts = self.adduct_embedder[adducts.long()]
            # embed_adducts_expand =embed_adducts[:, None, :].repeat_interleave(node_dim, 1)
            output_updated = torch.cat([output, embed_adducts], -1)
            output = output_updated

        # Decode
        output = self.decoder(output)
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
        graphs, masks = batch["graphs"], batch["graph_masks"]
        pred_spec = self.forward(graphs, batch["full_weight"], batch["adducts"])
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
