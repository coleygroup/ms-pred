# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel


# Import fairseq replacements
from .fairseq_replacements import FairseqDropout, quant_noise, utils


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        self_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention

        assert self.self_attention, "Only support self attention"

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        raise NotImplementedError

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        attn_bias: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
        """

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        import torch.nn.functional as F

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        # Reshape to (batch, seq_len, num_heads, head_dim) then transpose to (batch, num_heads, seq_len, head_dim)
        q = q.permute(1, 0, 2).contiguous().view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.permute(1, 0, 2).contiguous().view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.permute(1, 0, 2).contiguous().view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        # attn_mask: (tgt_len, src_len) or (bsz, num_heads, tgt_len, src_len)
        # key_padding_mask: (bsz, src_len)
        # attn_bias: (bsz * num_heads, tgt_len, src_len)
        # F.scaled_dot_product_attention expects attn_mask as (bsz, num_heads, tgt_len, src_len) or None
        fused_mask = None
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                fused_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, tgt_len, src_len)
            elif attn_mask.dim() == 4:
                fused_mask = attn_mask
        if key_padding_mask is not None:
            # Convert key_padding_mask to float mask for F.scaled_dot_product_attention
            # (bsz, src_len) -> (bsz, 1, 1, src_len)
            kpm = key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
            mask_value = float('-inf')
            if fused_mask is not None:
                fused_mask = fused_mask.clone()
                fused_mask = fused_mask.masked_fill(kpm, mask_value)
            else:
                fused_mask = torch.zeros((bsz, self.num_heads, tgt_len, src_len), device=q.device)
                fused_mask = fused_mask.masked_fill(kpm, mask_value)
        if attn_bias is not None:
            # attn_bias: (bsz * num_heads, tgt_len, src_len)
            attn_bias_reshaped = attn_bias.view(bsz, self.num_heads, tgt_len, src_len)
            if fused_mask is not None:
                fused_mask = fused_mask + attn_bias_reshaped
            else:
                fused_mask = attn_bias_reshaped

        attn_mask = fused_mask
        # Use EfficientAttention via F.scaled_dot_product_attention
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_module.p if self.training else 0.0,
                is_causal=False,
            )

        # attn_output: (bsz, num_heads, tgt_len, head_dim)
        attn = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim).permute(1, 0, 2)
        attn = self.out_proj(attn)

        return attn

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value