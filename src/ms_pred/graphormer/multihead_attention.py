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
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        attn_bias: Optional[Tensor],
    ):

        tgt_len, bsz, embed_dim = query.size()

        qkv = self.qkv_proj(query)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape to (B, H, L, D)
        q = q.view(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(-1, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.view(-1, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        # attn_bias: (B, H, L, S)
        attn_bias = attn_bias.view(bsz, self.num_heads, tgt_len, -1)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.dropout,
            is_causal=False
        )

        # back to (L, B, E)
        attn = attn.permute(2, 0, 1, 3).reshape(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        return attn