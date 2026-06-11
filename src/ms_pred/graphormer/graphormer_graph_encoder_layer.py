# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
import torch.nn as nn

from .multihead_attention import MultiheadAttention

class GraphormerGraphEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        init_fn: Callable = None,
        pre_layernorm: bool = False,
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.pre_layernorm = pre_layernorm

        self.dropout = nn.Dropout(dropout)
        self.activation_dropout = nn.Dropout(activation_dropout)

        # Initialize blocks
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, ffn_embedding_dim),
            get_activation_fn(activation_fn),
            nn.Dropout(activation_dropout),
            nn.Linear(ffn_embedding_dim, self.embedding_dim),
            nn.Dropout(dropout),
        )
        
        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        # x: T x B x C
        residual = x
        if self.pre_layernorm:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query=x,
            attn_bias=self_attn_bias,
        )
        x = self.dropout(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.pre_layernorm:
            x = self.final_layer_norm(x)
        x = residual + self.mlp(x)
        if not self.pre_layernorm:
            x = self.final_layer_norm(x)
        return x

def get_activation_fn(activation):
    """
    Returns the activation function given its name as a string.
    """
    if activation is None:
        return None
    activation = activation.lower()
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "silu" or activation == "swish":
        return nn.SiLU()
    else:
        raise RuntimeError(f"Unknown activation function: {activation}")