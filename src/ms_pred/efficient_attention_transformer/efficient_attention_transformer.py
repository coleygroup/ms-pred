import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def get_activation_fn(activation):
    if isinstance(activation, str):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    elif callable(activation):
        return activation
    else:
        raise ValueError("activation must be a string or callable")

def combine_masks(attn_mask, key_padding_mask, q_len, k_len, device):
    """
    Convert attn_mask and key_padding_mask into a single mask
    suitable for scaled_dot_product_attention.
    """
    final_mask = None

    # Expand key_padding_mask (B, K) -> (B, 1, 1, K) for broadcasting
    if key_padding_mask is not None:
        padding_mask = key_padding_mask[:, None, None, :].to(torch.bool)  # (B,1,1,K)
        final_mask = padding_mask

    # attn_mask (Q,K) -> (1,1,Q,K)
    if attn_mask is not None:
        attn_mask = attn_mask.to(torch.bool)
        final_mask = attn_mask if final_mask is None else final_mask | attn_mask
    return final_mask

class EfficientAttention(nn.Module):
    def __init__(self, dim, nhead, dropout=0.0):
        super().__init__()
        self.nhead = nhead
        self.head_dim = dim // nhead
        assert self.head_dim * nhead == dim

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv=None, attn_mask=None, key_padding_mask=None, is_causal=False):
        """
        x : (B, N, C) queries
        kv: (B, M, C) keys/values (if None, use x)
        """
        B, N, C = x.shape
        kv = x if kv is None else kv
        M = kv.shape[1]

        # Project
        qkv_x = self.qkv(x).reshape(B, N, 3, self.nhead, self.head_dim)
        q = qkv_x[:, :, 0].transpose(1, 2)  # (B,H,N,D)

        qkv_kv = self.qkv(kv).reshape(B, M, 3, self.nhead, self.head_dim)
        k = qkv_kv[:, :, 1].transpose(1, 2)  # (B,H,M,D)
        v = qkv_kv[:, :, 2].transpose(1, 2)

        # Combine masks
        final_mask = ~combine_masks(attn_mask, key_padding_mask, N, M, x.device)


        # EfficientAttention
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=final_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal
            )

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

class EfficientAttentionTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, nhead, dropout)
        self.norm2 = nn.LayerNorm(dim)

        # FFN (matches PyTorch exactly)
        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # Self-attention
        x = x + self.attn(
            self.norm1(x),
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            is_causal=False
        )

        # FFN
        x_residual = self.norm2(x)
        x_residual = self.linear1(x_residual)
        x_residual = self.activation(x_residual)
        x_residual = self.dropout(x_residual)
        x_residual = self.linear2(x_residual)
        x_residual = self.dropout2(x_residual)
        x = x + x_residual

        return x


class EfficientAttentionTransformerEncoder(nn.Module):
    def __init__(self, num_layers, dim, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.layers = nn.ModuleList([
            EfficientAttentionTransformerEncoderLayer(dim, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ])

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask, src_key_padding_mask)
        return x

class EfficientAttentionTransformerDecoderLayer(nn.Module):
    def __init__(self, dim, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = EfficientAttention(dim, nhead, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = EfficientAttention(dim, nhead, dropout)
        self.norm3 = nn.LayerNorm(dim)

        # FFN (matches PyTorch exactly)
        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    def forward(self, x, memory,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        # Self-attention
        x = x + self.self_attn(
            self.norm1(x),
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            is_causal=True
        )

        # Cross-attention
        x = x + self.cross_attn(
            self.norm2(x),
            kv=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            is_causal=False
        )

        # FFN
        x_residual = self.norm3(x)
        x_residual = self.linear1(x_residual)
        x_residual = self.activation(x_residual)
        x_residual = self.dropout(x_residual)
        x_residual = self.linear2(x_residual)
        x_residual = self.dropout2(x_residual)
        x = x + x_residual

        return x


class EfficientAttentionTransformerDecoder(nn.Module):
    def __init__(self, num_layers, dim, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.layers = nn.ModuleList([
            EfficientAttentionTransformerDecoderLayer(dim, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ])

    def forward(self, tgt, memory,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory,
                        tgt_mask, memory_mask,
                      tgt_key_padding_mask, memory_key_padding_mask)
        return tgt