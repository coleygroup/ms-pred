# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Replacement implementations for fairseq dependencies to avoid numpy compatibility issues.
These are simplified versions that provide the same interface as the original fairseq modules.
"""

import torch
from torch import nn


class FairseqDropout(nn.Module):
    """Replacement for fairseq.modules.FairseqDropout"""
    
    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.dropout = nn.Dropout(p)
    
    def forward(self, x):
        return self.dropout(x)


def quant_noise(module, p, block_size):
    """Replacement for fairseq.modules.quant_noise.quant_noise
    
    Simplified quantization noise - just returns the module unchanged.
    In a full implementation, this would add quantization noise during training.
    """
    return module


class utils:
    """Replacement for fairseq.utils"""
    
    @staticmethod
    def softmax(x, dim=-1, onnx_trace=False):
        """Replacement for fairseq.utils.softmax"""
        return torch.softmax(x, dim=dim)
    
    @staticmethod
    def item(x):
        """Replacement for fairseq.utils.item"""
        return x.item() if hasattr(x, 'item') else x

    @staticmethod
    def get_activation_fn(activation):
        """Replacement for fairseq.utils.get_activation_fn
        Returns the activation function given its name as a string.
        """
        if activation is None:
            return None
        activation = activation.lower()
        if activation == "relu":
            return torch.relu
        elif activation == "gelu":
            return torch.nn.functional.gelu
        elif activation == "tanh":
            return torch.tanh
        elif activation == "sigmoid":
            return torch.sigmoid
        elif activation == "silu" or activation == "swish":
            return torch.nn.functional.silu
        else:
            raise RuntimeError(f"Unknown activation function: {activation}")


class LayerNorm(nn.Module):
    """Replacement for fairseq.modules.LayerNorm"""
    
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.export = export
    
    def forward(self, x):
        return self.layer_norm(x)


def apply_quant_noise_(module, p, block_size):
    """Replacement for fairseq.modules.quant_noise.apply_quant_noise_
    
    Simplified version that just returns the module unchanged.
    """
    return module


class LayerDropModuleList(nn.ModuleList):
    """Replacement for fairseq.modules.LayerDropModuleList
    
    Simplified version that supports layer dropping during training.
    """
    
    def __init__(self, p=0.0, modules=None):
        super().__init__(modules)
        self.p = p
    
    def __getitem__(self, idx):
        if self.training and torch.rand(1).item() < self.p:
            # Skip this layer during training with probability p
            return lambda x, *args, **kwargs: (x, None)  # Return tuple to match expected output
        return super().__getitem__(idx)
