from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ----------------------------- Attention Blocks ----------------------------- #
class MHSA(nn.Module):
    """Multi-Head Self-Attention for E-SpecFormer.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        dropout: Dropout applied to attention weights.
        bias: Whether to use bias in linear projections.
        linattn: If True, use linear attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        linattn: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.linattn = linattn

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **kwargs)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        B, T, _ = x.size()
        qkv = self.qkv_proj(x)  # (B, T, 3*embed_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if not self.linattn:
            scaling = self.head_dim ** -0.5
            q = q * scaling
            attn_scores = torch.matmul(q, k.transpose(-2, -1))
            if attn_mask is not None:
                attn_scores += attn_mask
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, v)
        else:
            # LinAttn
            q = q / (q.abs().sum(dim=-1, keepdim=True) + 1e-6)
            k = k / (k.abs().sum(dim=-1, keepdim=True) + 1e-6)
            kv = torch.matmul(k.transpose(-2, -1), v)
            attn_output = torch.matmul(q, kv)

        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.embed_dim)
        attn_output_projected = self.out_proj(attn_output)
        return attn_output_projected, qkv, attn_output


class DTanh(nn.Module):
    """Dynamic Hypertangent Pre-Normalization layer."""

    def __init__(self, normalized_shape: int, channels_last: bool, alpha_init_value: float = 0.5) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self) -> str:
        return (
            f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, "
            f"channels_last={self.channels_last}"
        )


class LiTAN(nn.Module):
    """Linear Tanh Attention Network."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ffn: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        dtanh: bool = False,
        alpha: float = 0.5,
        linattn: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.attn = MHSA(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            bias=bias,
            linattn=linattn,
            **kwargs,
        )
        self.linear1 = nn.Linear(d_model, dim_ffn, bias=bias, **kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ffn, d_model, bias=bias, **kwargs)
        if dtanh:
            self.norm1 = DTanh(d_model, channels_last=True, alpha_init_value=alpha)
            self.norm2 = DTanh(d_model, channels_last=True, alpha_init_value=alpha)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=bias, **kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=bias, **kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _ffn(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x

    def forward(self, src: Tensor) -> Tensor:
        x = src
        x = x + self.dropout1(self.attn(self.norm1(x))[0])
        x = x + self._ffn(self.norm2(x))
        return x


class SpecEncoder(nn.Module):
    """Encoder for E-SpecFormer."""
    def __init__(self, encoder_layer: LiTAN, num_layers: int, norm=None) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm
        self.num_layers = num_layers

    def forward(self, src: Tensor) -> Tensor:
        x = src
        for layer in self.layers:
            x = layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ESpecFormer(nn.Module):
    """Edge-Efficient Spectrum Monitoring Transformer.

    Input shape: (B, 1, 2, T) where T is time length (e.g., 640 or 1024).
    """

    def __init__(
        self,
        num_classes: int = 24,
        d_model: int = 36,
        nhead: int = 2,
        num_layers: int = 2,
        dim_feedforward: int = 32,
        k: int = 16,
        s: int = 16,
        dtanh: bool = False,
        linattn: bool = False,
    ) -> None:
        super().__init__()

        # ConvTokenizer: 2D -> d_model dimensional tokens
        self.cnn = nn.Conv2d(1, d_model, kernel_size=(2, k), stride=(1, s))
        self.relu = nn.ReLU()

        encoder_layer = LiTAN(
            d_model=d_model,
            nhead=nhead,
            dim_ffn=dim_feedforward,
            dtanh=dtanh,
            alpha=0.5,
            linattn=linattn,
        )
        self.transformer = SpecEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(d_model, num_classes))

    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn(x)
        x = self.relu(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = self.fc(x)
        return x