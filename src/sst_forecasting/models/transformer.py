"""Transformer encoder for direct multi-step SST forecasting (E1).

Architecture
------------
Input:  ``(B, L, 1, H, W)``  — normalised SST anomaly, L=context, H×W=grid
Output: ``(B, h, H, W)``     — h-step-ahead normalised SST anomaly

Pipeline::

    flatten spatial     →  (B, L, H*W)
    Linear encoder      →  (B, L, d_model)
    + sinusoidal PE     →  (B, L, d_model)
    TransformerEncoder  →  (B, L, d_model)   [4 layers, 8 heads]
    mean-pool over L    →  (B, d_model)
    Linear decoder      →  (B, h * H*W)
    reshape             →  (B, h, H, W)

The model is trained in *normalised* space; caller multiplies by ``norm_std``
to recover °C RMSE for reporting.

Notes
-----
*  Sinusoidal positional encoding follows Vaswani et al. (2017): fixed, not
   learnable — appropriate when L is fixed at 90.
*  Causal masking is **not** applied: the encoder attends over all L positions
   for the deterministic forecasting task (not autoregressive generation).
*  ``torch.compile`` is left disabled for AVX1-only Sandy Bridge compatibility.

Usage
-----
>>> model = SpatialFlatTransformer(H=81, W=121, context_len=90, horizon=7)
>>> x = torch.randn(4, 90, 1, 81, 121)
>>> y = model(x)
>>> y.shape
torch.Size([4, 7, 81, 121])
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["SpatialFlatTransformer"]


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding (fixed, not learnable)
# ---------------------------------------------------------------------------


class SinusoidalPE(nn.Module):
    """Add fixed sinusoidal positional encoding to a ``(B, T, d)`` sequence."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute the (max_len, d_model) PE table once and register as buffer
        pe = torch.zeros(max_len, d_model)                       # (T, d)
        pos = torch.arange(max_len).unsqueeze(1).float()         # (T, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))              # (1, T, d)

    def forward(self, x: Tensor) -> Tensor:
        """Add PE to ``x`` of shape ``(B, T, d)``."""
        x = x + self.pe[:, : x.size(1)]                         # type: ignore[index]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Spatial-flat Transformer
# ---------------------------------------------------------------------------


class SpatialFlatTransformer(nn.Module):
    """Transformer encoder that flattens the spatial grid before attention.

    Parameters
    ----------
    H, W :
        Spatial grid dimensions (height × width grid cells).
    context_len :
        Number of input timesteps *L*.
    horizon :
        Number of output timesteps *h*.
    d_model :
        Transformer model dimension (also the spatial encoder output width).
    nhead :
        Number of attention heads.  Must divide ``d_model``.
    num_encoder_layers :
        Depth of the Transformer encoder (number of ``TransformerEncoderLayer``
        blocks stacked).
    dim_feedforward :
        Width of the two-layer FFN inside each encoder layer.
    dropout :
        Dropout probability for all dropout layers (attention, FFN, PE).
    """

    def __init__(
        self,
        H: int,
        W: int,
        context_len: int = 90,
        horizon: int = 7,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.H = H
        self.W = W
        self.L = context_len
        self.h = horizon
        self.d_model = d_model

        spatial_dim = H * W

        # ── Spatial encoder ───────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(spatial_dim, d_model),
            nn.ReLU(inplace=True),
        )

        # ── Positional encoding ───────────────────────────────────────────────
        self.pos_enc = SinusoidalPE(d_model, max_len=max(context_len + 1, 512), dropout=dropout)

        # ── Transformer encoder ───────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,        # expects (B, T, d)
            norm_first=False,        # post-LN (standard Vaswani 2017)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False,   # avoid potential shape issues on CPU
        )

        # ── Output head ───────────────────────────────────────────────────────
        self.decoder = nn.Linear(d_model, horizon * spatial_dim)

    # -------------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : ``(B, L, 1, H, W)``  normalised SST anomaly context.

        Returns
        -------
        ``(B, h, H, W)``  h-step ahead normalised SST anomaly forecast.
        """
        B, L, C, H, W = x.shape  # noqa: N806
        assert C == 1 and H == self.H and W == self.W

        # Flatten spatial: (B, L, H*W)
        x_flat = x.view(B, L, H * W)

        # Spatial encoder + PE: (B, L, d_model)
        enc = self.pos_enc(self.encoder(x_flat))

        # Transformer encoder: (B, L, d_model)
        out_seq = self.transformer(enc)

        # Mean-pool over temporal axis: (B, d_model)
        pooled = out_seq.mean(dim=1)

        # Output head: (B, h * H * W) → (B, h, H, W)
        out = self.decoder(pooled)
        return out.view(B, self.h, H, W)

    # -------------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
