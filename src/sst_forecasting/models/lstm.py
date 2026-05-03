"""Stacked LSTM for direct multi-step SST forecasting (E1).

Architecture
------------
Input:  ``(B, L, 1, H, W)``  — normalised SST anomaly, L=context, H×W=grid
Output: ``(B, h, H, W)``     — h-step-ahead normalised SST anomaly

Pipeline::

    flatten spatial  →  (B, L, H*W)
    Linear encoder   →  (B, L, d_spatial)
    Stacked LSTM     →  (B, hidden)   [last timestep]
    Dropout
    Linear decoder   →  (B, h * H*W)
    reshape          →  (B, h, H, W)

The model is trained in *normalised* space; caller multiplies by ``norm_std``
to recover °C RMSE for reporting.

Usage
-----
>>> model = SpatialFlatLSTM(H=81, W=121, context_len=90, horizon=7)
>>> x = torch.randn(4, 90, 1, 81, 121)
>>> y = model(x)
>>> y.shape
torch.Size([4, 7, 81, 121])
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["SpatialFlatLSTM"]


class SpatialFlatLSTM(nn.Module):
    """Encoder–decoder LSTM that flattens the spatial grid before the RNN.

    Parameters
    ----------
    H, W :
        Spatial grid dimensions (height × width grid cells).
    context_len :
        Number of input timesteps *L*.
    horizon :
        Number of output timesteps *h*.
    d_spatial :
        Width of the spatial encoder projection (input → RNN).
    hidden_size :
        LSTM hidden/cell state dimensionality.
    num_layers :
        Number of stacked LSTM layers.
    dropout :
        Dropout probability applied between LSTM layers and before the
        output projection.  Set to 0.0 when ``num_layers==1``.
    """

    def __init__(
        self,
        H: int,
        W: int,
        context_len: int = 90,
        horizon: int = 7,
        d_spatial: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.H = H
        self.W = W
        self.L = context_len
        self.h = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        spatial_dim = H * W

        # ── Spatial encoder ───────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(spatial_dim, d_spatial),
            nn.ReLU(inplace=True),
        )

        # ── Temporal LSTM ─────────────────────────────────────────────────────
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=d_spatial,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # ── Output head ───────────────────────────────────────────────────────
        self.dropout = nn.Dropout(p=dropout)
        self.decoder = nn.Linear(hidden_size, horizon * spatial_dim)

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

        # Spatial encoder: (B, L, d_spatial)
        enc = self.encoder(x_flat)

        # LSTM: take final hidden state from last layer → (B, hidden)
        _, (h_n, _) = self.lstm(enc)
        last_hidden = h_n[-1]                       # (B, hidden)

        # Output head: (B, h * H * W) → (B, h, H, W)
        out = self.decoder(self.dropout(last_hidden))
        return out.view(B, self.h, H, W)

    # -------------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
