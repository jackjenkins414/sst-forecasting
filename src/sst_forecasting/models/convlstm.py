"""Convolutional LSTM for direct multi-step SST forecasting (E2).

Architecture
------------
Input:  ``(B, L, 1, H, W)``  — normalised SST anomaly, L=context, H×W=grid
Output: ``(B, h, H, W)``     — h-step-ahead normalised SST anomaly

Pipeline::

    ConvLSTMCell × n_layers  (unrolled over L timesteps, full H×W grid)
    Last hidden state  →  (B, hidden_channels[-1], H, W)
    1×1 Conv decoder   →  (B, h, H, W)

Unlike ``SpatialFlatLSTM``, the hidden state is never flattened — spatial
structure is preserved throughout the recurrence.  With ``hidden_channels=[32, 64]``
this is ~260 k parameters vs ~9.7 M for the flat LSTM.

The model is trained in *normalised* space; caller multiplies by ``norm_std``
to recover °C RMSE for reporting.

Usage
-----
>>> model = SpatialConvLSTM(H=81, W=121, context_len=90, horizon=7)
>>> x = torch.randn(2, 90, 1, 81, 121)
>>> y = model(x)
>>> y.shape
torch.Size([2, 7, 81, 121])
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["ConvLSTMCell", "SpatialConvLSTM"]


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell (Shi et al. 2015).

    All four gates are computed with 2-D convolutions so the hidden state
    keeps its (H, W) spatial dimensions.  Both input and hidden projections
    are fused into one ``Conv2d`` call each for efficiency.

    Parameters
    ----------
    in_channels :
        Channels in the input tensor (1 for raw SST, or hidden size of the
        previous layer).
    hidden_channels :
        Channels in the hidden and cell states.
    kernel_size :
        Spatial kernel size; same-padding is applied automatically.
    bias :
        Whether to add bias to the input projection.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2  # same-padding keeps H, W unchanged

        # input → all 4 gates in one conv
        self.conv_input = nn.Conv2d(
            in_channels, 4 * hidden_channels,
            kernel_size=kernel_size, padding=pad, bias=bias,
        )
        # hidden → all 4 gates; no bias to avoid double-counting
        self.conv_hidden = nn.Conv2d(
            hidden_channels, 4 * hidden_channels,
            kernel_size=kernel_size, padding=pad, bias=False,
        )

    # ------------------------------------------------------------------

    def init_hidden(self, batch_size: int, H: int, W: int) -> tuple[Tensor, Tensor]:
        """Return zero (h_0, c_0) on the same device as the cell weights."""
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.hidden_channels, H, W, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, H, W, device=device)
        return h, c

    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,                  # (B, in_channels, H, W)
        hc: tuple[Tensor, Tensor],  # (h_{t-1}, c_{t-1})
    ) -> tuple[Tensor, Tensor]:
        """One timestep update. Returns (h_t, c_t), each ``(B, hidden, H, W)``."""
        h_prev, c_prev = hc
        gates = self.conv_input(x) + self.conv_hidden(h_prev)  # (B, 4*hidden, H, W)

        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t


# ─────────────────────────────────────────────────────────────────────────────


class SpatialConvLSTM(nn.Module):
    """Stacked ConvLSTM encoder + 1×1-Conv decoder for SST forecasting.

    Parameters
    ----------
    H, W :
        Spatial grid dimensions.
    context_len :
        Number of input timesteps *L*.
    horizon :
        Number of output timesteps *h*.
    hidden_channels :
        Hidden channel count per layer.  ``[32, 64]`` gives ~260 k params.
    kernel_size :
        Spatial kernel size for all gate convolutions (same-padding applied).
    dropout :
        Feature-map dropout (``nn.Dropout2d``) on the final hidden state.
    """

    def __init__(
        self,
        H: int,
        W: int,
        context_len: int = 90,
        horizon: int = 7,
        hidden_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.H = H
        self.W = W
        self.L = context_len
        self.h = horizon

        if hidden_channels is None:
            hidden_channels = [32, 64]
        self.hidden_channels = hidden_channels

        # build ConvLSTM stack
        cells: list[nn.Module] = []
        in_ch = 1  # input has 1 channel (normalised SST)
        for hc in hidden_channels:
            cells.append(ConvLSTMCell(in_ch, hc, kernel_size=kernel_size))
            in_ch = hc
        self.cells = nn.ModuleList(cells)

        self.drop = nn.Dropout2d(p=dropout)

        # one output channel per forecast lead time
        self.decoder = nn.Conv2d(
            hidden_channels[-1], horizon, kernel_size=1, bias=True
        )

    # -------------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : ``(B, L, 1, H, W)``  normalised SST anomaly context.

        Returns
        -------
        ``(B, h, H, W)``  h-step ahead normalised SST anomaly forecast.
        """
        B, L, C, H, W = x.shape  # noqa: N806
        assert C == 1 and H == self.H and W == self.W

        states: list[tuple[Tensor, Tensor]] = [
            cell.init_hidden(B, H, W) for cell in self.cells
        ]

        for t in range(L):
            x_t = x[:, t]  # (B, 1, H, W)
            for layer_idx, cell in enumerate(self.cells):
                x_t, c_t = cell(x_t, states[layer_idx])
                states[layer_idx] = (x_t, c_t)

        # x_t is h_L of the final layer
        out = self.decoder(self.drop(x_t))  # (B, horizon, H, W)
        return out

    # -------------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
