from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint

__all__ = ["ConvLSTMCell", "SpatialConvLSTM"]


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell (Shi et al. 2015)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2
        self.conv_input = nn.Conv2d(
            in_channels, 4 * hidden_channels,
            kernel_size=kernel_size, padding=pad, bias=bias,
        )
        self.conv_hidden = nn.Conv2d(
            hidden_channels, 4 * hidden_channels,
            kernel_size=kernel_size, padding=pad, bias=False,
        )

    def init_hidden(self, batch_size: int, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.hidden_channels, H, W, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, H, W, device=device)
        return h, c

    def forward(
        self,
        x: Tensor,
        hc: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        h_prev, c_prev = hc
        gates = self.conv_input(x) + self.conv_hidden(h_prev)

        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t


class SpatialConvLSTM(nn.Module):
    """Stacked ConvLSTM encoder with 1x1 conv decoder for multi-step SST forecasting.

    Input:  (B, L, 1, H, W)
    Output: (B, h, H, W)
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
        checkpoint_segments: int = 0,
    ) -> None:
        super().__init__()
        self.H = H
        self.W = W
        self.L = context_len
        self.h = horizon
        self.checkpoint_segments = checkpoint_segments

        if hidden_channels is None:
            hidden_channels = [32, 64]
        self.hidden_channels = hidden_channels

        cells: list[nn.Module] = []
        in_ch = 1
        for hc in hidden_channels:
            cells.append(ConvLSTMCell(in_ch, hc, kernel_size=kernel_size))
            in_ch = hc
        self.cells = nn.ModuleList(cells)

        for cell in self.cells:
            cell.conv_input  = cell.conv_input.to(memory_format=torch.channels_last)
            cell.conv_hidden = cell.conv_hidden.to(memory_format=torch.channels_last)

        self.drop = nn.Dropout2d(p=dropout)
        self.decoder = nn.Conv2d(hidden_channels[-1], horizon, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        B, L, C, H, W = x.shape
        assert C == 1 and H == self.H and W == self.W

        x = x.contiguous()
        states: list[tuple[Tensor, Tensor]] = [
            cell.init_hidden(B, H, W) for cell in self.cells
        ]

        if self.checkpoint_segments > 0 and torch.is_grad_enabled():
            seg_len = (L + self.checkpoint_segments - 1) // self.checkpoint_segments

            def run_segment(x_seg: Tensor, *states_flat: Tensor) -> tuple[Tensor, ...]:
                n = len(self.cells)
                seg_states: list[tuple[Tensor, Tensor]] = [
                    (states_flat[2 * i], states_flat[2 * i + 1]) for i in range(n)
                ]
                for t in range(x_seg.size(1)):
                    x_t = x_seg[:, t].to(memory_format=torch.channels_last)
                    for layer_idx in range(n):
                        x_t, c_t = self.cells[layer_idx](x_t, seg_states[layer_idx])
                        seg_states[layer_idx] = (x_t, c_t)
                return tuple(s for pair in seg_states for s in pair)

            for seg_start in range(0, L, seg_len):
                seg_end = min(seg_start + seg_len, L)
                states_flat_in = tuple(s for pair in states for s in pair)
                states_flat_out = grad_checkpoint(
                    run_segment,
                    x[:, seg_start:seg_end],
                    *states_flat_in,
                    use_reentrant=False,
                )
                states = [
                    (states_flat_out[2 * i], states_flat_out[2 * i + 1])
                    for i in range(len(self.cells))
                ]
            x_t = states[-1][0]

        else:
            for t in range(L):
                x_t = x[:, t].to(memory_format=torch.channels_last)
                for layer_idx in range(len(self.cells)):
                    x_t, c_t = self.cells[layer_idx](x_t, states[layer_idx])
                    states[layer_idx] = (x_t, c_t)

        out = self.decoder(self.drop(x_t))
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

