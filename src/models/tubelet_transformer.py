"""
SST Tubelet Transformer — factored spatiotemporal transformer for SST forecasting.

Input:  (B, L, 1, H, W)  normalised SST anomaly, L days of context
Output: (B, h, H, W)     h-step-ahead normalised SST anomaly

Architecture:
    Conv3d tubelet embed     -> (B, T', P, d)   T' = L/t_s, P = n_h * n_w
    + sinusoidal temporal PE + learned spatial PE
    N x FactoredBlock        -> temporal attn, spatial attn, FFN
    mean pool over T'        -> (B, P, d)
    linear head per patch    -> tile back to (B, h, H, W)

The Conv3d kernel spans t_s days x p_h x p_w cells, so each token
encodes joint spatiotemporal features over its block rather than
treating space and time as separate afterthoughts.
"""

import math
import torch
import torch.nn as nn


def _sinusoidal_pe(length, d_model):
    """Fixed sinusoidal PE table, shape (length, d_model)."""
    pe  = torch.zeros(length, d_model)
    pos = torch.arange(length).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class _FactoredBlock(nn.Module):
    """
    One transformer layer: temporal attention -> spatial attention -> FFN.

    Temporal pass: each patch independently attends over its T' time tokens.
    Spatial pass:  at each timestep, all P patches attend to each other.
    Both are cheaper than full T'*P x T'*P attention and preserve structure.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.t_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.s_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff     = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        B, T, P, d = x.shape

        # Temporal: reshape so each patch is a batch item, attend over T'
        xt = x.permute(0, 2, 1, 3).reshape(B * P, T, d)
        xt, _ = self.t_attn(xt, xt, xt)
        xt = xt.reshape(B, P, T, d).permute(0, 2, 1, 3)
        x = self.norm1(x + self.drop(xt))

        # Spatial: reshape so each timestep is a batch item, attend over P
        xs = x.reshape(B * T, P, d)
        xs, _ = self.s_attn(xs, xs, xs)
        xs = xs.reshape(B, T, P, d)
        x = self.norm2(x + self.drop(xs))

        x = self.norm3(x + self.drop(self.ff(x)))
        return x


class TubeletTransformer(nn.Module):
    """
    Factored spatiotemporal transformer for SST forecasting.

    Parameters
    ----------
    H, W        : grid height and width in cells (81, 121 for Coral Sea)
    context_len : input days L (must be divisible by t_s)
    horizon     : forecast steps h
    d_model     : token dimension
    n_heads     : attention heads (d_model must be divisible by n_heads)
    n_layers    : number of FactoredBlocks
    d_ff        : FFN hidden dimension
    t_s         : temporal stride — days per tubelet (90/5 = 18 time tokens)
    p_h, p_w    : spatial patch size in cells (9x11 divides 81x121 exactly)
    dropout     : applied in attention and FFN
    """

    def __init__(
        self,
        H:           int,
        W:           int,
        context_len: int   = 90,
        horizon:     int   = 7,
        d_model:     int   = 128,
        n_heads:     int   = 4,
        n_layers:    int   = 4,
        d_ff:        int   = 512,
        t_s:         int   = 5,
        p_h:         int   = 9,
        p_w:         int   = 11,
        dropout:     float = 0.1,
    ):
        super().__init__()
        assert context_len % t_s == 0,       "context_len must be divisible by t_s"
        assert H % p_h == 0 and W % p_w == 0, "grid dims must divide evenly by patch size"

        T_prime = context_len // t_s  # 18 temporal tokens
        n_h     = H // p_h            # 9
        n_w     = W // p_w            # 11
        P       = n_h * n_w           # 99 spatial patches

        self.T_prime        = T_prime
        self.P              = P
        self.n_h, self.n_w  = n_h, n_w
        self.p_h, self.p_w  = p_h, p_w
        self.horizon        = horizon

        # Each tubelet token covers t_s days x p_h x p_w cells jointly
        self.embed = nn.Conv3d(1, d_model, kernel_size=(t_s, p_h, p_w), stride=(t_s, p_h, p_w))

        # Fixed sinusoidal for time (position within the 18-token sequence),
        # learned for space (model discovers which patches co-vary)
        self.register_buffer("temporal_pe", _sinusoidal_pe(T_prime, d_model))
        self.spatial_pe = nn.Embedding(P, d_model)

        self.layers = nn.ModuleList(
            [_FactoredBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

        # ConvTranspose2d with kernel slightly larger than stride so adjacent
        # patches overlap in the reconstruction — eliminates hard patch boundaries.
        # kernel (p_h+2, p_w+2) with padding=1 preserves exact output size H×W.
        self.head = nn.ConvTranspose2d(
            d_model, horizon,
            kernel_size=(p_h + 2, p_w + 2),
            stride=(p_h, p_w),
            padding=(1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, 1, H, W) -> (B, horizon, H, W)"""
        B = x.shape[0]

        # Conv3d expects (B, C, D, H, W)
        x = self.embed(x.permute(0, 2, 1, 3, 4))          # (B, d, T', n_h, n_w)

        _, d, T, nh, nw = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T, nh * nw, d)  # (B, T', P, d)

        patch_ids = torch.arange(self.P, device=x.device)
        x = (x
             + self.temporal_pe.unsqueeze(1)               # (T', 1, d) -> broadcast over P
             + self.spatial_pe(patch_ids).unsqueeze(0))    # (1, P, d)  -> broadcast over T'

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x).mean(dim=1)             # mean pool over T' -> (B, P, d)
        x = x.view(B, self.n_h, self.n_w, d)    # (B, n_h, n_w, d)
        x = x.permute(0, 3, 1, 2)               # (B, d, n_h, n_w)
        return self.head(x)                      # (B, horizon, H, W)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
