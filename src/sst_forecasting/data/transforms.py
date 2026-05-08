"""Composable transforms applied to SST tensors.

All transforms are callable dataclasses that operate on ``torch.Tensor``
inputs. They are designed to be composed via ``torchvision.transforms.Compose``
or used individually.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


@dataclass
class Standardize:
    """Z-score normalise using precomputed training-period statistics.

    Parameters
    ----------
    mean, std:
        Scalar statistics derived from the training split **only** to avoid
        data leakage.  Stored in the Zarr store attributes ``norm_mean`` /
        ``norm_std``.

    Usage
    -----
    >>> norm = Standardize(mean=0.12, std=0.47)
    >>> x_norm = norm(x)          # forward
    >>> x_orig = norm.inverse(x_norm)  # back to original units
    """

    mean: float
    std: float

    def __call__(self, x: Tensor) -> Tensor:
        return (x - self.mean) / (self.std + 1e-8)

    def inverse(self, x: Tensor) -> Tensor:
        """Undo normalisation (e.g., to report RMSE in °C)."""
        return x * (self.std + 1e-8) + self.mean


# ---------------------------------------------------------------------------
# Land / NaN handling
# ---------------------------------------------------------------------------


@dataclass
class FillLand:
    """Replace NaN (land) cells with a constant fill value.

    Applied before tensors reach model layers to avoid NaN propagation.
    The default of 0.0 is appropriate for normalised anomalies (mean≈0).
    """

    fill_value: float = 0.0

    def __call__(self, x: Tensor) -> Tensor:
        return torch.nan_to_num(x, nan=self.fill_value)


# ---------------------------------------------------------------------------
# Spatial patching (for Transformer / patch-based models)
# ---------------------------------------------------------------------------


@dataclass
class SpatialPatchify:
    """Reshape spatial dims into a flat sequence of non-overlapping patches.

    Input shape:   ``(..., H, W)``
    Output shape:  ``(..., num_patches, patch_h * patch_w)``

    where ``num_patches = (H // patch_h) * (W // patch_w)``.

    Parameters
    ----------
    patch_h, patch_w:
        Patch height and width in grid cells.  H and W must be exactly
        divisible by the respective patch dimension.
    """

    patch_h: int = 8
    patch_w: int = 8

    def __call__(self, x: Tensor) -> Tensor:
        *leading, H, W = x.shape
        ph, pw = self.patch_h, self.patch_w
        if H % ph != 0 or W % pw != 0:
            raise ValueError(
                f"Spatial dims ({H}, {W}) must be divisible by patch size ({ph}, {pw})."
            )
        n_leading = len(leading)
        nH, nW = H // ph, W // pw

        # (..., H, W) -> (..., nH, ph, nW, pw)
        x = x.reshape(*leading, nH, ph, nW, pw)
        # (..., nH, ph, nW, pw) -> (..., nH, nW, ph, pw)
        x = x.permute(*range(n_leading), n_leading, n_leading + 2, n_leading + 1, n_leading + 3)
        # (..., nH, nW, ph, pw) -> (..., nH*nW, ph*pw)
        x = x.reshape(*leading, nH * nW, ph * pw)
        return x

    @property
    def patch_size(self) -> int:
        return self.patch_h * self.patch_w
