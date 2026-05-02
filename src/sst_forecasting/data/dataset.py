"""PyTorch Dataset for sliding-window SST forecasting.

Each sample ``(x, y)`` consists of:

* ``x`` — float32 tensor of shape ``(context_len, 1, H, W)``:
  the past *L* days of normalised SST anomaly (channel=1 for SST-only).
* ``y`` — float32 tensor of shape ``(horizon, H, W)``:
  the next *h* days of normalised SST anomaly to predict.

NaN (land) cells are replaced with 0.0 before being returned, which is
appropriate for normalised anomalies whose ocean mean is ≈ 0.

Usage
-----
>>> from sst_forecasting.data.dataset import SSTWindowDataset
>>> ds = SSTWindowDataset("data/processed/oisst_coralsea.zarr",
...                       split="train", context_len=90, horizon=7)
>>> x, y = ds[0]
>>> x.shape, y.shape
(torch.Size([90, 1, 80, 120]), torch.Size([7, 80, 120]))

For testing without a real Zarr store, pass a numpy array directly:

>>> import numpy as np
>>> arr = np.random.randn(200, 16, 16).astype("float32")
>>> ds = SSTWindowDataset(data_array=arr, context_len=90, horizon=7)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset

from sst_forecasting.data.splits import date_mask


class SSTWindowDataset(Dataset):
    """Sliding-window dataset over normalised SST anomaly fields.

    Parameters
    ----------
    zarr_path :
        Path to the Zarr store produced by
        :func:`sst_forecasting.data.preprocess.build_zarr_store`.
        Mutually exclusive with *data_array*.
    split :
        One of ``"train"``, ``"val"``, ``"test"``.  Windows are constrained
        to lie entirely within the split's date range — no window crosses a
        split boundary.
    context_len :
        Number of past timesteps fed to the model (L in the paper).
    horizon :
        Number of future timesteps to predict (h in the paper).
    data_array :
        Optional in-memory ``(T, H, W)`` float32 numpy array (for unit tests
        or toy experiments).  When supplied, *zarr_path* and *split* are
        ignored and all T timesteps are used.
    transform :
        Optional callable applied to ``x`` after land-fill and channel
        insertion.  Receives and returns a ``(L, 1, H, W)`` tensor.
    target_transform :
        Optional callable applied to ``y`` after land-fill.  Receives and
        returns a ``(h, H, W)`` tensor.
    fill_land :
        Value used to replace NaN (land) cells.  Default 0.0.

    Notes
    -----
    Zarr arrays are accessed lazily (chunk-level caching via zarr's default
    LRU cache) so the full dataset never needs to reside in RAM.  For small
    domains (≲ 300 MB) you can optionally pre-load by calling
    ``np.asarray(root["sst_norm"])`` yourself and passing the result via
    *data_array*.
    """

    def __init__(
        self,
        zarr_path: str | Path | None = None,
        *,
        split: str = "train",
        context_len: int = 90,
        horizon: int = 7,
        data_array: np.ndarray | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        fill_land: float = 0.0,
    ) -> None:
        if zarr_path is None and data_array is None:
            raise ValueError("Provide either zarr_path or data_array.")
        if zarr_path is not None and data_array is not None:
            raise ValueError("Provide zarr_path OR data_array, not both.")

        self.L = context_len
        self.h = horizon
        self.transform = transform
        self.target_transform = target_transform
        self.fill_land = fill_land

        if data_array is not None:
            # ── in-memory path (tests / toy experiments) ─────────────────────
            if data_array.ndim != 3:
                raise ValueError(f"data_array must be (T, H, W); got shape {data_array.shape}.")
            self._data = data_array
            self._start_idx = 0
            n_split = len(data_array)
        else:
            # ── Zarr path ─────────────────────────────────────────────────────
            zarr_path = Path(zarr_path)
            if not zarr_path.exists():
                raise FileNotFoundError(
                    f"Zarr store not found: {zarr_path}. Run build_zarr_store first."
                )
            root = zarr.open_group(str(zarr_path), mode="r")
            self._data = root["sst_norm"]  # lazy zarr.Array (T, H, W)

            # Convert stored int64 days-since-epoch to DatetimeIndex
            time_days: np.ndarray = root["time"][:]
            epoch = pd.Timestamp("1970-01-01")
            time_pd = pd.DatetimeIndex(
                [epoch + pd.Timedelta(days=int(d)) for d in time_days]
            )

            idx = np.where(date_mask(time_pd, split))[0]
            if len(idx) == 0:
                raise ValueError(
                    f"No timesteps found for split={split!r} in {zarr_path}."
                )

            # Verify that the split's time indices are contiguous (no gaps).
            # OISST is daily and complete, but we check to catch corrupted stores.
            gaps = np.diff(idx)
            if np.any(gaps != 1):
                raise ValueError(
                    f"Split '{split}' has non-contiguous time indices in {zarr_path}. "
                    "The Zarr store may be corrupt or missing days."
                )

            self._start_idx = int(idx[0])
            n_split = len(idx)

        # Number of valid windows within the split
        self._n_windows = n_split - context_len - horizon + 1
        if self._n_windows <= 0:
            raise ValueError(
                f"Split has {n_split} timesteps but context_len={context_len} + "
                f"horizon={horizon} requires at least {context_len + horizon}."
            )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._n_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self._n_windows:
            raise IndexError(f"Index {idx} out of range [0, {self._n_windows}).")

        t0 = self._start_idx + idx

        # Load from zarr / numpy; convert to float32 numpy array eagerly
        x_raw = np.asarray(self._data[t0 : t0 + self.L],          dtype=np.float32)  # (L, H, W)
        y_raw = np.asarray(self._data[t0 + self.L : t0 + self.L + self.h], dtype=np.float32)  # (h, H, W)

        x = torch.from_numpy(x_raw)   # (L, H, W)
        y = torch.from_numpy(y_raw)   # (h, H, W)

        # Replace NaN (land) with fill value
        x = torch.nan_to_num(x, nan=self.fill_land)
        y = torch.nan_to_num(y, nan=self.fill_land)

        # Add channel dimension: (L, H, W) → (L, 1, H, W)
        x = x.unsqueeze(1)

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """``(H, W)`` grid dimensions."""
        return self._data.shape[1], self._data.shape[2]

    @property
    def n_windows(self) -> int:
        """Total number of valid sliding windows (same as ``len(self)``)."""
        return self._n_windows
