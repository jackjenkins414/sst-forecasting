from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset

from src.data.splits import date_mask

# Cache of the preloaded, NaN-cleaned sst_norm field keyed by resolved store
# path. The train/val/test splits all read the SAME array (only their index
# ranges differ), so without this each split would re-decompress all ~7000
# (1,H,W) chunks (~70 s each). Loaded once, shared read-only across splits.
_FIELD_CACHE: dict[str, np.ndarray] = {}


def _load_field(zarr_path: Path) -> np.ndarray:
    key = str(zarr_path.resolve())
    field = _FIELD_CACHE.get(key)
    if field is None:
        root = zarr.open_group(str(zarr_path), mode="r")
        field = np.nan_to_num(
            np.asarray(root["sst_norm"][:], dtype=np.float32), nan=0.0
        )
        _FIELD_CACHE[key] = field
    return field


class SstWindowDataset(Dataset):
    """
    PyTorch Dataset for grid-based SST sliding-window forecasting.

    Reads normalised SST anomaly fields lazily from a Zarr store and
    returns sliding-window samples for one chronological split.

    Each item contains:
        x = sequence of past SST maps with a channel dimension
        y = sequence of future SST maps to predict

    Expected shapes:
        x: context_len x 1 x H x W
        y: horizon x H x W

    Land cells (NaN in the raw store) are replaced with 0.0,
    which is appropriate because the data is normalised anomalies
    centred near zero.
    """

    def __init__(
        self,
        zarr_path: str | Path,
        split: str = "train",
        context_len: int = 90,
        horizon: int = 7,
    ):
        zarr_path = Path(zarr_path)
        if not zarr_path.exists():
            raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

        # Preload the whole normalised field into RAM once (shared across
        # splits via _FIELD_CACHE), with land-NaN already replaced by 0.0. The
        # store is chunked (1, H, W) — one chunk per timestep — so a lazy
        # __getitem__ would decompress `context_len` (+ horizon) chunks per
        # sample on the main thread (num_workers=0), making training I/O-bound
        # on synchronous zarr reads. The full array is only ~277 MB (float32),
        # so holding it resident is trivial and turns __getitem__ into a pure
        # in-RAM slice. Numerically identical to the previous per-item
        # nan_to_num(..., nan=0.0).
        self._data = _load_field(zarr_path)

        # Convert stored int64 days-since-epoch to a DatetimeIndex
        root = zarr.open_group(str(zarr_path), mode="r")
        time_days = root["time"][:]
        epoch = pd.Timestamp("1970-01-01")
        time_pd = pd.DatetimeIndex(
            [epoch + pd.Timedelta(days=int(d)) for d in time_days]
        )

        split_idx = np.where(date_mask(time_pd, split))[0]
        if len(split_idx) == 0:
            raise ValueError(f"No timesteps found for split={split!r}.")

        self._start_idx = int(split_idx[0])
        self._n_windows = len(split_idx) - context_len - horizon + 1
        if self._n_windows <= 0:
            raise ValueError(
                f"Split {split!r} has {len(split_idx)} timesteps, "
                f"but context_len={context_len} + horizon={horizon} "
                f"requires at least {context_len + horizon}."
            )

        self.context_len = context_len
        self.horizon = horizon

    def __len__(self):
        return self._n_windows

    def __getitem__(self, idx):
        t0 = self._start_idx + idx

        # Pure in-RAM slices (NaN already cleaned in __init__). Copy so the
        # returned tensors own their memory (safe across DataLoader workers).
        x_raw = self._data[t0 : t0 + self.context_len].copy()
        y_raw = self._data[
            t0 + self.context_len : t0 + self.context_len + self.horizon
        ].copy()

        x = torch.from_numpy(x_raw)
        y = torch.from_numpy(y_raw)

        # Add channel dimension to x: (L, H, W) -> (L, 1, H, W)
        x = x.unsqueeze(1)

        return x, y