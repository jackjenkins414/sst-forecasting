from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset

from src.data.splits import date_mask


class SstWindowDataset(Dataset):
    """
    PyTorch Dataset for grid-based SST sliding-window forecasting.

    Preloads the full normalised SST anomaly array from the Zarr store into
    RAM on construction, then returns sliding-window samples for one
    chronological split entirely from in-memory numpy arrays.

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

        root = zarr.open_group(str(zarr_path), mode="r")
        # Load the full array into RAM once so every __getitem__ call is a
        # fast in-memory slice instead of a Lustre read.
        self._data = np.asarray(root["sst_norm"][:], dtype=np.float32)

        # Convert stored int64 days-since-epoch to a DatetimeIndex
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

        x_raw = np.asarray(
            self._data[t0 : t0 + self.context_len],
            dtype=np.float32,
        )
        y_raw = np.asarray(
            self._data[t0 + self.context_len : t0 + self.context_len + self.horizon],
            dtype=np.float32,
        )

        x = torch.from_numpy(x_raw)
        y = torch.from_numpy(y_raw)

        # Replace land NaN with 0.0
        x = torch.nan_to_num(x, nan=0.0)
        y = torch.nan_to_num(y, nan=0.0)

        # Add channel dimension to x: (L, H, W) -> (L, 1, H, W)
        x = x.unsqueeze(1)

        return x, y