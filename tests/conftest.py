"""Shared pytest fixtures for the SST Forecasting test suite.

The ``tiny_zarr`` session fixture builds a small synthetic Zarr store
(200 timesteps, 16×16 grid) that covers the training split date range.
It is created once per test session in a temporary directory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import zarr

from sst_forecasting.data.splits import SPLITS


@pytest.fixture(scope="session")
def tiny_zarr(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Return the path to a synthetic Zarr store for fast unit tests.

    Layout mirrors what ``build_zarr_store`` produces:
      time (T,), lat (H,), lon (W,),
      sst (T,H,W), sst_anom (T,H,W), sst_norm (T,H,W),
      climatology (366,H,W), land_mask (H,W)

    The 200 daily timesteps start on 1981-09-01, placing them entirely
    within the training split so they can be used by split="train" tests.
    """
    tmp = tmp_path_factory.mktemp("zarr")
    zarr_path = tmp / "test_sst.zarr"

    T, H, W = 200, 16, 16
    rng = np.random.default_rng(42)

    # ── Time ─────────────────────────────────────────────────────────────────
    epoch = pd.Timestamp("1970-01-01")
    start = pd.Timestamp("1981-09-01")
    start_day = (start - epoch).days
    time_days = np.arange(start_day, start_day + T, dtype=np.int64)

    # ── Spatial coords ────────────────────────────────────────────────────────
    lat_vals = np.linspace(-25.0, -5.0,  H, dtype=np.float32)
    lon_vals = np.linspace(140.0, 170.0, W, dtype=np.float32)

    # ── Land mask: top-left 2×2 block is land ─────────────────────────────────
    land = np.zeros((H, W), dtype=bool)
    land[:2, :2] = True

    # ── Synthetic SST fields ─────────────────────────────────────────────────
    sst_data = rng.normal(25.0, 2.0, (T, H, W)).astype(np.float32)
    sst_data[:, land] = np.nan

    anom_data = rng.normal(0.0, 0.5, (T, H, W)).astype(np.float32)
    anom_data[:, land] = np.nan

    norm_std  = 0.5
    norm_mean = 0.0
    norm_data = (anom_data / (norm_std + 1e-8)).astype(np.float32)
    norm_data[:, land] = np.nan

    clim_data = np.full((366, H, W), 25.0, dtype=np.float32)
    clim_data[:, land] = np.nan

    # ── Write Zarr store ──────────────────────────────────────────────────────
    root = zarr.open_group(str(zarr_path), mode="w")

    root.create_dataset("time",        data=time_days,         dtype="int64")
    root.create_dataset("lat",         data=lat_vals,          dtype="float32")
    root.create_dataset("lon",         data=lon_vals,          dtype="float32")
    root.create_dataset("sst",         data=sst_data,  chunks=(1, H, W), dtype="float32")
    root.create_dataset("sst_anom",    data=anom_data, chunks=(1, H, W), dtype="float32")
    root.create_dataset("sst_norm",    data=norm_data, chunks=(1, H, W), dtype="float32")
    root.create_dataset("climatology", data=clim_data, chunks=(1, H, W), dtype="float32")
    root.create_dataset("land_mask",   data=~land,             dtype="bool")

    root.attrs.update(
        {
            "norm_mean":   norm_mean,
            "norm_std":    norm_std,
            "train_start": SPLITS["train"][0],
            "train_end":   SPLITS["train"][1],
            "val_start":   SPLITS["val"][0],
            "val_end":     SPLITS["val"][1],
            "test_start":  SPLITS["test"][0],
            "test_end":    SPLITS["test"][1],
            "T": T, "H": H, "W": W,
        }
    )

    return str(zarr_path)


@pytest.fixture(scope="session")
def tiny_array() -> np.ndarray:
    """Return a small (200, 16, 16) float32 numpy array for in-memory tests."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((200, 16, 16)).astype(np.float32)
