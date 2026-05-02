"""Convert raw OISST NetCDF files to a Zarr store with climatology & normalisation.

Pipeline
--------
1. Load all yearly NetCDF files from *raw_dir* into a single xarray Dataset.
2. Normalise coordinate names to ``lat`` / ``lon`` (OISST ERDDAP uses
   ``latitude`` / ``longitude``).
3. Compute a day-of-year climatology using **training years only** to prevent
   data leakage into validation / test metrics.
4. Subtract the climatology to obtain SST anomalies.
5. Compute global mean and std of training anomalies; z-score normalise.
6. Build a land mask (True = ocean cell).
7. Write all arrays and metadata to a Zarr 2 store.

Zarr store layout
-----------------
├── time        (T,)      int64 — days since 1970-01-01
├── lat         (H,)      float32
├── lon         (W,)      float32
├── sst         (T,H,W)  float32 — raw SST °C, NaN = land
├── sst_anom    (T,H,W)  float32 — SST anomaly (sst − clim), NaN = land
├── sst_norm    (T,H,W)  float32 — z-scored anomaly (train stats)
├── climatology (366,H,W) float32 — DOY 1–366 mean over training years
└── land_mask   (H,W)    bool    — True = ocean

Group attributes: norm_mean, norm_std, split date strings.

Usage
-----
>>> from sst_forecasting.data.preprocess import build_zarr_store
>>> build_zarr_store("data/raw", "data/processed/oisst_coralsea.zarr")
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from sst_forecasting.data.splits import SPLITS

log = logging.getLogger(__name__)

# Training period boundaries (no leakage)
_TRAIN_START = pd.Timestamp(SPLITS["train"][0])
_TRAIN_END   = pd.Timestamp(SPLITS["train"][1])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_raw(raw_dir: Path, pattern: str = "oisst_v21_*.nc") -> xr.Dataset:
    """Open all matching NetCDF files as one xarray Dataset (lazy via dask)."""
    files = sorted(raw_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching {pattern!r} found in {raw_dir}. "
            "Run download_oisst first."
        )
    log.info("Loading %d NetCDF files from %s …", len(files), raw_dir)
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        engine="netcdf4",
        parallel=True,  # uses dask
    )
    return ds


def _normalise_coords(ds: xr.Dataset) -> xr.Dataset:
    """Rename ERDDAP coordinate names to canonical ``lat`` / ``lon``."""
    rename_map = {}
    if "latitude" in ds.coords and "lat" not in ds.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.coords and "lon" not in ds.coords:
        rename_map["longitude"] = "lon"
    if rename_map:
        ds = ds.rename(rename_map)
    return ds


def _compute_climatology(sst_train: xr.DataArray) -> xr.DataArray:
    """Day-of-year climatology computed from training data only.

    Parameters
    ----------
    sst_train : DataArray with dims ``(time, lat, lon)`` covering **only**
                training timesteps.

    Returns
    -------
    clim : DataArray with dims ``(dayofyear, lat, lon)``, indexed 1–366.
           DOY 366 is filled via nearest-neighbour from DOY 365 when no
           29 February appears in the training record.
    """
    clim = sst_train.groupby("time.dayofyear").mean("time", skipna=True)
    # Ensure all 366 DOY slots are present (fill missing leap-day if needed)
    clim = clim.reindex(dayofyear=np.arange(1, 367), method="nearest")
    return clim


def _compute_anomaly(sst: xr.DataArray, clim: xr.DataArray) -> xr.DataArray:
    """Subtract the DOY climatology from the full SST timeseries.

    Uses ``groupby`` so that each day is paired with the correct DOY entry.
    The ``dayofyear`` coordinate is dropped from the result.
    """
    anom = sst.groupby("time.dayofyear") - clim
    return anom.drop_vars("dayofyear", errors="ignore")


def _compute_norm_stats(anom_values: np.ndarray) -> tuple[float, float]:
    """Global mean and std of training anomalies, ignoring NaN (land).

    Returns
    -------
    (mean, std) as Python floats.  If std ≈ 0 (e.g. single-year test run where
    climatology == training data), returns std=1.0 with a warning so that
    sst_norm = sst_anom − mean (no scaling collapse) and downstream checks
    don't divide by zero.
    """
    valid = anom_values[~np.isnan(anom_values)]
    if len(valid) == 0:
        raise ValueError("Training anomaly array contains only NaN values.")
    mean = float(np.mean(valid))
    std  = float(np.std(valid))
    if std < 1e-6:
        log.warning(
            "Training anomaly std ≈ 0 (%.2e) — this happens when only one year "
            "is loaded because the DOY climatology equals the training data. "
            "Falling back to std=1.0 so sst_norm is not a constant field.",
            std,
        )
        std = 1.0
    return mean, std


def _build_land_mask(sst: xr.DataArray) -> np.ndarray:
    """Boolean mask of shape ``(H, W)``: True where at least one valid obs exists."""
    return np.any(~np.isnan(sst.values), axis=0)


def _time_to_days(time_index: pd.DatetimeIndex) -> np.ndarray:
    """Convert a DatetimeIndex to int64 days-since-1970-01-01."""
    epoch = pd.Timestamp("1970-01-01")
    return np.array([(t - epoch).days for t in time_index], dtype=np.int64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_zarr_store(
    raw_dir: str | Path,
    zarr_path: str | Path,
    *,
    raw_pattern: str = "oisst_v21_*.nc",
    sst_var: str = "sst",
    overwrite: bool = False,
) -> Path:
    """Convert raw OISST NetCDF files to a ready-to-train Zarr store.

    Parameters
    ----------
    raw_dir     : Directory containing the yearly ``oisst_v21_*.nc`` files.
    zarr_path   : Destination Zarr store path (will be created).
    raw_pattern : Glob pattern for input files inside *raw_dir*.
    sst_var     : Variable name inside the NetCDF files.
    overwrite   : If True, delete any existing store at *zarr_path* first.

    Returns
    -------
    Path : Path to the created Zarr store.

    Raises
    ------
    FileNotFoundError : No NetCDF files match *raw_pattern* in *raw_dir*.
    ValueError        : Training period contains no valid ocean data.
    """
    raw_dir   = Path(raw_dir)
    zarr_path = Path(zarr_path)

    if zarr_path.exists():
        if overwrite:
            shutil.rmtree(zarr_path)
            log.info("Removed existing store at %s.", zarr_path)
        else:
            log.info("Zarr store already exists at %s; skipping. Pass overwrite=True to rebuild.", zarr_path)
            return zarr_path

    # ── 1. Load ──────────────────────────────────────────────────────────────
    ds = _load_raw(raw_dir, pattern=raw_pattern)
    ds = _normalise_coords(ds)

    # Drop the altitude/zlev dimension if present (OISST has a scalar zlev)
    if "altitude" in ds.coords:
        ds = ds.squeeze("altitude", drop=True)
    if "zlev" in ds.coords:
        ds = ds.squeeze("zlev", drop=True)

    sst: xr.DataArray = ds[sst_var].sortby("time")

    # ── 2. Split masks ───────────────────────────────────────────────────────
    time_pd = pd.DatetimeIndex(sst.time.values)
    train_mask = (time_pd >= _TRAIN_START) & (time_pd <= _TRAIN_END)

    T = len(time_pd)
    lat_vals = sst.lat.values.astype(np.float32)
    lon_vals = sst.lon.values.astype(np.float32)
    H, W = len(lat_vals), len(lon_vals)

    log.info(
        "Timesteps total=%d  train=%d  |  lat=%d  lon=%d",
        T, int(train_mask.sum()), H, W,
    )

    # ── 3. Climatology from training years only ───────────────────────────────
    log.info("Computing day-of-year climatology (training years only) …")
    # Compute the full SST array once to avoid multiple dask evaluations
    log.info("Loading SST into memory (this may take a moment for large datasets) …")
    sst_values = sst.values.astype(np.float32)          # (T, H, W), NaN = land
    sst_train_da = xr.DataArray(
        sst_values[train_mask],
        dims=["time", "lat", "lon"],
        coords={
            "time": sst.time.values[train_mask],
            "lat":  sst.lat.values,
            "lon":  sst.lon.values,
        },
    )
    clim = _compute_climatology(sst_train_da)  # (366, H, W)

    # ── 4. Anomalies ──────────────────────────────────────────────────────────
    log.info("Computing SST anomalies …")
    # Reuse already-loaded sst_values; build a full DataArray for groupby
    sst_da = xr.DataArray(
        sst_values,
        dims=["time", "lat", "lon"],
        coords={
            "time": sst.time.values,
            "lat":  sst.lat.values,
            "lon":  sst.lon.values,
        },
    )
    sst_anom_da = _compute_anomaly(sst_da, clim)
    sst_anom_values = sst_anom_da.values.astype(np.float32)  # (T, H, W)

    # ── 5. Normalisation stats (training anomalies only) ──────────────────────
    log.info("Computing normalisation statistics …")
    norm_mean, norm_std = _compute_norm_stats(sst_anom_values[train_mask])
    log.info("  norm_mean=%.5f  norm_std=%.5f", norm_mean, norm_std)

    # ── 6. Normalised anomaly ─────────────────────────────────────────────────
    sst_norm_values = ((sst_anom_values - norm_mean) / (norm_std + 1e-8)).astype(np.float32)

    # ── 7. Land mask ──────────────────────────────────────────────────────────
    land_mask = np.any(~np.isnan(sst_values), axis=0)  # True = ocean; reuse sst_values

    # ── 8. Write Zarr ─────────────────────────────────────────────────────────
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Writing Zarr store → %s …", zarr_path)

    root = zarr.open_group(str(zarr_path), mode="w")

    time_days = _time_to_days(time_pd)

    root.create_dataset("time", data=time_days, dtype="int64", chunks=(T,))
    root.create_dataset("lat",  data=lat_vals,  dtype="float32")
    root.create_dataset("lon",  data=lon_vals,  dtype="float32")

    # Spatial arrays: chunk 1 timestep at a time for efficient sequential access
    root.create_dataset(
        "sst",
        data=sst_values,
        chunks=(1, H, W),
        dtype="float32",
        fill_value=float("nan"),
    )
    root.create_dataset(
        "sst_anom",
        data=sst_anom_values,
        chunks=(1, H, W),
        dtype="float32",
        fill_value=float("nan"),
    )
    root.create_dataset(
        "sst_norm",
        data=sst_norm_values,
        chunks=(1, H, W),
        dtype="float32",
        fill_value=float("nan"),
    )
    root.create_dataset(
        "climatology",
        data=clim.values.astype(np.float32),
        chunks=(1, H, W),
        dtype="float32",
        fill_value=float("nan"),
    )
    root.create_dataset(
        "land_mask",
        data=land_mask.astype(bool),
        dtype="bool",
    )

    # Store metadata as group attributes
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
            "lat_min":     float(lat_vals.min()),
            "lat_max":     float(lat_vals.max()),
            "lon_min":     float(lon_vals.min()),
            "lon_max":     float(lon_vals.max()),
            "T": T, "H": H, "W": W,
        }
    )

    log.info("Zarr store complete: T=%d H=%d W=%d  (%.1f MB)", T, H, W, _store_size_mb(zarr_path))
    return zarr_path


def load_zarr_metadata(zarr_path: str | Path) -> dict:
    """Read group-level attributes from an existing Zarr store.

    Useful for retrieving ``norm_mean`` / ``norm_std`` before constructing
    transforms, without loading any array data.
    """
    root = zarr.open_group(str(zarr_path), mode="r")
    return dict(root.attrs)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _store_size_mb(path: Path) -> float:
    """Approximate on-disk size of a directory in megabytes."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 ** 2)
