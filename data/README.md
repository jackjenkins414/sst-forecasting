# Data

Raw files and processed outputs are **not committed to git** — they are generated locally from a public NOAA source.

---

## Quickstart

```bash
# Step 1 — download 20 years of raw SST data (~280 MB, 5–10 min)
python scripts/download_oisst.py --output-dir data/raw

# Step 2 — build the processed Zarr store (~1 min)
python scripts/build_zarr.py

# Step 3 — verify everything is correct (~30 sec)
python scripts/validate_pipeline.py
```

After this you will have:

```
data/
  raw/
    oisst_v21_1981.nc  …  oisst_v21_2000.nc   (20 files, ~280 MB)
  processed/
    oisst_coralsea.zarr/                        (~564 MB)
```

---

## Source data

| Property | Value |
|---|---|
| Source | NOAA OISST v2.1 via CoastWatch ERDDAP |
| Variable | Sea Surface Temperature (SST) in °C |
| Resolution | 0.25° spatial, daily temporal |
| Full period | 1981-09-01 → 2000-12-31 (7,062 days) |
| Spatial crop | Coral Sea: 140°E–170°E × 25°S–5°S |
| Grid size | 81 × 121 cells (9,801 total) |
| Ocean cells | 7,890 (80.5%) |
| Land cells | 1,911 (19.5%) — coastlines, islands, PNG |

---

## Train / Val / Test splits

A **chronological hold-out split** is used — the timeline is cut at fixed points so the model is always evaluated on genuinely unseen future data.

**3-month gap buffers** are applied at both split boundaries. SST is highly autocorrelated day-to-day, so without a gap the model could benefit from autocorrelation at the edges rather than learning real structure. The gap periods are excluded from all splits.

| Split | Period | Days | Purpose |
|---|---|---|---|
| Train | 1981-09-01 → 1995-12-31 | 5,235 | Model learning |
| *Gap* | *1996-01-01 → 1996-03-31* | *91* | *Boundary buffer — excluded* |
| Val | 1996-04-01 → 1998-09-30 | 913 | Hyperparameter tuning, early stopping |
| *Gap* | *1998-10-01 → 1998-12-31* | *92* | *Boundary buffer — excluded* |
| Test | 1999-01-01 → 2000-12-31 | 731 | Final frozen evaluation only |

**Important:** the test set must not be used during model development. Only report test results once hyperparameters are finalised.

---

## What the preprocessing does

The raw NetCDF files go through three steps before reaching the model:

### 1. Climatology (`climatology` array)
For each calendar day-of-year (1–366), the average SST across all training years is computed per grid cell. This gives a `(366, 81, 121)` array representing the "expected" SST for any given day of year based on historical averages.

**Computed from training data only** — val and test years are never touched, so no future information leaks into this baseline.

### 2. Anomaly (`sst_anom` array)
The climatology is subtracted from the raw SST at every timestep:

```
sst_anom = sst - climatology[day_of_year]
```

This removes the seasonal cycle, leaving only departures from normal (e.g. El Niño warming, cold upwelling events). Models trained on anomalies learn to predict unusual behaviour rather than just reproducing the seasonal pattern.

### 3. Z-score normalisation (`sst_norm` array)
The anomalies are standardised using the mean and standard deviation computed from **training anomalies only**:

```
sst_norm = (sst_anom - train_mean) / train_std
```

Current stats: `mean = 0.0`, `std = 0.700°C`

This puts values in a roughly [-3, 3] range, which is better for neural network training than raw °C values.

**Land cells** remain as `NaN` throughout all three arrays. The dataset class fills them with `0.0` when loading — the model sees a consistent zero signal at every land cell and can learn to ignore it.

---

## Zarr store layout

The processed store at `data/processed/oisst_coralsea.zarr` contains:

| Array | Shape | dtype | Contents |
|---|---|---|---|
| `time` | `(7062,)` | int64 | Days since 1970-01-01 |
| `lat` | `(81,)` | float32 | Latitude values (−24.875 → −4.875) |
| `lon` | `(121,)` | float32 | Longitude values (140.125 → 170.125) |
| `sst` | `(7062, 81, 121)` | float32 | Raw SST in °C, NaN = land |
| `sst_anom` | `(7062, 81, 121)` | float32 | SST anomaly (sst − climatology), NaN = land |
| `sst_norm` | `(7062, 81, 121)` | float32 | Z-scored anomaly — **what models train on** |
| `climatology` | `(366, 81, 121)` | float32 | Mean SST per day-of-year (training years only) |
| `land_mask` | `(81, 121)` | bool | True = valid ocean cell |

Store attributes: `norm_mean`, `norm_std`, `train_start/end`, `val_start/end`, `test_start/end`, `T`, `H`, `W`.

---

## Dataset class

`SSTWindowDataset` wraps the Zarr store into a PyTorch sliding-window dataset. Each sample is a `(x, y)` pair:

- `x` — shape `(context_len, 1, H, W)` — the past `L` days of `sst_norm` fed to the model
- `y` — shape `(horizon, H, W)` — the next `h` days of `sst_norm` to predict

Windows slide forward one day at a time and are constrained to stay within the split boundaries (no window crosses a split).

Default settings: `context_len=90` days, `horizon=7` days.

| Split | Windows available |
|---|---|
| Train | 5,138 |
| Val | 817 |
| Test | 635 |

```python
from sst_forecasting.data.dataset import SSTWindowDataset

ds = SSTWindowDataset(
    "data/processed/oisst_coralsea.zarr",
    split="train",
    context_len=90,
    horizon=7,
)
x, y = ds[0]
# x: torch.Size([90, 1, 81, 121])
# y: torch.Size([7, 81, 121])
```

---

## Land masking in training

The `y` target returned by the dataset has land cells filled with `0.0`. **The training loss must mask these out** — otherwise the model is penalised for predicting non-zero values on land, wasting capacity.

Use the `land_mask` from the Zarr store:

```python
import zarr, torch
root = zarr.open_group("data/processed/oisst_coralsea.zarr", mode="r")
ocean_mask = torch.from_numpy(root["land_mask"][:]).to(device)  # (81, 121) bool

# inside training loop:
loss = F.mse_loss(pred[:, :, ocean_mask], target[:, :, ocean_mask])
```

Evaluation metrics (`rmse`, `mae`, `acc` in `sst_forecasting/utils/metrics.py`) already mask land automatically via `np.isfinite(truth)`.

---

## Troubleshooting

**Download stalls** — ERDDAP can take 2–3 min to prepare a year's file before streaming starts. The script retries automatically (3 attempts with exponential back-off).

**Resume interrupted download** — re-run `download_oisst.py`; files already on disk are skipped.

**Rebuild Zarr from scratch** — pass `--overwrite` to `build_zarr.py`.
