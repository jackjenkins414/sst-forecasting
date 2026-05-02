# Data

This directory is **not committed to git** (`data/raw/` and `data/processed/` are in `.gitignore`).
All team members generate it locally from the same public NOAA source — takes ~5–10 min and ~480 MB disk.

---

## Quickstart (run once per machine)

```bash
# 1. Download 20 years of OISST v2.1 from NOAA CoastWatch ERDDAP (~280 MB, ~5–10 min)
python scripts/download_oisst.py --output-dir data/raw

# 2. Build the Zarr store with climatology, anomalies, and z-normalisation (~200 MB, ~1 min)
python scripts/build_zarr.py

# 3. Verify everything is correct (downloads 1 year, runs all checks, cleans up)
python scripts/validate_pipeline.py
```

After step 2 you will have:

```
data/
  raw/
    oisst_v21_1981.nc  … oisst_v21_2000.nc   (20 files, ~280 MB total)
  processed/
    oisst_coralsea.zarr/                       (~200 MB)
```

---

## What each step does

### Step 1 — Download (`download_oisst.py`)

Fetches daily SST NetCDF files year-by-year from the NOAA CoastWatch ERDDAP server and saves them to `data/raw/`.
Each file covers one calendar year cropped to the Coral Sea region.
Already-downloaded files are skipped on re-run, so it is safe to interrupt and resume.

### Step 2 — Build Zarr store (`build_zarr.py`)

Converts the raw NetCDF files into a single **Zarr store** at `data/processed/oisst_coralsea.zarr`.

Zarr is a chunked, compressed array format optimised for ML training — unlike NetCDF, it serves any arbitrary time-slice in milliseconds without scanning the whole file. The store contains:

| Array | Shape | Contents |
|---|---|---|
| `sst` | `(T, 81, 121)` | Raw SST in °C; NaN = land |
| `sst_anom` | `(T, 81, 121)` | SST minus the day-of-year climatology |
| `sst_norm` | `(T, 81, 121)` | Z-scored anomaly — **what the models train on** |
| `climatology` | `(366, 81, 121)` | Mean SST per day-of-year (training years only) |
| `land_mask` | `(81, 121)` | `True` = valid ocean cell |

The climatology and z-normalisation statistics are computed from **training years only** so no val/test information leaks into the inputs.

### Step 3 — Validate (`validate_pipeline.py`)

An end-to-end sanity check that runs ~30 seconds. It:

1. Sends a HEAD request to ERDDAP to confirm the server is reachable
2. Downloads one year of real data (~14 MB) to a temp directory
3. Inspects the raw NetCDF — variable names, lat/lon extent, time coverage, SST value range
4. Builds a mini Zarr store and checks all array shapes, dtypes, and metadata attributes
5. Confirms the grid is exactly H=81 × W=121 at 0.25° spacing
6. Loads `SSTWindowDataset` and checks tensor shapes, dtypes, no NaN, correct sliding-window stride, and DataLoader batching

Exits 0 if all checks pass. Run this after cloning on a new machine before starting the full download.

---

## Dataset details

| Property | Value |
|---|---|
| Source | NOAA OISST v2.1, CoastWatch ERDDAP |
| Dataset ID | `ncdcOisst21Agg_LonPM180` |
| Variable | `sst` (°C, NaN = land) |
| Resolution | 0.25° daily |
| Period | 1981-09-01 → 2000-12-31 |
| Spatial crop | 140°E–170°E × 25°S–5°S (Coral Sea) |
| Grid size | H=81, W=121 |

**Splits (no leakage):**

| Split | Period |
|---|---|
| Train | 1981-09-01 → 1995-12-31 |
| Val | 1996-01-01 → 1998-12-31 |
| Test | 1999-01-01 → 2000-12-31 (frozen) |

Climatology and normalisation statistics are computed from **training years only**.

---

## Troubleshooting

**Download stalls or times out** — ERDDAP sometimes takes 2–3 min to prepare a full-year file server-side before streaming starts. This is normal. The script retries automatically (3 attempts with exponential back-off).

**Resume interrupted download** — just re-run `download_oisst.py`; files already on disk are skipped (`--skip-existing` is on by default).

**Rebuild Zarr from scratch** — pass `--overwrite` to `build_zarr.py`.
