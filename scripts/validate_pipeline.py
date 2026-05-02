"""End-to-end pipeline validation using a 1-year mini dataset (1982).

Downloads one year of OISST (~14 MB for the Coral Sea crop), builds the Zarr
store, then validates every requirement from PLAN.md §4 and §5 before you
commit to the full 20-year pull.

Usage
-----
    python scripts/validate_pipeline.py [--keep-raw] [--log-level DEBUG]

Options
-------
    --keep-raw      Keep the downloaded NetCDF file after validation.
    --output-dir    Base dir for raw/ and processed/ sub-dirs  [default: /tmp/sst_validate]
    --log-level     Logging verbosity  [default: INFO]

Exit code
---------
    0 — all checks passed
    1 — one or more checks failed (details printed to stdout)
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
import zarr

from sst_forecasting.data.dataset import SSTWindowDataset
from sst_forecasting.data.download import build_erddap_url, download_oisst
from sst_forecasting.data.preprocess import build_zarr_store, load_zarr_metadata
from sst_forecasting.data.splits import SPLITS

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

_failures: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        print(f"  {PASS} {label}")
    else:
        msg = f"{label}" + (f": {detail}" if detail else "")
        print(f"  {FAIL} {msg}")
        _failures.append(msg)


# ─────────────────────────────────────────────────────────────────────────────
# 1. URL smoke-test  (HEAD request, no download)
# ─────────────────────────────────────────────────────────────────────────────

def validate_url() -> None:
    import requests

    print("\n[1] ERDDAP URL smoke-test")
    url = build_erddap_url(
        dataset_id="ncdcOisst21Agg_LonPM180",
        variable="sst",
        start="1982-01-01",
        end="1982-01-01",
        lat_min=-25.0, lat_max=-5.0,
        lon_min=140.0, lon_max=170.0,
    )
    log.debug("Test URL: %s", url)
    try:
        r = requests.head(url, timeout=30, allow_redirects=True)
        check(r.status_code == 200,
              f"HEAD {url[:80]}… → HTTP {r.status_code}",
              f"Expected 200, got {r.status_code}")
    except requests.RequestException as exc:
        check(False, "ERDDAP reachable", str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Download 1982
# ─────────────────────────────────────────────────────────────────────────────

def validate_download(raw_dir: Path) -> list[Path]:
    print("\n[2] Download 1982 (Coral Sea crop)")
    paths = download_oisst(
        output_dir=raw_dir,
        start_year=1982,
        end_year=1982,
    )
    check(len(paths) == 1, "Exactly 1 NetCDF file downloaded")
    if paths:
        size_mb = paths[0].stat().st_size / 1024 ** 2
        check(size_mb > 0.1, f"File is non-empty ({size_mb:.1f} MB)")
        check(paths[0].suffix == ".nc", "File extension is .nc")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# 3. Inspect raw NetCDF
# ─────────────────────────────────────────────────────────────────────────────

def validate_netcdf(nc_path: Path) -> None:
    import xarray as xr

    print("\n[3] Raw NetCDF structure")
    ds = xr.open_dataset(nc_path, engine="netcdf4")
    log.debug("Variables: %s", list(ds.data_vars))
    log.debug("Coords: %s", list(ds.coords))
    log.debug("Dims: %s", dict(ds.dims))

    check("sst" in ds.data_vars, "Variable 'sst' present in NetCDF")

    has_lat  = "latitude" in ds.coords or "lat" in ds.coords
    has_lon  = "longitude" in ds.coords or "lon" in ds.coords
    has_time = "time" in ds.coords
    check(has_lat,  "Latitude coordinate present")
    check(has_lon,  "Longitude coordinate present")
    check(has_time, "Time coordinate present")

    lat = ds.coords.get("latitude", ds.coords.get("lat"))
    lon = ds.coords.get("longitude", ds.coords.get("lon"))
    if lat is not None and lon is not None:
        check(float(lat.min()) >= -25.5 and float(lat.max()) <= -4.5,
              f"Lat range within Coral Sea crop (-25 to -5): [{float(lat.min()):.2f}, {float(lat.max()):.2f}]")
        check(float(lon.min()) >= 139.5 and float(lon.max()) <= 170.5,
              f"Lon range within Coral Sea crop (140 to 170): [{float(lon.min()):.2f}, {float(lon.max()):.2f}]")

    time_pd = pd.DatetimeIndex(ds.time.values)
    check(len(time_pd) == 365 or len(time_pd) == 366,
          f"Time dim covers 1 year (365/366 days): got {len(time_pd)}")
    check(time_pd[0].year == 1982 and time_pd[-1].year == 1982,
          f"All timesteps in 1982: [{time_pd[0].date()}, {time_pd[-1].date()}]")

    sst_vals = ds["sst"].values
    ocean_frac = np.mean(~np.isnan(sst_vals))
    check(ocean_frac > 0.5, f"SST has >50% valid (ocean) cells: {ocean_frac:.1%}")
    check(np.nanmin(sst_vals) > 0 and np.nanmax(sst_vals) < 45,
          f"SST values in realistic range (0–45°C): [{np.nanmin(sst_vals):.1f}, {np.nanmax(sst_vals):.1f}]")

    ds.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Build Zarr store
# ─────────────────────────────────────────────────────────────────────────────

def validate_zarr_build(raw_dir: Path, zarr_path: Path) -> None:
    print("\n[4] Build Zarr store")
    build_zarr_store(raw_dir, zarr_path, overwrite=True)
    check(zarr_path.exists(), f"Zarr store created at {zarr_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Validate Zarr store contents against PLAN.md requirements
# ─────────────────────────────────────────────────────────────────────────────

def validate_zarr_contents(zarr_path: Path) -> None:
    print("\n[5] Zarr store contents and requirements (PLAN.md §4 / §5)")
    root = zarr.open_group(str(zarr_path), mode="r")

    # ── Array existence ───────────────────────────────────────────────────────
    for arr in ("time", "lat", "lon", "sst", "sst_anom", "sst_norm",
                "climatology", "land_mask"):
        check(arr in root, f"Array '{arr}' present in Zarr store")

    T = root["sst"].shape[0]
    H = root["sst"].shape[1]
    W = root["sst"].shape[2]

    # ── Shapes ────────────────────────────────────────────────────────────────
    check(root["sst"].shape       == (T, H, W), f"sst shape ({T},{H},{W})")
    check(root["sst_anom"].shape  == (T, H, W), f"sst_anom shape ({T},{H},{W})")
    check(root["sst_norm"].shape  == (T, H, W), f"sst_norm shape ({T},{H},{W})")
    check(root["climatology"].shape == (366, H, W), f"climatology shape (366,{H},{W})")
    check(root["land_mask"].shape == (H, W),    f"land_mask shape ({H},{W})")

    # ── dtypes ────────────────────────────────────────────────────────────────
    check(root["sst"].dtype      == np.float32, "sst dtype float32")
    check(root["sst_norm"].dtype == np.float32, "sst_norm dtype float32")
    check(root["time"].dtype     == np.int64,   "time dtype int64")
    check(root["land_mask"].dtype == np.dtype("bool"), "land_mask dtype bool")

    # ── Spatial extent ────────────────────────────────────────────────────────
    lat = root["lat"][:]
    lon = root["lon"][:]
    check(float(lat.min()) >= -25.5 and float(lat.max()) <= -4.5,
          f"lat range within Coral Sea [-25,-5]: [{lat.min():.2f},{lat.max():.2f}]")
    check(float(lon.min()) >= 139.5 and float(lon.max()) <= 170.5,
          f"lon range within Coral Sea [140,170]: [{lon.min():.2f},{lon.max():.2f}]")
    expected_H = round(((-5) - (-25)) / 0.25) + 1  # 81 cells
    expected_W = round((170 - 140) / 0.25) + 1      # 121 cells
    check(H == expected_H, f"H={H} == {expected_H} (0.25° spacing, 25°S–5°S)")
    check(W == expected_W, f"W={W} == {expected_W} (0.25° spacing, 140°E–170°E)")

    # ── Time ─────────────────────────────────────────────────────────────────
    epoch = pd.Timestamp("1970-01-01")
    time_pd = pd.DatetimeIndex([epoch + pd.Timedelta(days=int(d)) for d in root["time"][:]])
    check(len(time_pd) >= 365, f"At least 365 timesteps: {len(time_pd)}")
    check(time_pd[0].year == 1982, f"First timestep in 1982: {time_pd[0].date()}")

    # ── Metadata attributes ───────────────────────────────────────────────────
    attrs = dict(root.attrs)
    for key in ("norm_mean", "norm_std", "train_start", "train_end",
                "val_start", "val_end", "test_start", "test_end", "T", "H", "W"):
        check(key in attrs, f"Attribute '{key}' in store metadata")
    if "train_start" in attrs:
        check(attrs["train_start"] == SPLITS["train"][0],
              f"train_start matches split: {attrs['train_start']}")

    # ── No-leakage check: norm stats computed from training timesteps only ────
    # NOTE: With a single training year the DOY climatology equals the training
    # data exactly, so sst_anom ≡ 0.  preprocess.py falls back to std=1.0 in
    # that case.  Skip the sst_norm distribution checks here — they only make
    # sense with ≥2 loaded training years.
    ocean = root["land_mask"][:]
    epoch = pd.Timestamp("1970-01-01")
    time_pd_loaded = pd.DatetimeIndex(
        [epoch + pd.Timedelta(days=int(d)) for d in root["time"][:]]
    )
    unique_loaded_years = len(set(t.year for t in time_pd_loaded))
    _note = " (skipped: only 1 training year in mini-test)"
    if unique_loaded_years >= 2:
        if "norm_std" in attrs:
            check(attrs["norm_std"] > 0, f"norm_std > 0: {attrs.get('norm_std'):.5f}")
        sst_norm = root["sst_norm"][:]
        norm_vals = sst_norm[:, ocean]
        valid = norm_vals[~np.isnan(norm_vals)]
        if len(valid) > 0:
            check(abs(float(np.mean(valid))) < 0.5,
                  f"sst_norm mean ≈ 0 over training data (got {np.mean(valid):.4f})")
            check(0.5 < float(np.std(valid)) < 2.0,
                  f"sst_norm std ≈ 1 over training data (got {np.std(valid):.4f})")
    else:
        print(f"  (skipping norm-std checks{_note})")

    # ── Anomaly sanity: mean of climatology over DOY for a single cell ────────
    clim = root["climatology"][:]  # (366, H, W)
    mid_lat, mid_lon = H // 2, W // 2
    if ocean[mid_lat, mid_lon]:
        clim_cell = clim[:, mid_lat, mid_lon]
        check(np.all(np.isfinite(clim_cell)),
              f"Climatology finite at centre cell ({lat[mid_lat]:.2f}°, {lon[mid_lon]:.2f}°)")
        check(5 < float(np.mean(clim_cell)) < 35,
              f"Climatology centre cell mean in range 5–35°C: {np.mean(clim_cell):.2f}°C")

    # ── Land mask ─────────────────────────────────────────────────────────────
    ocean_frac = ocean.mean()
    check(0.5 < ocean_frac < 1.0,
          f"Land mask ocean fraction between 50–100%: {ocean_frac:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Validate SSTWindowDataset
# ─────────────────────────────────────────────────────────────────────────────

def validate_dataset(zarr_path: Path) -> None:
    print("\n[6] SSTWindowDataset (PLAN.md §5 tensor contract)")
    L, h = 90, 7

    # All 1982 data is in the training split — use split="train"
    try:
        ds = SSTWindowDataset(str(zarr_path), split="train", context_len=L, horizon=h)
    except Exception as exc:
        check(False, f"SSTWindowDataset construction: {exc}")
        return

    check(len(ds) > 0, f"Dataset has {len(ds)} windows (context={L}, horizon={h})")

    x, y = ds[0]
    H, W = ds.spatial_shape

    check(isinstance(x, torch.Tensor), "x is a torch.Tensor")
    check(isinstance(y, torch.Tensor), "y is a torch.Tensor")
    check(x.shape == torch.Size([L, 1, H, W]),
          f"x shape == (L=90, C=1, H={H}, W={W}): got {tuple(x.shape)}")
    check(y.shape == torch.Size([h, H, W]),
          f"y shape == (h=7, H={H}, W={W}): got {tuple(y.shape)}")
    check(x.dtype == torch.float32, f"x dtype float32: {x.dtype}")
    check(y.dtype == torch.float32, f"y dtype float32: {y.dtype}")
    check(not torch.isnan(x).any(), "x contains no NaN (land filled)")
    check(not torch.isnan(y).any(), "y contains no NaN (land filled)")

    # Consecutive windows slide by exactly 1 timestep
    x1, _ = ds[1]
    check(torch.allclose(x[1:], x1[:-1]),
          "Consecutive windows overlap by L-1 timesteps (correct stride=1)")

    # DataLoader round-trip
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, num_workers=0)
    batch_x, batch_y = next(iter(loader))
    check(batch_x.shape == torch.Size([4, L, 1, H, W]),
          f"DataLoader batch x shape (4,90,1,{H},{W}): got {tuple(batch_x.shape)}")
    check(batch_y.shape == torch.Size([4, h, H, W]),
          f"DataLoader batch y shape (4,7,{H},{W}): got {tuple(batch_y.shape)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end pipeline validation on a 1-year mini dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", default="/tmp/sst_validate", metavar="PATH")
    p.add_argument("--keep-raw",   action="store_true",
                   help="Keep the downloaded NetCDF file after validation.")
    p.add_argument("--log-level",  default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    base = Path(args.output_dir)
    raw_dir   = base / "raw"
    zarr_path = base / "processed" / "oisst_1982_test.zarr"

    print("=" * 60)
    print("SST Forecasting — Pipeline Validation (1982 mini dataset)")
    print("=" * 60)

    validate_url()
    nc_paths = validate_download(raw_dir)
    if nc_paths:
        validate_netcdf(nc_paths[0])
    validate_zarr_build(raw_dir, zarr_path)
    validate_zarr_contents(zarr_path)
    validate_dataset(zarr_path)

    if not args.keep_raw:
        shutil.rmtree(base, ignore_errors=True)
        log.info("Cleaned up temp dir %s", base)

    print("\n" + "=" * 60)
    if _failures:
        print(f"\033[91mFAILED — {len(_failures)} check(s) failed:\033[0m")
        for f in _failures:
            print(f"  • {f}")
        sys.exit(1)
    else:
        print("\033[92mAll checks passed — pipeline is ready.\033[0m")
        print("\nTo run the full download:")
        print("  python scripts/download_oisst.py --output-dir data/raw")
        print("  python scripts/build_zarr.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
