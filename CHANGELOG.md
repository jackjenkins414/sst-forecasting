# Changelog

All notable changes to this project will be documented here.

---

## [Unreleased]

### Added
- `PLAN.md` — comprehensive project plan covering research questions, dataset strategy (OISST 1981–2000 primary track A; Himawari-8 hourly stretch track B), model roster, Setonix/ROCm workflow with SLURM scripts, target repo structure, experiment matrix (E0–E8), 8-week timeline aligned with course deadlines (video 31 May, report 14 Jun), risks, and definition of done.
- `pyproject.toml` — package metadata, `sstf` CLI entry point, ruff + pytest config; `setuptools.build_meta` backend compatible with Python 3.12.
- `requirements.txt` — pinned dependencies: `torch`, `xarray`, `netCDF4`, `zarr<3`, `dask`, `scipy`, `hydra-core`, `omegaconf`, `matplotlib`, `cartopy`, `pytest`, `ruff`, `pre-commit`.
- `src/sst_forecasting/` — installable Python package (`pip install -e .`).
- **Data pipeline** (`src/sst_forecasting/data/`):
  - `splits.py` — canonical train / val / test date boundaries (train 1981-09 → 1995-12, val 1996 → 1998, test 1999 → 2000); `date_mask()` and `split_indices()` helpers; no split boundary is ever crossed by a window.
  - `transforms.py` — composable tensor transforms: `Standardize` (z-score with `.inverse()`), `FillLand` (NaN → constant), `SpatialPatchify` (non-overlapping patch tokenisation for Transformer / Informer inputs).
  - `download.py` — streams NOAA OISST v2.1 year-by-year from CoastWatch ERDDAP (`ncdcOisst21Agg_LonPM180`); Coral Sea default crop `[140°E–170°E] × [25°S–5°S]`; atomic write via `.tmp` rename; exponential-backoff retry; resumable (`skip_existing=True`).
  - `preprocess.py` — `build_zarr_store()`: loads yearly NetCDF files, computes DOY climatology and z-score normalisation **from training years only** (no leakage into val/test), writes Zarr 2 store containing `time`, `lat`, `lon`, `sst`, `sst_anom`, `sst_norm` (T×H×W float32), `climatology` (366×H×W), `land_mask` (H×W bool), and group attributes `norm_mean` / `norm_std` / split date strings.
  - `dataset.py` — `SSTWindowDataset`: sliding-window `torch.utils.data.Dataset` over `sst_norm`; each sample is `(x, y)` where `x ∈ ℝ^{L×1×H×W}` (context) and `y ∈ ℝ^{h×H×W}` (target); supports both Zarr path and in-memory numpy array (for unit tests); validates split contiguity; lazy Zarr access (no full array pre-load required).
- **Scripts**: `scripts/download_oisst.py` (CLI wrapper for `download_oisst()`), `scripts/build_zarr.py` (CLI wrapper for `build_zarr_store()`), `scripts/validate_pipeline.py` (6-stage end-to-end sanity check against real 1982 OISST data — ERDDAP reachability, download, raw NetCDF inspection, Zarr build, array/metadata validation, `SSTWindowDataset` window/shape/dtype checks; exits 0 on pass).
- **Hydra configs**: `configs/default.yaml` (root compose config), `configs/data/oisst_coralsea.yaml` (dataset + ERDDAP params), `configs/training/default.yaml` (Setonix single-GCD profile, bfloat16 AMP), `configs/training/debug.yaml` (1-epoch CPU smoke-test profile).
- **Tests** (`tests/`): `conftest.py` with session-scoped `tiny_zarr` (200 days × 16×16 synthetic Zarr, training-period timestamps) and `tiny_array` fixtures; `test_dataset.py` — 23 passing tests covering splits, dataset shapes/dtypes/NaN-freedom/window-sliding, error paths, and all three transforms.

### Changed
- `Context.md` — fully rewritten to consolidate all project knowledge previously split across `CLAUDE.md` and the old `Context.md`. Now includes: course deadlines, formalised RQ1–RQ3 with hypotheses H1–H3, models table with file paths and parameter budgets, experiment priority table (E0–E8), full evaluation metrics with bootstrap CI and skill-score formula, Setonix/ROCm gotchas and filesystem paths, quickstart commands for local and Setonix, repo conventions, and 8-week timeline with exit criteria. Corrected stale claim that OISST has sub-daily cadence — OISST is daily only; Himawari-8 sub-daily is Track B.
- `data/README.md` — expanded from stub to full team onboarding guide: 3-command quickstart, "What each step does" section (download → Zarr arrays table → validate 6-point list), dataset details table, splits table, Zarr-vs-NetCDF rationale, and troubleshooting section.
- `.gitignore` — added `data/raw/`, `data/processed/`, and `!data/README.md` exclusions so downloaded NetCDF and Zarr stores are never committed.

### Removed
- `CLAUDE.md` — deleted; all content merged into `Context.md`.

### Fixed
- `preprocess.py` `_compute_norm_stats()` — added graceful fallback (`std = 1.0` + WARNING log) when training anomaly std ≈ 0. Triggered only in single-year validation runs where DOY climatology equals the sole training year; the full 20-year dataset is unaffected.
- `validate_pipeline.py` — fixed `UnboundLocalError` for `ocean` variable (moved `land_mask` load outside the multi-year guard block); fixed norm-std skip condition to count unique years actually present in the Zarr time array rather than the split date range.

### Compute notes — 2 May 2026
GPU service units on both Pawsey Setonix (AMD MI250X) and NCI Gadi (NVIDIA A100) are running low for this quarter. **E0 baselines and the full data download/preprocessing pipeline must therefore be validated on Raijin (CPU cluster) first.**

Raijin node specs (5 nodes available):
- CPU: Intel Xeon E5-2670 — 2 sockets × 8 cores × 2 threads = **32 logical CPUs per node**
- NUMA: 2 nodes (node0: CPUs 0–7, 16–23; node1: CPUs 8–15, 24–31)
- Total usable: 5 nodes × 16 physical cores = **80 physical cores / 160 logical CPUs**
- No GPU, no ROCm/CUDA

Impact on workflow:
- `scripts/validate_pipeline.py` already runs CPU-only and passes.
- `configs/training/debug.yaml` uses `device: cpu` — use this config on Raijin.
- `torch.compile` must be disabled on Raijin (`compile: false` in config).
- Use `torch.set_num_threads(16)` (physical cores per node) and pin workers with `numactl --cpunodebind=0` / `--cpunodebind=1` to avoid NUMA cross-traffic.
- DataLoader `num_workers` ≤ 8 per NUMA node recommended.
- SLURM scripts for Raijin will target `--ntasks-per-node=1 --cpus-per-task=32`.
- Once GPU quota is refreshed or a small allocation is approved, E1–E3 GPU runs on Setonix proceed as planned.

### Data pipeline completed — 3 May 2026 ✓

Full 20-year OISST dataset downloaded, processed, and verified on local machine.

**Download** (`python scripts/download_oisst.py --output-dir data/raw`):
- 20 NetCDF files, 1981–2000, ~270 MB total on disk
- 2 ERDDAP read timeouts auto-recovered via retry (1983, 1991) — no data lost
- Total wall time: ~63 min (dominated by ERDDAP server-side prep per year)

**Zarr build** (`python scripts/build_zarr.py`):
- Completed in 21 seconds
- T=7,062 timesteps | H=81 | W=121 | train=5,234 days
- norm_mean=0.00000, norm_std=0.70023 (training years only, no leakage)
- Store size: 564 MB on disk (`data/processed/oisst_coralsea.zarr`)

**Validation** (`python scripts/validate_pipeline.py`) — all 6 stages passed ✓:
1. ERDDAP HEAD request → HTTP 200
2. 1982 download → 13.7 MB, non-empty .nc
3. Raw NetCDF: sst variable, lat/lon in crop window, 365 timesteps, SST 17.6–33.9°C, 80.5% ocean cells
4. Zarr build: store created, all 8 arrays present
5. Arrays/metadata: shapes, dtypes, H=81, W=121, all split attributes, climatology centre cell 26.68°C, land mask 80.5%
6. SSTWindowDataset: 269 windows, x=(90,1,81,121), y=(7,81,121), float32, no NaN, stride-1 overlap, DataLoader batch shapes correct

### Remaining Setonix GPU quota — 3 May 2026

~1 KSU remaining on Setonix for this quarter. Billing on GPU partition: ~96 SU/GCD-hour (1 GCD = half an MI250X card, minimum useful allocation). 1 KSU ≈ **10 hours of single-GCD time**.

Training estimates at our scale (5,138 windows, bfloat16, 1 GCD):
- LSTM / Transformer, bs=32: ~3–5 min/epoch → 50 epochs ≈ 2.5–4 h → fits in 1 KSU with room to spare
- Large batch (bs=64) roughly halves epoch time → even more headroom

**Plan for the remaining 1 KSU:**
- Reserve for E1 MVE runs only (LSTM seed 1 + Transformer seed 1 — ~6–8 h total)
- Do NOT use GPU quota on E0 baselines — they are pure numpy/CPU and run on Raijin
- Use `--gres=gpu:1` (single GCD) to minimise billing burn
- If a run approaches 8 h wall-time, checkpoint and resume rather than extend

### Next steps (priority order)
1. **E0 baselines** (P0 — due 10 May 2026): implement `src/sst_forecasting/models/baselines.py` (persistence, climatological mean, linear AR); evaluate at h ∈ {1, 7, 30} on test set; save to `experiments/results/baselines.json`. Run on Raijin CPU — zero GPU quota used.
2. **E1 MVE** (P0 — due 11 May 2026): LSTM vs Transformer at h=7, L=90, bs=64, single GCD on Setonix. Budget: ~6–8 KSU for both seeds combined — use the remaining ~1 KSU for seed 1 of LSTM; seed 2+ once quota refreshes or Raijin CPU fallback.
3. **Raijin SLURM scripts**: add `scripts/slurm/raijin_preprocess.sbatch` and `scripts/slurm/raijin_train_cpu.sbatch`.

## [0.1.0] — 2026-04-23

### Added
- Initial project scaffold: `src/`, `configs/`, `data/`, `experiments/`, `report/` directory structure
- Model stubs: `lstm.py`, `convlstm.py`, `informer.py`, `tft.py`
- Config files: `lstm.yaml`, `convlstm.yaml`, `informer.yaml`, `tft.yaml`
- Training and evaluation scripts: `train.py`, `evaluate.py`
- Utility modules: `metrics.py`, `visualisation.py`
- Data pipeline: `dataset.py`, `preprocess.py`
- `requirements.txt`
- `README.md` and `data/README.md`
- `Context.md` with full project overview, research question, models, datasets, evaluation plan, and repo structure
- Created branch `ayush`
