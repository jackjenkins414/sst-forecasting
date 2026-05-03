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

### Raijin setup — ordered steps (3 May 2026)

**On Raijin hpc-01 login node:**

```bash
# 1. Clone repo from GitHub (code only — data is gitignored)
git clone https://github.com/jackjenkins414/sst-forecasting.git
cd sst-forecasting
git checkout ayush

# 2. Create Python venv and install deps (CPU-only torch wheel, much smaller)
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .

# 3. Smoke test — confirm 23/23 tests pass
pytest -q

# 4. Copy the Zarr store from local Mac (run this on your Mac, not Raijin)
#    Raw NetCDF not needed — only the processed store is required for training
#    Uses the hpc-01 Host alias from ~/.ssh/config (ProxyJump already configured)
scp -r /Users/ayushsamuel/Downloads/Comp3242/Project/sst-forecasting/data/processed/oisst_coralsea.zarr \
    hpc-01:~/sst-forecasting/data/processed/oisst_coralsea.zarr

# 5. Verify Zarr loaded correctly on Raijin
python3 - <<'EOF'
import zarr
root = zarr.open("data/processed/oisst_coralsea.zarr", mode="r")
print("T =", root["sst"].shape[0], "H =", root["sst"].shape[1], "W =", root["sst"].shape[2])
print("norm_std =", root.attrs["norm_std"])
EOF
# Expected: T = 7062  H = 81  W = 121  norm_std = 0.70023

# 6. Run E0 baselines (once baselines.py is implemented)
python3 scripts/run_baselines.py \
    --zarr-path data/processed/oisst_coralsea.zarr \
    --horizons 1 7 30 \
    --output experiments/results/baselines.json
```

**Raijin CPU settings to apply before any training:**
- `device: cpu`, `compile: false` in config
- `torch.set_num_threads(16)` (physical cores per node, avoids hyperthreading overhead)
- `numactl --cpunodebind=0` or `--cpunodebind=1` per process to avoid NUMA cross-traffic
- DataLoader `num_workers=8` max per NUMA node
- SLURM: `--ntasks-per-node=1 --cpus-per-task=32`
- No `wandb` online mode — run `wandb offline` if wandb is used

### Data copy to hpc-01 — 3 May 2026 ✓

Zarr store successfully copied from local Mac to hpc-01 via `scp` through jumpbox (`u7508601@150.203.215.60`).

- Path: `~/sst-forecasting/data/processed/oisst_coralsea.zarr`
- Size on disk: 613 MB
- All 8 arrays present: `sst`, `sst_norm`, `sst_anom`, `climatology`, `land_mask`, `lat`, `lon`, `time`

### Python environment on hpc-01 — 3 May 2026 ✓

System `python3-venv` package was missing and apt was held by another sudo session, so used the existing miniconda installation instead of a venv.

- Conda env: `sst` (Python 3.12.3), at `/home/hpc/miniconda3/envs/sst`
- Activate with: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate sst`
- Installed: `torch==2.11.0+cpu` (from `download.pytorch.org/whl/cpu`) + all of `requirements.txt` + `pip install -e .`
- Smoke test: **`pytest` → 23 passed in 0.28 s ✓**

### Fixed
- `requirements.txt` line 9 — removed stray quotes around `zarr>=2.16,<3.0` that broke `pip install -r` parsing.

### Raijin infrastructure survey — 3 May 2026

Deep audit of head + compute nodes to fix the workflow constraints before any training run.

**Hardware (uniform across all 7 nodes hpc-01..07):**

| Component | Spec |
|---|---|
| CPU | Intel Xeon E5-2670 (Sandy Bridge) — 2 sockets × 8 cores × 2 threads = **16 phys / 32 log** |
| ISA | x86-64, AVX1, SSE4.2, AES — **NO AVX2, NO FMA, NO AVX-512** |
| Memory | hpc-01,02,04,05,07 = 64 GB; **hpc-03, hpc-06 = 128 GB** |
| NUMA | 2 nodes (node0: cores 0–7,16–23; node1: cores 8–15,24–31) |
| GPU | none |

**Network:**
- Mgmt Ethernet `192.168.2.0/24` — hpc-01 = `.101`, hpc-02 = `.52`, etc.
- InfiniBand `10.0.0.0/24`, hostname suffix `-ib` — Mellanox MT4099 (ConnectX-3 FDR), Active LinkUp on all nodes. Use IB IPs for any multi-node MPI / Gloo collectives.

**Software stack:**

| Stack | Version | Path / Notes |
|---|---|---|
| OS | Ubuntu 24.04 (kernel 6.8) | uniform |
| Intel oneAPI | 2026.0 | `/opt/intel/oneapi/setvars.sh` (NFS-shared **read-only** from hpc-01) |
| Intel MKL | 2026.0 | `MKLROOT=/opt/intel/oneapi/mkl/2026.0` |
| Intel ICX | 2026.0 | available after `setvars.sh` |
| Intel MPI | 2021.18 | `mpiicx`, `mpirun` from `$I_MPI_ROOT/bin` |
| GCC | 13.3 | system |
| OpenMPI / mpich | system `/usr/bin/mpirun` | fallback only |
| Python | 3.12.3 | system + miniconda env `sst` (only on hpc-01) |
| PyTorch | 2.11.0+cpu | conda env `sst` |
| `module` system | **not installed** | use `source /opt/intel/oneapi/setvars.sh` directly |

**Slurm:**
- 1 partition `batch` (default, infinite walltime), 6 idle worker nodes `hpc-[02-07]`.
- hpc-01 is the controller, **not in the worker pool** — but has identical hardware → can run "interactive" CPU jobs locally.
- No QoS / account constraints; no preemption.

**Filesystem (critical gap):**

| Path | Visible from | Mode |
|---|---|---|
| `/opt/intel` | all nodes | NFS ro from hpc-01 ✓ |
| `/home/hpc/sst-forecasting` | hpc-01 only | local ext4 — **NOT shared** |
| `/home/hpc/miniconda3` | hpc-01 only | local ext4 — **NOT shared** |

→ Slurm jobs on hpc-02..07 cannot see the repo, the venv, or the Zarr store. This **must** be solved before any `sbatch` submission. Two options:
1. **NFS-export `/home/hpc`** (rw, 192.168.2.0/24) following the existing `/opt/intel` pattern in `setup_nfs_clients.sh`. Mount on workers via `/etc/fstab`. *Preferred.*
2. **rsync stage-in per job** — slower, fragile, doubles disk usage.

### Constraints summary (apply to **every** training run on Raijin)

1. **CPU-only.** No GPU; PyTorch CPU build, MKL-backed (ATen oneDNN). `torch.compile = false` always.
2. **No AVX2.** Sandy Bridge SIMD is AVX1 only — many modern kernels fall back. Set `MKL_ENABLE_INSTRUCTIONS=AVX` to silence runtime warnings; verify torch eager-mode tensor ops do not crash.
3. **Hyperthreading harmful for compute-bound ML.** Always pin to 16 physical cores per node:
   - `--hint=nomultithread`, `--cpus-per-task=16`
   - `OMP_NUM_THREADS=16`, `MKL_NUM_THREADS=16`, `OPENBLAS_NUM_THREADS=16`, `NUMEXPR_NUM_THREADS=16`
   - `KMP_AFFINITY=granularity=fine,compact,1,0`
   - `torch.set_num_threads(16)`, `torch.set_num_interop_threads(2)`
4. **NUMA pinning.** For single-process: `numactl --cpunodebind=0 --membind=0`. For 2-process (rare): one per NUMA node.
5. **Node isolation.** `--exclusive` on every job — no MKL thread oversubscription from neighbours.
6. **Memory targeting.** Default to 64 GB nodes (`--nodelist=hpc-02,04,05,07`); reserve hpc-03, hpc-06 (128 GB) for big DL runs.
7. **Multi-node networking.** Only over IB (`hpc-NN-ib` / `10.0.0.x`). For PyTorch DDP CPU: `gloo` backend with `GLOO_SOCKET_IFNAME=ibp176s0`. Never use `192.168.2.x` for collectives.
8. **Walltime always set explicitly.** Default partition is infinite; we set `--time=01:00:00` (E0) up to `--time=08:00:00` (DL training) and `--signal=B:USR1@60` for graceful checkpointing.
9. **Reproducibility.** Every run captures git SHA, hostname, slurm jobid, full env, package versions, seed → `run.yaml` next to `metrics.json`.

### Raijin workflow plan — phased rollout

**Phase 1 — E0 baselines (today, 3 May 2026):**
- Run on **hpc-01 directly** (same hardware, has data + venv, not in Slurm pool → no NFS work needed).
- Pure numpy + scipy ridge regression; no PyTorch GPU paths exercised.
- Wall-time target ≤ 15 min for h ∈ {1, 7, 30}.
- Output: `experiments/results/baselines.json` + `experiments/results/baselines_run.yaml`.

**Phase 2 — NFS + Slurm template parity (next):**
- Add `/home/hpc → 192.168.2.0/24(rw,async,no_subtree_check,no_root_squash)` to hpc-01 `/etc/exports`.
- Mount on hpc-02..07 (`/etc/fstab` entry following `setup_nfs_clients.sh` pattern).
- Validate by re-running E0 via `sbatch scripts/slurm/raijin_baselines.sbatch` on hpc-02 — bit-exact metrics expected.
- Same SLURM template (with model-name override) becomes the basis for E1+.

**Phase 3 — E1 MVE single-node CPU (LSTM vs Transformer at h=7):**
- 1 node × 1 task × 16 threads; bs=16 (memory headroom for 2-layer LSTM with hidden=256).
- Use `hpc-03` or `hpc-06` (128 GB) to avoid OOM during initial profiling; drop to 64-GB nodes once footprint confirmed.

**Phase 4 — multi-node DDP (only if E3 demands more compute):**
- `torchrun` with `--rdzv-backend=c10d --rdzv-endpoint=hpc-XX-ib:29500`.
- Backend `gloo` over IB; world_size = nodes (1 process per node, NOT 1 per core).

### Slurm batch script safeguards (`scripts/slurm/raijin_baselines.sbatch`)

Every batch script in this project follows the same template, encoded in the E0 script as the canonical reference:

| Concern | Implementation |
|---|---|
| Node isolation | `#SBATCH --exclusive --nodes=1 --ntasks=1 --cpus-per-task=16 --hint=nomultithread` |
| Wall-time guard | `#SBATCH --time=01:00:00` + `#SBATCH --signal=B:USR1@60` for graceful exit |
| Memory limit | `#SBATCH --mem=0` (whole node, exclusive anyway) |
| Auto-requeue | `#SBATCH --requeue` + write `.requeue_count` to detect runaway loops |
| Output isolation | `#SBATCH --output=experiments/results/%x-%j/slurm.out --error=...slurm.err` |
| Provenance | `git rev-parse HEAD`, `hostname`, `$SLURM_JOB_ID`, `pip freeze`, full env dump → `run.yaml` |
| Thread pinning | `OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS / KMP_AFFINITY` exported |
| NUMA binding | `numactl --cpunodebind=0 --membind=0 python …` |
| Fail-fast | `set -euo pipefail`; `srun --kill-on-bad-exit=1` |
| Determinism | `PYTHONHASHSEED`, `NUMPY_SEED` exported; passed to script via `--seed` |
| Result pollution | Output dir is per-jobid (`%x-%j`); never overwritten; symlink `latest/` for convenience |
| AVX2 fallback noise | `MKL_ENABLE_INSTRUCTIONS=AVX` |

### Next steps (priority order)
1. ~~**E0 baselines**~~ ✓ 3 May 2026
2. ~~**NFS + sbatch validation**~~ ✓ 3 May 2026
3. ~~**E1 MVE — LSTM vs Transformer at h=7**~~ ✓ 3 May 2026 — see results below
4. **E1 multi-seed** (P0): re-run both models with seeds 0/1/2 for reportable mean ± std; diagnose Transformer underperformance (try LR=1e-4 + warmup, patch tokenisation).
5. **E2 — GRU + ConvLSTM** (P1): extend model roster at h=7.
6. **E3 — horizon sweep** (P1): all models at h ∈ {1, 7, 30}.
7. **Models scaffolding**: drop stale empty top-level `src/` directories.

### E1 MVE results — 3 May 2026 ✓

LSTM (Slurm 117) and Transformer (Slurm 119) trained at h=7, L=90, seed=42 on hpc-03 (128 GB, NFS).

#### h=7 days — mean RMSE °C over forecast steps 1–7

| Model | RMSE °C | SS vs persistence | Beats LinearAR? | Wall time | Params |
|---|---|---|---|---|---|
| Persistence (E0) | 0.6959 | 0.000 | — | — | 0 |
| Climatology (E0) | 0.7933 | −0.140 | no | — | 0 |
| LinearAR-30 (E0) | 0.6292 | +0.096 | — | — | <1 k |
| **LSTM** | **0.6138** | **+0.118** | **yes (+0.015 °C)** | 1.86 h | 9.71 M |
| Transformer | 0.6994 | −0.005 | no (−0.070 °C) | 0.50 h | 10.63 M |

**LSTM per-step RMSE (days 1→7):** 0.595, 0.601, 0.608, 0.614, 0.620, 0.626, 0.633 °C  
**Transformer per-step RMSE (days 1→7):** 0.695, 0.697, 0.700, 0.700, 0.700, 0.701, 0.703 °C

Key findings:
- **LSTM beats all baselines** including LinearAR (SS=+0.118). H1 partially supported.
- **Transformer fails to beat persistence** (SS=−0.005). Likely cause: 9801-dim flattened spatial input overwhelms attention; LR=1e-3 too aggressive. Transformer trains 3.7× faster (parallel vs sequential).
- Both ran 16 epochs; early stopping not triggered. Best val epoch: LSTM 0.627 °C, Transformer 0.713 °C.

### NFS export + worker node setup — 3 May 2026 ✓

`/home/hpc` (containing the repo, conda env, and Zarr store) is now NFS-exported
from hpc-01 to all six compute nodes.

**On hpc-01 (NFS server):**
- Added to `/etc/exports`: `/home/hpc 192.168.2.0/24(rw,async,no_subtree_check,no_root_squash)`
- Applied with `exportfs -ra` (no service restart needed — NFS server was already running)

**On hpc-02..07 (clients):**
- Installed `nfs-common` on hpc-06 (missing; all others already had it)
- Added fstab entry: `hpc-01:/home/hpc /home/hpc nfs rw,async,defaults,_netdev 0 0`
- Mounted with `mount -a` — all 6 nodes confirmed ✓

**Validation:** `ssh hpc-0N "ls /home/hpc/sst-forecasting/data/processed/oisst_coralsea.zarr/.zgroup"` → zarr ✓ on all 6 nodes.

**sbatch bugs fixed in `raijin_baselines.sbatch`:**
1. `source setvars.sh` with `set -eu` — `setvars.sh` calls `exit` internally and would kill the parent shell. Fixed with the subshell+env pattern from `run_e0_local.sh`.
2. `$KMP_AFFINITY` referenced after `unset KMP_AFFINITY` with `set -u` → unbound variable. Fixed with `${KMP_AFFINITY:-}`.

**sbatch E0 parity run (job 116, hpc-02) — bit-exact ✓:**

Results on hpc-02 via NFS are identical to hpc-01 local run:

| Model | h=1 | h=7 | h=30 |
|---|---|---|---|
| Persistence | 0.3170 °C | 0.6959 °C | 1.0781 °C |
| Climatology | 0.7946 °C | 0.7933 °C | 0.7884 °C |
| LinearAR(30) | 0.2993 °C | 0.6292 °C | 0.9245 °C |

NFS setup and sbatch workflow are fully validated. ✓

---

### E1 MVE implementation — 3 May 2026

E1 experiment files added. Both models operate in **normalised SST-anomaly space**;
RMSE is converted to °C by multiplying by `norm_std = 0.70023`.

#### Architecture: `SpatialFlatLSTM` (`src/sst_forecasting/models/lstm.py`)

```
(B, L, 1, H, W)
  → flatten spatial       (B, L, H*W = 9801)
  → Linear(9801 → 64) + ReLU
  → LSTM(64, hidden=128, layers=2)   [last hidden state]
  → Dropout(0.1)
  → Linear(128 → 7×9801)
  → reshape               (B, 7, H=81, W=121)
```
Parameters: ~9.7 M.  Estimated training time: ~20 min / 50 epochs on 16-core Raijin node.

#### Architecture: `SpatialFlatTransformer` (`src/sst_forecasting/models/transformer.py`)

```
(B, L, 1, H, W)
  → flatten spatial       (B, L, H*W = 9801)
  → Linear(9801 → 128) + ReLU
  → SinusoidalPE(d=128, L=90)
  → TransformerEncoder(4 layers, 8 heads, ffn=256)
  → mean-pool over L      (B, 128)
  → Linear(128 → 7×9801)
  → reshape               (B, 7, H=81, W=121)
```
Parameters: ~10.6 M.  Slightly slower than LSTM per epoch due to attention O(L²).

#### Training script: `scripts/train_e1.py`
- Plain argparse (no Hydra) for direct `sbatch` submission
- Loss: MSE over ocean cells only (land cells masked via zarr `land_mask`)
- Optimiser: Adam(lr=1e-3, weight_decay=1e-4) + ReduceLROnPlateau(patience=5, factor=0.5)
- Early stopping: patience=10 epochs on val MSE
- Outputs: `best_model.pt`, `last_model.pt`, `metrics.json`, `run.yaml`, `training_log.csv`
- Smoke-tested: 2 epochs on 64 training windows → LSTM and Transformer both converge ✓

#### SLURM: `scripts/slurm/raijin_e1.sbatch`
- Default: LSTM on hpc-03 (128 GB, `--nodelist=hpc-03`)
- Override: `sbatch --export=ALL,MODEL=transformer raijin_e1.sbatch`
- `--time=08:00:00`, `--signal=B:USR1@120` for graceful checkpoint on timeout
- Same threading + NUMA config as E0 baseline script

#### New configs
- `configs/lstm.yaml` — LSTM hyperparameters (d_spatial=64, hidden=128, layers=2)
- `configs/transformer.yaml` — Transformer hyperparameters (d_model=128, nhead=8, layers=4)
- `configs/training/raijin.yaml` — CPU training profile (bs=16, no AMP, compile=false)

#### Tests: `tests/test_models_forward.py`
17 tests covering:
- Output shape `(B, h, H, W)` for both models
- float32 dtype, no NaN, gradient flow, eval-mode determinism
- Batch independence (no batch-norm leakage)
- Positional encoding wired in (Transformer)
- Cross-architecture I/O interface parity

**Full test suite: 52/52 passed ✓**



Run: `bash scripts/run_e0_local.sh` on hpc-01 (Xeon E5-2670, 16 phys cores, 64 GB RAM).  
Wall time: **58.5 s** total for all 3 horizons × 3 models × 1000 bootstrap resamples.  
Output: `experiments/results/e0_local/baselines.json` + `run.yaml`.  
git SHA: `7fa29a7e` · seed: 42 · ar\_context: 30 days · test split: 731 days (1999-01-01 – 2000-12-31)

#### Results — RMSE °C [95% CI] / ACC / skill vs persistence

| Model | h=1 d | h=7 d | h=30 d |
|---|---|---|---|
| **Persistence** | 0.3170 [0.3121, 0.3220] / ACC=0.908 | 0.6959 [0.6872, 0.7055] / ACC=0.567 | 1.0781 [1.0590, 1.0967] / ACC=0.255 |
| **Climatology** | 0.7946 [0.7833, 0.8061] / ACC=n/a · SS=−1.507 | 0.7933 [0.7820, 0.8068] / ACC=n/a · SS=−0.140 | 0.7884 [0.7766, 0.8012] / ACC=n/a · SS=**+0.269** |
| **LinearAR(30)** | 0.2993 [0.2945, 0.3042] / ACC=0.916 · SS=**+0.056** | 0.6292 [0.6186, 0.6402] / ACC=0.587 · SS=**+0.096** | 0.9245 [0.9065, 0.9408] / ACC=0.237 · SS=**+0.142** |

Key observations:
- Persistence dominates at h=1 (RMSE 0.317 °C); LinearAR beats it by only 5.6%.
- LinearAR gives meaningful gains at h=7 (+9.6%) and h=30 (+14.2%) vs persistence.
- Climatology beats persistence only at h=30 (SS=+0.269), confirming seasonal signal dominates at long range.
- Climatology ACC=n/a is expected — anomalies relative to itself are identically 0, denominator undefined.

#### Performance optimisations applied (to reach 58.5 s)

| Change | Before | After |
|---|---|---|
| `LinearAR.fit` XtX: `np.einsum` → `np.matmul` (MKL GEMM) | 73.5 s | 15.3 s |
| Predictions: per-origin Python loop → `predict_batch` (one GEMM) | ~730 serial calls | 1.05 s |
| Bootstrap RMSE/MAE: per-window pre-reduction → `(n_boot, N)` index resampling | slow loop | ~instant |
| Bootstrap ACC: Python loop → algebraic decomposition into 5 per-window scalars | hung >5 min | ~instant |
| NUMA: `--cpunodebind=0` (8 cores) → `--interleave=all` (both sockets, 16 cores) | 1 socket | both |
| Thread binding: `KMP_AFFINITY=compact` → `OMP_PLACES=cores OMP_PROC_BIND=close` | KMP override | portable |



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
