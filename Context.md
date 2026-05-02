# Project Context

## Title
Ocean Temperature Forecasting: Recurrent vs Attention-Based Architectures (COMP3242)

## Overview
Multi-step sea surface temperature (SST) forecasting using deep learning, comparing recurrent architectures (LSTM, GRU, ConvLSTM) against attention-based models (Transformer encoder, Informer, TFT) across varying prediction horizons (1-day, 7-day, 30-day). Evaluates how each architecture captures temporal dependencies in ocean dynamics.

**Course:** COMP3242 Deep Learning, Semester 1, 2026
**Deliverables:**
- Video presentation — 31 May 2026 (16%)
- Final report (4–8 pp) + source code — 14 Jun 2026 (24%)

P0 experiments must be complete by **10 May 2026** so the video has real numbers.

---

## Research Questions

- **RQ1.** At which horizon $h \in \{1, 7, 30\}$ days do attention models outperform recurrent models?
- **RQ2.** How does context length $L \in \{30, 90, 180\}$ days interact with architecture?
- **RQ3.** Does ConvLSTM's spatial bias beat purely temporal models at matched parameter budgets?

**Hypotheses:**
- H1: Recurrent models match or beat attention at h=1 (strong locality, small effective context).
- H2: Attention dominates at h=30 (long-range dependencies, teleconnections).
- H3: ConvLSTM beats both on per-cell RMSE in high-variability regions, but not basin-mean.

---

## Models

| Model | File | Params |
|---|---|---|
| Persistence / Climatology / Linear AR | `src/sst_forecasting/models/baselines.py` | 0 / <1 k |
| Stacked LSTM + GRU | `src/sst_forecasting/models/lstm.py` | ~1–3 M |
| ConvLSTM | `src/sst_forecasting/models/convlstm.py` | ~1–3 M |
| Transformer encoder | `src/sst_forecasting/models/transformer.py` | ~1–3 M |
| Informer | `src/sst_forecasting/models/informer.py` | ~1–3 M |
| TFT *(stretch)* | `src/sst_forecasting/models/tft.py` | ~1–3 M |

All models matched to ±20% parameter count and equal GPU wall-clock budget.
**Every deep model must beat persistence, climatology, and Linear AR at h=1 — failure means a bug.**

All models take input `x ∈ ℝ^{B × L × C × H × W}` and predict `y ∈ ℝ^{B × h × H × W}`.

---

## Dataset

**Primary:** NOAA OISST v2.1 (0.25°, daily), 1981–2000, Coral Sea crop `[140°E–170°E] × [25°S–5°S]`.
OISST is **daily only**. Sub-daily cadence is Track B (Himawari-8, stretch goal, not primary).

| Split | Period |
|---|---|
| Train | 1981-09-01 → 1995-12-31 |
| Val | 1996-01-01 → 1998-12-31 |
| Test | 1999-01-01 → 2000-12-31 (frozen) |

Climatology and normalisation computed from training years only — no leakage.

**Secondary (stretch):** GLORYS12v1 — 1/12° daily multi-variate (SST + salinity + U/V + MLD). Subset server-side via Copernicus Marine Toolbox. Never download the full archive (~16 TB).

**Track B (stretch):** Himawari-8 AHI hourly SST (2015–present) — tests generalisation to sub-daily cadence. Only attempted after Track A is fully reproducible.

---

## Experiment Priorities

| ID | Experiment | Horizon | Priority |
|---|---|---|---|
| E0 | Baselines | 1, 7, 30 | **P0** |
| E1 | LSTM vs Transformer | 7 | **P0 (MVE)** |
| E2 | + GRU, ConvLSTM | 7 | P1 |
| E3 | Horizon sweep × all models | 1, 7, 30 | P1 |
| E4 | Context-length sweep | 7 | P1 |
| E5 | Transformer ablations | 7 | P2 |
| E6 | Multivariate GLORYS12 | 7 | P2 |
| E7 | Compute-cost benchmark | 7 | P2 |
| E8 | Himawari-8 hourly *(stretch)* | 1 | P3 |

---

## Evaluation

### Metrics (in `src/sst_forecasting/utils/metrics.py`)
All reported with bootstrap 95% CI over test years:
- RMSE (°C) — global and per-grid-cell
- MAE (°C)
- Anomaly Correlation Coefficient (ACC)
- Skill score vs persistence: `SS = 1 - RMSE_model / RMSE_persistence`

### Baselines (must all be beaten at h=1)
- Persistence forecast: $\hat{y}_{t+h} = y_t$
- Daily climatology: $\hat{y}_{t+h} = \overline{y}_{\text{doy}(t+h)}$ over training years
- Linear AR(L): ridge regression per grid cell on L past days

### Ablations (priority order)
1. Context length L ∈ {30, 90, 180} × architecture
2. Horizon h ∈ {1, 7, 30} × architecture
3. Transformer: remove positional encoding; heads ∈ {1, 4, 8}; pre- vs post-LN
4. Multivariate (GLORYS12) vs SST-only *(stretch)*
5. Compute cost: s/epoch, peak GPU mem, inference throughput

### Visualisations
- Per-grid-cell RMSE heatmaps (model × horizon)
- Transformer attention maps over the time axis at h ∈ {1, 7, 30}
- RMSE-vs-horizon curves with 95% CI (the key report figure)
- Sample forecast animations *(stretch, for video)*

---

## Compute: Setonix + ROCm

Pawsey Supercomputing Centre, AMD MI250X GPUs (2 GCDs per card, 8 GCDs per node).

**Key gotchas:**
- Use Pawsey ROCm container — do not build PyTorch from source.
- Prefer `bfloat16` over `fp16` on MI250X.
- Keep a config flag to disable `torch.compile` (ROCm support is uneven).
- NCCL → RCCL on ROCm; `NCCL_DEBUG=INFO` only when debugging.
- `gpu-dev` partition for jobs <1 h; `gpu` for production.

**Filesystem on Setonix:**

| Path | Use |
|---|---|
| `$MYSCRATCH/sst-forecasting/raw/` | Downloaded NetCDF (temporary) |
| `$MYSCRATCH/sst-forecasting/processed/` | Zarr store |
| `$MYGROUP/sst-forecasting/checkpoints/` | Model checkpoints |
| `$MYGROUP/sst-forecasting/logs/` | TensorBoard / W&B offline logs |

---

## Quickstart

### Local (CPU)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -e .
pre-commit install
pytest -q
python scripts/download_oisst.py --output-dir data/raw
python scripts/build_zarr.py
python scripts/validate_pipeline.py   # end-to-end sanity check
```

### Setonix
```bash
sbatch scripts/slurm/preprocess.sbatch       # OISST → zarr (~1–2 h, CPU)
sbatch scripts/slurm/train_single_gpu.sbatch # MVE single GCD
sbatch scripts/slurm/train_multi_gpu.sbatch  # 8-GCD DDP
sbatch scripts/slurm/sweep.sbatch            # hyperparameter array sweep
```

---

## Repo Conventions

- **Hydra/OmegaConf** for all configs; compose with `+experiment=...`.
- **No code in notebooks** — notebooks only call into `src/`.
- **One model = one file**, each exposes `build_model(cfg) -> nn.Module`.
- **Deterministic by default:** seed via `cfg.training.seed`.
- Every run writes: `run.yaml`, `metrics.json`, `ckpt_best.pt`, `test_preds.nc`.
- All results = 3-seed mean ± std. Final report tagged at `v1.0-report` on `main`.
- AI usage logged in `report/ai_usage.md` (required by course guidelines).

---

## Timeline

| Week | Focus | Exit criterion |
|---|---|---|
| W1 — 27 Apr | Repo bootstrap, Setonix access, data download | `pytest -q` passes locally |
| W2 — 04 May | OISST download + zarr pipeline; E0 baselines | E0 metrics in `experiments/results/baselines.json` |
| W3 — 11 May | MVE: LSTM vs Transformer E1 | E1 reproducible on Setonix |
| W4 — 18 May | GRU + ConvLSTM E2; horizon sweep E3 | First RMSE-vs-horizon plot |
| W5 — 25 May | Context-length E4; record video | Video submitted 31 May |
| W6 — 01 Jun | Transformer ablations E5; multivariate E6 | Ablation tables populated |
| W7 — 08 Jun | Compute-cost E7; write report | Draft circulated by Wed |
| W8 — 11–14 Jun | Polish figures, proofread, submit | Submit 14 Jun |

---

## Repo Structure

```
configs/          # Hydra/OmegaConf configs (data/, model/, training/, experiment/)
data/
  raw/            # Downloaded NetCDF — not committed to git
  processed/      # Zarr store — not committed to git
  README.md       # Data setup instructions
src/sst_forecasting/
  data/           # download.py, preprocess.py, dataset.py, splits.py, transforms.py
  models/         # baselines.py, lstm.py, convlstm.py, transformer.py, informer.py, tft.py
  training/       # train.py, loop.py, evaluate.py, callbacks.py, ddp_utils.py
  utils/          # metrics.py, visualisation.py, logging.py, seeding.py, rocm.py
  cli.py          # sstf train / sstf eval entry points
scripts/
  download_oisst.py
  build_zarr.py
  validate_pipeline.py
  slurm/          # preprocess.sbatch, train_single_gpu.sbatch, train_multi_gpu.sbatch, sweep.sbatch
experiments/
  results/        # JSON metrics, test_preds.nc
report/
  figures/        # Generated plots
  ai_usage.md     # Required by course guidelines
```

