# Project Plan — SST Forecasting: Recurrent vs Attention

**Course:** COMP3242 (Deep Learning), Semester 1, 2026
**Team branch:** `ayush`
**Last updated:** 2026-04-23

> This document is the single source of truth for the project. It extends `Context.md` with concrete technical decisions, a reproducible Setonix/ROCm workflow, a full repository layout, and a milestone timeline aligned with the course deliverables (Video 31 May, Final Report 14 Jun).

---

## 1. Executive Summary

We benchmark recurrent (LSTM, GRU), hybrid spatio-temporal (ConvLSTM), and attention-based (Transformer encoder, Informer, TFT) sequence models on multi-step sea surface temperature (SST) forecasting. The minimum viable experiment (MVE) is a head-to-head comparison of one recurrent model vs one attention model at a single horizon on a single cropped region. All other experiments are tracked as **stretch goals** and only attempted once the MVE is reproducible end-to-end on Setonix.

**Deliverables & weights (from guidelines):**
| # | Deliverable | Due | Weight |
|---|---|---|---|
| 1 | Project Registration (title + abstract) | 20 Apr 2026 | 0% (submitted) |
| 2 | 5-minute video presentation | 31 May 2026 | 16% |
| 3 | Final report (4–8 pages) + source code | 14 Jun 2026 | 24% |

COMP3242 only requires analysis/implementation of existing methods — no novel contribution required. We should still produce clean, insightful analysis.

---

## 2. Research Questions & Hypotheses

**RQ1 (primary).** At what forecast horizon $h \in \{1, 7, 30\}$ days do attention-based models measurably outperform recurrent models on SST?

**RQ2.** How does input context length $L \in \{30, 90, 180\}$ days interact with architecture choice?

**RQ3.** Does spatial inductive bias (ConvLSTM) outperform purely temporal models on a gridded domain at matched parameter budgets?

**H1.** Recurrent models (LSTM/GRU) will match or beat attention models at $h=1$ due to strong locality and small effective context.
**H2.** Attention models will dominate at $h=30$ where long-range dependencies and non-local teleconnections matter.
**H3.** ConvLSTM will beat both families when evaluated per-grid-cell RMSE in regions with strong mesoscale variability (e.g., western boundary currents), but offer no advantage in basin-mean metrics.

---

## 3. Related Work (to read & cite)

Short reading list — one paper per team-member per week until MVE is running.

- **Shi et al. (2015)** — *ConvLSTM for precipitation nowcasting* (canonical ConvLSTM paper).
- **Vaswani et al. (2017)** — *Attention is All You Need*.
- **Zhou et al. (2021)** — *Informer* (efficient long-sequence transformer, ProbSparse attention).
- **Lim et al. (2021)** — *Temporal Fusion Transformer (TFT)*.
- **Pathak et al. (2022)** — *FourCastNet*: ML weather forecasting at scale.
- **Lam et al. (2023, Science)** — *GraphCast*.
- **de Burgh-Day & Leeuwenburg (2023)** — review of ML for operational weather/ocean forecasting.
- **Taylor & Feng (2022)** or similar — LSTM/ConvLSTM for SST anomaly prediction (find 1–2 direct SST-forecasting refs).

Track these in `report/references.bib`.

---

## 4. Datasets

### 4.1 Primary: NOAA OISST v2.1 (daily)

- **Resolution:** 0.25° global, daily.
- **Period used:** 1981-09-01 → 2000-12-31 (≈7 060 days) as the "standardised pre-climate-change" baseline — avoids the post-2000 acceleration in ocean heat uptake and gives a stationary-ish target for model comparison.
- **Variable:** SST only.
- **Access:** NOAA NCEI THREDDS / ERDDAP; NetCDF4.
- **Region crop (default):** Coral Sea bounding box `[140°E, 170°E] × [25°S, 5°S]` → ~120×80 grid cells. Small enough to iterate fast on a single MI250X GCD.

### 4.2 Sub-daily addendum (1–2 hour cadence)

NOAA OISST is **daily only**. True 1–2 hour SST does not exist before the geostationary era (~2000+) and so is incompatible with a pre-climate-change baseline. We resolve this by running **two parallel tracks**:

| Track | Dataset | Cadence | Period | Purpose |
|---|---|---|---|---|
| **A (main)** | NOAA OISST v2.1 | daily | 1981–2000 | Core benchmark for RQ1–RQ3 |
| **B (stretch)** | Himawari-8 AHI L3C SST (or GOES-R ABI) | hourly | 2015–present | Test whether architectures generalise to sub-daily cadence; short window, modern climate |

Track B is only attempted after Track A is fully reproducible and if time allows. The report frames Track A as the primary study and Track B as a generalisation probe.

### 4.3 Tertiary (optional): GLORYS12v1

- 1/12° daily multi-variate reanalysis (SST + salinity + U/V currents + MLD).
- Used only for the multi-variate ablation (stretch goal) on the Coral Sea crop.
- Full global archive is ~16 TB — we will **never download the full archive**; we subset server-side via Copernicus Marine Toolbox.

### 4.4 Data splits

| Split | Years | Notes |
|---|---|---|
| Train | 1981-09 → 1995-12 | ~14 yr |
| Val | 1996-01 → 1998-12 | 3 yr (model selection + early stopping) |
| Test | 1999-01 → 2000-12 | 2 yr (held out, frozen until final eval) |

All anomalies computed against the **climatology of the training years only** to avoid leakage.

---

## 5. Models

All models take input tensor `x ∈ ℝ^{B × L × C × H × W}` (batch, context length, variables, lat, lon) and predict `y ∈ ℝ^{B × h × H × W}` (SST at $h$ future steps). For purely temporal baselines (LSTM/GRU/Transformer) we flatten or patchify the spatial dims.

| Model | File | Key hyperparams (default) | Param budget target |
|---|---|---|---|
| Persistence | `src/models/baselines.py` | — | 0 |
| Climatology | `src/models/baselines.py` | — | 0 |
| Linear AR | `src/models/baselines.py` | Ridge, $L$-step lag | <1 k |
| Stacked LSTM | `src/models/lstm.py` | hidden=256, layers=2, patch=8×8 | ~1–3 M |
| Stacked GRU | `src/models/lstm.py` (shared) | hidden=256, layers=2 | ~1–3 M |
| ConvLSTM | `src/models/convlstm.py` | 3×3 kernel, 2 layers, 64 channels | ~1–3 M |
| Transformer encoder | `src/models/transformer.py` | d_model=256, heads=8, layers=4 | ~1–3 M |
| Informer | `src/models/informer.py` | ProbSparse, distilling | ~1–3 M |
| TFT (stretch) | `src/models/tft.py` | — | ~1–3 M |

**Fair-comparison rule.** All comparison runs are tuned to approximately the same parameter count (±20 %) and given the same compute budget (wall-clock GPU-hours). Reported in Table 1 of the final report.

---

## 6. Baselines (must pass)

Every deep model must beat all three of these on the test set at $h=1$ before any claim is made:

1. **Persistence:** $\hat{y}_{t+h} = y_t$.
2. **Daily climatology:** $\hat{y}_{t+h} = \overline{y}_{\text{doy}(t+h)}$ over training years.
3. **Linear AR(L):** ridge regression per grid cell on $L$ past days.

If a model cannot beat persistence at $h=1$ there is a bug.

---

## 7. Evaluation

### 7.1 Metrics (implemented in `src/utils/metrics.py`)

- **RMSE** (°C), global and per-grid-cell.
- **MAE** (°C).
- **Anomaly Correlation Coefficient (ACC)** against daily climatology.
- **Skill score vs persistence:** $\text{SS} = 1 - \text{RMSE}_{\text{model}} / \text{RMSE}_{\text{persistence}}$.

All metrics reported with bootstrap 95 % CI over the test years.

### 7.2 Ablations (in priority order)

1. Context length $L \in \{30, 90, 180\}$ × architecture. *(MVE adjacent.)*
2. Forecast horizon $h \in \{1, 7, 30\}$ × architecture.
3. Transformer: remove positional encoding; heads $\in \{1, 4, 8\}$; pre- vs post-LN.
4. Multivariate (SST+SAL+U+V, GLORYS12) vs SST-only. *(Stretch.)*
5. Compute cost: s/epoch, peak GPU mem, inference throughput (forecasts/sec).

### 7.3 Visualisations

- Per-grid-cell RMSE heatmaps (one figure per model × horizon).
- Transformer attention maps over the time axis at $h \in \{1,7,30\}$.
- RMSE-vs-horizon curves with 95 % CI (the money plot for the report).
- Sample forecast animations (stretch; for the video).

---

## 8. Compute Environment: Setonix + ROCm

Setonix (Pawsey Supercomputing Centre) nodes with AMD MI250X GPUs. Each MI250X exposes **2 GCDs** which SLURM treats as 2 logical GPUs.

### 8.1 Software stack

- OS modules: `module load rocm/<latest>`, `module load singularity/<latest>`.
- PyTorch via Pawsey-provided ROCm container image (preferred) or `pip install torch --index-url https://download.pytorch.org/whl/rocm6.x`.
- **Do not build PyTorch from source.** Use the Pawsey container; it is pre-validated for MI250X.
- `xarray` + `dask` for NetCDF I/O; `zarr` for on-scratch caching.

### 8.2 Filesystem plan

| Path | Use |
|---|---|
| `$MYSCRATCH/sst-forecasting/raw/` | Downloaded NetCDF (tempoary, purged) |
| `$MYSCRATCH/sst-forecasting/processed/` | Zarr store, train-ready tensors |
| `$MYSOFTWARE/envs/` | Python venv / conda env |
| `$MYGROUP/sst-forecasting/checkpoints/` | Model checkpoints (kept) |
| `$MYGROUP/sst-forecasting/logs/` | TensorBoard / W&B offline logs |

Repo lives in `$HOME` or `$MYSOFTWARE`; data never in `$HOME`.

### 8.3 SLURM workflow

- `scripts/slurm/train_single_gpu.sbatch` — 1 GCD, debug/MVE runs.
- `scripts/slurm/train_multi_gpu.sbatch` — `torchrun --nproc-per-node=8` within one node (4 MI250X × 2 GCDs).
- `scripts/slurm/sweep.sbatch` — SLURM array job for hyperparameter sweeps.
- `scripts/slurm/preprocess.sbatch` — CPU-only, for one-off data prep.

Always use `#SBATCH --partition=gpu-dev` for <1 h debug jobs; `gpu` for production.

### 8.4 ROCm gotchas to watch

- `torch.compile` support on ROCm can be uneven — keep a config flag to disable it.
- Mixed precision: use `torch.cuda.amp` (works on ROCm) with `bfloat16` on MI250X (preferred over fp16).
- NCCL → RCCL on ROCm; set `NCCL_DEBUG=INFO` only when debugging.
- Pin `HSA_FORCE_FINE_GRAIN_PCIE=1` only if we see collective perf issues.

---

## 9. Repository Structure (target)

```
sst-forecasting/
├── README.md
├── LICENSE
├── Context.md                    # Background / motivation (exists)
├── Changelog.md                  # Change log (exists)
├── PLAN.md                       # THIS FILE
├── pyproject.toml                # Package + tool config (ruff, pytest)
├── requirements.txt              # Pinned deps (cpu + rocm variants)
├── requirements-rocm.txt         # Setonix / MI250X
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml      # Lint + unit tests on CPU
│
├── configs/                      # Hydra / OmegaConf configs
│   ├── default.yaml
│   ├── data/
│   │   ├── oisst_coralsea.yaml
│   │   ├── oisst_global.yaml
│   │   └── glorys12_coralsea.yaml
│   ├── model/
│   │   ├── persistence.yaml
│   │   ├── climatology.yaml
│   │   ├── linear_ar.yaml
│   │   ├── lstm.yaml
│   │   ├── gru.yaml
│   │   ├── convlstm.yaml
│   │   ├── transformer.yaml
│   │   ├── informer.yaml
│   │   └── tft.yaml
│   ├── training/
│   │   ├── default.yaml
│   │   ├── debug.yaml
│   │   └── setonix.yaml
│   └── experiment/               # Composed configs per experiment row
│       ├── mve_lstm_vs_transformer_h7.yaml
│       ├── ablation_context_length.yaml
│       └── ablation_horizon.yaml
│
├── src/sst_forecasting/
│   ├── __init__.py
│   ├── data/
│   │   ├── download.py           # ERDDAP / Copernicus fetchers
│   │   ├── preprocess.py         # NetCDF → zarr, normalisation, climatology
│   │   ├── dataset.py            # Torch Dataset: windowed (x, y) pairs
│   │   ├── splits.py             # Train/val/test temporal splits (no leakage)
│   │   └── transforms.py         # Standardise, patchify, land-mask
│   ├── models/
│   │   ├── baselines.py          # persistence, climatology, linear AR
│   │   ├── lstm.py               # LSTM + GRU (shared base)
│   │   ├── convlstm.py
│   │   ├── transformer.py
│   │   ├── informer.py
│   │   ├── tft.py
│   │   └── registry.py           # model factory
│   ├── training/
│   │   ├── train.py              # entry point (Hydra main)
│   │   ├── loop.py               # train/val loop, AMP, DDP
│   │   ├── evaluate.py
│   │   ├── callbacks.py          # early stop, ckpt, lr sched
│   │   └── ddp_utils.py          # RCCL init
│   ├── utils/
│   │   ├── metrics.py            # RMSE / MAE / ACC / SS
│   │   ├── visualisation.py      # heatmaps, attention maps, curves
│   │   ├── logging.py            # tb + optional W&B
│   │   ├── seeding.py
│   │   └── rocm.py               # device/env helpers
│   └── cli.py                    # `sstf train ...`, `sstf eval ...`
│
├── scripts/
│   ├── download_oisst.py
│   ├── build_zarr.py
│   ├── run_baselines.py
│   └── slurm/
│       ├── preprocess.sbatch
│       ├── train_single_gpu.sbatch
│       ├── train_multi_gpu.sbatch
│       └── sweep.sbatch
│
├── tests/                        # pytest, CPU-only, tiny fixtures
│   ├── test_dataset.py
│   ├── test_metrics.py
│   ├── test_models_forward.py
│   └── fixtures/
│
├── notebooks/                    # Exploratory only; not imported from src
│   ├── 01_oisst_inspect.ipynb
│   ├── 02_climatology.ipynb
│   └── 03_result_plots.ipynb
│
├── data/                         # gitignored payloads, .gitkeep only
│   ├── raw/
│   └── processed/
│
├── experiments/
│   ├── results/                  # CSV / JSON of per-run metrics
│   └── checkpoints/              # symlink to $MYGROUP on Setonix
│
└── report/
    ├── main.tex
    ├── references.bib
    └── figures/
```

### 9.1 Key conventions

- **Hydra/OmegaConf** for all configs so we can compose `+experiment=...` and keep one entry point.
- **No code in notebooks** — notebooks only call into `src/`.
- **One model = one file**, each exposes `build_model(cfg) -> nn.Module`.
- **Deterministic by default:** seed in `cfg.training.seed`; set CuDNN/ROCm deterministic flags.
- **Every run writes** `{results_dir}/run.yaml`, `metrics.json`, `ckpt_best.pt`, `test_preds.nc`.

---

## 10. End-to-End Workflow

### 10.1 Local dev (MacBook, CPU)

1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt -e .`
3. `pre-commit install`
4. Tiny smoke dataset: 30 days × 16×16 grid fixture under `tests/fixtures/`.
5. `pytest -q`  → all green before pushing.
6. `sstf train +experiment=mve_lstm_vs_transformer_h7 training=debug` — runs 1 epoch on CPU.

### 10.2 Setonix (MI250X)

1. `ssh setonix.pawsey.org.au`
2. `cd $MYSOFTWARE && git clone ... && cd sst-forecasting`
3. `sbatch scripts/slurm/preprocess.sbatch` (downloads OISST → zarr). ~1–2 h, CPU.
4. Confirm shapes: `python scripts/inspect_zarr.py`.
5. `sbatch scripts/slurm/train_single_gpu.sbatch --config mve_lstm_vs_transformer_h7`.
6. Monitor: `squeue -u $USER`; TensorBoard offline → rsync logs back.
7. Eval: `sbatch scripts/slurm/evaluate.sbatch --run <run_id>`.

### 10.3 Reproducibility

- Every `sbatch` submission stamps commit SHA + full config into the run dir.
- All random seeds fixed per run; at least 3 seeds per reported number.
- Final report points to a tagged git commit `v1.0-report`.

---

## 11. Experiment Matrix

| ID | Config | Horizon | L | Track | Priority |
|---|---|---|---|---|---|
| E0 | baselines (persist/clim/AR) | 1,7,30 | 90 | A | P0 (MVE prereq) |
| E1 | lstm vs transformer | 7 | 90 | A | **P0 (MVE)** |
| E2 | gru, convlstm added | 7 | 90 | A | P1 |
| E3 | horizon sweep × all models | 1,7,30 | 90 | A | P1 |
| E4 | context-length sweep | 7 | 30,90,180 | A | P1 |
| E5 | transformer ablations (PE, heads, LN) | 7 | 90 | A | P2 |
| E6 | multivariate (GLORYS12) | 7 | 90 | A | P2 |
| E7 | compute-cost benchmark | 7 | 90 | A | P2 |
| E8 | Himawari-8 hourly generalisation | 1 | 48 h | B | P3 (stretch) |

P0 must be finished by **10 May 2026** so the video (due 31 May) has real numbers.

---

## 12. Timeline

Working backward from **Final Report 14 Jun 2026**.

| Week (start Mon) | Focus | Exit criterion |
|---|---|---|
| W1 — **27 Apr** | Repo bootstrap, Setonix access, data download plan | `sstf train training=debug` runs locally; Setonix login confirmed |
| W2 — **04 May** | OISST download + zarr pipeline; baselines E0 | E0 metrics in `experiments/results/baselines.json` |
| W3 — **11 May** | MVE: LSTM vs Transformer at $h=7$ (E1) | E1 reproducible end-to-end on Setonix |
| W4 — **18 May** | Add GRU + ConvLSTM (E2); horizon sweep E3 | First RMSE-vs-horizon plot generated |
| W5 — **25 May** | Finish context-length E4; **record video** | Video submitted **Sun 31 May** |
| W6 — **01 Jun** | Transformer ablations E5; start multivariate E6 | Ablation tables populated |
| W7 — **08 Jun** | Compute-cost E7; write report; freeze results | Draft report circulated to team by Wed |
| W8 — **11–14 Jun** | Polish figures, final proofread, submit | **Submit 14 Jun** |

---

## 13. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Pawsey queue congestion near deadline | High | High | Finish MVE by W3; use `gpu-dev` for iteration |
| ROCm PyTorch incompatibility with some layer | Med | Med | Use Pawsey container; have CPU fallback |
| Data download slower than expected | Med | High | Start W1 Day 1; fall back to OISST-only |
| Scope creep (4 models × 3 horizons × 3 $L$) | High | Med | Experiment priorities P0→P3; freeze on 08 Jun |
| GLORYS12 preprocessing blows budget | Med | Low | Skip E6 if not ready by W6 |
| Disagreement on metrics / report ownership | Low | Med | Assign sections per team member in W5 |

---

## 14. Team & Contributions

*(To be filled in per the course declaration requirement.)*

| Member | UID | Primary ownership |
|---|---|---|
| Ayush | — | Repo / infra / Setonix / Transformer |
| TBD | — | Data pipeline / ConvLSTM |
| TBD | — | LSTM+GRU / evaluation |
| TBD | — | Visualisation / report |

Generative-AI usage log tracked in `report/ai_usage.md` (required by the guidelines).

---

## 15. Definition of Done (for the report)

- [ ] All P0 + P1 experiments have metrics with 3-seed mean ± std.
- [ ] RMSE-vs-horizon plot with 95 % CI reproducible from `notebooks/03_result_plots.ipynb`.
- [ ] Per-grid-cell RMSE heatmap for each model at $h=7$.
- [ ] Attention-map figure for Transformer at $h \in \{1,7,30\}$.
- [ ] Compute-cost table (s/epoch, GB peak, forecasts/sec).
- [ ] 4–8 page PDF with declarations + AI-usage statement.
- [ ] Tagged commit `v1.0-report` on `main`.
- [ ] `README.md` has one-command reproduce instructions for the MVE.

---

## 16. Open Questions (resolve in W1)

1. Team size & members finalised?
2. Pawsey project allocation code & quota?
3. Do we have W&B access, or offline TensorBoard only?
4. Is Track B (Himawari-8 hourly) worth the complexity, or drop it now?
5. LaTeX vs Markdown-to-PDF for the report?
