# How far ahead can you "sea"?

Recurrent and attention-based deep learning architectures for multi-step sea
surface temperature (SST) anomaly forecasting over the Coral Sea.

This repository benchmarks seven sequence architectures (RNN, Stacked Spatial
LSTM, ConvLSTM, Spatial Flat Transformer, ProbSparse Informer, Patch
Transformer, Tubelet Transformer) under one identical input–output contract,
dataset, and loss. The headline finding is that **spatial inductive bias — not
model family or parameter count — is decisive**: a 1.24 M-parameter ConvLSTM and
a 1.44 M-parameter Tubelet Transformer are joint-best at 0.503 °C mean RMSE,
while the four architectures that flatten each map fall below persistence
despite up to 30× more parameters.

ANU COMP3242/6242 group project. Jack Jenkins, Ayush Samuel, Khaled El-hassan,
Isaac Jaensch.

## Task contract

Every model consumes and produces identical tensors, so any performance gap
reflects architecture rather than data or training target:

```
input  X : (90, 1, 81, 121)   90 days of normalised daily anomaly maps
output Y : (7, 81, 121)        next 7 days of normalised anomalies
```

The grid is the Coral Sea (25°S–5°S, 140°E–170°E), 81 × 121 cells. Models are
trained on z-scored anomalies and evaluated in °C after denormalisation, over
ocean cells only (land is masked out of the loss and all metrics).

## Repository layout

```
src/
  data/        Zarr-backed sliding-window Dataset, dataloaders, split dates, preprocessing helpers
  models/      convlstm, lstm, transformer, informer, patch_transformer, tubelet_transformer
  baselines/   persistence baseline, simple RNN
  training/    train loop, anomaly-weighted MSE loss, evaluation/prediction
  utils/       RMSE / skill-score metrics
scripts/       data pipeline, per-model experiments, Optuna HPO, retraining, AR rollout, figures
experiments/   saved run summaries, best-model artifacts, aggregated figures
report/        figures used in the LaTeX report
configs/       (currently empty — models are configured via module-level constants in each script)
data/          raw NetCDF and processed Zarr store (gitignored; see "Data" below)
```

## Setup

Python 3.10+ (the code uses `int | None` union syntax). Training runs on CPU
(an M3 MacBook is fine); a CUDA GPU helps but is not required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All scripts are run from the repository root and add it to `sys.path`
themselves, so no `pip install -e .` step is needed:

```bash
python3 scripts/<script_name>.py
```

## Data

The models read a processed **Zarr store** at
`data/processed/oisst_coralsea.zarr` (~612 MB, **zarr v2** format). It is
gitignored because it is too large for version control, so it must be present
locally before training. The store must contain:

- `sst_norm` — z-scored daily anomaly field, shape `(time, 81, 121)`, land cells as NaN
- `time` — int64 days since `1970-01-01`

Splits are assigned chronologically by date (`src/data/splits.py`): train
1981-09-01 → 1995-12-31, validation 1996-01-01 → 1998-12-31, test 1999-01-01 →
2000-12-31. Climatology and z-score statistics are fit on the training years
only to prevent leakage.

The source product is NOAA OISST v2.1 (daily 0.25° SST), cropped to the Coral
Sea. See https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/

> **Note — data pipeline is work in progress.** The scripts
> `scripts/download_oisst.py`, `scripts/build_zarr.py`, and
> `scripts/validate_pipeline.py` reference a `sst_forecasting` package that is
> not yet present in this repository, so they do **not** run as-is. Regenerating
> the Zarr store from raw NetCDF is therefore not currently automated end-to-end.
> Obtain the prepared `oisst_coralsea.zarr` store and place it at
> `data/processed/oisst_coralsea.zarr` to reproduce the experiments below.

## Reproducing the experiments

All commands assume the activated venv and the Zarr store in place.

**Train and evaluate a single architecture.** Each script trains one model,
evaluates it on the test split against persistence, and writes a timestamped run
to `experiments/results/run_YYYYMMDD_HHMMSS/` (config, metrics, checkpoint,
summary figure):

```bash
python3 scripts/run_tubelet_experiment.py
python3 scripts/run_patch_transformer_experiment.py
python3 scripts/run_lstm_experiment.py
python3 scripts/run_transformer_experiment.py
python3 scripts/run_probsparse_informer_experiment.py
python3 scripts/run_rnn_baseline_experiment.py
```

**Hyperparameter search (Optuna).** Studies are persisted to
`experiments/optuna_<model>.db`; add `--show` to print a study summary without
running trials:

```bash
python3 scripts/run_optuna_convlstm.py --n_trials 50
python3 scripts/run_optuna_lstm.py
python3 scripts/run_optuna_transformer.py
python3 scripts/run_optuna_informer.py
```

**Retrain the best configuration per model.** Writes artifacts to
`experiments/best_<model>/` (best config, loss curves, per-step RMSE/skill,
denormalised predictions/targets, checkpoint). Runs all models by default, or a
subset:

```bash
python3 scripts/retrain_best.py
python3 scripts/retrain_best.py --models tubelet convlstm
# choices: tubelet lstm informer convlstm transformer patch_transformer rnn
```

**Long-horizon autoregressive rollout.** Rolls each trained model out to 49 days
by feeding back its own 7-day predictions:

```bash
python3 scripts/run_ar_rollout_avg.py   # phase-averaged rollout (reported in the paper)
python3 scripts/run_ar_rollout.py
```

**Anomaly-weighted loss ablation** (vanilla MSE vs the tuned weight α*):

```bash
python3 scripts/run_alpha_ablation.py
```

**Figures and aggregation.**

```bash
python3 scripts/aggregate_seed_results.py   # mean ± std across seeds for the main table
python3 scripts/plot_ar_aggregate.py        # long-horizon figure with seed bands
python3 scripts/compare_runs.py             # HP-vs-RMSE scatter + RMSE-per-step curves
python3 scripts/report_model.py --all       # per-model report panels
```

## Results

Seven-day test RMSE (°C). Only the three spatially-structured models beat
persistence; the ranking tracks spatial structure, not parameter count.

| Family    | Model                    | Params  | Mean RMSE | Skill   |
|-----------|--------------------------|---------|-----------|---------|
| Recurrent | RNN                      | 9.5 M   | 0.605     | −0.100  |
| Recurrent | Stacked Spatial LSTM     | 18.6 M  | 0.583     | −0.059  |
| Recurrent | **ConvLSTM**             | 1.24 M  | **0.503** | +0.125  |
| Attention | Spatial Flat Transformer | 21.3 M  | 0.610     | −0.110  |
| Attention | ProbSparse Informer      | 11.2 M  | 0.629     | −0.148  |
| Attention | Patch Transformer        | 0.71 M  | 0.512     | +0.106  |
| Attention | **Tubelet Transformer**  | 1.44 M  | **0.503** | +0.122  |
| Baseline  | Persistence              | —       | 0.580     | 0.000   |

Under autoregressive rollout the three skillful models stay below the
climatology error floor for roughly five to six weeks — far beyond their 7-day
training horizon — with the ConvLSTM extrapolating furthest (~45 days).

## Reproducibility notes

- Random seed fixed to 42; multi-seed runs use seeds 1–3.
- Hardware was not standardised across models (A100 on NCI Gadi for the RNN and
  ConvLSTM; RTX 4070 locally for the rest), which is a caveat for strict bit-level
  reproducibility but does not favour the strongest models.