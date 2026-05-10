# Contributing a new model

Here we briefly detail how the SST forecasting pipeline is organised and how to add your own model (ConvLSTM, RNN, Transformer, etc.) without breaking anyone else's work.

## How the pipeline fits together

The pipeline is built around a shared Coral Sea Zarr dataset. Most of the data work is done up front — your model just needs to consume grid tensors and produce grid forecasts.

```
data/processed/oisst_coralsea.zarr
        |
        v
SstWindowDataset  (src/data/dataset.py)
        |   returns:  x = (context_len, 1, H, W)
        |             y = (horizon, H, W)
        v
DataLoader  (src/data/dataloaders.py)
        |   batches:  x = (B, context_len, 1, H, W)
        |             y = (B, horizon, H, W)
        v
Your model  (src/models/<your_model>.py)
        |   produces: pred = (B, horizon, H, W)
        v
train_model  (src/training/train.py)
        |   masked MSE over ocean cells, gradient clipping
        v
predict  (src/training/evaluate.py)
        |   returns predictions and targets as numpy
        v
metrics.py  (src/utils/metrics.py)
            rmse, mae, rmse_per_step, skill_score
```

The Zarr store is built once by Ayush's pipeline (on a separate branch) and contains:

- `sst_norm` — z-scored SST anomalies, the model input/output
- `land_mask` — boolean grid, True = ocean
- `climatology` — day-of-year mean over training years
- group attributes `norm_mean`, `norm_std` for denormalisation

## What each file does

### `src/data/`

- **`splits.py`** — train/val/test date boundaries and a `date_mask()` helper. **Don't change.**
- **`dataset.py`** — `SstWindowDataset`. Reads windows lazily from the Zarr store. Returns `(L, 1, H, W)` context and `(h, H, W)` target. **Don't change unless you need a different sample format** (e.g. patch-based input for a Transformer).
- **`dataloaders.py`** — `create_dataloaders()` builds train/val/test DataLoaders directly from a Zarr path. Reusable as-is for any model that consumes the same `(x, y)` format.

### `src/models/`

- **`lstm.py`** — `StackedSpatialLSTM`. Reference model.

This is where your new model goes. One file per model. Add whatever you guys like, such as `convlstm.py`, `transformer.py`, `cnn_lstm.py`, etc.

### `src/training/`

- **`train.py`** — `train_model()`. Generic training loop. Accepts any `nn.Module` whose forward takes `(B, L, 1, H, W)` and returns `(B, h, H, W)`. Supports masked MSE loss (`land_mask` kwarg) and gradient clipping (`grad_clip` kwarg). **Reuse as-is.**
- **`evaluate.py`** — `predict()`. Runs a model over a DataLoader, returns numpy arrays. **Reuse as-is.**

### `src/baselines/`

- **`persistence.py`** — `persistence_forecast()`. Predicts every future day as the last context day. **Reuse as-is for comparison.**
- **`rnn.py`** - `RNN`. A baseline recurrent neural network model for evaluating the enhanced models in `src/models/`. **Reuse as-is for comparison.**

### `src/utils/`

- **`metrics.py`** — `rmse`, `mae`, `rmse_per_step`, `skill_score`. All accept an optional `land_mask` to ignore land cells. **Reuse as-is.**

### `scripts/`

- **`run_lstm_experiment.py`** — End-to-end experiment for the LSTM. Use this as a template for your own model's experiment script.

## How to add a new model

### Step 1 — Write the model

Create `src/models/<your_model>.py` with a single `nn.Module` subclass.

The contract:

```
forward input shape:  (B, L, 1, H, W)
forward output shape: (B, h, H, W)
```

Where `L = context_len` (default 90) and `h = horizon` (default 7).

Constructor signature is up to you, but at minimum it needs to know `H`, `W`, `context_len`, and `horizon`. Look at `src/models/lstm.py` for an example.

### Step 2 — Write an experiment script

Copy `scripts/run_lstm_experiment.py` to `scripts/run_<your_model>_experiment.py`.

Change:
- The model import at the top
- The model instantiation in `main()`
- Any model-specific hyperparameters in the `# Configuration` block

Everything else (loading the Zarr, building DataLoaders, training, evaluating, persistence baseline, metrics) stays the same.

### Step 3 — Run it

```bash
python3 scripts/run_<your_model>_experiment.py
```

You'll see per-epoch train/val loss, then a summary printing per-horizon RMSE for both your model and persistence, plus skill scores.

### Step 4 — Open a PR

- One branch per model: `convlstm/<your-name>`, `transformer/<your-name>`, etc.
- Don't modify files outside `src/models/` and `scripts/` unless you have a reason — if you do, call it out in the PR description.
- The PR description should include the test RMSE per horizon and how it compares to persistence and the LSTM reference.

## Things that need to change in shared code (rare)

If your model genuinely needs a different sample format — for example, a Transformer that wants patches `(B, L, num_patches, patch_size)` instead of grids — then `SstWindowDataset` needs an option for it. In that case:

1. Add the new behaviour as an opt-in flag (e.g. `patchify=True`) so existing models keep working.
2. Open a separate PR for the dataset change before the model PR. Easier to review, and it lets the team agree on the interface.

Don't fork `dataset.py` into `dataset_transformer.py`. One dataset class, configurable behaviour.

## Quick checklist before opening your PR

- [ ] Your model's forward pass takes `(B, L, 1, H, W)` and returns `(B, h, H, W)`
- [ ] Your experiment script runs end-to-end without errors
- [ ] You compared against the persistence baseline at horizon 7
- [ ] You reported per-horizon RMSE in the PR description
- [ ] You didn't modify shared modules (`train.py`, `metrics.py`, etc.) without flagging it

## Reference numbers

For sanity-checking your runs against the LSTM baseline:

| | Test RMSE day 1 | Test RMSE day 7 | Test RMSE mean |
|---|---|---|---|
| Persistence | 0.33 °C | 0.70 °C | 0.58 °C |
| Recurrent Neural Network (8 epochs) | 0.58 °C | 0.63 °C | 0.61 °C |
| StackedSpatialLSTM (8 epochs) | 0.60 °C | 0.63 °C | 0.62 °C |

Persistence beats the LSTM at short horizons (less than eq 3 days) and the LSTM beats persistence at long horizons (\geq 4 days). A model that doesn't beat persistence at *any* horizon is probably misconfigured.