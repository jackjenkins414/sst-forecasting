"""
Optuna hyperparameter search for SpatialConvLSTM.

Each trial saves results under experiments/results/ in the same format as
the Tubelet HPO so compare_runs.py can read everything together.
Study is persisted to experiments/optuna_convlstm.db.

Usage
-----
    python scripts/run_optuna_convlstm.py              # 20 trials (default)
    python scripts/run_optuna_convlstm.py --n_trials 50
    python scripts/run_optuna_convlstm.py --show       # print study summary only
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.optuna_plots import save_study_plots

import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
import torch
import torch.optim as optim
import zarr

from src.data.dataloaders import create_dataloaders
from src.sst_forecasting.models.convlstm import SpatialConvLSTM
from src.training.losses import AnomalyWeightedMSE
from src.training.train import train_model
from src.training.evaluate import predict
from src.utils.metrics import rmse_per_step, skill_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Fixed — not part of the search
# ---------------------------------------------------------------------------

ZARR_PATH   = PROJECT_ROOT / "data/processed/oisst_coralsea.zarr"
RESULTS_DIR = PROJECT_ROOT / "experiments/results"
DB_PATH     = PROJECT_ROOT / "experiments/optuna_convlstm.db"

CONTEXT_LEN         = 90
HORIZON             = 7
BATCH_SIZE_DEFAULT  = 4   # preferred batch size; reduced to 2 automatically for
                          # memory-heavy configs (see _batch_for_config)
BATCH_SIZE_SMALL    = 2
NUM_EPOCHS          = 50
EARLY_STOP_PATIENCE = 5
RANDOM_SEED         = 42

# ---------------------------------------------------------------------------
# VRAM budget / estimation
#
# Dominant cost: full BPTT over CONTEXT_LEN timesteps.
# Per (timestep, layer) PyTorch saves 7 tensors of shape (B, hidden, H, W)
# for backward: h_t, c_t, sigmoid(i), sigmoid(f), tanh(g), sigmoid(o), tanh(c_t).
#
# activation_bytes = CONTEXT_LEN * n_layers * 7 * hidden * B * H * W * 4
#
# Verified against known outcomes:
#   hidden=32, n_layers=2, B=4  → ~7.8 GB  (completed ✓)
#   hidden=64, n_layers=3, B=4  → ~20.5 GB (crashed   ✗)
# ---------------------------------------------------------------------------

_GRID_H          = 81
_GRID_W          = 121
_SAVED_PER_STEP  = 7     # float32 (B, hidden, H, W) tensors kept per (t, layer)
_OVERHEAD_GB     = 1.5   # weights + Adam states + input batch + CUDA overhead
VRAM_BUDGET_GB   = 9.0   # safe ceiling on 12 GB card; 3 GB headroom for OS/drivers


def _estimate_peak_gb(hidden_dim: int, n_layers: int, batch_size: int) -> float:
    """Estimated peak VRAM in GB for a given ConvLSTM config."""
    activation = (
        CONTEXT_LEN * n_layers * _SAVED_PER_STEP
        * hidden_dim * batch_size
        * _GRID_H * _GRID_W
        * 4  # bytes per float32
    )
    return activation / 1e9 + _OVERHEAD_GB


def _batch_for_config(hidden_dim: int, n_layers: int) -> int | None:
    """Return the largest safe batch size, or None if the config must be pruned.

    Decision table (9 GB budget, 12 GB card):
      B=4 OK  if estimated peak <= 9.0 GB  (n_layers * hidden <= ~75)
      B=2 OK  if estimated peak <= 9.0 GB  (n_layers * hidden <= ~152)
      PRUNE   if even B=2 exceeds budget
    """
    if _estimate_peak_gb(hidden_dim, n_layers, BATCH_SIZE_DEFAULT) <= VRAM_BUDGET_GB:
        return BATCH_SIZE_DEFAULT
    if _estimate_peak_gb(hidden_dim, n_layers, BATCH_SIZE_SMALL) <= VRAM_BUDGET_GB:
        return BATCH_SIZE_SMALL
    return None

# ---------------------------------------------------------------------------
# Search space
#
# hidden_dim  — channel count used for every ConvLSTM layer
# n_layers    — how many ConvLSTM layers to stack
# kernel_size — spatial conv kernel (3 = local, 5 = broader receptive field)
# ---------------------------------------------------------------------------

SEARCH_SPACE = {
    "lr":            FloatDistribution(1e-4, 2e-3, log=True),
    "hidden_dim":    CategoricalDistribution([16, 32, 64, 96]),
    "n_layers":      IntDistribution(1, 4),
    "kernel_size":   CategoricalDistribution([3, 5]),
    "dropout":       FloatDistribution(0.0, 0.35),
    "lr_factor":     FloatDistribution(0.4, 0.8),
    "anomaly_alpha": FloatDistribution(0.0, 0.20),
}

# Baseline config injected as trial 0.
# anomaly_alpha=0.0 guarantees a plain-MSE baseline is always tested.
SEED_CONFIG = {
    "lr":            5e-4,
    "hidden_dim":    32,
    "n_layers":      2,
    "kernel_size":   3,
    "dropout":       0.1,
    "lr_factor":     0.5,
    "anomaly_alpha": 0.0,
}


# ---------------------------------------------------------------------------
# Seed study with previous ConvLSTM runs
# ---------------------------------------------------------------------------

def _params_from_config(config: dict) -> dict | None:
    if config.get("model_type") != "convlstm":
        return None
    mapping = {
        "lr":          config.get("learning_rate"),
        "hidden_dim":  config.get("hidden_dim"),
        "n_layers":    config.get("n_layers"),
        "kernel_size": config.get("kernel_size"),
        "dropout":     config.get("dropout"),
        "lr_factor":   config.get("lr_factor"),
        "anomaly_alpha": config.get("anomaly_alpha"),
    }
    if any(v is None for v in mapping.values()):
        return None
    for name, dist in SEARCH_SPACE.items():
        val = mapping[name]
        if isinstance(dist, CategoricalDistribution):
            if val not in dist.choices:
                return None
        elif isinstance(dist, (FloatDistribution, IntDistribution)):
            if not (dist.low <= val <= dist.high):
                return None
    return mapping


def load_previous_runs(study: optuna.Study) -> int:
    loaded = 0
    for run_dir in sorted(RESULTS_DIR.glob("run_*/")):
        config_f  = run_dir / "config.json"
        metrics_f = run_dir / "metrics.json"
        if not config_f.exists() or not metrics_f.exists():
            continue
        with open(config_f)  as f: config  = json.load(f)
        with open(metrics_f) as f: metrics = json.load(f)
        params = _params_from_config(config)
        if params is None:
            continue
        if any(t.params == params for t in study.trials):
            continue
        mean_rmse = metrics["mean_rmse"]
        if isinstance(mean_rmse, dict):
            mean_rmse = mean_rmse["model"]
        study.add_trial(optuna.trial.create_trial(
            params=params, distributions=SEARCH_SPACE, value=mean_rmse,
        ))
        loaded += 1
    return loaded


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(land_mask_np, norm_mean, norm_std,
                   loaders_b4, loaders_b2, device):

    land_mask_torch = torch.from_numpy(land_mask_np).to(device)
    H, W = land_mask_np.shape

    def objective(trial: optuna.Trial) -> float:
        lr          = trial.suggest_float("lr",        **_dist_kwargs(SEARCH_SPACE["lr"]))
        hidden_dim  = trial.suggest_categorical("hidden_dim",  SEARCH_SPACE["hidden_dim"].choices)
        n_layers    = trial.suggest_int("n_layers",    SEARCH_SPACE["n_layers"].low,
                                                       SEARCH_SPACE["n_layers"].high)
        kernel_size = trial.suggest_categorical("kernel_size", SEARCH_SPACE["kernel_size"].choices)
        dropout     = trial.suggest_float("dropout",   SEARCH_SPACE["dropout"].low,
                                                       SEARCH_SPACE["dropout"].high)
        lr_factor   = trial.suggest_float("lr_factor", SEARCH_SPACE["lr_factor"].low,
                                                        SEARCH_SPACE["lr_factor"].high)
        anomaly_alpha = trial.suggest_float("anomaly_alpha", SEARCH_SPACE["anomaly_alpha"].low,
                                                              SEARCH_SPACE["anomaly_alpha"].high)

        # Pick batch size based on estimated peak VRAM for this config.
        # Falls back to B=2 for memory-heavy configs; prunes if too large even then.
        batch_size = _batch_for_config(hidden_dim, n_layers)
        if batch_size is None:
            est = _estimate_peak_gb(hidden_dim, n_layers, BATCH_SIZE_SMALL)
            print(f"  Trial {trial.number:03d} pruned: est {est:.1f} GB at B=2 "
                  f"exceeds {VRAM_BUDGET_GB} GB budget "
                  f"(hidden={hidden_dim}, layers={n_layers})")
            raise optuna.exceptions.TrialPruned()

        train_loader, val_loader, test_loader = (
            loaders_b4 if batch_size == BATCH_SIZE_DEFAULT else loaders_b2
        )
        if batch_size != BATCH_SIZE_DEFAULT:
            est = _estimate_peak_gb(hidden_dim, n_layers, batch_size)
            print(f"  Trial {trial.number:03d} using B={batch_size} "
                  f"(est {est:.1f} GB, hidden={hidden_dim}, layers={n_layers})")

        hidden_channels = [hidden_dim] * n_layers

        run_dir = RESULTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}"
        run_dir.mkdir(parents=True, exist_ok=True)

        model     = None
        optimizer = None
        try:
            config = {
                "model_type":          "convlstm",
                "context_len":         CONTEXT_LEN,
                "horizon":             HORIZON,
                "batch_size":          batch_size,
                "hidden_dim":          hidden_dim,
                "n_layers":            n_layers,
                "hidden_channels":     hidden_channels,
                "kernel_size":         kernel_size,
                "dropout":             dropout,
                "anomaly_alpha":       anomaly_alpha,
                "num_epochs":          NUM_EPOCHS,
                "early_stop_patience": EARLY_STOP_PATIENCE,
                "learning_rate":       lr,
                "weight_decay":        1e-4,
                "grad_clip":           1.0,
                "lr_factor":           lr_factor,
                "lr_patience":         5,
                "random_seed":         RANDOM_SEED,
                "optuna_trial":        trial.number,
            }
            with open(run_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)

            model = SpatialConvLSTM(
                H=H, W=W, context_len=CONTEXT_LEN, horizon=HORIZON,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                dropout=dropout,
            ).to(device)

            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = AnomalyWeightedMSE(alpha=anomaly_alpha)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=lr_factor, patience=5,
                threshold=1e-3, min_lr=1e-5, cooldown=2,
            )

            def epoch_callback(epoch, val_loss):
                trial.report(val_loss, epoch)
                return trial.should_prune()

            train_losses, val_losses = train_model(
                model=model, train_loader=train_loader, val_loader=val_loader,
                criterion=criterion, optimizer=optimizer, device=device,
                num_epochs=NUM_EPOCHS, land_mask=land_mask_torch, grad_clip=1.0,
                scheduler=scheduler, early_stop_patience=EARLY_STOP_PATIENCE,
                epoch_callback=epoch_callback,
            )

            test_preds_norm, test_targets_norm = predict(model, test_loader, device)
            test_preds   = test_preds_norm   * norm_std + norm_mean
            test_targets = test_targets_norm * norm_std + norm_mean
            rmse_steps   = rmse_per_step(test_preds, test_targets, land_mask=land_mask_np)
            mean_rmse    = float(rmse_steps.mean())

            metrics = {
                "epochs_trained": len(train_losses),
                "best_val_loss":  float(min(val_losses)),
                "mean_rmse":      {"model": mean_rmse},
                "rmse_per_step":  {f"day_{i+1}": float(r) for i, r in enumerate(rmse_steps)},
            }
            with open(run_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            print(f"  Trial {trial.number:03d} | mean RMSE {mean_rmse:.4f} | "
                  f"{len(train_losses)} epochs | {run_dir.name}")
            return mean_rmse

        except optuna.exceptions.TrialPruned:
            raise
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"  Trial {trial.number:03d} unexpected OOM at B={batch_size} "
                  f"(hidden={hidden_dim}, layers={n_layers}) — pruned")
            raise optuna.exceptions.TrialPruned()
        finally:
            if model is not None:
                del model
            if optimizer is not None:
                del optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return objective


def _dist_kwargs(dist) -> dict:
    kwargs = {"name": None, "low": dist.low, "high": dist.high}
    if hasattr(dist, "log") and dist.log:
        kwargs["log"] = True
    return {k: v for k, v in kwargs.items() if k != "name"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _fix_zombie_trials(study: optuna.Study) -> int:
    """Mark any RUNNING trials left by a previous crash as FAIL."""
    zombies = [t for t in study.trials if t.state.name == "RUNNING"]
    if not zombies:
        return 0
    storage = study._storage
    for t in zombies:
        storage.set_trial_state_values(t._trial_id, state=optuna.trial.TrialState.FAIL)
        print(f"  Marked zombie trial {t.number} (params={t.params}) as FAIL")
    return len(zombies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--show",     action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name="convlstm_hpo",
        storage=f"sqlite:///{DB_PATH}",
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    n_zombies = _fix_zombie_trials(study)
    if n_zombies:
        print(f"Fixed {n_zombies} zombie trial(s) left by a previous crash")

    if not any(t.params == SEED_CONFIG for t in study.trials):
        study.enqueue_trial(SEED_CONFIG)
        print("Queued baseline config (alpha=0) as trial 0")

    n_loaded = load_previous_runs(study)
    print(f"Loaded {n_loaded} previous ConvLSTM runs into study "
          f"({len(study.trials)} total trials so far)")

    if args.show:
        _print_summary(study)
        return

    root         = zarr.open_group(str(ZARR_PATH), mode="r")
    norm_mean    = float(root.attrs["norm_mean"])
    norm_std     = float(root.attrs["norm_std"])
    land_mask_np = np.array(root["land_mask"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    loaders_b4 = create_dataloaders(
        zarr_path=ZARR_PATH, context_len=CONTEXT_LEN,
        horizon=HORIZON, batch_size=BATCH_SIZE_DEFAULT,
    )
    loaders_b2 = create_dataloaders(
        zarr_path=ZARR_PATH, context_len=CONTEXT_LEN,
        horizon=HORIZON, batch_size=BATCH_SIZE_SMALL,
    )

    objective = make_objective(
        land_mask_np, norm_mean, norm_std,
        loaders_b4, loaders_b2, device,
    )

    print(f"Starting {args.n_trials} new trials on {device}")
    print(f"VRAM budget: {VRAM_BUDGET_GB} GB  (B=4 up to ~{VRAM_BUDGET_GB:.0f} GB est, B=2 for larger)\n")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    _print_summary(study)
    save_study_plots(study, model_name="convlstm")


def _print_summary(study: optuna.Study):
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        print("No completed trials yet.")
        return
    best = study.best_trial
    print("\n--- ConvLSTM Optuna Study Summary ---")
    print(f"  Completed trials : {len(trials)}")
    print(f"  Best mean RMSE   : {best.value:.4f}")
    print("  Best params:")
    for k, v in best.params.items():
        print(f"    {k:25s} {v}")
    print("\n  Top 5 trials:")
    for i, t in enumerate(sorted(trials, key=lambda t: t.value)[:5], 1):
        print(f"  {i}. RMSE={t.value:.4f}  " +
              "  ".join(f"{k}={v}" for k, v in t.params.items()))


if __name__ == "__main__":
    main()
