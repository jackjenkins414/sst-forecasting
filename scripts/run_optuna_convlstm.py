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
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

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
BATCH_SIZE          = 8   # ConvLSTM is memory-heavy (full H×W hidden states)
NUM_EPOCHS          = 50
EARLY_STOP_PATIENCE = 5
RANDOM_SEED         = 42

# ---------------------------------------------------------------------------
# Search space
#
# hidden_dim  — channel count used for every ConvLSTM layer
# n_layers    — how many ConvLSTM layers to stack
# kernel_size — spatial conv kernel (3 = local, 5 = broader receptive field)
# ---------------------------------------------------------------------------

SEARCH_SPACE = {
    "lr":          FloatDistribution(1e-4, 2e-3, log=True),
    "hidden_dim":  CategoricalDistribution([16, 32, 64, 96]),
    "n_layers":    IntDistribution(1, 4),
    "kernel_size": CategoricalDistribution([3, 5]),
    "dropout":     FloatDistribution(0.0, 0.35),
    "lr_factor":   FloatDistribution(0.4, 0.8),
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
                   train_loader, val_loader, test_loader, device):

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

        hidden_channels = [hidden_dim] * n_layers

        run_dir = RESULTS_DIR / datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "model_type":          "convlstm",
            "context_len":         CONTEXT_LEN,
            "horizon":             HORIZON,
            "batch_size":          BATCH_SIZE,
            "hidden_dim":          hidden_dim,
            "n_layers":            n_layers,
            "hidden_channels":     hidden_channels,
            "kernel_size":         kernel_size,
            "dropout":             dropout,
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
        criterion = AnomalyWeightedMSE(alpha=0.0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor, patience=5,
            threshold=1e-3, min_lr=1e-5, cooldown=2,
        )

        def epoch_callback(epoch, val_loss):
            trial.report(val_loss, epoch)
            return trial.should_prune()

        try:
            train_losses, val_losses = train_model(
                model=model, train_loader=train_loader, val_loader=val_loader,
                criterion=criterion, optimizer=optimizer, device=device,
                num_epochs=NUM_EPOCHS, land_mask=land_mask_torch, grad_clip=1.0,
                scheduler=scheduler, early_stop_patience=EARLY_STOP_PATIENCE,
                epoch_callback=epoch_callback,
            )
        except optuna.exceptions.TrialPruned:
            raise

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

    return objective


def _dist_kwargs(dist) -> dict:
    kwargs = {"name": None, "low": dist.low, "high": dist.high}
    if hasattr(dist, "log") and dist.log:
        kwargs["log"] = True
    return {k: v for k, v in kwargs.items() if k != "name"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    train_loader, val_loader, test_loader = create_dataloaders(
        zarr_path=ZARR_PATH, context_len=CONTEXT_LEN,
        horizon=HORIZON, batch_size=BATCH_SIZE,
    )

    objective = make_objective(
        land_mask_np, norm_mean, norm_std,
        train_loader, val_loader, test_loader, device,
    )

    print(f"Starting {args.n_trials} new trials on {device}\n")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    _print_summary(study)


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
