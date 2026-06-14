"""
Optuna hyperparameter search for the Tubelet Transformer.

Seeds the Bayesian optimiser with all previous run results so it starts
from what we already know, then searches intelligently for better configs.
Bad trials are pruned mid-training so GPU time isn't wasted.

Each trial saves its results under experiments/results/ like a normal run.
The study is persisted to experiments/optuna_study.db so it survives restarts
and every new run you do (manually or via this script) adds to the knowledge.

Usage
-----
    python scripts/run_optuna_search.py              # 20 trials (default)
    python scripts/run_optuna_search.py --n_trials 50
    python scripts/run_optuna_search.py --show       # print study summary only
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
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
import torch
import torch.optim as optim
import zarr

from src.data.dataloaders import create_dataloaders
from src.models.tubelet_transformer import TubeletTransformer
from src.training.losses import AnomalyWeightedMSE
from src.training.train import train_model
from src.training.evaluate import predict
from src.utils.metrics import rmse_per_step, skill_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Fixed - not part of the search

ZARR_PATH   = PROJECT_ROOT / "data/processed/oisst_coralsea.zarr"
RESULTS_DIR = PROJECT_ROOT / "experiments/results"
DB_PATH     = PROJECT_ROOT / "experiments/optuna_study.db"

CONTEXT_LEN         = 90
HORIZON             = 7
BATCH_SIZE          = 8   # halved from 16 - gives deeper trials headroom on 12 GB VRAM
D_MODEL             = 128
D_FF                = 512
T_S                 = 5
P_H                 = 3   # fine patches - ablation proved this is better
P_W                 = 11
NUM_EPOCHS          = 50
EARLY_STOP_PATIENCE = 5   # fixed - keeps all trials comparable in length
RANDOM_SEED         = 42

# Search space - edit ranges here to change what Optuna explores
#
# FloatDistribution(low, high, log=True)  - good for LR (spans orders of magnitude)
# FloatDistribution(low, high)            - linear range
# IntDistribution(low, high)             - integers inclusive
# CategoricalDistribution([a, b, c])     - discrete choices
#
# n_heads must divide d_model evenly.
# d_model=128: valid heads = 4, 8, 16   (16 gives only 8 dims/head - narrow but ok)
# d_model=256: valid heads = 8, 16      (16 gives 16 dims/head - meaningful)

SEARCH_SPACE = {
    "lr":       FloatDistribution(3e-4, 1.5e-3, log=True), # best runs clustered ~7e-4
    "d_model":  CategoricalDistribution([128, 256]),        # 256 needs B=8 to avoid OOM
    "n_heads":  CategoricalDistribution([8, 16]),           # 8 proven; 16 worth trying with d=256
    "n_layers": IntDistribution(4, 6),                      # 2-3 layers clearly underfit
    "lr_factor":FloatDistribution(0.55, 0.75),              # best runs ~0.67
    "alpha":    FloatDistribution(0.0, 0.2),                # best runs ~0.07-0.15; >0.5 hurts
    "dropout":  FloatDistribution(0.15, 0.30),              # best runs ~0.20
}

# Seed study with previous runs

def _params_from_config(config: dict) -> dict | None:
    """Extract search-space params from a saved config.json.
    Returns None if any param falls outside the defined search space bounds."""
    mapping = {
        "lr":       config.get("learning_rate", config.get("lr")),
        "d_model":  config.get("d_model", 128),
        "n_heads":  config.get("n_heads"),
        "n_layers": config.get("n_layers"),
        "lr_factor":config.get("lr_factor"),
        "alpha":    config.get("anomaly_alpha", 0.0),
        "dropout":  config.get("dropout"),
    }
    if any(v is None for v in mapping.values()):
        return None

    # Validate against search space bounds
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
    """Inject existing run results into the study as completed trials."""
    loaded = 0
    for run_dir in sorted(RESULTS_DIR.glob("run_*/")):
        config_file  = run_dir / "config.json"
        metrics_file = run_dir / "metrics.json"
        if not config_file.exists() or not metrics_file.exists():
            continue

        with open(config_file)  as f: config  = json.load(f)
        with open(metrics_file) as f: metrics = json.load(f)

        params = _params_from_config(config)
        if params is None:
            continue

        # Skip if this trial already exists in the study (avoid duplicates on restart)
        already = any(t.params == params for t in study.trials)
        if already:
            continue

        objective_value = metrics["mean_rmse"]["model"]

        study.add_trial(
            optuna.trial.create_trial(
                params=params,
                distributions=SEARCH_SPACE,
                value=objective_value,
            )
        )
        loaded += 1

    return loaded

# Objective

def make_objective(root, land_mask_np, lat, lon, norm_mean, norm_std,
                   train_loader, val_loader, test_loader, device):

    land_mask_torch = torch.from_numpy(land_mask_np).to(device)
    H, W = land_mask_np.shape

    def objective(trial: optuna.Trial) -> float:
        lr        = trial.suggest_float("lr", **_dist_kwargs(SEARCH_SPACE["lr"]))
        d_model   = trial.suggest_categorical("d_model", SEARCH_SPACE["d_model"].choices)
        n_heads   = trial.suggest_categorical("n_heads", SEARCH_SPACE["n_heads"].choices)
        n_layers  = trial.suggest_int("n_layers", SEARCH_SPACE["n_layers"].low, SEARCH_SPACE["n_layers"].high)
        lr_factor = trial.suggest_float("lr_factor", SEARCH_SPACE["lr_factor"].low, SEARCH_SPACE["lr_factor"].high)
        alpha     = trial.suggest_float("alpha", SEARCH_SPACE["alpha"].low, SEARCH_SPACE["alpha"].high)
        dropout   = trial.suggest_float("dropout", SEARCH_SPACE["dropout"].low, SEARCH_SPACE["dropout"].high)

        # n_heads must divide d_model - skip invalid combos
        if d_model % n_heads != 0:
            raise optuna.exceptions.TrialPruned()

        d_ff = d_model * 4  # maintain standard 4x FFN ratio

        run_dir = RESULTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}"
        run_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "model_type": "tubelet",
            "context_len": CONTEXT_LEN, "horizon": HORIZON, "batch_size": BATCH_SIZE,
            "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers, "d_ff": d_ff,
            "t_s": T_S, "p_h": P_H, "p_w": P_W, "dropout": dropout,
            "num_epochs": NUM_EPOCHS, "early_stop_patience": EARLY_STOP_PATIENCE,
            "learning_rate": lr, "weight_decay": 1e-4, "grad_clip": 1.0,
            "lr_factor": lr_factor, "lr_patience": 5, "random_seed": RANDOM_SEED,
            "anomaly_alpha": alpha, "optuna_trial": trial.number,
        }
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        model = TubeletTransformer(
            H=H, W=W, context_len=CONTEXT_LEN, horizon=HORIZON,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
            t_s=T_S, p_h=P_H, p_w=P_W, dropout=dropout,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = AnomalyWeightedMSE(alpha=alpha)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor, patience=5,
            threshold=1e-3, min_lr=1e-5, cooldown=2,
        )

        # Pruning callback - reports val loss each epoch; Optuna kills bad trials early
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

        # Evaluate
        test_preds_norm, test_targets_norm = predict(model, test_loader, device)
        test_preds   = test_preds_norm   * norm_std + norm_mean
        test_targets = test_targets_norm * norm_std + norm_mean
        rmse_steps   = rmse_per_step(test_preds, test_targets, land_mask=land_mask_np)
        mean_rmse    = float(rmse_steps.mean())

        # Save metrics
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
    """Convert a distribution object to suggest_float keyword args."""
    kwargs = {"name": None, "low": dist.low, "high": dist.high}
    if hasattr(dist, "log") and dist.log:
        kwargs["log"] = True
    return {k: v for k, v in kwargs.items() if k != "name"}

# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--show",     action="store_true", help="Print study summary and exit")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name="tubelet_hpo_v2",
        storage=f"sqlite:///{DB_PATH}",
        direction="minimize",       # minimise mean RMSE
        load_if_exists=True,        # resume if study already exists
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    # Seed with everything we've already run
    n_loaded = load_previous_runs(study)
    print(f"Loaded {n_loaded} previous runs into study "
          f"({len(study.trials)} total trials so far)")

    if args.show:
        _print_summary(study)
        return

    # Load data once - shared across all trials
    root         = zarr.open_group(str(ZARR_PATH), mode="r")
    norm_mean    = float(root.attrs["norm_mean"])
    norm_std     = float(root.attrs["norm_std"])
    land_mask_np = np.array(root["land_mask"])
    lat          = np.array(root["lat"])
    lon          = np.array(root["lon"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    train_loader, val_loader, test_loader = create_dataloaders(
        zarr_path=ZARR_PATH, context_len=CONTEXT_LEN,
        horizon=HORIZON, batch_size=BATCH_SIZE,
    )

    objective = make_objective(
        root, land_mask_np, lat, lon, norm_mean, norm_std,
        train_loader, val_loader, test_loader, device,
    )

    print(f"Starting {args.n_trials} new trials on {device}\n")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    _print_summary(study)
    save_study_plots(study, model_name="tubelet")

def _print_summary(study: optuna.Study):
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        print("No completed trials yet.")
        return

    best = study.best_trial
    print("\n--- Optuna Study Summary ---")
    print(f"  Completed trials : {len(trials)}")
    print(f"  Best mean RMSE   : {best.value:.4f}")
    print(f"  Best params:")
    for k, v in best.params.items():
        print(f"    {k:25s} {v}")

    print("\n  Top 5 trials:")
    top = sorted(trials, key=lambda t: t.value)[:5]
    for i, t in enumerate(top, 1):
        print(f"  {i}. RMSE={t.value:.4f}  " +
              "  ".join(f"{k}={v}" for k, v in t.params.items()))

if __name__ == "__main__":
    main()
