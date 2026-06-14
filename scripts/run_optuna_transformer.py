"""
Optuna hyperparameter search for Ayush's SpatialFlatTransformer.

12 trials, patience=5, max_epochs=15.
Persisted to experiments/optuna_transformer.db.
Results saved in the same format as run_tubelet_experiment.py so
compare_runs.py can plot them alongside Tubelet results.

Usage
-----
    python scripts/run_optuna_transformer.py              # 12 trials (default)
    python scripts/run_optuna_transformer.py --n_trials 6
    python scripts/run_optuna_transformer.py --show       # summary only
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import zarr

from sst_forecasting.data.dataset import SSTWindowDataset
from sst_forecasting.models.transformer import SpatialFlatTransformer
from src.baselines.persistence import persistence_forecast
from src.utils.metrics import rmse_per_step, skill_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

ZARR_PATH   = PROJECT_ROOT / "data/processed/oisst_coralsea.zarr"
RESULTS_DIR = PROJECT_ROOT / "experiments/results"
DB_PATH     = PROJECT_ROOT / "experiments/optuna_transformer.db"

CONTEXT_LEN         = 90
HORIZON             = 7
BATCH_SIZE          = 8
MAX_EPOCHS          = 50   # bumped from 15 to match LSTM/Informer/Tubelet budget
EARLY_STOP_PATIENCE = 5
RANDOM_SEED         = 42

SEARCH_SPACE = {
    "lr":       FloatDistribution(1e-4, 1e-3, log=True),
    "d_model":  CategoricalDistribution([128, 256]),
    "n_heads":  CategoricalDistribution([4, 8]),
    "n_layers": IntDistribution(2, 5),
    "ffn_dim":  CategoricalDistribution([256, 512]),
    "dropout":  FloatDistribution(0.05, 0.25),
    "lr_factor":FloatDistribution(0.4, 0.7),
}

def _make_loader(zarr_path, split, batch_size, context_len, horizon):
    ds = SSTWindowDataset(
        zarr_path=str(zarr_path),
        split=split,
        context_len=context_len,
        horizon=horizon,
    )
    return DataLoader(
        ds, batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0, drop_last=False,
    )

def make_objective(land_mask_np, norm_mean, norm_std,
                   train_loader, val_loader, test_loader,
                   pers_rmse_steps, device):

    H, W = land_mask_np.shape
    criterion = nn.MSELoss()
    ocean_mask = torch.from_numpy(land_mask_np).to(device)   # (H, W) bool

    def objective(trial: optuna.Trial) -> float:
        lr       = trial.suggest_float("lr", **_dist_kwargs(SEARCH_SPACE["lr"]))
        d_model  = trial.suggest_categorical("d_model", SEARCH_SPACE["d_model"].choices)
        n_heads  = trial.suggest_categorical("n_heads", SEARCH_SPACE["n_heads"].choices)
        n_layers = trial.suggest_int("n_layers", SEARCH_SPACE["n_layers"].low, SEARCH_SPACE["n_layers"].high)
        ffn_dim  = trial.suggest_categorical("ffn_dim", SEARCH_SPACE["ffn_dim"].choices)
        dropout  = trial.suggest_float("dropout", SEARCH_SPACE["dropout"].low, SEARCH_SPACE["dropout"].high)
        lr_factor= trial.suggest_float("lr_factor", SEARCH_SPACE["lr_factor"].low, SEARCH_SPACE["lr_factor"].high)

        if d_model % n_heads != 0:
            raise optuna.exceptions.TrialPruned()

        run_dir = RESULTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}"
        run_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "model_type": "transformer",
            "context_len": CONTEXT_LEN, "horizon": HORIZON, "batch_size": BATCH_SIZE,
            "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
            "ffn_dim": ffn_dim, "dropout": dropout,
            "num_epochs": MAX_EPOCHS, "early_stop_patience": EARLY_STOP_PATIENCE,
            "learning_rate": lr, "lr_factor": lr_factor, "lr_patience": 5,
            "weight_decay": 1e-4, "grad_clip": 1.0,
            "anomaly_alpha": 0.0, "random_seed": RANDOM_SEED,
            "optuna_trial": trial.number,
        }
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        model = SpatialFlatTransformer(
            H=H, W=W, context_len=CONTEXT_LEN, horizon=HORIZON,
            d_model=d_model, nhead=n_heads, num_encoder_layers=n_layers,
            dim_feedforward=ffn_dim, dropout=dropout,
        ).to(device)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor, patience=5, min_lr=1e-6,
        )

        best_val  = float("inf")
        best_state= None
        no_improve= 0
        train_losses, val_losses = [], []

        for epoch in range(MAX_EPOCHS):
            # --- train ---
            model.train()
            t_loss, n_batches = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                mask = ocean_mask.unsqueeze(0).unsqueeze(0).expand_as(pred)
                loss = criterion(pred[mask], y[mask])
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                t_loss += loss.item(); n_batches += 1
            train_losses.append(t_loss / max(n_batches, 1))

            # --- val ---
            model.eval()
            v_loss, n_batches = 0.0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    mask = ocean_mask.unsqueeze(0).unsqueeze(0).expand_as(pred)
                    v_loss += criterion(pred[mask], y[mask]).item()
                    n_batches += 1
            val_loss = v_loss / max(n_batches, 1)
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            trial.report(val_loss, epoch)
            if trial.should_prune():
                print(f"  Trial {trial.number:03d} pruned at epoch {epoch+1}")
                raise optuna.exceptions.TrialPruned()

            if no_improve >= EARLY_STOP_PATIENCE:
                break

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        # --- test ---
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in test_loader:
                pred = model(x.to(device)).cpu().numpy()
                all_preds.append(pred * norm_std + norm_mean)
                all_targets.append(y.numpy() * norm_std + norm_mean)

        test_preds   = np.concatenate(all_preds,   axis=0)  # (N, h, H, W)
        test_targets = np.concatenate(all_targets, axis=0)

        rmse_steps = rmse_per_step(test_preds, test_targets, land_mask=land_mask_np)
        mean_rmse  = float(rmse_steps.mean())

        metrics = {
            "epochs_trained": len(train_losses),
            "best_val_loss":  best_val,
            "mean_rmse": {
                "model":       mean_rmse,
                "persistence": float(pers_rmse_steps.mean()),
                "skill":       float(skill_score(mean_rmse, float(pers_rmse_steps.mean()))),
            },
            "rmse_per_step": {
                f"day_{i+1}": {
                    "model":       float(mr),
                    "persistence": float(pr),
                    "skill":       float(skill_score(mr, pr)),
                }
                for i, (mr, pr) in enumerate(zip(rmse_steps, pers_rmse_steps))
            },
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

def _compute_persistence_baseline(zarr_path, land_mask_np, norm_mean, norm_std,
                                   context_len, horizon, batch_size):
    test_loader = _make_loader(zarr_path, "test", batch_size, context_len, horizon)
    all_X, all_y = [], []
    for bx, by in test_loader:
        all_X.append(bx.numpy())
        all_y.append(by.numpy())
    X_norm = np.concatenate(all_X, axis=0)  # (N, L, 1, H, W)
    y_norm = np.concatenate(all_y, axis=0)

    pers_norm = persistence_forecast(X_norm, horizon=horizon)
    pers      = pers_norm * norm_std + norm_mean
    targets   = y_norm   * norm_std + norm_mean
    return rmse_per_step(pers, targets, land_mask=land_mask_np)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=12)
    parser.add_argument("--show",     action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name="transformer_hpo_v1",
        storage=f"sqlite:///{DB_PATH}",
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3),
    )

    if args.show:
        _print_summary(study)
        return

    root      = zarr.open_group(str(ZARR_PATH), mode="r")
    norm_mean = float(root.attrs["norm_mean"])
    norm_std  = float(root.attrs["norm_std"])
    land_mask_np = np.array(root["land_mask"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    train_loader = _make_loader(ZARR_PATH, "train", BATCH_SIZE, CONTEXT_LEN, HORIZON)
    val_loader   = _make_loader(ZARR_PATH, "val",   BATCH_SIZE, CONTEXT_LEN, HORIZON)
    test_loader  = _make_loader(ZARR_PATH, "test",  BATCH_SIZE, CONTEXT_LEN, HORIZON)

    print("Computing persistence baseline...")
    pers_rmse_steps = _compute_persistence_baseline(
        ZARR_PATH, land_mask_np, norm_mean, norm_std,
        CONTEXT_LEN, HORIZON, BATCH_SIZE,
    )
    print(f"Persistence mean RMSE: {pers_rmse_steps.mean():.4f}")

    objective = make_objective(
        land_mask_np, norm_mean, norm_std,
        train_loader, val_loader, test_loader,
        pers_rmse_steps, device,
    )

    print(f"Starting {args.n_trials} transformer trials on {device}\n")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    _print_summary(study)

def _print_summary(study: optuna.Study):
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        print("No completed trials yet.")
        return
    best = study.best_trial
    print("\n--- Transformer HPO Summary ---")
    print(f"  Completed trials : {len(trials)}")
    print(f"  Best mean RMSE   : {best.value:.4f}")
    print("  Best params:")
    for k, v in best.params.items():
        print(f"    {k:20s} {v}")
    print("\n  Top 5 trials:")
    for i, t in enumerate(sorted(trials, key=lambda t: t.value)[:5], 1):
        print(f"  {i}. RMSE={t.value:.4f}  " +
              "  ".join(f"{k}={v}" for k, v in t.params.items()))

if __name__ == "__main__":
    main()
