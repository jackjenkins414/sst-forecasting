"""
Optuna hyperparameter search for the SstPatchTransformer (cluster version).

Mirrors the other models' HPO scripts so results drop straight into
compare_runs.py / retrain_best.py: each trial writes
experiments/results/run_*/{config.json,metrics.json} and the study is
persisted to experiments/optuna_patch.db (study name: patch_transformer_hpo).

Patches are fixed at 9x11 — the only sizes that evenly divide the 81x121 Coral
Sea grid (81 = 3^4, 121 = 11^2). The search covers the Transformer
hyperparameters. n_heads always divides d_model for every choice below.

This version is sized for a big-VRAM cluster GPU (e.g. A100): the full d_model
range and a configurable batch size are exposed. On a 12 GB card the larger
configs will OOM — there is an OOM->prune safety net, but prefer --batch-size 8
and the smaller d_model values there.

Usage
-----
    python scripts/run_optuna_patch_transformer.py --n_trials 30 --batch-size 32
    python scripts/run_optuna_patch_transformer.py --show         # summary only
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.optuna_plots import save_study_plots

import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
import torch
import torch.optim as optim
import zarr

from src.data.dataloaders import create_dataloaders
from src.models.patch_transformer import SstPatchTransformer
from src.training.losses import AnomalyWeightedMSE
from src.training.train import train_model
from src.training.evaluate import predict
from src.utils.metrics import rmse_per_step

optuna.logging.set_verbosity(optuna.logging.WARNING)

# A100 optimisations: TF32 matmul/cuDNN (A100 natively supports TF32),
# cuDNN autotuner (fixed input shapes means one-time cost).
# Only active when --a100 flag is passed from the PBS script.
def _apply_a100_opts():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("[optuna] A100 opts: TF32=True, cudnn.benchmark=True")

# ---------------------------------------------------------------------------
# Fixed — not part of the search
# ---------------------------------------------------------------------------

ZARR_PATH   = PROJECT_ROOT / "data/processed/oisst_coralsea.zarr"
RESULTS_DIR = PROJECT_ROOT / "experiments/results"
DB_PATH     = PROJECT_ROOT / "experiments/optuna_patch.db"

CONTEXT_LEN         = 90
HORIZON             = 7
PATCH_HEIGHT        = 9    # divides 81
PATCH_WIDTH         = 11   # divides 121
NUM_EPOCHS          = 50
EARLY_STOP_PATIENCE = 5
RANDOM_SEED         = 42

# ---------------------------------------------------------------------------
# Search space.  Full range for cluster GPUs.  Every n_heads divides every
# d_model, so no invalid attention configs are proposed.
# ---------------------------------------------------------------------------

SEARCH_SPACE = {
    "lr":            FloatDistribution(1e-4, 2e-3, log=True),
    "d_model":       CategoricalDistribution([64, 128, 256, 512]),
    "n_blocks":      IntDistribution(2, 4),
    "n_heads":       CategoricalDistribution([4, 8, 16]),
    "d_ff":          CategoricalDistribution([256, 512, 1024]),
    "dropout":       FloatDistribution(0.0, 0.30),
    "lr_factor":     FloatDistribution(0.4, 0.8),
    "anomaly_alpha": FloatDistribution(0.0, 0.20),
}

# Jack's validated default, injected as the first trial (alpha=0 => plain MSE).
SEED_CONFIG = {
    "lr":            1e-3,
    "d_model":       128,
    "n_blocks":      2,
    "n_heads":       4,
    "d_ff":          512,
    "dropout":       0.1,
    "lr_factor":     0.5,
    "anomaly_alpha": 0.0,
}


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(land_mask_np, norm_mean, norm_std, loaders, batch_size, device):
    train_loader, val_loader, test_loader = loaders
    land_mask_torch = torch.from_numpy(land_mask_np).to(device)
    H, W = land_mask_np.shape

    def objective(trial: optuna.Trial) -> float:
        lr            = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
        d_model       = trial.suggest_categorical("d_model", [64, 128, 256, 512])
        n_blocks      = trial.suggest_int("n_blocks", 2, 4)
        n_heads       = trial.suggest_categorical("n_heads", [4, 8, 16])
        d_ff          = trial.suggest_categorical("d_ff", [256, 512, 1024])
        dropout       = trial.suggest_float("dropout", 0.0, 0.30)
        lr_factor     = trial.suggest_float("lr_factor", 0.4, 0.8)
        anomaly_alpha = trial.suggest_float("anomaly_alpha", 0.0, 0.20)

        run_dir = RESULTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}"
        run_dir.mkdir(parents=True, exist_ok=True)

        model = None
        optimizer = None
        try:
            config = {
                "model_type":          "patch_transformer",
                "context_len":         CONTEXT_LEN,
                "horizon":             HORIZON,
                "batch_size":          batch_size,
                "patch_height":        PATCH_HEIGHT,
                "patch_width":         PATCH_WIDTH,
                "d_model":             d_model,
                "n_blocks":            n_blocks,
                "n_heads":             n_heads,
                "d_ff":                d_ff,
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

            model = SstPatchTransformer(
                height=H, width=W,
                patch_height=PATCH_HEIGHT, patch_width=PATCH_WIDTH,
                seq_len=CONTEXT_LEN, horizon=HORIZON,
                d_model=d_model, n_blocks=n_blocks, n_heads=n_heads,
                d_ff=d_ff, dropout=dropout,
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
                  f"{len(train_losses)} epochs | d_model={d_model} blocks={n_blocks} "
                  f"heads={n_heads} d_ff={d_ff} | {run_dir.name}")
            return mean_rmse

        except optuna.exceptions.TrialPruned:
            raise
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"  Trial {trial.number:03d} OOM (d_model={d_model}, blocks={n_blocks}, "
                  f"B={batch_size}) — pruned. Lower --batch-size if this is frequent.")
            raise optuna.exceptions.TrialPruned()
        finally:
            if model is not None:
                del model
            if optimizer is not None:
                del optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _fix_zombie_trials(study: optuna.Study) -> int:
    zombies = [t for t in study.trials if t.state.name == "RUNNING"]
    storage = study._storage
    for t in zombies:
        storage.set_trial_state_values(t._trial_id, state=optuna.trial.TrialState.FAIL)
        print(f"  Marked zombie trial {t.number} as FAIL")
    return len(zombies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials",   type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Per-GPU batch size (32-64 on an A100; use 8 on a 12GB card).")
    parser.add_argument("--show",       action="store_true")
    parser.add_argument("--a100",       action="store_true",
                        help="Enable A100-specific optimisations (TF32, cuDNN benchmark).")
    args = parser.parse_args()

    if args.a100:
        _apply_a100_opts()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name="patch_transformer_hpo",
        storage=f"sqlite:///{DB_PATH}",
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    n_zombies = _fix_zombie_trials(study)
    if n_zombies:
        print(f"Fixed {n_zombies} zombie trial(s) from a previous crash")

    if not any(t.params == SEED_CONFIG for t in study.trials):
        study.enqueue_trial(SEED_CONFIG)
        print("Queued Jack's default config (alpha=0) as the first trial")

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

    loaders = create_dataloaders(
        zarr_path=ZARR_PATH, context_len=CONTEXT_LEN,
        horizon=HORIZON, batch_size=args.batch_size,
    )

    objective = make_objective(land_mask_np, norm_mean, norm_std,
                               loaders, args.batch_size, device)

    print(f"Starting {args.n_trials} trials on {device}  "
          f"(patch {PATCH_HEIGHT}x{PATCH_WIDTH}, B={args.batch_size})\n")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    _print_summary(study)
    save_study_plots(study, model_name="patch_transformer")


def _print_summary(study: optuna.Study):
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        print("No completed trials yet.")
        return
    best = study.best_trial
    print("\n--- Patch Transformer Optuna Study Summary ---")
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
