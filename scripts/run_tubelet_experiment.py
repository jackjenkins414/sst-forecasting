"""
Tubelet Transformer experiment — train, evaluate, and visualise results.

Mirrors the structure of run_lstm_experiment.py so results are directly
comparable. Each run is saved to experiments/results/run_YYYYMMDD_HHMMSS/
with config.json, metrics.json, a checkpoint, and a summary figure.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import zarr

from src.data.dataloaders import create_dataloaders
from src.models.tubelet_transformer import TubeletTransformer
from src.training.losses import AnomalyWeightedMSE
from src.training.train import train_model
from src.training.evaluate import predict
from src.baselines.persistence import persistence_forecast
from src.utils.metrics import rmse, rmse_per_step, mae, skill_score


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ZARR_PATH = PROJECT_ROOT / "data/processed/oisst_coralsea.zarr"
RESULTS_DIR = PROJECT_ROOT / "experiments/results"

CONTEXT_LEN = 90
HORIZON = 7
BATCH_SIZE = 16

D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 4
D_FF     = 512
T_S      = 5    # days per tubelet  -> T' = 90/5 = 18 time tokens
P_H      = 3    # patch height      -> 81/3  = 27 patch rows
P_W      = 11   # patch width       -> 121/11 = 11 patch cols
DROPOUT  = 0.1

NUM_EPOCHS          = 50
EARLY_STOP_PATIENCE = 5
LEARNING_RATE       = 1e-3
WEIGHT_DECAY        = 1e-4
GRAD_CLIP           = 1.0
LR_FACTOR           = 0.5    # multiply LR by this when plateau detected
LR_PATIENCE         = 5      # epochs without improvement before reducing LR
ANOMALY_ALPHA       = 1.0    # anomaly-weighted loss strength (0 = standard MSE)

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_results(
    train_losses, val_losses,
    model_rmse_per_step, persistence_rmse_per_step,
    spatial_rmse,
    sample_pred, sample_truth,
    lat, lon,
    horizon,
    save_path,
):
    """Six-panel summary figure saved to save_path."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Tubelet Transformer — SST Forecasting (Coral Sea)", fontsize=14)

    days = np.arange(1, horizon + 1)
    extent = [lon[0], lon[-1], lat[0], lat[-1]]  # [W, E, S, N]

    # --- [0,0] Training curves ---
    ax = axes[0, 0]
    ax.plot(train_losses, label="train")
    ax.plot(val_losses,   label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (normalised)")
    ax.set_title("Training & Validation Loss")
    ax.legend()

    # --- [0,1] RMSE per forecast day ---
    ax = axes[0, 1]
    ax.plot(days, model_rmse_per_step,       marker="o", label="Tubelet Transformer")
    ax.plot(days, persistence_rmse_per_step, marker="s", label="Persistence", linestyle="--")
    ax.set_xlabel("Forecast day")
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("RMSE per Forecast Step")
    ax.legend()

    # --- [0,2] Skill score per forecast day ---
    ax = axes[0, 2]
    skills = [skill_score(m, p) for m, p in zip(model_rmse_per_step, persistence_rmse_per_step)]
    colours = ["steelblue" if s >= 0 else "tomato" for s in skills]
    ax.bar(days, skills, color=colours)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Forecast day")
    ax.set_ylabel("Skill score vs persistence")
    ax.set_title("Skill Score (positive = better than persistence)")

    # --- [1,0] Spatial RMSE heatmap ---
    ax = axes[1, 0]
    im = ax.imshow(
        spatial_rmse,
        origin="lower", extent=extent,
        cmap="YlOrRd", vmin=0,
        aspect="auto",
    )
    fig.colorbar(im, ax=ax, label="RMSE (°C)")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Spatial RMSE — mean over test set & all steps")

    # --- [1,1] Sample prediction at day HORIZON ---
    vmax = float(np.nanmax(np.abs(sample_pred)))
    vmax = max(vmax, 0.5)

    ax = axes[1, 1]
    im = ax.imshow(
        sample_pred, origin="lower", extent=extent,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
    )
    fig.colorbar(im, ax=ax, label="SST anomaly (°C)")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"Predicted anomaly — day {horizon}")

    # --- [1,2] Corresponding ground truth ---
    ax = axes[1, 2]
    im = ax.imshow(
        sample_truth, origin="lower", extent=extent,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
    )
    fig.colorbar(im, ax=ax, label="SST anomaly (°C)")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"Ground truth anomaly — day {horizon}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run a Tubelet Transformer experiment.")
    p.add_argument("--context_len",         type=int,   default=CONTEXT_LEN)
    p.add_argument("--horizon",             type=int,   default=HORIZON)
    p.add_argument("--batch_size",          type=int,   default=BATCH_SIZE)
    p.add_argument("--d_model",             type=int,   default=D_MODEL)
    p.add_argument("--n_heads",             type=int,   default=N_HEADS)
    p.add_argument("--n_layers",            type=int,   default=N_LAYERS)
    p.add_argument("--d_ff",                type=int,   default=D_FF)
    p.add_argument("--t_s",                 type=int,   default=T_S)
    p.add_argument("--p_h",                 type=int,   default=P_H)
    p.add_argument("--p_w",                 type=int,   default=P_W)
    p.add_argument("--dropout",             type=float, default=DROPOUT)
    p.add_argument("--num_epochs",          type=int,   default=NUM_EPOCHS)
    p.add_argument("--early_stop_patience", type=int,   default=EARLY_STOP_PATIENCE)
    p.add_argument("--lr",                  type=float, default=LEARNING_RATE)
    p.add_argument("--weight_decay",        type=float, default=WEIGHT_DECAY)
    p.add_argument("--grad_clip",           type=float, default=GRAD_CLIP)
    p.add_argument("--lr_factor",           type=float, default=LR_FACTOR)
    p.add_argument("--lr_patience",         type=int,   default=LR_PATIENCE)
    p.add_argument("--seed",                type=int,   default=RANDOM_SEED)
    p.add_argument("--alpha",               type=float, default=ANOMALY_ALPHA)
    return p.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Each run gets its own timestamped directory
    run_dir = RESULTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Persist all hyperparameters immediately so the run is self-documented
    config = {
        "model_type":          "tubelet",
        "context_len":         args.context_len,
        "horizon":             args.horizon,
        "batch_size":          args.batch_size,
        "d_model":             args.d_model,
        "n_heads":             args.n_heads,
        "n_layers":            args.n_layers,
        "d_ff":                args.d_ff,
        "t_s":                 args.t_s,
        "p_h":                 args.p_h,
        "p_w":                 args.p_w,
        "dropout":             args.dropout,
        "num_epochs":          args.num_epochs,
        "early_stop_patience": args.early_stop_patience,
        "learning_rate":       args.lr,
        "weight_decay":        args.weight_decay,
        "grad_clip":           args.grad_clip,
        "lr_factor":           args.lr_factor,
        "lr_patience":         args.lr_patience,
        "random_seed":         args.seed,
        "anomaly_alpha":       args.alpha,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load grid metadata from Zarr
    root = zarr.open_group(str(ZARR_PATH), mode="r")
    norm_mean    = float(root.attrs["norm_mean"])
    norm_std     = float(root.attrs["norm_std"])
    land_mask_np = np.array(root["land_mask"])   # (H, W) bool, True = ocean
    lat          = np.array(root["lat"])          # (81,)
    lon          = np.array(root["lon"])          # (121,)
    H, W         = land_mask_np.shape

    print(f"Grid: {H}x{W}  ocean cells: {int(land_mask_np.sum())}")
    print(f"norm_mean={norm_mean:.5f}  norm_std={norm_std:.5f}")

    train_loader, val_loader, test_loader = create_dataloaders(
        zarr_path=ZARR_PATH,
        context_len=args.context_len,
        horizon=args.horizon,
        batch_size=args.batch_size,
    )

    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    land_mask_torch = torch.from_numpy(land_mask_np).to(device)

    model = TubeletTransformer(
        H=H, W=W,
        context_len=args.context_len,
        horizon=args.horizon,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        t_s=args.t_s,
        p_h=args.p_h,
        p_w=args.p_w,
        dropout=args.dropout,
    ).to(device)

    print(f"Parameters: {model.count_parameters():,}")
    print(f"Device: {device}")
    print(f"Time tokens T': {model.T_prime}  Spatial patches P: {model.P}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = AnomalyWeightedMSE(alpha=args.alpha)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience,
        threshold=1e-3, min_lr=1e-5, cooldown=2,
    )

    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        land_mask=land_mask_torch,
        grad_clip=args.grad_clip,
        scheduler=scheduler,
        early_stop_patience=args.early_stop_patience,
    )

    ckpt_path = run_dir / "tubelet_ckpt.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    # Evaluate on test set
    test_preds_norm, test_targets_norm = predict(model, test_loader, device)

    test_preds   = test_preds_norm   * norm_std + norm_mean
    test_targets = test_targets_norm * norm_std + norm_mean

    model_rmse_steps = rmse_per_step(test_preds, test_targets, land_mask=land_mask_np)
    model_rmse_mean  = float(model_rmse_steps.mean())
    model_mae        = mae(test_preds, test_targets, land_mask=land_mask_np)

    print("Evaluating persistence baseline...")
    all_X, all_y = [], []
    for bx, by in test_loader:
        all_X.append(bx.numpy())
        all_y.append(by.numpy())
    test_X_norm = np.concatenate(all_X, axis=0)
    test_y_norm = np.concatenate(all_y, axis=0)

    pers_preds_norm = persistence_forecast(test_X_norm, horizon=HORIZON)
    pers_preds      = pers_preds_norm * norm_std + norm_mean
    pers_targets    = test_y_norm     * norm_std + norm_mean

    pers_rmse_steps = rmse_per_step(pers_preds, pers_targets, land_mask=land_mask_np)
    pers_rmse_mean  = float(pers_rmse_steps.mean())
    pers_mae        = mae(pers_preds, pers_targets, land_mask=land_mask_np)

    rmse_skill = skill_score(model_rmse_mean, pers_rmse_mean)
    mae_skill  = skill_score(model_mae, pers_mae)

    # Save metrics
    metrics = {
        "epochs_trained": len(train_losses),
        "best_val_loss":  float(min(val_losses)),
        "mean_rmse": {
            "model": model_rmse_mean,
            "persistence": pers_rmse_mean,
            "skill": rmse_skill,
        },
        "mean_mae": {
            "model": model_mae,
            "persistence": pers_mae,
            "skill": mae_skill,
        },
        "rmse_per_step": {
            f"day_{i+1}": {
                "model":       float(mr),
                "persistence": float(pr),
                "skill":       float(skill_score(mr, pr)),
            }
            for i, (mr, pr) in enumerate(zip(model_rmse_steps, pers_rmse_steps))
        },
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {run_dir / 'metrics.json'}")

    # Print summary
    print("\nTubelet Transformer — Test Results")
    print("-----------------------------------")
    print(f"  Context: {args.context_len} days  Horizon: {args.horizon} days  Device: {device}")
    print(f"  T'={model.T_prime} time tokens  P={model.P} spatial patches")
    print()
    print("  RMSE per step (°C):")
    for step, (mr, pr) in enumerate(zip(model_rmse_steps, pers_rmse_steps), start=1):
        ss = skill_score(mr, pr)
        print(f"    day {step}: model={mr:.4f}  persistence={pr:.4f}  skill={ss:+.4f}")
    print(f"  Mean RMSE : model={model_rmse_mean:.4f}  persistence={pers_rmse_mean:.4f}")
    print(f"  Mean MAE  : model={model_mae:.4f}  persistence={pers_mae:.4f}")
    print(f"  RMSE skill: {rmse_skill:+.4f}")
    print(f"  MAE  skill: {mae_skill:+.4f}")

    # Spatial RMSE heatmap
    spatial_rmse = np.sqrt(np.mean((test_preds - test_targets) ** 2, axis=(0, 1)))
    spatial_rmse[~land_mask_np] = np.nan

    # Sample with largest ocean-mean absolute anomaly at final horizon step
    ocean_abs = np.abs(test_targets[:, -1, :, :])[:, land_mask_np].mean(axis=1)
    sample_idx = int(np.argmax(ocean_abs))

    sample_pred  = test_preds[sample_idx, -1].copy()
    sample_truth = test_targets[sample_idx, -1].copy()
    sample_pred[~land_mask_np]  = np.nan
    sample_truth[~land_mask_np] = np.nan

    plot_results(
        train_losses=train_losses,
        val_losses=val_losses,
        model_rmse_per_step=model_rmse_steps,
        persistence_rmse_per_step=pers_rmse_steps,
        spatial_rmse=spatial_rmse,
        sample_pred=sample_pred,
        sample_truth=sample_truth,
        lat=lat,
        lon=lon,
        horizon=args.horizon,
        save_path=run_dir / "tubelet_results.png",
    )


if __name__ == "__main__":
    main()
