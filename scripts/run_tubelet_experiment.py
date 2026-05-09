"""
Tubelet Transformer experiment — train, evaluate, and visualise results.

Mirrors the structure of run_lstm_experiment.py so results are directly
comparable. Figures are saved to experiments/results/.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import torch.optim as optim
import zarr

from src.data.dataloaders import create_dataloaders
from src.models.tubelet_transformer import TubeletTransformer
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
P_H      = 9    # patch height      -> 81/9  = 9 patch rows
P_W      = 11   # patch width       -> 121/11 = 11 patch cols
DROPOUT  = 0.1

NUM_EPOCHS          = 50   # upper bound — early stopping will trigger well before this
EARLY_STOP_PATIENCE = 7    # stop if val loss hasn't improved for 7 epochs
LEARNING_RATE       = 1e-3
WEIGHT_DECAY        = 1e-4
GRAD_CLIP           = 1.0

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
    norm_std,
    save_path,
):
    """Six-panel summary figure saved to save_path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Tubelet Transformer — SST Forecasting (Coral Sea)", fontsize=14)

    days = np.arange(1, HORIZON + 1)
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

    # --- [1,0] Spatial RMSE heatmap (mean over all test samples & steps) ---
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

    # --- [1,1] Sample prediction at day 7 (denormalised °C anomaly) ---
    # Pick the test sample with the largest mean absolute target anomaly
    vmax = float(np.nanmax(np.abs(sample_pred)))
    vmax = max(vmax, 0.5)  # floor so colorbar is readable on near-zero samples

    ax = axes[1, 1]
    im = ax.imshow(
        sample_pred, origin="lower", extent=extent,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
    )
    fig.colorbar(im, ax=ax, label="SST anomaly (°C)")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"Predicted anomaly — day {HORIZON}")

    # --- [1,2] Corresponding ground truth ---
    ax = axes[1, 2]
    im = ax.imshow(
        sample_truth, origin="lower", extent=extent,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
    )
    fig.colorbar(im, ax=ax, label="SST anomaly (°C)")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"Ground truth anomaly — day {HORIZON}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
        context_len=CONTEXT_LEN,
        horizon=HORIZON,
        batch_size=BATCH_SIZE,
    )

    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    land_mask_torch = torch.from_numpy(land_mask_np).to(device)

    model = TubeletTransformer(
        H=H, W=W,
        context_len=CONTEXT_LEN,
        horizon=HORIZON,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        t_s=T_S,
        p_h=P_H,
        p_w=P_W,
        dropout=DROPOUT,
    ).to(device)

    print(f"Parameters: {model.count_parameters():,}")
    print(f"Device: {device}")
    print(f"Time tokens T': {model.T_prime}  Spatial patches P: {model.P}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    # Halve LR whenever val loss doesn't improve for 3 consecutive epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, threshold=1e-3, min_lr=1e-5, cooldown=2,
    )

    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        land_mask=land_mask_torch,
        grad_clip=GRAD_CLIP,
        scheduler=scheduler,
        early_stop_patience=EARLY_STOP_PATIENCE,
    )

    # Save checkpoint
    ckpt_path = RESULTS_DIR / "tubelet_ckpt.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    # Evaluate on test set
    test_preds_norm, test_targets_norm = predict(model, test_loader, device)

    # Denormalise to physical °C space for reporting
    test_preds   = test_preds_norm   * norm_std + norm_mean
    test_targets = test_targets_norm * norm_std + norm_mean

    model_rmse_steps = rmse_per_step(test_preds, test_targets, land_mask=land_mask_np)
    model_rmse_mean  = float(model_rmse_steps.mean())
    model_mae        = mae(test_preds, test_targets, land_mask=land_mask_np)

    # Persistence baseline — use raw test inputs collected from the loader
    print("Evaluating persistence baseline...")
    all_X, all_y = [], []
    for bx, by in test_loader:
        all_X.append(bx.numpy())
        all_y.append(by.numpy())
    test_X_norm = np.concatenate(all_X, axis=0)
    test_y_norm = np.concatenate(all_y, axis=0)

    pers_preds_norm = persistence_forecast(test_X_norm, horizon=HORIZON)
    pers_preds      = pers_preds_norm   * norm_std + norm_mean
    pers_targets    = test_y_norm       * norm_std + norm_mean

    pers_rmse_steps = rmse_per_step(pers_preds, pers_targets, land_mask=land_mask_np)
    pers_rmse_mean  = float(pers_rmse_steps.mean())
    pers_mae        = mae(pers_preds, pers_targets, land_mask=land_mask_np)

    rmse_skill_score = skill_score(model_rmse_mean, pers_rmse_mean)
    mae_skill_score  = skill_score(model_mae, pers_mae)

    # Summary
    print("\nTubelet Transformer — Test Results")
    print("-----------------------------------")
    print(f"  Context: {CONTEXT_LEN} days  Horizon: {HORIZON} days  Device: {device}")
    print(f"  T'={model.T_prime} time tokens  P={model.P} spatial patches")
    print()
    print("  RMSE per step (°C):")
    for step, (mr, pr) in enumerate(zip(model_rmse_steps, pers_rmse_steps), start=1):
        ss = skill_score(mr, pr)
        print(f"    day {step}: model={mr:.4f}  persistence={pr:.4f}  skill={ss:+.4f}")
    print(f"  Mean RMSE : model={model_rmse_mean:.4f}  persistence={pers_rmse_mean:.4f}")
    print(f"  Mean MAE  : model={model_mae:.4f}  persistence={pers_mae:.4f}")
    print(f"  RMSE skill: {rmse_skill_score:+.4f}")
    print(f"  MAE  skill: {mae_skill_score:+.4f}")

    # Spatial RMSE heatmap: mean over all test samples and all horizon steps
    # Land cells stay NaN so they show as blank in the figure
    spatial_rmse = np.sqrt(np.mean((test_preds - test_targets) ** 2, axis=(0, 1)))  # (H, W)
    spatial_rmse[~land_mask_np] = np.nan

    # Pick the test sample with the largest ocean-mean absolute target anomaly
    # at the final horizon step — likely to show a meaningful forecast pattern
    ocean_abs = np.abs(test_targets[:, -1, :, :])[:, land_mask_np].mean(axis=1)
    sample_idx = int(np.argmax(ocean_abs))

    sample_pred  = test_preds[sample_idx, -1]           # (H, W) at day HORIZON
    sample_truth = test_targets[sample_idx, -1]          # (H, W)
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
        norm_std=norm_std,
        save_path=RESULTS_DIR / "tubelet_results.png",
    )


if __name__ == "__main__":
    main()
