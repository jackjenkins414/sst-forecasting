"""Retrain PatchTransformer with the best Optuna hyperparameters and save checkpoint.

Usage (from repo root, inside the venv)
----------------------------------------
    python scripts/retrain_best_patch_transformer.py [--a100]

Writes to  experiments/results/best_patch_transformer/
    best_model.pt   — model state dict (best val loss)
    config.json     — exact hyperparameters used
    metrics.json    — val/test RMSE per step
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataloaders import create_dataloaders
from src.models.patch_transformer import SstPatchTransformer
from src.training.losses import AnomalyWeightedMSE
from src.training.train import train_model

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ZARR_PATH   = ROOT / "data/processed/oisst_coralsea.zarr"
RESULTS_DIR = ROOT / "experiments/results/best_patch_transformer"

# ---------------------------------------------------------------------------
# Best Optuna config  (trial 6, val RMSE 0.5067)
# ---------------------------------------------------------------------------
BEST_CONFIG = {
    "model_type":          "patch_transformer",
    "context_len":         90,
    "horizon":             7,
    "batch_size":          64,
    "patch_height":        9,
    "patch_width":         11,
    "d_model":             64,
    "n_blocks":            4,
    "n_heads":             4,
    "d_ff":                1024,
    "dropout":             0.19126724140656393,
    "anomaly_alpha":       0.09444298503238986,
    "num_epochs":          50,
    "early_stop_patience": 5,
    "learning_rate":       0.0008880965698768723,
    "weight_decay":        1e-4,
    "grad_clip":           1.0,
    "lr_factor":           0.7548850970305306,
    "lr_patience":         5,
    "random_seed":         42,
    "optuna_trial":        6,
    "optuna_study_val_rmse": 0.5067,
}


def _apply_a100_opts() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("[retrain] A100 opts: BF16 autocast=ON, TF32=True, cudnn.benchmark=True")


def predict(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            out = model(X).cpu()
            preds.append(out.numpy())
            targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def rmse_per_step(preds, targets, land_mask=None):
    diff = preds - targets          # (N, horizon, H, W)
    if land_mask is not None:
        mask = ~land_mask[None, None]   # (1, 1, H, W)
        diff = diff * mask
        count = mask.sum() * preds.shape[0]
    else:
        count = np.prod(diff.shape[0::1]) // diff.shape[1]
    sq = (diff ** 2)
    if land_mask is not None:
        sq_mean = sq.sum(axis=(0, 2, 3)) / count
    else:
        sq_mean = sq.mean(axis=(0, 2, 3))
    return np.sqrt(sq_mean)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--a100", action="store_true", help="Enable A100 TF32/BF16 opts")
    args = parser.parse_args()

    if args.a100:
        _apply_a100_opts()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[retrain] device={device}")

    torch.manual_seed(BEST_CONFIG["random_seed"])
    np.random.seed(BEST_CONFIG["random_seed"])

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    print(f"[retrain] Loading zarr from {ZARR_PATH}")
    root      = zarr.open_group(str(ZARR_PATH), mode="r")
    norm_mean = float(root["norm_mean"][()])
    norm_std  = float(root["norm_std"][()])
    land_mask_np = np.isnan(np.asarray(root["sst"][0]))

    cfg = BEST_CONFIG
    loaders = create_dataloaders(
        zarr_path=ZARR_PATH,
        context_len=cfg["context_len"],
        horizon=cfg["horizon"],
        batch_size=cfg["batch_size"],
    )
    train_loader, val_loader, test_loader = loaders

    H, W = land_mask_np.shape
    land_mask_torch = torch.from_numpy(land_mask_np).to(device)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = SstPatchTransformer(
        height=H, width=W,
        patch_height=cfg["patch_height"], patch_width=cfg["patch_width"],
        seq_len=cfg["context_len"], horizon=cfg["horizon"],
        d_model=cfg["d_model"], n_blocks=cfg["n_blocks"], n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"], dropout=cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[retrain] Model params: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    criterion = AnomalyWeightedMSE(alpha=cfg["anomaly_alpha"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg["lr_factor"], patience=cfg["lr_patience"],
        threshold=1e-3, min_lr=1e-5, cooldown=2,
    )

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    print(f"[retrain] Training up to {cfg['num_epochs']} epochs (patience={cfg['early_stop_patience']}) ...")
    train_losses, val_losses = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, device=device,
        num_epochs=cfg["num_epochs"], land_mask=land_mask_torch, grad_clip=cfg["grad_clip"],
        scheduler=scheduler, early_stop_patience=cfg["early_stop_patience"],
    )

    # -----------------------------------------------------------------------
    # Evaluate on test set
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), RESULTS_DIR / "best_model.pt")
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[retrain] Done!")
    print(f"  Epochs trained : {len(train_losses)}")
    print(f"  Best val loss  : {min(val_losses):.4f}")
    print(f"  Test mean RMSE : {mean_rmse:.4f}")
    for i, r in enumerate(rmse_steps):
        print(f"    day {i+1}: {r:.4f}")
    print(f"\n  Saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
