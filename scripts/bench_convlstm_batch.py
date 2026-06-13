#!/usr/bin/env python
"""
Batch-size validation benchmark for the ConvLSTM retrain on H200.

Motivation
----------
After fixing the data pipeline (RAM-preload + workers) and enabling BF16
autocast, the HPO-selected batch size B=2 is still ~8 min/epoch because at
B=2 each epoch runs ~2,570 iterations of the sequential 90-step recurrence on
tiny tensors (launch/iteration-bound; the convs are too small for the H200's
tensor cores to matter). Raising the batch size is now a real speed lever
(fewer iterations + bigger convs), BUT batch size can change the trained model
(gradient-noise / generalization). The learning rate is square-root-scaled
with batch size (Adam rule).

This benchmark trains seed-1 to early-stop at several batch sizes and reports,
for each: epochs, wall time, time/epoch, best val loss, and the same
denormalised masked val RMSE used by retrain_best.py (`mean_rmse`). Use it to
decide whether a larger batch is *equivalent* to the B=2 reference before
switching the production multi-seed run.

Nothing is saved — this only measures. Run on a GPU node.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.optim as optim
import zarr

# retrain_best sets up sys.path to the project root on import.
from scripts.retrain_best import (
    find_best_config, find_best_ablation_alpha, MODEL_BUILDERS, ZARR_PATH,
)
from src.data.dataloaders import create_dataloaders
from src.training.losses import AnomalyWeightedMSE
from src.training.train import train_model
from src.training.evaluate import predict
from src.utils.metrics import rmse_per_step

MODEL_TYPE = "convlstm"
SEED = 1


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def val_rmse(model, val_loader, device, norm_mean, norm_std, land_mask) -> float:
    """Denormalised, land-masked mean RMSE over the val set (matches the
    `mean_rmse` metric retrain_best.py reports on the test set)."""
    preds_norm, targets_norm = predict(model, val_loader, device)
    preds   = preds_norm   * norm_std + norm_mean
    targets = targets_norm * norm_std + norm_mean
    steps = rmse_per_step(preds, targets, land_mask=land_mask)
    return float(np.mean(steps))


def run_one(batch_size: int, config: dict, device, norm_mean, norm_std,
            land_mask) -> dict:
    set_seed(SEED)

    base_batch = config.get("batch_size", 8)
    lr = config["learning_rate"]
    if batch_size != base_batch:
        lr_scale = (batch_size / base_batch) ** 0.5
        lr = lr * lr_scale
    else:
        lr_scale = 1.0

    use_cuda = (device.type == "cuda")
    H, W = land_mask.shape
    train_loader, val_loader, _ = create_dataloaders(
        zarr_path=ZARR_PATH,
        context_len=config["context_len"],
        horizon=config["horizon"],
        batch_size=batch_size,
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
    )
    n_batches = len(train_loader)

    model = MODEL_BUILDERS[MODEL_TYPE](config, H, W).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=config.get("weight_decay", 1e-4))
    criterion = AnomalyWeightedMSE(alpha=config.get("anomaly_alpha", 0.0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.get("lr_factor", 0.5),
        patience=config.get("lr_patience", 5),
        threshold=1e-3, min_lr=1e-5, cooldown=2,
    )
    land_mask_t = torch.from_numpy(land_mask).to(device)

    print(f"\n=== B={batch_size}  (lr ×{lr_scale:.3g} = {lr:.2e}, "
          f"{n_batches} batches/epoch) ===", flush=True)
    t0 = time.time()
    train_losses, val_losses = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, device=device,
        num_epochs=config.get("num_epochs", 50),
        land_mask=land_mask_t,
        grad_clip=config.get("grad_clip", 1.0),
        scheduler=scheduler,
        early_stop_patience=config.get("early_stop_patience", 5),
        use_amp=use_cuda,
    )
    wall = time.time() - t0
    n_epochs = len(val_losses)
    vrmse = val_rmse(model, val_loader, device, norm_mean, norm_std, land_mask)

    return {
        "batch": batch_size,
        "lr": lr,
        "lr_scale": lr_scale,
        "batches_per_epoch": n_batches,
        "epochs": n_epochs,
        "wall_s": wall,
        "s_per_epoch": wall / max(n_epochs, 1),
        "best_val_loss": float(min(val_losses)),
        "val_rmse": vrmse,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[2, 16, 32],
                    help="Batch sizes to benchmark (default: 2 16 32).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    config = find_best_config(MODEL_TYPE)
    alpha = find_best_ablation_alpha(MODEL_TYPE)
    if alpha is not None:
        config = dict(config); config["anomaly_alpha"] = alpha
    print(f"HPO config: base_batch={config.get('batch_size')}  "
          f"lr={config['learning_rate']:.2e}  alpha={config.get('anomaly_alpha'):.4f}")

    root = zarr.open_group(str(ZARR_PATH), mode="r")
    norm_mean = float(root.attrs["norm_mean"])
    norm_std  = float(root.attrs["norm_std"])
    land_mask = np.array(root["land_mask"])

    results = [run_one(b, config, device, norm_mean, norm_std, land_mask)
               for b in args.batches]

    # --- Comparison table + verdict ---
    ref = next((r for r in results if r["batch"] == config.get("batch_size")), results[0])
    print("\n" + "=" * 78)
    print(f"{'batch':>6} {'lr':>10} {'b/ep':>6} {'epochs':>7} "
          f"{'s/epoch':>9} {'wall_min':>9} {'val_loss':>10} {'val_rmse':>9} {'ΔRMSE%':>8}")
    print("-" * 78)
    for r in results:
        d_rmse = 100.0 * (r["val_rmse"] - ref["val_rmse"]) / ref["val_rmse"]
        speed = ref["s_per_epoch"] / r["s_per_epoch"]
        print(f"{r['batch']:>6} {r['lr']:>10.2e} {r['batches_per_epoch']:>6} "
              f"{r['epochs']:>7} {r['s_per_epoch']:>9.1f} {r['wall_s']/60:>9.1f} "
              f"{r['best_val_loss']:>10.5f} {r['val_rmse']:>9.4f} {d_rmse:>+7.2f}%"
              f"   ({speed:.1f}x vs ref)")
    print("=" * 78)
    print(f"Reference = B={ref['batch']} (HPO).  A larger batch is 'equivalent' "
          f"if ΔRMSE% is within seed noise (~±1-2%).")
    print("Pick the largest batch whose ΔRMSE% stays within tolerance for the "
          "fastest fidelity-validated production run.")


if __name__ == "__main__":
    main()
