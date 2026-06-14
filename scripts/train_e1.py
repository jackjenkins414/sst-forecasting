#!/usr/bin/env python3
"""Train E1 MVE: LSTM or Transformer at h=7, L=90 on Coral Sea OISST.

This script is intentionally self-contained (plain argparse, no Hydra) so it
can be submitted directly via ``sbatch`` without a config file system.

Outputs (written to ``--output-dir``)
--------------------------------------
``best_model.pt``     checkpoint with lowest val MSE
``last_model.pt``     checkpoint after final epoch
``metrics.json``      per-epoch train/val MSE + final test metrics
``run.yaml``          full provenance (git SHA, args, env, package versions)
``training_log.csv``  epoch-level CSV: epoch,train_mse,val_mse,lr,elapsed_s

Usage
-----
# LSTM at h=7 (E1 main run)
python scripts/train_e1.py --model lstm --horizon 7 \\
    --zarr-path data/processed/oisst_coralsea.zarr \\
    --output-dir experiments/results/e1_lstm_h7

# Transformer at h=7
python scripts/train_e1.py --model transformer --horizon 7 \\
    --zarr-path data/processed/oisst_coralsea.zarr \\
    --output-dir experiments/results/e1_transformer_h7

# Quick smoke-test (1 epoch, small batch)
python scripts/train_e1.py --model lstm --horizon 7 --max-epochs 1 \\
    --batch-size 4 --max-train-windows 64 --max-val-windows 16

Notes
-----
*  Loss is MSE computed only over **ocean cells** (land filled with 0 by the
   dataset but masked out using the zarr ``land_mask``).
*  Training is in *normalised* SST-anomaly space.  RMSE is converted to °C
   by multiplying by ``norm_std`` from the zarr store attributes.
*  Early stopping monitors val MSE (patience configurable).
*  ``torch.compile`` is always disabled for Sandy Bridge AVX1 compatibility.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
import zarr
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from sst_forecasting.data.dataset import SSTWindowDataset
from sst_forecasting.models.convlstm import SpatialConvLSTM
from sst_forecasting.models.lstm import SpatialFlatLSTM
from sst_forecasting.models.transformer import SpatialFlatTransformer
from sst_forecasting.utils.metrics import rmse as rmse_metric

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train E1 LSTM or Transformer on Coral Sea OISST."
    )
    # Data
    p.add_argument("--zarr-path", default="data/processed/oisst_coralsea.zarr",
                   help="Path to Zarr store.")
    p.add_argument("--horizon", type=int, default=7,
                   help="Forecast horizon h in days.")
    p.add_argument("--context-len", type=int, default=90,
                   help="Context window L in days.")

    # Model
    p.add_argument("--model", choices=["lstm", "transformer", "convlstm"], required=True,
                   help="Model architecture.")
    # LSTM hyperparams
    p.add_argument("--lstm-d-spatial", type=int, default=64)
    p.add_argument("--lstm-hidden", type=int, default=128)
    p.add_argument("--lstm-layers", type=int, default=2)
    # ConvLSTM hyperparams
    p.add_argument("--convlstm-hidden", type=int, nargs="+", default=[32, 64],
                   help="Hidden channels per ConvLSTM layer, e.g. --convlstm-hidden 32 64 128")
    p.add_argument("--convlstm-kernel", type=int, default=3)
    # Transformer hyperparams
    p.add_argument("--tf-d-model", type=int, default=128)
    p.add_argument("--tf-nhead", type=int, default=8)
    p.add_argument("--tf-layers", type=int, default=4)
    p.add_argument("--tf-ffn-dim", type=int, default=256)
    # Shared
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers. 0 = main process (safest on NFS).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience (epochs).")
    p.add_argument("--lr-patience", type=int, default=5,
                   help="LR scheduler patience (epochs).")
    p.add_argument("--lr-factor", type=float, default=0.5,
                   help="LR reduction factor on plateau.")
    p.add_argument("--seed", type=int, default=42)

    # Debug / truncation
    p.add_argument("--max-train-windows", type=int, default=0,
                   help="Truncate training set (0=use all).")
    p.add_argument("--max-val-windows", type=int, default=0,
                   help="Truncate val set (0=use all).")
    p.add_argument("--preload-zarr", action="store_true", default=True,
                   help="Load full zarr SST array into RAM before training "
                        "(eliminates NFS delegation overhead; ~270 MB, fits on any node).")
    p.add_argument("--no-preload-zarr", dest="preload_zarr", action="store_false",
                   help="Disable zarr preloading (lazy NFS reads per batch).")

    # Output
    p.add_argument("--output-dir", default="experiments/results/e1_debug",
                   help="Directory for checkpoints and metrics.")
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"

def _build_model(args: argparse.Namespace, H: int, W: int) -> nn.Module:
    if args.model == "lstm":
        return SpatialFlatLSTM(
            H=H, W=W,
            context_len=args.context_len,
            horizon=args.horizon,
            d_spatial=args.lstm_d_spatial,
            hidden_size=args.lstm_hidden,
            num_layers=args.lstm_layers,
            dropout=args.dropout,
        )
    elif args.model == "convlstm":
        return SpatialConvLSTM(
            H=H, W=W,
            context_len=args.context_len,
            horizon=args.horizon,
            hidden_channels=args.convlstm_hidden,
            kernel_size=args.convlstm_kernel,
            dropout=args.dropout,
        )
    else:
        return SpatialFlatTransformer(
            H=H, W=W,
            context_len=args.context_len,
            horizon=args.horizon,
            d_model=args.tf_d_model,
            nhead=args.tf_nhead,
            num_encoder_layers=args.tf_layers,
            dim_feedforward=args.tf_ffn_dim,
            dropout=args.dropout,
        )

def _make_loader(
    zarr_path: str, split: str, args: argparse.Namespace, max_windows: int
) -> DataLoader:
    ds = SSTWindowDataset(
        zarr_path=zarr_path,
        split=split,
        context_len=args.context_len,
        horizon=args.horizon,
    )
    if max_windows > 0 and len(ds) > max_windows:
        ds = Subset(ds, list(range(max_windows)))
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Train / val loops
# ─────────────────────────────────────────────────────────────────────────────

def _epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.MSELoss,
    ocean_mask: torch.Tensor,  # (H, W) bool
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
    device: torch.device,
) -> float:
    """Run one train or val epoch; return mean MSE over ocean cells."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n_batches = 0

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x = x.to(device, non_blocking=True)    # (B, L, 1, H, W)
            y = y.to(device, non_blocking=True)    # (B, h, H, W)

            pred = model(x)                        # (B, h, H, W)

            # MSE over ocean cells only
            # ocean_mask: (H, W) → broadcast over (B, h, H, W)
            mask = ocean_mask.unsqueeze(0).unsqueeze(0)        # (1, 1, H, W)
            loss = criterion(pred[mask.expand_as(pred)],
                             y[mask.expand_as(y)])

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # Seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

    # ── Load metadata from Zarr ───────────────────────────────────────────────
    root = zarr.open_group(args.zarr_path, mode="r")
    norm_std: float = float(root.attrs["norm_std"])
    norm_mean: float = float(root.attrs["norm_mean"])
    land_mask_arr: np.ndarray = np.array(root["land_mask"])   # (H, W) bool True=ocean
    H, W = land_mask_arr.shape
    ocean_mask = torch.from_numpy(land_mask_arr).to(device)    # (H, W) bool, True=ocean

    print(f"[train_e1] Grid: H={H}, W={W}, ocean cells={int(ocean_mask.sum())}")
    print(f"[train_e1] norm_mean={norm_mean:.5f}, norm_std={norm_std:.5f}")

    # ── Data loaders ──────────────────────────────────────────────────────────
    train_loader = _make_loader(args.zarr_path, "train", args, args.max_train_windows)
    val_loader   = _make_loader(args.zarr_path, "val",   args, args.max_val_windows)

    # Preload zarr into RAM to avoid NFS delegation-RPC overhead (~2s, 270 MB)
    if args.preload_zarr:
        print("[train_e1] preloading zarr sst_norm into RAM ...", flush=True)
        preloaded_sst = np.array(root["sst_norm"])   # (T, H, W) float32
        for loader in [train_loader, val_loader]:
            ds = loader.dataset
            if isinstance(ds, Subset):
                ds = ds.dataset
            if hasattr(ds, "_data") and not isinstance(ds._data, np.ndarray):
                ds._data = preloaded_sst
        print(f"[train_e1] zarr preloaded: {preloaded_sst.nbytes / 1e6:.0f} MB", flush=True)

    print(f"[train_e1] train batches={len(train_loader)}, val batches={len(val_loader)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = _build_model(args, H, W).to(device)
    n_params = model.count_parameters()
    print(f"[train_e1] model={args.model}, params={n_params:,}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor,
        patience=args.lr_patience, min_lr=1e-6,
    )
    criterion = nn.MSELoss()

    # ── CSV log ───────────────────────────────────────────────────────────────
    csv_path = out_dir / "training_log.csv"
    csv_fh = open(csv_path, "w", newline="")
    writer = csv.writer(csv_fh)
    writer.writerow(["epoch", "train_mse", "val_mse", "lr", "elapsed_s"])

    # ── Training loop with early stopping ────────────────────────────────────
    best_val_mse = float("inf")
    no_improve_count = 0
    history: list[dict] = []
    t0 = time.monotonic()

    for epoch in range(1, args.max_epochs + 1):
        t_ep = time.monotonic()
        train_mse = _epoch(model, train_loader, criterion, ocean_mask, optimizer, args.grad_clip, device)
        val_mse   = _epoch(model, val_loader,   criterion, ocean_mask, None,      0.0,            device)

        scheduler.step(val_mse)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.monotonic() - t0
        ep_time = time.monotonic() - t_ep

        # Convert MSE in norm space → RMSE in °C
        train_rmse_c = float(np.sqrt(train_mse)) * norm_std
        val_rmse_c   = float(np.sqrt(val_mse))   * norm_std

        print(
            f"[epoch {epoch:03d}/{args.max_epochs}] "
            f"train RMSE={train_rmse_c:.4f}°C  "
            f"val RMSE={val_rmse_c:.4f}°C  "
            f"lr={current_lr:.2e}  "
            f"ep={ep_time:.1f}s  total={elapsed:.0f}s"
        )
        writer.writerow([epoch, train_mse, val_mse, current_lr, elapsed])
        csv_fh.flush()

        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse,
                         "train_rmse_c": train_rmse_c, "val_rmse_c": val_rmse_c,
                         "lr": current_lr, "elapsed_s": elapsed})

        # Best checkpoint
        if val_mse < best_val_mse - 1e-6:
            best_val_mse = val_mse
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_mse": val_mse, "args": vars(args)},
                       out_dir / "best_model.pt")
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= args.patience:
            print(f"[train_e1] early stopping at epoch {epoch} (patience={args.patience})")
            break

    csv_fh.close()

    # ── Last checkpoint ───────────────────────────────────────────────────────
    torch.save({"epoch": epoch, "model_state": model.state_dict(),
                "val_mse": val_mse, "args": vars(args)},
               out_dir / "last_model.pt")

    total_time = time.monotonic() - t0
    print(f"[train_e1] training done in {total_time:.1f}s")

    # ── Quick test evaluation on best checkpoint ──────────────────────────────
    print("[train_e1] loading best model for test evaluation ...")
    ckpt = torch.load(out_dir / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_ds = SSTWindowDataset(
        zarr_path=args.zarr_path,
        split="test",
        context_len=args.context_len,
        horizon=args.horizon,
    )
    if args.preload_zarr and hasattr(test_ds, "_data") and not isinstance(test_ds._data, np.ndarray):
        test_ds._data = preloaded_sst  # reuse already-loaded array
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    all_preds, all_truths = [], []
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x.to(device)).cpu().numpy()
            # Denormalise: multiply by norm_std (mean≈0 for anomalies)
            all_preds.append(pred * norm_std + norm_mean)
            all_truths.append(y.numpy() * norm_std + norm_mean)

    preds_all  = np.concatenate(all_preds,  axis=0)   # (N_test, h, H, W)
    truths_all = np.concatenate(all_truths, axis=0)

    # Apply land mask to truths (set land to NaN for metric masking)
    land = ~land_mask_arr                              # True = land
    truths_all[:, :, land] = np.nan

    # RMSE per horizon step (averaged over all windows and ocean cells)
    test_rmse_per_h = []
    for hi in range(args.horizon):
        r = rmse_metric(preds_all[:, hi], truths_all[:, hi])
        test_rmse_per_h.append(float(r))
    test_rmse_mean = float(np.mean(test_rmse_per_h))

    print(f"[train_e1] test RMSE (°C) per step: {[f'{r:.4f}' for r in test_rmse_per_h]}")
    print(f"[train_e1] test RMSE mean over {args.horizon} steps: {test_rmse_mean:.4f}°C")

    # ── Write metrics.json ────────────────────────────────────────────────────
    metrics = {
        "model": args.model,
        "horizon": args.horizon,
        "context_len": args.context_len,
        "n_params": n_params,
        "best_val_mse": best_val_mse,
        "best_val_rmse_c": float(np.sqrt(best_val_mse)) * norm_std,
        "test_rmse_per_step_c": test_rmse_per_h,
        "test_rmse_mean_c": test_rmse_mean,
        "epochs_trained": epoch,
        "total_train_time_s": total_time,
        "norm_std": norm_std,
        "history": history,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train_e1] metrics written to {out_dir / 'metrics.json'}")

    # ── Write run.yaml ────────────────────────────────────────────────────────
    repo = Path(__file__).parent.parent
    provenance = {
        "model": args.model,
        "args": vars(args),
        "git_sha": _git_sha(repo),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none"),
        "n_params": n_params,
        "best_epoch": ckpt["epoch"],
        "best_val_rmse_c": float(np.sqrt(best_val_mse)) * norm_std,
        "test_rmse_mean_c": test_rmse_mean,
    }
    with open(out_dir / "run.yaml", "w") as f:
        yaml.dump(provenance, f, default_flow_style=False)
    print(f"[train_e1] run.yaml written to {out_dir / 'run.yaml'}")

if __name__ == "__main__":
    main()
