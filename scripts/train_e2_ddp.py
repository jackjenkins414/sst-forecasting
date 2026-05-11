#!/usr/bin/env python3
"""Train E2 SpatialConvLSTM with PyTorch DistributedDataParallel (DDP).

Designed for 2-node CPU training on Raijin hpc-03 + hpc-06 (128 GB RAM each)
over IPoIB (gloo backend, GLOO_SOCKET_IFNAME=ibp176s0, 56 Gb/s FDR).

This script is ConvLSTM-specific.  For single-node LSTM / Transformer / ConvLSTM
training use scripts/train_e1.py instead.

DDP topology
------------
world_size = 2  (one process per node, launched by torchrun)
rank 0 → hpc-03 (IB: 10.0.0.3)  — rendezvous host, writes all output files
rank 1 → hpc-06 (IB: 10.0.0.6)  — participant

Data parallelism
----------------
DistributedSampler splits the 5138 training windows ~evenly (≈2569 per rank).
DDP hooks all-reduce gradients after each backward pass; every optimizer.step()
is therefore identical on both ranks.  Val MSE is explicitly all-reduced before
the early-stopping / LR-scheduler decision so both ranks always make the same
choice.

Effective global batch size = per-rank batch_size × world_size.

Outputs (rank 0 only, shared via NFS)
--------------------------------------
best_model.pt       checkpoint with lowest all-reduced val MSE
last_model.pt       checkpoint after the final completed epoch
metrics.json        per-epoch losses + final test metrics
run.yaml            full provenance (git SHA, args, platform)
training_log.csv    epoch,train_mse,val_mse,lr,elapsed_s

Launch (via sbatch — see scripts/slurm/raijin_e2_convlstm_ddp.sbatch)
-----------------------------------------------------------------------
    export GLOO_SOCKET_IFNAME=ibp176s0
    srun --ntasks=2 --ntasks-per-node=1 \\
        numactl --interleave=all \\
        torchrun \\
            --nproc_per_node=1 --nnodes=2 \\
            --rdzv-backend=c10d \\
            --rdzv-endpoint=hpc-03:29500 \\
            --rdzv-id=$SLURM_JOB_ID \\
        scripts/train_e2_ddp.py [args]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
import zarr
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, DistributedSampler, Subset

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from sst_forecasting.data.dataset import SSTWindowDataset
from sst_forecasting.models.convlstm import SpatialConvLSTM
from sst_forecasting.utils.metrics import rmse as rmse_metric

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DDP training for SpatialConvLSTM on Coral Sea OISST."
    )
    # Data
    p.add_argument("--zarr-path", default="data/processed/oisst_coralsea.zarr",
                   help="Path to Zarr store (relative to cwd or absolute).")
    p.add_argument("--horizon", type=int, default=7,
                   help="Forecast horizon h in days.")
    p.add_argument("--context-len", type=int, default=90,
                   help="Context window L in days.")

    # ConvLSTM architecture
    p.add_argument("--convlstm-hidden", type=int, nargs="+", default=[32, 64],
                   metavar="CH",
                   help="Hidden channel widths per ConvLSTM layer (space-separated). "
                        "Default: 32 64  (~260 k params).")
    p.add_argument("--convlstm-kernel", type=int, default=3,
                   help="Spatial kernel size for ConvLSTM gates.")
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16,
                   help="Per-rank batch size.  Effective global bs = batch_size × world_size.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=10,
                   help="Early-stopping patience (epochs).")
    p.add_argument("--lr-patience", type=int, default=5,
                   help="LR scheduler patience (epochs).")
    p.add_argument("--lr-factor", type=float, default=0.5,
                   help="LR reduction factor on plateau.")
    p.add_argument("--seed", type=int, default=42)

    # Debug / truncation
    p.add_argument("--max-train-windows", type=int, default=0,
                   help="Truncate training set for smoke tests (0 = use all).")
    p.add_argument("--max-val-windows", type=int, default=0,
                   help="Truncate val set for smoke tests (0 = use all).")
    p.add_argument("--preload-zarr", action="store_true", default=True,
                   help="Load full zarr SST array into RAM before training "
                        "(each rank loads independently; ~270 MB, fits on 128 GB nodes).")
    p.add_argument("--no-preload-zarr", dest="preload_zarr", action="store_false",
                   help="Disable zarr preloading (lazy NFS reads per batch).")

    # Output
    p.add_argument("--output-dir", default="experiments/results/e2_convlstm_ddp_debug",
                   help="Directory for checkpoints and metrics (rank 0 writes, NFS shared).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# DDP helpers
# ─────────────────────────────────────────────────────────────────────────────


def _ddp_init() -> tuple[int, int]:
    """Initialise gloo process group from torchrun env vars.

    torchrun sets RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    before spawning this script.  ``init_method="env://"`` reads them.

    Returns
    -------
    (rank, world_size)
    """
    dist.init_process_group(backend="gloo", init_method="env://")
    return dist.get_rank(), dist.get_world_size()


def _allreduce_mse(total_loss: float, n_batches: int) -> float:
    """All-reduce (loss_sum, batch_count) across all ranks; return global mean MSE.

    Using sum-of-sums / sum-of-counts rather than mean-of-means avoids bias
    when batch counts differ across ranks (e.g. last val batch has fewer samples).
    """
    t = torch.tensor([total_loss, float(n_batches)], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t[0].item() / max(t[1].item(), 1.0)


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


# ─────────────────────────────────────────────────────────────────────────────
# Train / val loop
# ─────────────────────────────────────────────────────────────────────────────


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.MSELoss,
    ocean_mask: torch.Tensor,   # (H, W) bool, True = ocean cell
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
    device: torch.device,
) -> tuple[float, int]:
    """Run one train or val pass.

    Returns
    -------
    (total_loss_sum, n_batches)
        Raw sums suitable for cross-rank all-reduce via ``_allreduce_mse``.
    """
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
            mask = ocean_mask.unsqueeze(0).unsqueeze(0)    # (1, 1, H, W)
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

    return total_loss, n_batches


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()

    # ── DDP init ──────────────────────────────────────────────────────────────
    rank, world_size = _ddp_init()
    is_rank0 = rank == 0

    # Deterministic seeding.  All ranks use the same seed for model init so
    # weights start identically (DDP also broadcasts rank-0 weights on __init__).
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu")
    n_threads = int(os.environ.get("OMP_NUM_THREADS", "16"))
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(2)

    if is_rank0:
        print(f"[ddp] world_size={world_size}  rank={rank}  host={socket.gethostname()}", flush=True)
        print(f"[ddp] backend=gloo  threads={n_threads}  GLOO_SOCKET_IFNAME="
              f"{os.environ.get('GLOO_SOCKET_IFNAME', 'not set')}", flush=True)

    # ── Output directory ──────────────────────────────────────────────────────
    # Rank 0 creates; barrier ensures it exists before rank 1 needs it.
    out_dir = Path(args.output_dir)
    if is_rank0:
        out_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    # ── Zarr metadata (all ranks read independently over NFS) ─────────────────
    root = zarr.open_group(args.zarr_path, mode="r")
    norm_std: float = float(root.attrs["norm_std"])
    norm_mean: float = float(root.attrs["norm_mean"])
    land_mask_arr: np.ndarray = np.array(root["land_mask"])   # (H, W) bool
    H, W = land_mask_arr.shape
    ocean_mask = torch.from_numpy(land_mask_arr).to(device)   # (H, W) bool

    if is_rank0:
        print(f"[ddp] Grid: H={H}, W={W}, ocean cells={int(ocean_mask.sum())}")
        print(f"[ddp] norm_mean={norm_mean:.5f}, norm_std={norm_std:.5f}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = SSTWindowDataset(
        zarr_path=args.zarr_path, split="train",
        context_len=args.context_len, horizon=args.horizon,
    )
    val_ds = SSTWindowDataset(
        zarr_path=args.zarr_path, split="val",
        context_len=args.context_len, horizon=args.horizon,
    )
    if args.max_train_windows > 0 and len(train_ds) > args.max_train_windows:
        train_ds = Subset(train_ds, list(range(args.max_train_windows)))
    if args.max_val_windows > 0 and len(val_ds) > args.max_val_windows:
        val_ds = Subset(val_ds, list(range(args.max_val_windows)))

    # Preload zarr into local RAM.  Each rank performs the full NFS read
    # independently (~270 MB) into its own node's DRAM.  After preload, all
    # training reads are local memory accesses with zero NFS traffic.
    preloaded_sst: np.ndarray | None = None
    if args.preload_zarr:
        print(f"[ddp rank={rank}] preloading zarr sst_norm into RAM ...", flush=True)
        preloaded_sst = np.array(root["sst_norm"])   # (T, H, W) float32
        for ds in (train_ds, val_ds):
            inner = ds.dataset if isinstance(ds, Subset) else ds
            if hasattr(inner, "_data") and not isinstance(inner._data, np.ndarray):
                inner._data = preloaded_sst
        print(f"[ddp rank={rank}] preloaded {preloaded_sst.nbytes / 1e6:.0f} MB", flush=True)

    # ── Distributed samplers ──────────────────────────────────────────────────
    # Train: shuffle=True so each epoch each rank sees a different shard.
    # Val: shuffle=False; DistributedSampler pads to an even split so counts
    #      may differ by ≤1 between ranks — handled by _allreduce_mse.
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed,
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=0, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=0, pin_memory=False, drop_last=False,
    )

    if is_rank0:
        eff_bs = args.batch_size * world_size
        print(
            f"[ddp] train windows={len(train_ds)}, val windows={len(val_ds)}"
            f"  |  per-rank bs={args.batch_size} (effective global bs={eff_bs})"
            f"  |  train batches≈{len(train_loader)}, val batches≈{len(val_loader)}"
        )

    # ── Model ──────────────────────────────────────────────────────────────────
    # Same seed on all ranks → identical weight initialisation.
    # DDP.__init__ also broadcasts rank-0 parameters to all other ranks.
    torch.manual_seed(args.seed)
    model = SpatialConvLSTM(
        H=H, W=W,
        context_len=args.context_len,
        horizon=args.horizon,
        hidden_channels=args.convlstm_hidden,
        kernel_size=args.convlstm_kernel,
        dropout=args.dropout,
    ).to(device)

    model = DDP(model)   # gloo CPU backend; no device_ids needed for CPU

    n_params = model.module.count_parameters()
    if is_rank0:
        print(f"[ddp] SpatialConvLSTM params={n_params:,}  world_size={world_size}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor,
        patience=args.lr_patience, min_lr=1e-6,
    )
    criterion = nn.MSELoss()

    # ── CSV log (rank 0 only) ──────────────────────────────────────────────────
    csv_fh = None
    csv_writer = None
    if is_rank0:
        csv_path = out_dir / "training_log.csv"
        csv_fh = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow(["epoch", "train_mse", "val_mse", "lr", "elapsed_s"])

    # ── Graceful USR1 handler ──────────────────────────────────────────────────
    # Slurm sends USR1 before the walltime limit.  We finish the current epoch
    # then save last_model.pt before exiting so the job can be requeued cleanly.
    _stop_early: list[bool] = [False]

    def _on_sigusr1(signum: int, frame: object) -> None:
        _stop_early[0] = True

    signal.signal(signal.SIGUSR1, _on_sigusr1)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_mse = float("inf")
    no_improve_count = 0
    history: list[dict] = []
    t0 = time.monotonic()
    epoch = 0
    val_mse_global = float("inf")

    for epoch in range(1, args.max_epochs + 1):
        # Update sampler epoch so each epoch draws a different shard on each rank
        train_sampler.set_epoch(epoch)
        t_ep = time.monotonic()

        train_total, train_n = _epoch(
            model, train_loader, criterion, ocean_mask, optimizer, args.grad_clip, device
        )
        val_total, val_n = _epoch(
            model, val_loader, criterion, ocean_mask, None, 0.0, device
        )

        # Cross-rank all-reduce: both ranks now have identical global MSE values.
        train_mse_global = _allreduce_mse(train_total, train_n)
        val_mse_global   = _allreduce_mse(val_total,   val_n)

        # LR scheduler uses global val MSE → same LR on all ranks.
        scheduler.step(val_mse_global)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed  = time.monotonic() - t0
        ep_time  = time.monotonic() - t_ep
        train_rmse_c = float(np.sqrt(train_mse_global)) * norm_std
        val_rmse_c   = float(np.sqrt(val_mse_global))   * norm_std

        if is_rank0:
            print(
                f"[epoch {epoch:03d}/{args.max_epochs}] "
                f"train RMSE={train_rmse_c:.4f}°C  "
                f"val RMSE={val_rmse_c:.4f}°C  "
                f"lr={current_lr:.2e}  "
                f"ep={ep_time:.1f}s  total={elapsed:.0f}s",
                flush=True,
            )
            if csv_writer is not None:
                csv_writer.writerow([epoch, train_mse_global, val_mse_global,
                                     current_lr, elapsed])
                csv_fh.flush()   # type: ignore[union-attr]

        history.append({
            "epoch": epoch,
            "train_mse": train_mse_global,
            "val_mse": val_mse_global,
            "train_rmse_c": train_rmse_c,
            "val_rmse_c": val_rmse_c,
            "lr": current_lr,
            "elapsed_s": elapsed,
        })

        # Best checkpoint — save model.module to strip the DDP wrapper so the
        # checkpoint can be loaded without dist being initialised (e.g. test eval).
        if val_mse_global < best_val_mse - 1e-6:
            best_val_mse = val_mse_global
            if is_rank0:
                torch.save(
                    {"epoch": epoch, "model_state": model.module.state_dict(),
                     "val_mse": val_mse_global, "args": vars(args)},
                    out_dir / "best_model.pt",
                )
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Early stopping — uses all-reduced val MSE so both ranks decide identically.
        if no_improve_count >= args.patience:
            if is_rank0:
                print(f"[ddp] early stopping at epoch {epoch} (patience={args.patience})")
            break

        # Graceful walltime exit (fires when Slurm USR1 reaches the Python process).
        if _stop_early[0]:
            if is_rank0:
                print(f"[ddp] caught USR1 at epoch {epoch} — saving last checkpoint",
                      flush=True)
            break

    # ── Last checkpoint + close CSV (rank 0 only) ─────────────────────────────
    if is_rank0:
        torch.save(
            {"epoch": epoch, "model_state": model.module.state_dict(),
             "val_mse": val_mse_global, "args": vars(args)},
            out_dir / "last_model.pt",
        )
        if csv_fh is not None:
            csv_fh.close()

    total_time = time.monotonic() - t0

    # ── Sync and tear down process group ──────────────────────────────────────
    dist.barrier()
    dist.destroy_process_group()

    # ── Test evaluation (rank 0 only — pure local, no dist ops) ───────────────
    if is_rank0:
        print("[ddp] loading best model for test evaluation ...", flush=True)
        ckpt = torch.load(out_dir / "best_model.pt", map_location="cpu",
                          weights_only=True)

        eval_model = SpatialConvLSTM(
            H=H, W=W,
            context_len=args.context_len,
            horizon=args.horizon,
            hidden_channels=args.convlstm_hidden,
            kernel_size=args.convlstm_kernel,
            dropout=args.dropout,
        )
        eval_model.load_state_dict(ckpt["model_state"])
        eval_model.eval()

        test_ds = SSTWindowDataset(
            zarr_path=args.zarr_path, split="test",
            context_len=args.context_len, horizon=args.horizon,
        )
        if args.preload_zarr and preloaded_sst is not None:
            if hasattr(test_ds, "_data") and not isinstance(test_ds._data, np.ndarray):
                test_ds._data = preloaded_sst
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)

        all_preds: list[np.ndarray] = []
        all_truths: list[np.ndarray] = []
        with torch.no_grad():
            for x, y in test_loader:
                pred = eval_model(x.to(device)).cpu().numpy()
                all_preds.append(pred * norm_std + norm_mean)
                all_truths.append(y.numpy() * norm_std + norm_mean)

        preds_all  = np.concatenate(all_preds,  axis=0)   # (N_test, h, H, W)
        truths_all = np.concatenate(all_truths, axis=0)

        land = ~land_mask_arr                              # True = land
        truths_all[:, :, land] = np.nan

        test_rmse_per_h = [
            float(rmse_metric(preds_all[:, hi], truths_all[:, hi]))
            for hi in range(args.horizon)
        ]
        test_rmse_mean = float(np.mean(test_rmse_per_h))

        print(f"[ddp] test RMSE per step: {[f'{r:.4f}' for r in test_rmse_per_h]}")
        print(f"[ddp] test RMSE mean ({args.horizon} steps): {test_rmse_mean:.4f}°C")

        # metrics.json
        metrics = {
            "model": "convlstm_ddp",
            "ddp_world_size": world_size,
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
        print(f"[ddp] metrics written to {out_dir / 'metrics.json'}")

        # run.yaml
        repo = Path(__file__).parent.parent
        provenance = {
            "model": "convlstm_ddp",
            "args": vars(args),
            "ddp_world_size": world_size,
            "git_sha": _git_sha(repo),
            "hostname_rank0": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none"),
            "n_params": n_params,
            "best_epoch": int(ckpt["epoch"]),
            "best_val_rmse_c": float(np.sqrt(best_val_mse)) * norm_std,
            "test_rmse_mean_c": test_rmse_mean,
        }
        with open(out_dir / "run.yaml", "w") as f:
            yaml.dump(provenance, f, default_flow_style=False)

        print(f"[ddp] outputs written to {out_dir}")
        print(f"[ddp] total training time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
