"""
VRAM smoke test for the ConvLSTM HPO memory guard.

Validates that `_estimate_peak_gb()` in run_optuna_convlstm.py is conservative
by running a real forward+backward+step on the worst-case configs that the
guard *allows* at each batch tier, then comparing measured peak VRAM
(torch.cuda.max_memory_allocated) against the estimate and the 9 GB budget.

This converts the calibrated-but-unmeasured budget into a measured one before
committing to an overnight study. Runs only a handful of optimizer steps per
config, so it finishes in a minute or two.

    python scripts/smoke_test_convlstm_vram.py

Exit code 0 if every allowed config stays under the SAFETY_GB line, 1 otherwise.
"""

import os
import sys
from pathlib import Path

# Reduce allocator fragmentation on long-lived studies (and this test).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
import torch.optim as optim
import zarr

from src.data.dataloaders import create_dataloaders
from src.sst_forecasting.models.convlstm import SpatialConvLSTM
from src.training.losses import AnomalyWeightedMSE
from scripts.run_optuna_convlstm import (
    CONTEXT_LEN, HORIZON, VRAM_BUDGET_GB,
    _estimate_peak_gb, _batch_for_config,
)

ZARR_PATH = PROJECT_ROOT / "data/processed/oisst_coralsea.zarr"
N_STEPS = 3              # optimizer steps per config (peak is reached on step 1)
SAFETY_GB = 10.0        # measured peak must stay below this on the 12 GB card

# Worst-case *allowed* configs per batch tier (all estimate ~7.8 GB) plus the
# deepest stacks, where real BPTT memory is most likely to exceed the model.
TEST_CONFIGS = [
    # (hidden_dim, n_layers) — chosen as the highest-estimate configs that run
    (64, 1),   # B=4 tier, est 7.8
    (32, 2),   # B=4 tier, est 7.8
    (16, 4),   # B=4 tier, est 7.8, deepest at B=4
    (64, 2),   # B=2 tier, est 7.8
    (32, 4),   # B=2 tier, est 7.8, deepest at B=2
    (96, 1),   # B=2 tier, est 6.2, widest single layer
]


def log(msg: str) -> None:
    print(msg, flush=True)


def measure_peak_gb(hidden_dim: int, n_layers: int, batch_size: int,
                    land_mask_np, device) -> float:
    """Run a few real train steps and return measured peak VRAM in GB."""
    H, W = land_mask_np.shape
    land_mask = torch.from_numpy(land_mask_np).to(device)

    train_loader, _, _ = create_dataloaders(
        zarr_path=ZARR_PATH, context_len=CONTEXT_LEN,
        horizon=HORIZON, batch_size=batch_size,
    )

    model = SpatialConvLSTM(
        H=H, W=W, context_len=CONTEXT_LEN, horizon=HORIZON,
        hidden_channels=[hidden_dim] * n_layers, kernel_size=5, dropout=0.1,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = AnomalyWeightedMSE(alpha=0.1)

    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    steps = 0
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        preds = model(batch_X)
        mask = land_mask.expand_as(preds)
        loss = criterion(preds[mask], batch_y[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        steps += 1
        if steps >= N_STEPS:
            break

    peak = torch.cuda.max_memory_allocated(device) / 1e9
    reserved = torch.cuda.max_memory_reserved(device) / 1e9

    del model, optimizer, train_loader
    torch.cuda.empty_cache()
    return peak, reserved


def main() -> int:
    if not torch.cuda.is_available():
        log("CUDA not available — cannot run VRAM smoke test.")
        return 1

    device = torch.device("cuda")
    log(f"Device: {torch.cuda.get_device_name(device)}")
    log(f"Budget (guard): {VRAM_BUDGET_GB} GB   Safety line (measured): {SAFETY_GB} GB")
    log(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}\n")

    root = zarr.open_group(str(ZARR_PATH), mode="r")
    land_mask_np = np.array(root["land_mask"])

    np.random.seed(42)
    torch.manual_seed(42)

    header = f"{'hidden':>6} {'layers':>6} {'B':>2} {'est_GB':>7} {'peak_GB':>8} {'resv_GB':>8}  result"
    log(header)
    log("-" * len(header))

    all_ok = True
    for hidden_dim, n_layers in TEST_CONFIGS:
        batch_size = _batch_for_config(hidden_dim, n_layers)
        if batch_size is None:
            log(f"{hidden_dim:6d} {n_layers:6d}  - {'(pruned)':>7}  guard prunes this config — skipping")
            continue
        est = _estimate_peak_gb(hidden_dim, n_layers, batch_size)
        try:
            peak, reserved = measure_peak_gb(
                hidden_dim, n_layers, batch_size, land_mask_np, device
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log(f"{hidden_dim:6d} {n_layers:6d} {batch_size:2d} {est:7.1f} {'OOM':>8} {'':>8}  FAIL (real OOM)")
                torch.cuda.empty_cache()
                all_ok = False
                continue
            raise

        ok = peak < SAFETY_GB
        all_ok &= ok
        verdict = "ok" if ok else "FAIL (over safety line)"
        # Flag if estimate ever undercounts the real peak — the dangerous case.
        if peak > est:
            verdict += f"  [est undercounts by {peak - est:.1f} GB]"
        log(f"{hidden_dim:6d} {n_layers:6d} {batch_size:2d} {est:7.1f} {peak:8.2f} {reserved:8.2f}  {verdict}")

    log("")
    if all_ok:
        log("PASS — every allowed config stayed under the safety line.")
        log("The guard is safe to run the full study.")
        return 0
    log("FAIL — at least one allowed config exceeded the safety line.")
    log("Lower VRAM_BUDGET_GB in run_optuna_convlstm.py before running the study.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
