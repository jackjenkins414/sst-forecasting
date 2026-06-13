"""
Re-trains the best Optuna config for each model and saves all artifacts
needed by report_model.py and the final comparison plots.

Runs all 4 models sequentially (default), or a chosen subset:

    python scripts/retrain_best.py                    # all models
    python scripts/retrain_best.py --models tubelet lstm

Artifacts saved under experiments/best_<model>/:
    config.json         best HP configuration
    loss_curves.json    {"train": [...], "val": [...]} per epoch
    summary.json        RMSE/step, skill/step, BIC, n_params, example date
    predictions.npy     (N, horizon, H, W) denormalised test predictions
    targets.npy         (N, horizon, H, W) denormalised test targets
    model.pt            model state dict
"""

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
import torch.optim as optim
import zarr

from src.data.dataloaders import create_dataloaders
from src.baselines.persistence import persistence_forecast
from src.training.losses import AnomalyWeightedMSE
from src.training.train import train_model
from src.training.evaluate import predict
from src.utils.metrics import rmse_per_step, skill_score

ZARR_PATH    = PROJECT_ROOT / "data/processed/oisst_coralsea.zarr"
RESULTS_DIR  = PROJECT_ROOT / "experiments/results"
BEST_DIR     = PROJECT_ROOT / "experiments"
RANDOM_SEED  = 42
EXAMPLE_IDX  = 100   # fixed test-window index used for heatmap across all models

ALL_MODELS = ["tubelet", "lstm", "informer", "convlstm", "transformer", "patch_transformer", "rnn"]


# ---------------------------------------------------------------------------
# Model type inference (handles old Tubelet runs that predate the model_type key)
# ---------------------------------------------------------------------------

def infer_model_type(config: dict) -> str:
    if "patch_height" in config and "n_blocks" in config:
        return "patch_transformer"
    if "t_s" in config and "p_h" in config:
        return "tubelet"
    if "hidden_size" in config and "d_spatial" in config:
        return "lstm"
    if "factor" in config and "label_len" in config:
        return "informer"
    if "hidden_dim" in config and "n_layers" in config:
        return "convlstm"
    if "ffn_dim" in config and "n_heads" in config:
        return "transformer"
    return "unknown"


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_tubelet(config, H, W):
    from src.models.tubelet_transformer import TubeletTransformer
    return TubeletTransformer(
        H=H, W=W,
        context_len=config["context_len"], horizon=config["horizon"],
        d_model=config["d_model"], n_heads=config["n_heads"],
        n_layers=config["n_layers"], d_ff=config["d_ff"],
        t_s=config["t_s"], p_h=config["p_h"], p_w=config["p_w"],
        dropout=config["dropout"],
    )


def _build_lstm(config, H, W):
    from src.models.lstm import StackedSpatialLSTM
    return StackedSpatialLSTM(
        H=H, W=W,
        context_len=config["context_len"], horizon=config["horizon"],
        d_spatial=config["d_spatial"], hidden_size=config["hidden_size"],
        num_layers=config["num_layers"], dropout=config["dropout"],
    )


def _build_informer(config, H, W):
    from src.models.informer import ProbSparseInformer
    return ProbSparseInformer(
        height=H, width=W,
        context_len=config["context_len"], horizon=config["horizon"],
        d_model=config["d_model"], n_heads=config["n_heads"],
        n_encoder_layers=config["n_encoder_layers"],
        n_decoder_layers=config["n_decoder_layers"],
        d_ff=config["d_ff"], dropout=config["dropout"],
        factor=config["factor"], label_len=config["label_len"],
    )


def _build_convlstm(config, H, W):
    from src.models.convlstm import SpatialConvLSTM
    hidden_channels = config.get("hidden_channels") or \
                      [config["hidden_dim"]] * config["n_layers"]
    return SpatialConvLSTM(
        H=H, W=W,
        context_len=config["context_len"], horizon=config["horizon"],
        hidden_channels=hidden_channels,
        kernel_size=config["kernel_size"], dropout=config["dropout"],
    )


def _build_transformer(config, H, W):
    from src.models.transformer import SstFlatTransformer as SpatialFlatTransformer
    return SpatialFlatTransformer(
        height=H, width=W,
        seq_len=config["context_len"], horizon=config["horizon"],
        d_model=config["d_model"], n_heads=config["n_heads"],
        n_blocks=config["n_layers"],
        d_ff=config["ffn_dim"],
        dropout=config["dropout"],
    )


def _build_patch_transformer(config, H, W):
    from src.models.patch_transformer import SstPatchTransformer
    return SstPatchTransformer(
        height=H, width=W,
        patch_height=config["patch_height"], patch_width=config["patch_width"],
        seq_len=config["context_len"], horizon=config["horizon"],
        d_model=config["d_model"], n_blocks=config["n_blocks"],
        n_heads=config["n_heads"], d_ff=config["d_ff"], dropout=config["dropout"],
    )


def _build_rnn(config, H, W):
    from src.baselines.rnn import RNN
    return RNN(
        H=H, W=W,
        context_len=config["context_len"], horizon=config["horizon"],
        d_spatial=config["d_spatial"], hidden_size=config["hidden_size"],
        num_layers=config["num_layers"], dropout=config["dropout"],
    )


MODEL_BUILDERS = {
    "tubelet":     _build_tubelet,
    "lstm":        _build_lstm,
    "informer":    _build_informer,
    "convlstm":    _build_convlstm,
    "transformer": _build_transformer,
    "patch_transformer": _build_patch_transformer,
    "rnn":         _build_rnn,
}


# ---------------------------------------------------------------------------
# Find best config + best ablation alpha
# ---------------------------------------------------------------------------

def find_best_config(model_type: str) -> dict:
    best, best_rmse = None, float("inf")
    for d in sorted(RESULTS_DIR.glob("run_*/")):
        cf, mf = d / "config.json", d / "metrics.json"
        if not cf.exists() or not mf.exists():
            continue
        config = json.load(open(cf))
        mt = config.get("model_type") or config.get("model") or infer_model_type(config)
        if mt != model_type:
            continue
        if config.get("ablation"):
            continue
        metrics = json.load(open(mf))
        rmse = metrics["mean_rmse"]
        if isinstance(rmse, dict):
            rmse = rmse["model"]
        if rmse < best_rmse:
            best_rmse, best = rmse, config
    if best is None:
        raise SystemExit(f"No completed search run found for model={model_type!r}. "
                         f"Run the HPO script first.")
    print(f"  Best {model_type} HPO config: mean RMSE {best_rmse:.4f} "
          f"(alpha={best.get('anomaly_alpha', 0):.3f})")
    return best


def find_best_ablation_alpha(model_type: str) -> float | None:
    """Return the alpha with lowest RMSE from ablation runs, or None if none exist."""
    best_alpha, best_rmse = None, float("inf")
    for d in sorted(RESULTS_DIR.glob("run_*/")):
        cf, mf = d / "config.json", d / "metrics.json"
        if not cf.exists() or not mf.exists():
            continue
        config = json.load(open(cf))
        mt = config.get("model_type") or config.get("model") or infer_model_type(config)
        if mt != model_type or not config.get("ablation"):
            continue
        metrics = json.load(open(mf))
        rmse = metrics["mean_rmse"]
        if isinstance(rmse, dict):
            rmse = rmse["model"]
        if rmse < best_rmse:
            best_rmse, best_alpha = rmse, config.get("anomaly_alpha", 0.0)
    if best_alpha is not None:
        print(f"  Best ablation alpha for {model_type}: {best_alpha:.3f} "
              f"(RMSE {best_rmse:.4f})")
    else:
        print(f"  No ablation results found for {model_type} — "
              f"using HPO alpha as-is")
    return best_alpha


# ---------------------------------------------------------------------------
# BIC
# ---------------------------------------------------------------------------

def compute_bic(model: torch.nn.Module,
                preds_norm: np.ndarray,
                targets_norm: np.ndarray,
                land_mask: np.ndarray) -> tuple[float, int]:
    """BIC = k·ln(n) + n·ln(MSE_test), Gaussian-noise approximation."""
    k = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ocean = land_mask.astype(bool)           # (H, W)
    # preds_norm: (N, h, H, W) — only ocean pixels, normalised space
    diff = (preds_norm - targets_norm)[:, :, ocean]   # (N, h, ocean)
    n    = int(diff.size)
    mse  = float(np.mean(diff ** 2))
    bic  = k * np.log(n) + n * np.log(mse)
    return bic, k


# ---------------------------------------------------------------------------
# Persistence baseline
# ---------------------------------------------------------------------------

def compute_persistence_rmse(test_loader, norm_mean, norm_std,
                              land_mask, horizon, device) -> np.ndarray:
    all_X, all_y = [], []
    for bx, by in test_loader:
        all_X.append(bx.numpy())
        all_y.append(by.numpy())
    X_norm = np.concatenate(all_X, axis=0)   # (N, L, 1, H, W)
    y_norm = np.concatenate(all_y, axis=0)   # (N, h, H, W)

    pers_norm = persistence_forecast(X_norm, horizon=horizon)
    pers      = pers_norm * norm_std + norm_mean
    targets   = y_norm   * norm_std + norm_mean
    return rmse_per_step(pers, targets, land_mask=land_mask)


# ---------------------------------------------------------------------------
# Date helper
# ---------------------------------------------------------------------------

def idx_to_date(zarr_root, window_start_idx: int, context_len: int) -> str:
    """Return the calendar date of the first forecast day for a test window."""
    try:
        time_arr = np.array(zarr_root["time"])
        # time values are days since 1970-01-01
        epoch = date(1970, 1, 1)
        # Test split start in the time array
        test_start_str = zarr_root.attrs.get("test_start", "")
        if test_start_str:
            ts = date.fromisoformat(test_start_str)
            ts_days = (ts - epoch).days
            # Find the first index in time_arr >= ts_days
            test_arr_idx = int(np.searchsorted(time_arr, ts_days))
        else:
            test_arr_idx = 0
        forecast_arr_idx = test_arr_idx + window_start_idx + context_len
        if forecast_arr_idx < len(time_arr):
            d = epoch + timedelta(days=int(time_arr[forecast_arr_idx]))
            return d.isoformat()
    except Exception:
        pass
    return f"test_window_{window_start_idx}"


# ---------------------------------------------------------------------------
# Retrain one model
# ---------------------------------------------------------------------------

def retrain_one(model_type: str, device: torch.device,
                zarr_root, norm_mean: float, norm_std: float,
                land_mask: np.ndarray, seed: int | None = None,
                batch_size_override: int | None = None):
    H, W = land_mask.shape
    # seed=None -> canonical dir (backward compatible); explicit seed -> suffix.
    suffix = f"_seed{seed}" if seed is not None else ""
    save_dir = BEST_DIR / f"best_{model_type}{suffix}"
    save_dir.mkdir(parents=True, exist_ok=True)

    config = find_best_config(model_type)

    # Transformer HPO ran with max_epochs=15; give the final retrain the full budget.
    if model_type == "transformer" and config.get("num_epochs", 50) < 50:
        config = dict(config)
        config["num_epochs"] = 50

    # Override alpha with ablation result if available
    best_alpha = find_best_ablation_alpha(model_type)
    if best_alpha is not None:
        config = dict(config)
        config["anomaly_alpha"] = best_alpha

    eff_seed = RANDOM_SEED if seed is None else seed
    config = dict(config)
    config["random_seed"] = eff_seed
    np.random.seed(eff_seed)
    torch.manual_seed(eff_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eff_seed)

    batch_size = batch_size_override if batch_size_override is not None else config.get("batch_size", 8)
    if batch_size_override is not None and batch_size_override != config.get("batch_size"):
        print(f"  batch_size overridden: {config.get('batch_size')} → {batch_size}")
    use_cuda = (device.type == "cuda")
    # Overlap host-side batch assembly + H2D copy with GPU compute. The dataset
    # holds its field in RAM, so workers fork copy-on-write (no per-worker reload).
    n_workers = 4 if use_cuda else 0
    train_loader, val_loader, test_loader = create_dataloaders(
        zarr_path=ZARR_PATH,
        context_len=config["context_len"],
        horizon=config["horizon"],
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=use_cuda,
    )

    model = MODEL_BUILDERS[model_type](config, H, W).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = optim.Adam(model.parameters(),
                           lr=config["learning_rate"],
                           weight_decay=config.get("weight_decay", 1e-4))
    criterion = AnomalyWeightedMSE(alpha=config.get("anomaly_alpha", 0.0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=config.get("lr_factor", 0.5),
        patience=config.get("lr_patience", 5),
        threshold=1e-3, min_lr=1e-5, cooldown=2,
    )
    land_mask_t = torch.from_numpy(land_mask).to(device)

    print(f"  Training {model_type} ({n_params:,} params) on {device}"
          f"{' [BF16 AMP]' if use_cuda else ''}...")
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

    print(f"  Evaluating on test set...")
    preds_norm, targets_norm = predict(model, test_loader, device)
    preds   = preds_norm   * norm_std + norm_mean
    targets = targets_norm * norm_std + norm_mean

    rmse_steps  = rmse_per_step(preds, targets, land_mask=land_mask)
    pers_steps  = compute_persistence_rmse(
        test_loader, norm_mean, norm_std, land_mask,
        config["horizon"], device,
    )
    skill_steps = 1.0 - rmse_steps / pers_steps
    bic, _      = compute_bic(model, preds_norm, targets_norm, land_mask)

    example_date = idx_to_date(zarr_root, EXAMPLE_IDX, config["context_len"])

    # Save all artifacts
    np.save(save_dir / "predictions.npy", preds)
    np.save(save_dir / "targets.npy",     targets)
    torch.save(model.state_dict(), save_dir / "model.pt")

    with open(save_dir / "loss_curves.json", "w") as f:
        json.dump({"train": train_losses, "val": val_losses}, f, indent=2)

    with open(save_dir / "summary.json", "w") as f:
        json.dump({
            "model_type":      model_type,
            "n_params":        n_params,
            "epochs_trained":  len(train_losses),
            "mean_rmse":       float(rmse_steps.mean()),
            "mean_skill":      float(skill_steps.mean()),
            "bic":             float(bic),
            "rmse_steps":      rmse_steps.tolist(),
            "pers_rmse_steps": pers_steps.tolist(),
            "skill_steps":     skill_steps.tolist(),
            "example_idx":     EXAMPLE_IDX,
            "example_date":    example_date,
        }, f, indent=2)

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"  {model_type} done — RMSE={rmse_steps.mean():.4f}  "
          f"skill={skill_steps.mean():.4f}  BIC={bic:.0f}  "
          f"params={n_params:,}  epochs={len(train_losses)}")
    print(f"  Saved to {save_dir}")
    return rmse_steps.mean()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS,
        help="Which models to retrain (default: all)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed. Default (None) uses canonical seed 42 and writes to "
             "experiments/best_<model>/. Pass an explicit int (e.g. 1, 2, 3) for "
             "variance studies — outputs land in experiments/best_<model>_seed<N>/.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override the config batch_size. Useful when retraining on a GPU with "
             "more VRAM than was used during HPO (e.g. H200 141 GB vs original run). "
             "Does not affect model quality — only training throughput.",
    )
    args = parser.parse_args()

    root         = zarr.open_group(str(ZARR_PATH), mode="r")
    norm_mean    = float(root.attrs["norm_mean"])
    norm_std     = float(root.attrs["norm_std"])
    land_mask    = np.array(root["land_mask"])
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Models to retrain: {args.models}")
    if args.seed is not None:
        print(f"Seed: {args.seed} (writing to best_<model>_seed{args.seed}/)")
    print()

    results = {}
    for model_type in args.models:
        print(f"=== {model_type.upper()} ===")
        try:
            rmse = retrain_one(model_type, device, root,
                               norm_mean, norm_std, land_mask,
                               seed=args.seed,
                               batch_size_override=args.batch_size)
            results[model_type] = rmse
        except SystemExit as e:
            print(f"  Skipped: {e}")
        print()

    if results:
        print("=== Summary ===")
        for mt, rmse in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {mt:12s}  RMSE={rmse:.4f}")


if __name__ == "__main__":
    main()
