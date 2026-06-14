"""
Post-search anomaly-alpha ablation.

Finds the best HPO run for a given model, then re-trains it across a grid of
anomaly_alpha values with EVERY other hyperparameter held fixed. This isolates
the effect of anomaly weighting at the model's tuned optimum - answering
"does anomaly weighting help this architecture?" as a controlled comparison
(unlike the joint search, where alpha is confounded with the other params).

Each ablation run is saved under experiments/results/ tagged with
ablation=True so compare_runs.py still reads it, and an alpha-response curve
is saved to experiments/alpha_ablation_<model>.png.

Usage
-----
    python scripts/run_alpha_ablation.py --model lstm
    python scripts/run_alpha_ablation.py --model informer --alphas 0,0.05,0.1,0.2
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
import torch.optim as optim
import zarr

from src.data.dataloaders import create_dataloaders
from src.training.losses import AnomalyWeightedMSE
from src.training.train import train_model
from src.training.evaluate import predict
from src.utils.metrics import rmse_per_step

ZARR_PATH   = PROJECT_ROOT / "data/processed/oisst_coralsea.zarr"
RESULTS_DIR = PROJECT_ROOT / "experiments/results"

DEFAULT_ALPHAS = [0.0, 0.05, 0.10, 0.20]
ALPHA_MAX = 0.20  # anomaly_alpha search-space ceiling

def _auto_alpha_grid(optimal_alpha: float | None) -> list[float]:
    """Bracket the tuned optimum: {0, a/2, a, 2a}, clamped to [0, ALPHA_MAX], deduped.

    Anchors the sweep on the model's own tuned alpha (a*) instead of arbitrary
    values, while still spanning below/above it to reveal the response curve.
    Falls back to a fixed spread when a* ~= 0, where a relative grid degenerates.
    """
    if optimal_alpha is None or optimal_alpha < 1e-3:
        return list(DEFAULT_ALPHAS)
    raw = [0.0, optimal_alpha / 2, optimal_alpha, min(2 * optimal_alpha, ALPHA_MAX)]
    return sorted({round(a, 4) for a in raw})

# Model builders - one per architecture, reading hyperparams from a config dict

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

def _build_convlstm(config, H, W):
    from src.sst_forecasting.models.convlstm import SpatialConvLSTM
    hidden_channels = config.get("hidden_channels")
    if hidden_channels is None:
        hidden_channels = [config["hidden_dim"]] * config["n_layers"]
    return SpatialConvLSTM(
        H=H, W=W,
        context_len=config["context_len"], horizon=config["horizon"],
        hidden_channels=hidden_channels,
        kernel_size=config["kernel_size"], dropout=config["dropout"],
    )

MODEL_BUILDERS = {
    "lstm":     _build_lstm,
    "informer": _build_informer,
    "tubelet":  _build_tubelet,
    "convlstm": _build_convlstm,
}

# Find the best (non-ablation) run for a model

def _infer_model_type(config: dict) -> str:
    """Identify model for runs predating the model_type key (e.g. old Tubelet)."""
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

def find_best_config(model_type: str) -> dict:
    best = None
    best_rmse = float("inf")
    for d in sorted(RESULTS_DIR.glob("run_*/")):
        cf, mf = d / "config.json", d / "metrics.json"
        if not cf.exists() or not mf.exists():
            continue
        config = json.load(open(cf))
        mt = config.get("model_type") or config.get("model") or _infer_model_type(config)
        if mt != model_type:
            continue
        if config.get("ablation"):           # don't seed from a prior ablation
            continue
        metrics = json.load(open(mf))
        rmse = metrics["mean_rmse"]
        if isinstance(rmse, dict):
            rmse = rmse["model"]
        if rmse < best_rmse:
            best_rmse, best = rmse, config
    if best is None:
        raise SystemExit(f"No completed search run found for model_type={model_type!r}")
    print(f"Best {model_type} config: mean RMSE {best_rmse:.4f} "
          f"(alpha was {best.get('anomaly_alpha')})")
    return best

# Run the ablation

def run_ablation(model_type: str, alphas: list[float] | None = None):
    builder = MODEL_BUILDERS[model_type]
    base = find_best_config(model_type)

    if alphas is None:
        alphas = _auto_alpha_grid(base.get("anomaly_alpha"))
    print(f"  alpha grid: {alphas}  (tuned optimum alpha*={base.get('anomaly_alpha')})")

    root         = zarr.open_group(str(ZARR_PATH), mode="r")
    norm_mean    = float(root.attrs["norm_mean"])
    norm_std     = float(root.attrs["norm_std"])
    land_mask_np = np.array(root["land_mask"])
    H, W = land_mask_np.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    land_mask_torch = torch.from_numpy(land_mask_np).to(device)
    seed = base.get("random_seed", 42)

    train_loader, val_loader, test_loader = create_dataloaders(
        zarr_path=ZARR_PATH,
        context_len=base["context_len"], horizon=base["horizon"],
        batch_size=base["batch_size"],
    )

    results = []
    for alpha in alphas:
        # Reset seeds so the ONLY difference between runs is alpha
        np.random.seed(seed)
        torch.manual_seed(seed)

        config = dict(base)
        config["anomaly_alpha"] = alpha
        config["ablation"] = True
        config["optuna_trial"] = None

        run_dir = RESULTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        model = builder(config, H, W).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"],
                               weight_decay=config.get("weight_decay", 1e-4))
        criterion = AnomalyWeightedMSE(alpha=alpha)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=config.get("lr_factor", 0.5),
            patience=config.get("lr_patience", 5),
            threshold=1e-3, min_lr=1e-5, cooldown=2,
        )

        train_losses, val_losses = train_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=optimizer, device=device,
            num_epochs=config["num_epochs"], land_mask=land_mask_torch,
            grad_clip=config.get("grad_clip", 1.0), scheduler=scheduler,
            early_stop_patience=config.get("early_stop_patience", 5),
        )

        preds_norm, targets_norm = predict(model, test_loader, device)
        preds   = preds_norm   * norm_std + norm_mean
        targets = targets_norm * norm_std + norm_mean
        rmse_steps = rmse_per_step(preds, targets, land_mask=land_mask_np)
        mean_rmse  = float(rmse_steps.mean())

        metrics = {
            "epochs_trained": len(train_losses),
            "best_val_loss":  float(min(val_losses)),
            "mean_rmse":      {"model": mean_rmse},
            "rmse_per_step":  {f"day_{i+1}": float(r) for i, r in enumerate(rmse_steps)},
        }
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"  alpha={alpha:.3f} | mean RMSE {mean_rmse:.4f} | "
              f"{len(train_losses)} epochs | {run_dir.name}")
        results.append((alpha, mean_rmse))

    _plot_curve(model_type, results, base)
    return results

def _plot_curve(model_type, results, base):
    alphas = [a for a, _ in results]
    rmses  = [r for _, r in results]
    best_i = int(np.argmin(rmses))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, rmses, marker="o", linewidth=2)
    ax.scatter([alphas[best_i]], [rmses[best_i]], color="red", zorder=5,
               s=120, label=f"best α={alphas[best_i]:.3f} ({rmses[best_i]:.4f})")
    # alpha=0 reference line
    if 0.0 in alphas:
        ax.axhline(rmses[alphas.index(0.0)], color="grey", linestyle="--",
                   alpha=0.6, label=f"α=0 baseline ({rmses[alphas.index(0.0)]:.4f})")
    ax.set_xlabel("anomaly_alpha")
    ax.set_ylabel("Test mean RMSE (°C)")
    ax.set_title(f"Anomaly-α ablation - {model_type}\n"
                 f"(all other hyperparams fixed at tuned optimum)")
    ax.legend()
    ax.grid(alpha=0.3)
    out = PROJECT_ROOT / "experiments" / f"alpha_ablation_{model_type}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_BUILDERS))
    parser.add_argument("--alphas", type=str, default=None,
                        help="Comma-separated alpha grid. Default: bracket the "
                             "tuned optimum {0, a*/2, a*, 2a*}.")
    args = parser.parse_args()

    alphas = (None if args.alphas is None
              else [float(a) for a in args.alphas.split(",")])

    print(f"=== Alpha ablation: {args.model} ===")
    results = run_ablation(args.model, alphas)

    print("\n--- Ablation summary ---")
    base_rmse = dict(results).get(0.0)
    for alpha, rmse in results:
        delta = ""
        if base_rmse is not None and alpha != 0.0:
            d = rmse - base_rmse
            delta = f"  ({'+' if d >= 0 else ''}{d:.4f} vs α=0)"
        print(f"  alpha={alpha:.3f}: RMSE={rmse:.4f}{delta}")

if __name__ == "__main__":
    main()
