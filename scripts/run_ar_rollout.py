"""
Autoregressive rollout: how far can the trained models predict?

Each model was trained for direct 7-day forecasting. This script rolls them
forward day-by-day past that horizon by feeding day-1 of the model's own
prediction back into the context window, then predicting again, repeated for
MAX_HORIZON days. Errors compound, and the model eventually converges to
climatology - the "useful horizon" is the last day where it still beats it.

Usage
-----
    python scripts/run_ar_rollout.py                       # 30-day rollout, all 3 models
    python scripts/run_ar_rollout.py --max-horizon 60
    python scripts/run_ar_rollout.py --models patch_transformer convlstm
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.dataset import SstWindowDataset
from src.utils.metrics import rmse_per_step, skill_score
from src.baselines.persistence import persistence_forecast
from scripts.retrain_best import MODEL_BUILDERS

ZARR_PATH = PROJECT_ROOT / "data/processed/oisst_coralsea.zarr"
BEST_DIR  = PROJECT_ROOT / "experiments"
OUT_DIR   = PROJECT_ROOT / "experiments/ar_rollout"

DEFAULT_MODELS = ["patch_transformer", "convlstm", "tubelet"]
COLOURS = {
    "patch_transformer": "#edc948",
    "convlstm":          "#59a14f",
    "tubelet":           "#4e79a7",
}
DISPLAY = {
    "patch_transformer": "Patch Transformer",
    "convlstm":          "ConvLSTM",
    "tubelet":           "Tubelet Transformer",
}

# Core: load a model and run an AR rollout on a batch of contexts

def load_model(model_type: str, device: torch.device, H: int, W: int,
               seed: int | None = None):
    # Read the config that the retrained checkpoint was built with - directly
    # from best_<model>[_seed<N>]/config.json. Avoids relying on the original
    # HPO run dir, which may not exist on every branch.
    suffix = f"_seed{seed}" if seed is not None else ""
    best = BEST_DIR / f"best_{model_type}{suffix}"
    config = json.load(open(best / "config.json"))
    model = MODEL_BUILDERS[model_type](config, H, W).to(device)
    model.load_state_dict(torch.load(best / "model.pt", map_location=device))
    model.eval()
    print(f"  loaded {model_type} from {best / 'model.pt'}")
    return model

def autoregressive_predict(model, x: torch.Tensor, max_horizon: int,
                           device: torch.device) -> np.ndarray:
    """
    x:  (B, 90, 1, H, W) normalised context.
    Returns (B, max_horizon, H, W) of model's own rolled-out predictions,
    still in normalised space.
    """
    ctx = x.to(device)
    preds = []
    with torch.no_grad():
        for _ in range(max_horizon):
            out = model(ctx)        # (B, h=7, H, W)
            day1 = out[:, 0]        # (B, H, W) - keep only day-1
            preds.append(day1)
            # append day-1 prediction to context as a new (1, 1, H, W) frame,
            # drop the oldest frame
            ctx = torch.cat([ctx[:, 1:], day1[:, None, None]], dim=1)
    return torch.stack(preds, dim=1).cpu().numpy()

# Plot

def plot_ar(rmse: dict, skill: dict, useful: dict, max_h: int, out_path: Path):
    days = list(range(1, max_h + 1))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left: RMSE vs forecast day ---
    axL.plot(days, rmse["climatology"], "--", color="grey", lw=1.6, label="Climatology")
    axL.plot(days, rmse["persistence"], ":",  color="black", lw=1.5, label="Persistence")
    model_keys = [m for m in rmse if m not in ("persistence", "climatology")]
    for mt in model_keys:
        axL.plot(days, rmse[mt], color=COLOURS.get(mt, "#999"), lw=2.0,
                 marker="o", markersize=4, label=DISPLAY.get(mt, mt))
    axL.axvline(7, color="grey", linestyle=":", alpha=0.6, lw=1.0)
    ylo, _ = axL.get_ylim()
    axL.text(7.2, ylo + 0.01, "training horizon",
             fontsize=8, color="grey", style="italic", va="bottom")
    axL.set_xlabel("Forecast day")
    axL.set_ylabel(r"RMSE ($^\circ$C)")
    axL.set_title("RMSE vs lead time - autoregressive rollout")
    axL.legend(fontsize=9, loc="lower right")
    axL.grid(alpha=0.3)

    # --- Right: skill vs climatology ---
    axR.axhline(0, color="black", lw=1.0, linestyle="--")
    for mt in model_keys:
        sk = skill[mt]
        axR.plot(days, sk, color=COLOURS.get(mt, "#999"), lw=2.0,
                 marker="o", markersize=4, label=DISPLAY.get(mt, mt))
        d = useful.get(mt, 0)
        if d > 0:
            axR.axvline(d, color=COLOURS.get(mt, "#999"),
                        linestyle=":", alpha=0.55, lw=1.0)
    axR.axvline(7, color="grey", linestyle=":", alpha=0.6, lw=1.0)

    # Useful-horizon labels in a single compact box (avoids overlapping annotations)
    if useful:
        txt = "\n".join(f"{DISPLAY.get(mt, mt)}: useful to day {useful[mt]}"
                        for mt in model_keys)
        axR.text(0.02, 0.04, txt, transform=axR.transAxes,
                 fontsize=9, va="bottom",
                 bbox=dict(boxstyle="round,pad=0.4",
                           facecolor="white", edgecolor="#ccc", alpha=0.95))

    axR.set_xlabel("Forecast day")
    axR.set_ylabel("Skill vs climatology")
    axR.set_title("Skill score - first crossing of zero is the useful horizon")
    axR.legend(fontsize=9, loc="upper right")
    axR.grid(alpha=0.3)

    fig.suptitle("Autoregressive rollout - how far can we predict?",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-horizon", type=int, default=30,
                        help="How many days to roll out (default 30).")
    parser.add_argument("--batch-size",  type=int, default=8,
                        help="Inference batch size (default 8 - fits ConvLSTM "
                             "without BPTT activations).")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        choices=DEFAULT_MODELS,
                        help="Which models to roll out.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Load best_<model>_seed<N>/ instead of the canonical "
                             "best_<model>/; outputs go to ar_rollout/seed<N>/. "
                             "Defaults to the canonical (seed-42) checkpoints.")
    args = parser.parse_args()
    out_dir = OUT_DIR / f"seed{args.seed}" if args.seed is not None else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = zarr.open_group(str(ZARR_PATH), mode="r")
    norm_mean = float(root.attrs["norm_mean"])
    norm_std  = float(root.attrs["norm_std"])
    land_mask = np.array(root["land_mask"]).astype(bool)
    H, W = land_mask.shape

    # Extended-horizon test dataset: each window comes with max_horizon days of truth
    dataset = SstWindowDataset(ZARR_PATH, "test", 90, args.max_horizon)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, pin_memory=True, persistent_workers=True)
    print(f"Test windows: {len(dataset)} | max_horizon: {args.max_horizon} | "
          f"batch: {args.batch_size} | device: {device}")

    # --- Gather X (90-day contexts) and y (extended truth) once ---
    print("\nLoading test windows...")
    all_X, all_y = [], []
    for x, y in loader:
        all_X.append(x.numpy())
        all_y.append(y.numpy())
    X_norm = np.concatenate(all_X, axis=0)   # (N, 90, 1, H, W)
    y_norm = np.concatenate(all_y, axis=0)   # (N, max_h, H, W)
    del all_X, all_y, loader
    y_denorm = y_norm * norm_std + norm_mean
    del y_norm
    print(f"  X_norm: {X_norm.shape}, y_denorm: {y_denorm.shape}")

    # --- Baselines ---
    # Persistence: predict every future day = last context day
    pers_denorm = persistence_forecast(X_norm, args.max_horizon) * norm_std + norm_mean
    pers_rmse = rmse_per_step(pers_denorm, y_denorm, land_mask=land_mask).tolist()
    del pers_denorm

    # Climatology: predicting climatology = predicting zero in anomaly space
    # (anomaly = SST - climatology, so anomaly-forecast = 0 is climatology-forecast)
    clim_pred = np.full_like(y_denorm, norm_mean)
    clim_rmse = rmse_per_step(clim_pred, y_denorm, land_mask=land_mask).tolist()
    del clim_pred
    print(f"\nPersistence day-1 RMSE: {pers_rmse[0]:.3f} | "
          f"day-{args.max_horizon} RMSE: {pers_rmse[-1]:.3f}")
    print(f"Climatology  RMSE (~flat across horizons): "
          f"{np.mean(clim_rmse):.3f} (range {min(clim_rmse):.3f}-{max(clim_rmse):.3f})")

    # --- Per-model AR rollout ---
    results: dict[str, list[float]] = {"persistence": pers_rmse,
                                       "climatology": clim_rmse}
    skill_results: dict[str, list[float]] = {}
    useful: dict[str, int] = {}

    for mt in args.models:
        print(f"\n=== {mt} ===")
        try:
            model = load_model(mt, device, H, W, seed=args.seed)
        except (FileNotFoundError, ModuleNotFoundError, ImportError) as e:
            print(f"  SKIP {mt}: {e}", file=sys.stderr)
            continue
        n = X_norm.shape[0]
        n_batches = (n + args.batch_size - 1) // args.batch_size
        all_preds = []
        for bi, start in enumerate(range(0, n, args.batch_size), start=1):
            x_batch = torch.from_numpy(X_norm[start:start + args.batch_size])
            preds_norm = autoregressive_predict(model, x_batch,
                                                args.max_horizon, device)
            all_preds.append(preds_norm)
            if bi % 10 == 0 or bi == n_batches:
                print(f"  batch {bi:>3d}/{n_batches}", flush=True)
        preds_arr = np.concatenate(all_preds, axis=0)
        del all_preds
        preds_denorm = preds_arr * norm_std + norm_mean
        del preds_arr

        per_day = rmse_per_step(preds_denorm, y_denorm,
                                land_mask=land_mask).tolist()
        results[mt] = per_day

        sk = [skill_score(per_day[h], clim_rmse[h])
              for h in range(args.max_horizon)]
        skill_results[mt] = sk
        # Useful horizon: last day before skill first drops to <= 0.
        # We still roll out the full max_horizon so the skill curve in
        # the JSON / plot remains complete; this is purely a summary number.
        useful_day = 0
        for h, v in enumerate(sk):
            if v > 0:
                useful_day = h + 1   # 1-indexed
            else:
                break
        useful[mt] = useful_day
        print(f"  day-1 RMSE: {per_day[0]:.3f}  |  "
              f"day-{args.max_horizon} RMSE: {per_day[-1]:.3f}")
        print(f"  useful horizon: {useful_day} days (skill > 0)")

        del preds_denorm, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save one file per model. The deterministic baselines are stored alongside
    # each model's data so every per-model file is self-contained.
    for mt in args.models:
        if mt not in results:   # may have been skipped on import/load error
            continue
        json.dump({"persistence": results["persistence"],
                   "climatology": results["climatology"],
                   mt:           results[mt]},
                  open(out_dir / f"rmse_per_day_{mt}.json", "w"), indent=2)
        json.dump({mt: skill_results[mt]},
                  open(out_dir / f"skill_vs_climatology_{mt}.json", "w"), indent=2)
        json.dump({mt: useful[mt]},
                  open(out_dir / f"useful_horizon_{mt}.json", "w"), indent=2)

    plot_ar(results, skill_results, useful, args.max_horizon,
            out_dir / "ar_long_horizon.png")
    print(f"\nOutputs written to {out_dir}")
    print(f"Useful horizons: {useful}")

if __name__ == "__main__":
    main()
