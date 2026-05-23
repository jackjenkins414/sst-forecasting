"""
Compare all experiment runs visually.

Reads every run_*/config.json + metrics.json and produces two figures:

  Figure 1 — Hyperparameter vs RMSE scatter plots
      One subplot per searchable hyperparameter.  Each point is one run,
      coloured by model type.

  Figure 2 — RMSE-per-step curves (overlaid)
      Background lines coloured by model type; top-5 runs highlighted.
      Persistence baseline shown for reference.
      A second panel shows skill-score curves overlaid the same way.

Usage
-----
    python scripts/compare_runs.py
    python scripts/compare_runs.py --out experiments/comparison.png
    python scripts/compare_runs.py --top_n 10
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches

RESULTS_DIR = PROJECT_ROOT / "experiments/results"
HORIZON     = 7
DAYS        = np.arange(1, HORIZON + 1)

# Colour per model architecture — add entries here as new models are registered
MODEL_COLOURS = {
    "tubelet":     "#4e79a7",  # blue
    "lstm":        "#f28e2b",  # orange
    "convlstm":    "#59a14f",  # green
    "rnn":         "#e15759",  # red
    "transformer": "#b07aa1",  # purple
    "unknown":     "#9c9c9c",  # grey
}

# Hyperparameters to scatter — common and model-specific.
# Runs that don't have a key just get skipped in that subplot.
PARAMS = {
    # shared
    "learning_rate": "Learning rate",
    "dropout":       "Dropout",
    # tubelet / transformer
    "n_heads":       "Attention heads",
    "n_layers":      "Layers",
    "lr_factor":     "LR decay factor",
    "anomaly_alpha": "Anomaly α",
    # lstm / rnn
    "hidden_size":   "Hidden size",
    "d_spatial":     "Spatial dim",
    # convlstm
    "hidden_dim":    "ConvLSTM hidden channels",
    "kernel_size":   "Conv kernel size",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_runs() -> list[dict]:
    runs = []
    for run_dir in sorted(RESULTS_DIR.glob("run_*/")):
        config_f  = run_dir / "config.json"
        metrics_f = run_dir / "metrics.json"
        if not config_f.exists() or not metrics_f.exists():
            continue

        with open(config_f)  as f: config  = json.load(f)
        with open(metrics_f) as f: metrics = json.load(f)

        # Normalise key names across old and new run formats
        config.setdefault("learning_rate", config.pop("lr", config.get("learning_rate")))
        model_type = config.get("model_type", "unknown")

        rmse_steps = []
        pers_steps_raw = []
        for d in DAYS:
            entry = metrics["rmse_per_step"][f"day_{d}"]
            if isinstance(entry, dict):
                rmse_steps.append(entry["model"])
                pers_steps_raw.append(entry.get("persistence"))
            else:
                rmse_steps.append(entry)
                pers_steps_raw.append(None)

        mean_rmse = metrics["mean_rmse"]
        if isinstance(mean_rmse, dict):
            mean_rmse = mean_rmse["model"]

        runs.append({
            "name":       run_dir.name,
            "model_type": model_type,
            "config":     config,
            "mean_rmse":  mean_rmse,
            "rmse_steps": np.array(rmse_steps, dtype=float),
            "pers_steps": np.array(pers_steps_raw, dtype=float)
                          if pers_steps_raw[0] is not None else None,
            "epochs":     metrics.get("epochs_trained", "?"),
        })

    runs.sort(key=lambda r: r["mean_rmse"])
    return runs


# ---------------------------------------------------------------------------
# Figure 1: hyperparameter scatter plots
# ---------------------------------------------------------------------------

def plot_param_scatter(runs: list[dict], save_path: Path):
    # Only show subplots where at least one run has the param
    active_params = {
        k: v for k, v in PARAMS.items()
        if any(r["config"].get(k) is not None for r in runs)
    }
    if not active_params:
        print("No hyperparameter data found — skipping scatter plot.")
        return

    ncols = 3
    nrows = (len(active_params) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    fig.suptitle("Hyperparameter vs Mean RMSE (°C) — all runs", fontsize=13)
    axes = axes.flatten()

    model_types = sorted({r["model_type"] for r in runs})

    for ax, (key, label) in zip(axes, active_params.items()):
        for mt in model_types:
            mt_runs = [r for r in runs if r["model_type"] == mt and r["config"].get(key) is not None]
            if not mt_runs:
                continue
            xs = [r["config"][key] for r in mt_runs]
            ys = [r["mean_rmse"]   for r in mt_runs]
            colour = MODEL_COLOURS.get(mt, MODEL_COLOURS["unknown"])
            ax.scatter(xs, ys, color=colour, s=60, edgecolors="black",
                       linewidths=0.4, label=mt, zorder=3, alpha=0.85)

        # Log scale for learning rate
        if key == "learning_rate":
            ax.set_xscale("log")

        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Mean RMSE (°C)", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Legend — one entry per model type, placed on the last active axis
    handles = [
        mpatches.Patch(color=MODEL_COLOURS.get(mt, MODEL_COLOURS["unknown"]), label=mt)
        for mt in model_types
    ]
    axes[len(active_params) - 1].legend(handles=handles, fontsize=8, title="Model")

    for ax in axes[len(active_params):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2: overlaid RMSE-per-step + skill curves
# ---------------------------------------------------------------------------

def plot_curves(runs: list[dict], save_path: Path, top_n: int = 5):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("RMSE & Skill per Forecast Day — all runs", fontsize=13)

    # Persistence from the first run that has it (same signal for all runs)
    pers_run = next((r for r in runs if r["pers_steps"] is not None), None)
    if pers_run:
        ax1.plot(DAYS, pers_run["pers_steps"], "k--", linewidth=2,
                 label="Persistence", zorder=5)
        ax2.axhline(0, color="black", linewidth=1.2, zorder=5)

    # Background lines — coloured by model type, faint
    for r in runs:
        colour = MODEL_COLOURS.get(r["model_type"], MODEL_COLOURS["unknown"])
        ax1.plot(DAYS, r["rmse_steps"], color=colour, alpha=0.20, linewidth=0.9)
        if pers_run:
            skill = 1 - r["rmse_steps"] / pers_run["pers_steps"]
            ax2.plot(DAYS, skill, color=colour, alpha=0.20, linewidth=0.9)

    # Top N — bold, labelled, coloured by model type (darker shade)
    for i, r in enumerate(runs[:top_n]):
        colour = MODEL_COLOURS.get(r["model_type"], MODEL_COLOURS["unknown"])
        label  = f"#{i+1} [{r['model_type']}] {r['name'][-13:]} {r['mean_rmse']:.4f}°C"
        ax1.plot(DAYS, r["rmse_steps"], color=colour, linewidth=2.2,
                 marker="o", markersize=4, label=label, zorder=4)
        if pers_run:
            skill = 1 - r["rmse_steps"] / pers_run["pers_steps"]
            ax2.plot(DAYS, skill, color=colour, linewidth=2.2,
                     marker="o", markersize=4, label=label, zorder=4)

    # Model-type legend patch (background line key)
    model_types = sorted({r["model_type"] for r in runs})
    type_handles = [
        mpatches.Patch(color=MODEL_COLOURS.get(mt, MODEL_COLOURS["unknown"]),
                       alpha=0.5, label=f"{mt} (all runs)")
        for mt in model_types
    ]

    ax1.set_xlabel("Forecast day");  ax1.set_ylabel("RMSE (°C)")
    ax1.set_title("RMSE per step")
    ax1.legend(fontsize=7, handles=ax1.get_legend_handles_labels()[0]
               + type_handles)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Forecast day");  ax2.set_ylabel("Skill vs persistence")
    ax2.set_title("Skill score per step")
    ax2.legend(fontsize=7, handles=ax2.get_legend_handles_labels()[0]
               + type_handles)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str,
                        default=str(PROJECT_ROOT / "experiments"))
    parser.add_argument("--top_n",  type=int, default=5,
                        help="How many top runs to highlight in curve plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs()
    if not runs:
        print("No completed runs found in", RESULTS_DIR)
        return

    model_counts = {}
    for r in runs:
        model_counts[r["model_type"]] = model_counts.get(r["model_type"], 0) + 1
    counts_str = "  ".join(f"{mt}={n}" for mt, n in sorted(model_counts.items()))
    print(f"Loaded {len(runs)} runs  [{counts_str}]")
    print(f"Best overall: {runs[0]['name']}  [{runs[0]['model_type']}]  "
          f"RMSE={runs[0]['mean_rmse']:.4f}")

    plot_param_scatter(runs, out_dir / "comparison_params.png")
    plot_curves(runs,        out_dir / "comparison_curves.png", top_n=args.top_n)

    print("\nTop 5 runs:")
    for i, r in enumerate(runs[:5], 1):
        cfg = r["config"]
        print(f"  {i}. [{r['model_type']:12s}] RMSE={r['mean_rmse']:.4f}  "
              f"lr={cfg.get('learning_rate', float('nan')):.2e}  "
              f"dropout={cfg.get('dropout', float('nan')):.2f}  "
              f"epochs={r['epochs']}")


if __name__ == "__main__":
    main()
