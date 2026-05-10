"""
Compare all experiment runs visually.

Reads every run_*/config.json + metrics.json and produces two figures:

  Figure 1 — Hyperparameter vs RMSE scatter plots
      One subplot per searchable hyperparameter.  Each point is one run.
      Makes it easy to see which parameters correlate with lower error.

  Figure 2 — RMSE-per-step curves (overlaid)
      Every run as a faint line; top-5 runs highlighted and labelled.
      Persistence baseline shown for reference.
      A second panel shows skill-score curves overlaid the same way.

Usage
-----
    python scripts/compare_runs.py
    python scripts/compare_runs.py --out experiments/comparison.png
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

RESULTS_DIR = PROJECT_ROOT / "experiments/results"
HORIZON     = 7
DAYS        = np.arange(1, HORIZON + 1)

# Hyperparameters to plot (config key → display label)
PARAMS = {
    "learning_rate": "Learning rate",
    "n_heads":       "Attention heads",
    "n_layers":      "Layers",
    "lr_factor":     "LR factor",
    "dropout":       "Dropout",
    "anomaly_alpha": "Anomaly α",
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
        config.setdefault("anomaly_alpha", config.pop("anomaly_alpha", 0.0))
        config.setdefault("learning_rate", config.pop("lr", config.get("learning_rate")))

        rmse_steps = [
            metrics["rmse_per_step"][f"day_{d}"]["model"]
            if isinstance(metrics["rmse_per_step"][f"day_{d}"], dict)
            else metrics["rmse_per_step"][f"day_{d}"]
            for d in DAYS
        ]
        pers_steps = [
            metrics["rmse_per_step"][f"day_{d}"]["persistence"]
            if isinstance(metrics["rmse_per_step"][f"day_{d}"], dict)
            else None
            for d in DAYS
        ]

        runs.append({
            "name":       run_dir.name,
            "config":     config,
            "mean_rmse":  metrics["mean_rmse"]["model"]
                          if isinstance(metrics["mean_rmse"], dict)
                          else metrics["mean_rmse"],
            "rmse_steps": np.array(rmse_steps, dtype=float),
            "pers_steps": np.array(pers_steps, dtype=float) if pers_steps[0] is not None else None,
            "epochs":     metrics.get("epochs_trained", "?"),
        })

    runs.sort(key=lambda r: r["mean_rmse"])
    return runs


# ---------------------------------------------------------------------------
# Figure 1: hyperparameter scatter plots
# ---------------------------------------------------------------------------

def plot_param_scatter(runs: list[dict], save_path: Path):
    n_params = len(PARAMS)
    ncols    = 3
    nrows    = (n_params + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    fig.suptitle("Hyperparameter vs Mean RMSE (°C) — all runs", fontsize=13)
    axes = axes.flatten()

    mean_rmses = np.array([r["mean_rmse"] for r in runs])
    vmin, vmax = mean_rmses.min(), mean_rmses.max()
    cmap       = cm.RdYlGn_r  # low RMSE = green, high = red

    for ax, (key, label) in zip(axes, PARAMS.items()):
        vals = [r["config"].get(key) for r in runs]
        # Skip if all None
        if all(v is None for v in vals):
            ax.set_visible(False)
            continue

        xs  = [v for v, r in zip(vals, runs) if v is not None]
        ys  = [r["mean_rmse"] for v, r in zip(vals, runs) if v is not None]
        cs  = [r["mean_rmse"] for v, r in zip(vals, runs) if v is not None]

        sc = ax.scatter(xs, ys, c=cs, cmap=cmap, vmin=vmin, vmax=vmax,
                        s=80, edgecolors="black", linewidths=0.5, zorder=3)

        # Annotate the best point
        best_idx = int(np.argmin(ys))
        ax.annotate(f"  {ys[best_idx]:.4f}", (xs[best_idx], ys[best_idx]),
                    fontsize=7, color="darkgreen")

        # Log scale for LR
        if key == "learning_rate":
            ax.set_xscale("log")

        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Mean RMSE (°C)", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for ax in axes[n_params:]:
        ax.set_visible(False)

    fig.colorbar(sc, ax=axes[:n_params], shrink=0.6, label="Mean RMSE (°C)")
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

    mean_rmses = np.array([r["mean_rmse"] for r in runs])
    norm       = plt.Normalize(mean_rmses.min(), mean_rmses.max())
    cmap       = cm.RdYlGn_r

    # Persistence from the best run (same across all runs)
    pers_run = next((r for r in runs if r["pers_steps"] is not None), None)
    if pers_run:
        ax1.plot(DAYS, pers_run["pers_steps"], "k--", linewidth=2,
                 label="Persistence", zorder=5)
        ax2.axhline(0, color="black", linewidth=1.2, zorder=5)

    # All runs — faint background lines
    for r in runs:
        colour = cmap(norm(r["mean_rmse"]))
        ax1.plot(DAYS, r["rmse_steps"], color=colour, alpha=0.25, linewidth=1)
        if pers_run:
            skill = 1 - r["rmse_steps"] / pers_run["pers_steps"]
            ax2.plot(DAYS, skill, color=colour, alpha=0.25, linewidth=1)

    # Top N — bold and labelled
    colours_top = plt.cm.tab10(np.linspace(0, 0.9, top_n))
    for i, r in enumerate(runs[:top_n]):
        label  = f"#{i+1} {r['name'][-15:]} RMSE={r['mean_rmse']:.4f}"
        colour = colours_top[i]
        ax1.plot(DAYS, r["rmse_steps"], color=colour, linewidth=2.2,
                 marker="o", markersize=4, label=label, zorder=4)
        if pers_run:
            skill = 1 - r["rmse_steps"] / pers_run["pers_steps"]
            ax2.plot(DAYS, skill, color=colour, linewidth=2.2,
                     marker="o", markersize=4, label=label, zorder=4)

    ax1.set_xlabel("Forecast day");  ax1.set_ylabel("RMSE (°C)")
    ax1.set_title("RMSE per step");  ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

    ax2.set_xlabel("Forecast day");  ax2.set_ylabel("Skill vs persistence")
    ax2.set_title("Skill score per step")
    ax2.legend(fontsize=7);          ax2.grid(alpha=0.3)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=[ax1, ax2], shrink=0.7, label="Mean RMSE (°C)")

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

    print(f"Loaded {len(runs)} runs")
    print(f"Best: {runs[0]['name']}  RMSE={runs[0]['mean_rmse']:.4f}  "
          f"params={runs[0]['config']}")

    plot_param_scatter(runs, out_dir / "comparison_params.png")
    plot_curves(runs,        out_dir / "comparison_curves.png", top_n=args.top_n)

    print("\nTop 5 runs:")
    for i, r in enumerate(runs[:5], 1):
        cfg = r["config"]
        print(f"  {i}. RMSE={r['mean_rmse']:.4f}  "
              f"lr={cfg.get('learning_rate'):.2e}  "
              f"heads={cfg.get('n_heads')}  layers={cfg.get('n_layers')}  "
              f"alpha={cfg.get('anomaly_alpha', 0):.2f}  "
              f"dropout={cfg.get('dropout'):.2f}  "
              f"epochs={r['epochs']}")


if __name__ == "__main__":
    main()
