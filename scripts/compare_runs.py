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
    "informer":    "#76b7b2",  # teal
    "patch_transformer": "#edc948",  # yellow
    "unknown":     "#9c9c9c",  # grey
}


def _infer_model_type(config: dict) -> str:
    """Identify model for runs predating the model_type key (old Tubelet/Transformer)."""
    if config.get("model_type"):
        return config["model_type"]
    if "t_s" in config and "p_h" in config:
        return "tubelet"
    if "patch_height" in config and "patch_width" in config:
        return "patch_transformer"
    if "factor" in config and "label_len" in config:
        return "informer"
    if "hidden_dim" in config and "kernel_size" in config:
        return "convlstm"
    if "hidden_size" in config and "d_spatial" in config:
        return "lstm"
    if "ffn_dim" in config and "n_heads" in config:
        return "transformer"
    return "unknown"

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
        model_type = _infer_model_type(config)

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

# ---------------------------------------------------------------------------
# Final report — one clean line per model, from retrain_best.py artifacts
# ---------------------------------------------------------------------------

def plot_final_comparison(out_dir: Path):
    """
    Reads experiments/best_<model>/summary.json for each available model and
    produces a single report-quality figure:
      Left panel  — RMSE per forecast day, one line per architecture
      Right panel — Skill per forecast day, one line per architecture
      Bottom      — metrics summary table (RMSE, skill, BIC, params)
    """
    summaries = {}
    for mt in MODEL_COLOURS:
        p = PROJECT_ROOT / "experiments" / f"best_{mt}" / "summary.json"
        if p.exists():
            summaries[mt] = json.load(open(p))

    if not summaries:
        print("No best_<model>/ artifacts found — run retrain_best.py first.")
        return

    fig = plt.figure(figsize=(16, 10))
    import matplotlib.gridspec as gridspec
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            height_ratios=[2, 1],
                            hspace=0.45, wspace=0.30,
                            left=0.07, right=0.97, top=0.93, bottom=0.05)

    ax_rmse  = fig.add_subplot(gs[0, 0])
    ax_skill = fig.add_subplot(gs[0, 1])
    ax_tbl   = fig.add_subplot(gs[1, :])

    fig.suptitle("Final Model Comparison — best HPO config per architecture",
                 fontsize=13, fontweight="bold")

    # Reference persistence from the first summary that has it
    pers_plotted = False
    table_rows   = []

    for mt, s in sorted(summaries.items(), key=lambda x: x[1]["mean_rmse"]):
        colour      = MODEL_COLOURS.get(mt, "#9c9c9c")
        rmse_steps  = np.array(s["rmse_steps"])
        pers_steps  = np.array(s["pers_rmse_steps"])
        skill_steps = np.array(s["skill_steps"])

        if not pers_plotted:
            ax_rmse.plot(DAYS, pers_steps, "k--", linewidth=1.6,
                         label="Persistence", zorder=5)
            pers_plotted = True

        ax_rmse.plot(DAYS, rmse_steps,  color=colour, linewidth=2.2,
                     marker="o", markersize=5, label=mt)
        ax_skill.plot(DAYS, skill_steps, color=colour, linewidth=2.2,
                      marker="o", markersize=5, label=mt)

        table_rows.append([
            mt,
            f"{s['mean_rmse']:.4f}",
            f"{s['mean_skill']:.4f}",
            f"{s['rmse_steps'][0]:.4f}",
            f"{s['rmse_steps'][-1]:.4f}",
            f"{s['bic']:.0f}",
            f"{s['n_params']:,}",
        ])

    ax_rmse.set_xlabel("Forecast day"); ax_rmse.set_ylabel("RMSE (°C)")
    ax_rmse.set_title("RMSE per forecast day")
    ax_rmse.legend(fontsize=9); ax_rmse.grid(alpha=0.3)

    ax_skill.axhline(0, color="black", linewidth=1.0, linestyle="--")
    ax_skill.set_xlabel("Forecast day"); ax_skill.set_ylabel("Skill vs persistence")
    ax_skill.set_title("Skill score per forecast day")
    ax_skill.legend(fontsize=9); ax_skill.grid(alpha=0.3)

    # Summary table
    ax_tbl.axis("off")
    col_labels = ["Model", "Mean RMSE", "Mean Skill",
                  "Day-1 RMSE", "Day-7 RMSE", "BIC", "Params"]
    tbl = ax_tbl.table(
        cellText=table_rows, colLabels=col_labels,
        cellLoc="center", loc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    for (r, _), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#d0d8e8")
            cell.set_text_props(fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f5f5f5")
    ax_tbl.set_title("Metrics summary", fontsize=10, pad=12)

    out = out_dir / "final_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str,
                        default=str(PROJECT_ROOT / "experiments"))
    parser.add_argument("--top_n",  type=int, default=5,
                        help="How many top runs to highlight in curve plots")
    parser.add_argument("--final",  action="store_true",
                        help="Generate report-quality final comparison from "
                             "retrain_best.py artifacts (experiments/best_*/)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.final:
        plot_final_comparison(out_dir)
        return

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
