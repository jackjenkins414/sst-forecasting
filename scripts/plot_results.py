#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS  = Path(__file__).parent.parent / "experiments" / "results"
FIGURES  = Path(__file__).parent.parent / "report" / "figures"

LSTM_DIR      = RESULTS / "sstf_e1_lstm-117"
CONVLSTM_DIR  = RESULTS / "sstf_e2_convlstm_a100-168099976.gadi-pbs"
BASELINES_JSON = RESULTS / "e0_local" / "baselines.json"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_training_curves(lstm, conv, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, data, title in zip(axes, [lstm, conv], ["LSTM", "ConvLSTM"]):
        epochs  = [e["epoch"]       for e in data["history"]]
        train_c = [e["train_rmse_c"] for e in data["history"]]
        val_c   = [e["val_rmse_c"]   for e in data["history"]]
        ax.plot(epochs, train_c, label="train")
        ax.plot(epochs, val_c,   label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RMSE (°C)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)
    print("Saved training_curves.png")


def plot_rmse_per_step(lstm, conv, baseline_h7, out):
    steps = list(range(1, 8))
    persist_rmse = baseline_h7["persistence"]["rmse_C"]
    ar_rmse      = baseline_h7["linear_ar"]["rmse_C"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, lstm["test_rmse_per_step_c"], "o-", label=f"LSTM (mean {lstm['test_rmse_mean_c']:.3f}°C)")
    ax.plot(steps, conv["test_rmse_per_step_c"], "s-", label=f"ConvLSTM (mean {conv['test_rmse_mean_c']:.3f}°C)")
    ax.axhline(persist_rmse, ls="--", color="gray",  label=f"Persistence ({persist_rmse:.3f}°C)")
    ax.axhline(ar_rmse,      ls=":",  color="brown", label=f"Linear AR ({ar_rmse:.3f}°C)")
    ax.set_xlabel("Forecast step (days ahead)")
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("Per-step test RMSE at h=7 (Coral Sea, 1999–2000)")
    ax.set_xticks(steps)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "rmse_per_step_h7.png", dpi=150)
    plt.close(fig)
    print("Saved rmse_per_step_h7.png")


def plot_model_comparison(lstm, conv, baseline_h7, out):
    persist = baseline_h7["persistence"]["rmse_C"]
    ar      = baseline_h7["linear_ar"]["rmse_C"]

    models = ["Persistence", "Linear AR", "LSTM", "ConvLSTM"]
    rmses  = [persist, ar, lstm["test_rmse_mean_c"], conv["test_rmse_mean_c"]]
    skills = [0.0,
              1 - ar / persist,
              1 - lstm["test_rmse_mean_c"] / persist,
              1 - conv["test_rmse_mean_c"] / persist]
    colors = ["#aaaaaa", "#888888", "#4c72b0", "#dd8452"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    bars = ax1.bar(models, rmses, color=colors)
    ax1.set_ylabel("Mean test RMSE (°C)")
    ax1.set_title("Model comparison — mean test RMSE at h=7")
    for bar, v in zip(bars, rmses):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    bars2 = ax2.bar(models, skills, color=colors)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Skill score vs persistence")
    ax2.set_title("Skill score (SS = 1 − RMSE_model / RMSE_persist)")
    for bar, v in zip(bars2, skills):
        offset = 0.005 if v >= 0 else -0.02
        ax2.text(bar.get_x() + bar.get_width() / 2, v + offset,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "model_comparison_h7.png", dpi=150)
    plt.close(fig)
    print("Saved model_comparison_h7.png")


def main():
    p = argparse.ArgumentParser(description="Generate result figures from saved metrics.")
    p.add_argument("--lstm-dir",     default=str(LSTM_DIR))
    p.add_argument("--convlstm-dir", default=str(CONVLSTM_DIR))
    p.add_argument("--baselines",    default=str(BASELINES_JSON))
    p.add_argument("--out-dir",      default=str(FIGURES))
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    lstm      = load_json(Path(args.lstm_dir)     / "metrics.json")
    conv      = load_json(Path(args.convlstm_dir) / "metrics.json")
    baselines = load_json(args.baselines)
    baseline_h7 = next(r for r in baselines["results"] if r["horizon"] == 7)

    plot_training_curves(lstm, conv, out)
    plot_rmse_per_step(lstm, conv, baseline_h7, out)
    plot_model_comparison(lstm, conv, baseline_h7, out)


if __name__ == "__main__":
    main()
