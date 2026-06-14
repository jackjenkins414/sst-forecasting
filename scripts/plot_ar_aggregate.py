"""
Aggregate the phase-averaged AR rollout across seeds into a single figure with
mean lines and shaded seed bands (no re-running of models -- reads existing JSONs).

For each model we stack the per-day RMSE from seeds 1-3, then plot the mean with a
shaded band. Baselines (persistence, climatology) are deterministic across seeds, so
they are drawn as single mean lines.

Band style (--band):
  std    : mean +/- 1 standard deviation across seeds (default).
  minmax : full envelope between the best and worst seed.
  ci95   : 95% t-interval (n=3, t=4.303) -- wide; use with caution.

Usage:
    python scripts/plot_ar_aggregate.py
    python scripts/plot_ar_aggregate.py --band minmax
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AVG_DIR = PROJECT_ROOT / "experiments/ar_rollout_avg"
SEEDS = [1, 2, 3]
MODELS = ["patch_transformer", "tubelet", "convlstm"]
COLOURS = {"patch_transformer": "#edc948", "convlstm": "#59a14f", "tubelet": "#4e79a7"}
DISPLAY = {"patch_transformer": "Patch Transformer", "convlstm": "ConvLSTM",
           "tubelet": "Tubelet Transformer"}
T_CRIT_N3 = 4.303  # t_{0.975, df=2}

def _band(stack: np.ndarray, kind: str):
    """stack: (n_seeds, n_days) -> (centre, lo, hi)."""
    mean = stack.mean(axis=0)
    if kind == "minmax":
        return mean, stack.min(axis=0), stack.max(axis=0)
    sd = stack.std(axis=0, ddof=1)
    if kind == "ci95":
        half = T_CRIT_N3 * sd / np.sqrt(stack.shape[0])
        return mean, mean - half, mean + half
    return mean, mean - sd, mean + sd          # std (default)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", choices=["std", "minmax", "ci95"], default="std")
    args = ap.parse_args()

    # Load per-seed arrays
    rmse = {m: [] for m in MODELS}
    skill = {m: [] for m in MODELS}
    pers, clim = [], []
    for s in SEEDS:
        for m in MODELS:
            d = json.load(open(AVG_DIR / f"seed{s}" / f"rmse_per_day_{m}.json"))
            rmse[m].append(d[m])
            c = np.array(d["climatology"])
            skill[m].append((1 - np.array(d[m]) / c).tolist())
        pers.append(d["persistence"])
        clim.append(d["climatology"])

    rmse = {m: np.array(v) for m, v in rmse.items()}
    skill = {m: np.array(v) for m, v in skill.items()}
    pers = np.array(pers).mean(axis=0)
    clim = np.array(clim).mean(axis=0)
    days = np.arange(1, rmse[MODELS[0]].shape[1] + 1)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left: RMSE ---
    axL.plot(days, clim, "--", color="grey", lw=1.6, label="Climatology")
    axL.plot(days, pers, ":", color="black", lw=1.5, label="Persistence")
    for m in MODELS:
        c, lo, hi = _band(rmse[m], args.band)
        col = COLOURS[m]
        axL.plot(days, c, color=col, lw=2.0, label=DISPLAY[m])
        axL.fill_between(days, lo, hi, color=col, alpha=0.12, linewidth=0)
        axL.plot(days, lo, ls=":", color=col, lw=1.0, alpha=0.9)
        axL.plot(days, hi, ls=":", color=col, lw=1.0, alpha=0.9)
    axL.axvline(7, color="grey", linestyle=":", alpha=0.6, lw=1.0)
    axL.set_xlabel("Forecast day")
    axL.set_ylabel(r"RMSE ($^\circ$C)")
    axL.set_title("RMSE vs lead time (mean $\\pm$ seed band)")
    axL.legend(fontsize=9, loc="lower right")
    axL.grid(alpha=0.3)

    # --- Right: skill vs climatology ---
    axR.axhline(0, color="black", lw=1.0, linestyle="--")
    for m in MODELS:
        c, lo, hi = _band(skill[m], args.band)
        col = COLOURS[m]
        axR.plot(days, c, color=col, lw=2.0, label=DISPLAY[m])
        axR.fill_between(days, lo, hi, color=col, alpha=0.12, linewidth=0)
        axR.plot(days, lo, ls=":", color=col, lw=1.0, alpha=0.9)
        axR.plot(days, hi, ls=":", color=col, lw=1.0, alpha=0.9)
    axR.axvline(7, color="grey", linestyle=":", alpha=0.6, lw=1.0)
    axR.set_xlabel("Forecast day")
    axR.set_ylabel("Skill vs climatology")
    axR.set_title("Skill vs climatology (first zero-crossing = useful horizon)")
    axR.legend(fontsize=9, loc="upper right")
    axR.grid(alpha=0.3)

    band_lbl = {"std": "mean $\\pm$ 1 s.d.", "minmax": "min--max envelope",
                "ci95": "mean, 95% t-CI"}[args.band]
    fig.suptitle(f"Phase-averaged autoregressive rollout, aggregated over 3 seeds "
                 f"({band_lbl})", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = AVG_DIR / f"aggregate_ar_long_horizon_{args.band}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
