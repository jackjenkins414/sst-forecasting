"""
Aggregate multi-seed retraining + AR rollout results into mean +/- std summaries
suitable for the paper's main results table and the long-horizon figure.

Reads:
  experiments/best_<model>_seed*/summary.json   (per-seed 7-day RMSE/skill)
  experiments/ar_rollout/seed*/rmse_per_day.json
  experiments/ar_rollout/seed*/skill_vs_climatology.json
  experiments/ar_rollout/seed*/useful_horizon.json

Writes:
  experiments/aggregate/<model>_summary_stats.json   per-model 7-day mean +/- std
  experiments/aggregate/ar_rollout_stats.json        per-model long-horizon stats
  experiments/aggregate/ar_long_horizon_mean.png     RMSE + skill with shaded bands
  experiments/aggregate/results_table.md             Markdown table for the paper

Usage
-----
    python scripts/aggregate_seed_results.py
    python scripts/aggregate_seed_results.py --models tubelet patch_transformer
    python scripts/aggregate_seed_results.py --include-canonical
        # also pull experiments/best_<model>/ (seed-42 canonical) into the stats

A model is silently skipped if it has fewer than --min-seeds seeded runs
(default 2 — std is undefined for n=1).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

BEST_DIR     = PROJECT_ROOT / "experiments"
AR_DIR       = PROJECT_ROOT / "experiments/ar_rollout"
OUT_DIR      = PROJECT_ROOT / "experiments/aggregate"

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

SEED_DIR_RE = re.compile(r"best_(?P<model>.+)_seed(?P<seed>\d+)$")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_seeded_summaries(model: str, include_canonical: bool) -> list[tuple[int, Path]]:
    """Return [(seed, summary.json path)] for a given model."""
    hits: list[tuple[int, Path]] = []
    for d in sorted(BEST_DIR.glob(f"best_{model}_seed*")):
        m = SEED_DIR_RE.match(d.name)
        if m and m.group("model") == model and (d / "summary.json").exists():
            hits.append((int(m.group("seed")), d / "summary.json"))
    if include_canonical:
        canonical = BEST_DIR / f"best_{model}" / "summary.json"
        if canonical.exists():
            hits.append((42, canonical))
    return hits


def find_seeded_ar(model: str) -> list[tuple[int, Path]]:
    """Return [(seed, seed_dir)] for AR rollout outputs that include this model.

    Accepts both the new per-model layout (rmse_per_day_<model>.json) and the
    legacy combined layout (rmse_per_day.json with a <model> key).
    """
    hits: list[tuple[int, Path]] = []
    for d in sorted(AR_DIR.glob("seed*")):
        m = re.match(r"seed(\d+)$", d.name)
        if not m:
            continue
        # Per-model file present -> definite hit.
        if (d / f"rmse_per_day_{model}.json").exists():
            hits.append((int(m.group(1)), d))
            continue
        # Legacy combined file with the model key inside.
        combined = d / "rmse_per_day.json"
        if combined.exists() and model in json.load(open(combined)):
            hits.append((int(m.group(1)), d))
    return hits


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def stack_per_day(arrays: list[list[float]]) -> np.ndarray:
    return np.stack([np.asarray(a, dtype=float) for a in arrays], axis=0)


def useful_horizon_from_skill(skill: list[float]) -> int:
    """Last day before skill first drops to <= 0 (1-indexed).

    This is the first-crossing definition. Wobbles back into positive skill
    after the initial crossing are ignored, which keeps the summary number
    consistent across noisy seeds.
    """
    useful_day = 0
    for h, v in enumerate(skill):
        if v > 0:
            useful_day = h + 1
        else:
            break
    return useful_day


def aggregate_short_horizon(model: str, summaries: list[tuple[int, Path]]) -> dict:
    rmse_steps  = []
    skill_steps = []
    mean_rmse   = []
    mean_skill  = []
    epochs      = []
    n_params    = None
    for _, p in summaries:
        s = json.load(open(p))
        rmse_steps.append(s["rmse_steps"])
        skill_steps.append(s["skill_steps"])
        mean_rmse.append(s["mean_rmse"])
        mean_skill.append(s["mean_skill"])
        epochs.append(s.get("epochs_trained"))
        n_params = s.get("n_params", n_params)
    rmse_arr  = stack_per_day(rmse_steps)
    skill_arr = stack_per_day(skill_steps)
    return {
        "model": model,
        "n_seeds": len(summaries),
        "seeds": [s for s, _ in summaries],
        "n_params": n_params,
        "epochs_trained": epochs,
        "rmse_per_day_mean": rmse_arr.mean(axis=0).tolist(),
        "rmse_per_day_std":  rmse_arr.std(axis=0, ddof=1).tolist()
                              if len(summaries) > 1 else [0.0] * rmse_arr.shape[1],
        "skill_per_day_mean": skill_arr.mean(axis=0).tolist(),
        "skill_per_day_std":  skill_arr.std(axis=0, ddof=1).tolist()
                               if len(summaries) > 1 else [0.0] * skill_arr.shape[1],
        "mean_rmse_mean": float(np.mean(mean_rmse)),
        "mean_rmse_std":  float(np.std(mean_rmse, ddof=1)) if len(mean_rmse) > 1 else 0.0,
        "mean_skill_mean": float(np.mean(mean_skill)),
        "mean_skill_std":  float(np.std(mean_skill, ddof=1)) if len(mean_skill) > 1 else 0.0,
    }


def aggregate_ar(model: str, ar_dirs: list[tuple[int, Path]]) -> dict:
    rmse_list  = []
    skill_list = []
    useful     = []
    pers_ref   = None
    clim_ref   = None
    for _, d in ar_dirs:
        # Prefer per-model files; fall back to the legacy combined layout.
        per_model = d / f"rmse_per_day_{model}.json"
        if per_model.exists():
            rmse  = json.load(open(per_model))
            skill = json.load(open(d / f"skill_vs_climatology_{model}.json"))
        else:
            rmse  = json.load(open(d / "rmse_per_day.json"))
            skill = json.load(open(d / "skill_vs_climatology.json"))
        rmse_list.append(rmse[model])
        skill_list.append(skill[model])
        # Derive useful horizon from the saved skill curve so the
        # first-crossing definition is applied consistently, regardless of
        # when the underlying rollout was run.
        useful.append(useful_horizon_from_skill(skill[model]))
        # Baselines are deterministic but we average defensively
        if pers_ref is None:
            pers_ref = rmse["persistence"]
            clim_ref = rmse["climatology"]
    # Ensure consistent horizon across seeds (truncate to the minimum if needed)
    min_h = min(len(r) for r in rmse_list)
    if any(len(r) != min_h for r in rmse_list):
        print(f"WARNING: {model} rollouts have inconsistent horizons; truncating to {min_h}", file=sys.stderr)
    rmse_list = [r[:min_h] for r in rmse_list]
    skill_list = [r[:min_h] for r in skill_list]
    if pers_ref is not None:
        pers_ref = pers_ref[:min_h]
        clim_ref = clim_ref[:min_h]

    rmse_arr  = stack_per_day(rmse_list)
    skill_arr = stack_per_day(skill_list)
    n = len(ar_dirs)
    return {
        "model": model,
        "n_seeds": n,
        "seeds": [s for s, _ in ar_dirs],
        "max_horizon": rmse_arr.shape[1],
        "rmse_per_day_mean": rmse_arr.mean(axis=0).tolist(),
        "rmse_per_day_std":  rmse_arr.std(axis=0, ddof=1).tolist()
                              if n > 1 else [0.0] * rmse_arr.shape[1],
        "skill_per_day_mean": skill_arr.mean(axis=0).tolist(),
        "skill_per_day_std":  skill_arr.std(axis=0, ddof=1).tolist()
                               if n > 1 else [0.0] * skill_arr.shape[1],
        "useful_horizon_mean": float(np.mean(useful)),
        "useful_horizon_std":  float(np.std(useful, ddof=1)) if n > 1 else 0.0,
        "useful_horizon_per_seed": useful,
        "persistence_rmse": pers_ref,
        "climatology_rmse": clim_ref,
    }


# ---------------------------------------------------------------------------
# Plot + table
# ---------------------------------------------------------------------------

def plot_ar_mean(ar_stats: dict, out_path: Path):
    """Mean curves with shaded +/- std bands, one panel for RMSE, one for skill."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Baselines from any of the models (they're identical across seeds)
    any_model = next(iter(ar_stats.values()))
    days = list(range(1, any_model["max_horizon"] + 1))
    axL.plot(days, any_model["climatology_rmse"], "--", color="grey",
             lw=1.6, label="Climatology")
    axL.plot(days, any_model["persistence_rmse"], ":",  color="black",
             lw=1.5, label="Persistence")

    for mt, st in ar_stats.items():
        col  = COLOURS.get(mt, "#999")
        name = DISPLAY.get(mt, mt)
        mean = np.asarray(st["rmse_per_day_mean"])
        std  = np.asarray(st["rmse_per_day_std"])
        axL.plot(days, mean, color=col, lw=2.0, marker="o", markersize=4,
                 label=f"{name} (n={st['n_seeds']})")
        axL.fill_between(days, mean - std, mean + std, color=col, alpha=0.18,
                         linewidth=0)

        sk_mean = np.asarray(st["skill_per_day_mean"])
        sk_std  = np.asarray(st["skill_per_day_std"])
        axR.plot(days, sk_mean, color=col, lw=2.0, marker="o", markersize=4,
                 label=f"{name} (n={st['n_seeds']})")
        axR.fill_between(days, sk_mean - sk_std, sk_mean + sk_std, color=col,
                         alpha=0.18, linewidth=0)

    axL.axvline(7, color="grey", linestyle=":", alpha=0.6, lw=1.0)
    axL.set_xlabel("Forecast day"); axL.set_ylabel(r"RMSE ($^\circ$C)")
    axL.set_title("Mean RMSE vs lead time (shaded: +/- 1 std)")
    axL.legend(fontsize=9, loc="lower right"); axL.grid(alpha=0.3)

    axR.axhline(0, color="black", lw=1.0, linestyle="--")
    axR.axvline(7, color="grey", linestyle=":", alpha=0.6, lw=1.0)
    axR.set_xlabel("Forecast day"); axR.set_ylabel("Skill vs climatology")
    axR.set_title("Skill score (mean across seeds)")
    axR.legend(fontsize=9, loc="upper right"); axR.grid(alpha=0.3)

    fig.suptitle("Autoregressive rollout — averaged over multi-seed retraining",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_results_table(short_stats: dict, ar_stats: dict, out_path: Path):
    lines = []
    lines.append("# Multi-seed results (mean +/- std)\n")
    lines.append("## 7-day forecast (training horizon)\n")
    lines.append("| Model | n seeds | mean RMSE (degC) | mean skill | params |")
    lines.append("|-------|---------|------------------|-----------|--------|")
    for mt in sorted(short_stats):
        s = short_stats[mt]
        params = f"{s['n_params']:,}" if s["n_params"] is not None else "?"
        lines.append(
            f"| {DISPLAY.get(mt, mt)} | {s['n_seeds']} | "
            f"{s['mean_rmse_mean']:.4f} +/- {s['mean_rmse_std']:.4f} | "
            f"{s['mean_skill_mean']:+.4f} +/- {s['mean_skill_std']:.4f} | "
            f"{params} |"
        )

    if ar_stats:
        lines.append("\n## Autoregressive long horizon\n")
        lines.append("| Model | n seeds | useful horizon (days) |")
        lines.append("|-------|---------|----------------------|")
        for mt in sorted(ar_stats):
            s = ar_stats[mt]
            lines.append(
                f"| {DISPLAY.get(mt, mt)} | {s['n_seeds']} | "
                f"{s['useful_horizon_mean']:.1f} +/- {s['useful_horizon_std']:.1f} "
                f"(per seed: {s['useful_horizon_per_seed']}) |"
            )

    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Which models to aggregate.")
    parser.add_argument("--min-seeds", type=int, default=2,
                        help="Skip models with fewer than this many seeded runs.")
    parser.add_argument("--include-canonical", action="store_true",
                        help="Also include experiments/best_<model>/ (seed-42 "
                             "canonical) in the short-horizon aggregation.")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    short_stats: dict[str, dict] = {}
    ar_stats:    dict[str, dict] = {}

    for mt in args.models:
        summaries = find_seeded_summaries(mt, args.include_canonical)
        if len(summaries) >= args.min_seeds:
            short_stats[mt] = aggregate_short_horizon(mt, summaries)
            seeds = [s for s, _ in summaries]
            print(f"  {mt}: short-horizon n={len(summaries)} (seeds {seeds})")
        else:
            print(f"  {mt}: short-horizon SKIP (only {len(summaries)} seeded run(s))")

        ar_dirs = find_seeded_ar(mt)
        if len(ar_dirs) >= args.min_seeds:
            ar_stats[mt] = aggregate_ar(mt, ar_dirs)
            seeds = [s for s, _ in ar_dirs]
            print(f"  {mt}: AR rollout n={len(ar_dirs)} (seeds {seeds})")
        else:
            print(f"  {mt}: AR rollout SKIP (only {len(ar_dirs)} seeded run(s))")

    if short_stats:
        for mt, s in short_stats.items():
            with open(OUT_DIR / f"{mt}_summary_stats.json", "w") as f:
                json.dump(s, f, indent=2)

    if ar_stats:
        with open(OUT_DIR / "ar_rollout_stats.json", "w") as f:
            json.dump(ar_stats, f, indent=2)
        plot_ar_mean(ar_stats, OUT_DIR / "ar_long_horizon_mean.png")

    if short_stats or ar_stats:
        write_results_table(short_stats, ar_stats, OUT_DIR / "results_table.md")
        print(f"\nWritten to {OUT_DIR}/")
    else:
        print("\nNothing to aggregate yet.")


if __name__ == "__main__":
    main()
