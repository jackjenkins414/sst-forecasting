"""
Final analysis pipeline — run after all HPO searches are complete.

Executes in order:
  1. Alpha ablations      (tubelet, lstm, informer, convlstm)
  2. Retrain best configs (all 4 models, ablation-informed alpha)
  3. Per-model reports    (report_model.py --all)
  4. Final comparison     (compare_runs.py --final)

Usage
-----
    python scripts/run_final_pipeline.py
    python scripts/run_final_pipeline.py --skip_ablation   # if ablations already done
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH     = PROJECT_ROOT / "experiments" / "final_pipeline_log.txt"


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def run(cmd: list[str], label: str) -> bool:
    log(f"START  {label}")
    log(f"       {' '.join(cmd)}")
    t0     = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    if result.returncode != 0:
        log(f"FAILED {label} (exit {result.returncode}, {elapsed/60:.1f} min)")
        return False
    log(f"DONE   {label} ({elapsed/60:.1f} min)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_ablation", action="store_true",
                        help="Skip alpha ablations (if already completed)")
    args = parser.parse_args()

    py = sys.executable
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log("=== Final pipeline started ===")

    # ------------------------------------------------------------------
    # 1. Alpha ablations
    # ------------------------------------------------------------------
    if not args.skip_ablation:
        for model in ["tubelet", "lstm", "informer", "convlstm"]:
            ok = run(
                [py, "scripts/run_alpha_ablation.py", "--model", model],
                f"alpha ablation — {model}",
            )
            if not ok:
                log(f"Ablation failed for {model} — continuing with remaining models")
    else:
        log("SKIP   alpha ablations (--skip_ablation)")

    # ------------------------------------------------------------------
    # 2. Retrain best configs (picks ablation-best alpha automatically)
    # ------------------------------------------------------------------
    ok = run(
        [py, "scripts/retrain_best.py"],
        "retrain best configs (all models)",
    )
    if not ok:
        log("retrain_best.py failed — aborting pipeline")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Per-model report PNGs
    # ------------------------------------------------------------------
    ok = run(
        [py, "scripts/report_model.py", "--all"],
        "per-model reports",
    )
    if not ok:
        log("report_model.py failed — continuing to final comparison")

    # ------------------------------------------------------------------
    # 4. Final cross-model comparison
    # ------------------------------------------------------------------
    run(
        [py, "scripts/compare_runs.py", "--final"],
        "final comparison plot",
    )

    log("=== Final pipeline complete ===")
    log(f"    Outputs in experiments/best_*/  and  experiments/")


if __name__ == "__main__":
    main()
