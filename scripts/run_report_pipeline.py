"""
Report pipeline (no alpha ablation): retrain each model's best HPO config,
generate its per-model report + prediction plots, then refresh the cross-model
comparison. Fast models run first so plots appear early; ConvLSTM (slow, B=2)
runs last.

For each model: retrain_best.py --models <m>  ->  report_model.py --model <m>
Then: compare_runs.py --final

A failure on one model is logged and skipped, not fatal.
Log: experiments/report_pipeline_log.txt
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH     = PROJECT_ROOT / "experiments" / "report_pipeline_log.txt"

# Fast attention/recurrent models first; ConvLSTM last (slow B=2 ~ overnight).
MODELS = ["tubelet", "lstm", "informer", "transformer", "convlstm"]

def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

def run(cmd: list[str], label: str) -> bool:
    log(f"START  {label}")
    t0 = time.time()
    rc = subprocess.run([str(c) for c in cmd], cwd=str(PROJECT_ROOT)).returncode
    dt = (time.time() - t0) / 60
    if rc != 0:
        log(f"FAILED {label} (exit {rc}, {dt:.1f} min)")
        return False
    log(f"DONE   {label} ({dt:.1f} min)")
    return True

def main():
    py = sys.executable
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log("=== Report pipeline started (retrain -> report per model, no ablation) ===")

    for m in MODELS:
        if not run([py, "-u", "scripts/retrain_best.py", "--models", m],
                   f"retrain {m}"):
            log(f"skipping report for {m} (retrain failed)")
            continue
        run([py, "-u", "scripts/report_model.py", "--model", m], f"report {m}")

    run([py, "-u", "scripts/compare_runs.py", "--final"], "final cross-model comparison")
    log("=== Report pipeline complete ===")
    log("  Per-model artifacts: experiments/best_<model>/ and report PNGs")
    log("  Comparison: experiments/comparison_*.png")

if __name__ == "__main__":
    main()
