"""
Presentation extras queue:
  1. Retrain + report Patch Transformer.
  2. alpha=0 (plain-MSE) point for the anomaly-tuned models, for the fair
     "no anomaly vs tuned alpha*" comparison. Flat Transformer already trained at
     alpha=0, so it's excluded. ConvLSTM (slow B=2) runs last.
  3. Refresh the cross-model comparison (compare_runs --final) so it includes Patch.

Failures are logged and skipped, not fatal.
Log: experiments/preso_extras_log.txt
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH     = PROJECT_ROOT / "experiments" / "preso_extras_log.txt"

# alpha=0 only for models that *tuned* alpha; convlstm last (slowest, B=2).
ALPHA0_MODELS = ["tubelet", "lstm", "informer", "convlstm"]

def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

def run(cmd, label):
    log(f"START  {label}")
    t0 = time.time()
    rc = subprocess.run([str(c) for c in cmd], cwd=str(PROJECT_ROOT)).returncode
    dt = (time.time() - t0) / 60
    log(f"{'DONE ' if rc == 0 else 'FAILED'} {label} ({dt:.1f} min, exit {rc})")
    return rc == 0

def main():
    py = sys.executable
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log("=== Presentation extras started ===")

    # 1. Patch Transformer retrain + report
    if run([py, "-u", "scripts/retrain_best.py", "--models", "patch_transformer"],
           "retrain patch_transformer"):
        run([py, "-u", "scripts/report_model.py", "--model", "patch_transformer"],
            "report patch_transformer")

    # 2. alpha=0 fair-comparison point (fast models first, convlstm last)
    for m in ALPHA0_MODELS:
        run([py, "-u", "scripts/run_alpha_ablation.py", "--model", m, "--alphas", "0"],
            f"alpha=0 ablation {m}")

    # 3. Refresh cross-model comparison (now includes patch)
    run([py, "-u", "scripts/compare_runs.py", "--final"], "final comparison (with patch)")

    log("=== Presentation extras complete ===")

if __name__ == "__main__":
    main()
