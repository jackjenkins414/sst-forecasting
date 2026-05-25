"""
Queue: ConvLSTM HPO (17 remaining trials) → final analysis pipeline.

Run once and leave it — each step starts only when the previous one finishes.

    python scripts/run_convlstm_then_final.py
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH     = PROJECT_ROOT / "experiments" / "queue_log_2.txt"


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def run(cmd: list[str], label: str) -> bool:
    log(f"START  {label}")
    t0     = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    if result.returncode != 0:
        log(f"FAILED {label} (exit {result.returncode}, {elapsed/60:.1f} min)")
        return False
    log(f"DONE   {label} ({elapsed/60:.1f} min)")
    return True


def main():
    py = sys.executable
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log("=== Queue started: ConvLSTM HPO -> final pipeline ===")

    ok = run(
        [py, "-u", "scripts/run_optuna_convlstm.py", "--n_trials", "17"],
        "ConvLSTM HPO (17 trials)",
    )
    if not ok:
        log("ConvLSTM HPO failed — aborting queue")
        sys.exit(1)

    ok = run(
        [py, "-u", "scripts/run_final_pipeline.py"],
        "Final analysis pipeline",
    )
    if not ok:
        log("Final pipeline failed")
        sys.exit(1)

    log("=== Queue complete ===")
    log("  Artifacts: experiments/best_*/")
    log("  Reports:   experiments/report_*.png")
    log("  Comparison: experiments/final_comparison.png")


if __name__ == "__main__":
    main()
