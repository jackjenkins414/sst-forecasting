"""
Sequential retrain + AR-rollout queue across (models x seeds).

Per (model, seed) pair runs:
    retrain_best.py  --models <m> --seed <N>
    run_ar_rollout.py --models <m> --seed <N>

A failure on one pair is logged and the queue continues with the next pair.
Log: experiments/seed_queue_log.txt

Usage
-----
    python scripts/run_seed_queue.py --models tubelet patch_transformer --seeds 1 2 3
    python scripts/run_seed_queue.py --models convlstm --seeds 1 2 3   # cluster (Ayush)
    python scripts/run_seed_queue.py --models tubelet --seeds 1 2 3 --skip-rollout
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH     = PROJECT_ROOT / "experiments" / "seed_queue_log.txt"


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def run(cmd, label: str) -> bool:
    log(f"START  {label}")
    t0 = time.time()
    rc = subprocess.run([str(c) for c in cmd], cwd=str(PROJECT_ROOT)).returncode
    dt = (time.time() - t0) / 60
    log(f"{'DONE ' if rc == 0 else 'FAILED'} {label} ({dt:.1f} min, exit {rc})")
    return rc == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="Models to sweep, in queue order.")
    parser.add_argument("--seeds",  nargs="+", type=int, required=True,
                        help="Seeds to sweep, in queue order.")
    parser.add_argument("--skip-rollout", action="store_true",
                        help="Only retrain; skip the AR rollout step.")
    parser.add_argument("--skip-retrain", action="store_true",
                        help="Only run AR rollout; assume checkpoints already exist.")
    args = parser.parse_args()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    log(f"=== seed queue started: models={args.models} seeds={args.seeds} ===")

    for m in args.models:
        for s in args.seeds:
            label_base = f"{m} seed={s}"
            if not args.skip_retrain:
                ok = run([py, "-u", "scripts/retrain_best.py",
                         "--models", m, "--seed", str(s)],
                        f"retrain {label_base}")
                if not ok and not args.skip_rollout:
                    log(f"skipping rollout for {label_base} (retrain failed)")
                    continue
            if not args.skip_rollout:
                run([py, "-u", "scripts/run_ar_rollout.py",
                     "--models", m, "--seed", str(s)],
                    f"rollout {label_base}")

    log("=== seed queue complete ===")


if __name__ == "__main__":
    main()
