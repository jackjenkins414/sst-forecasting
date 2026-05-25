"""
Smart pipeline queue:
  1. Wait for current Tubelet Optuna trial to finish
  2. Kill the Tubelet Optuna process (between trials, no work lost)
  3. Run Transformer Optuna (12 trials, patience=5, max_epochs=15)
  4. Generate comparison plots + copy to OneDrive Desktop
  5. Resume Tubelet Optuna for 15 more trials
  6. Final comparison plots + copy to OneDrive Desktop

Log: experiments/pipeline_log.txt
"""

import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONEDRIVE_DESKTOP = Path.home() / "OneDrive" / "Desktop"
CURRENT_TRIAL_DIR = PROJECT_ROOT / "experiments/results/run_20260511_003356"

sys.path.insert(0, str(PROJECT_ROOT))
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def log(msg):
    print(msg, flush=True)


def run(cmd, desc):
    log(f"\n>>> {desc}")
    result = subprocess.run([str(c) for c in cmd], cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        log(f"WARNING: exited with code {result.returncode}")
    return result.returncode


def copy_pngs(label=""):
    for fname in ["comparison_params.png", "comparison_curves.png"]:
        src = PROJECT_ROOT / "experiments" / fname
        if src.exists():
            stem = Path(fname).stem
            dst_name = f"{stem}{label}.png"
            dst = ONEDRIVE_DESKTOP / dst_name
            shutil.copy2(src, dst)
            log(f"Copied -> {dst}")


def kill_optuna_process():
    """Kill the process running run_optuna_search.py via wmic."""
    try:
        result = subprocess.run(
            ["wmic", "process", "where",
             "commandline like '%run_optuna_search%'",
             "get", "processid", "/format:list"],
            capture_output=True, text=True,
        )
        pids = [
            line.split("=")[1].strip()
            for line in result.stdout.splitlines()
            if line.startswith("ProcessId=") and line.split("=")[1].strip()
        ]
        for pid in pids:
            log(f"Killing Tubelet Optuna process PID {pid}")
            subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
        if not pids:
            log("No Tubelet Optuna process found (may have already finished)")
    except Exception as e:
        log(f"WARNING: could not kill Optuna process: {e}")


def wait_for_current_trial():
    metrics = CURRENT_TRIAL_DIR / "metrics.json"
    log(f"Waiting for current trial to finish: {CURRENT_TRIAL_DIR.name}")
    while not metrics.exists():
        time.sleep(30)
    log("Current trial done.")
    time.sleep(60)  # let Optuna write to DB before we kill it


def main():
    # Step 1: Wait for trial 11 to finish, then kill Optuna
    wait_for_current_trial()
    kill_optuna_process()
    time.sleep(5)

    # Step 2: Transformer Optuna (12 trials)
    run(
        [sys.executable, "-u", "scripts/run_optuna_transformer.py", "--n_trials", "12"],
        "Transformer Optuna (12 trials, patience=5, max_epochs=15)",
    )

    # Step 3: Mid-pipeline plots
    run([sys.executable, "-u", "scripts/compare_runs.py"], "Generating comparison plots (mid)")
    copy_pngs(label="_mid")

    # Step 4: Resume Tubelet Optuna for 15 more trials
    run(
        [sys.executable, "-u", "scripts/run_optuna_search.py", "--n_trials", "15"],
        "Resuming Tubelet Optuna (15 more trials)",
    )

    # Step 5: Final plots
    run([sys.executable, "-u", "scripts/compare_runs.py"], "Generating final comparison plots")
    copy_pngs(label="_final")

    log("\n=== Pipeline complete! ===")
    log("  Transformer HPO  -> experiments/optuna_transformer.db")
    log("  Tubelet HPO      -> experiments/optuna_study.db")
    log("  Final plots      -> OneDrive Desktop (*_final.png)")


if __name__ == "__main__":
    main()
