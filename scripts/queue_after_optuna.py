"""
Waits for the Optuna study to finish, then:
  1. Runs compare_runs.py and copies PNGs to OneDrive Desktop
  2. Trains Ayush's SpatialFlatTransformer for 25 epochs
  3. Prints a summary when done

Run as background process — output goes to experiments/queue_log.txt
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONEDRIVE_DESKTOP = Path.home() / "OneDrive" / "Desktop"
DB_PATH = PROJECT_ROOT / "experiments" / "optuna_study.db"

optuna.logging.set_verbosity(optuna.logging.WARNING)


def log(msg):
    print(msg, flush=True)


def wait_for_optuna():
    log("Waiting for Optuna study to finish (no RUNNING trials)...")
    while True:
        study = optuna.load_study(
            study_name="tubelet_hpo_v2",
            storage=f"sqlite:///{DB_PATH}",
        )
        running = [t for t in study.trials if t.state.name == "RUNNING"]
        done = [t for t in study.trials if t.state.name == "COMPLETE"]
        log(f"  {len(done)} complete, {len(running)} running ...")
        if not running:
            log(f"Optuna done. {len(done)} completed trials.")
            best = study.best_trial
            log(f"Best RMSE: {best.value:.4f}  params: {best.params}")
            break
        time.sleep(120)  # check every 2 minutes


def run(cmd, desc):
    log(f"\n>>> {desc}")
    log(f"    {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        log(f"WARNING: command exited with code {result.returncode}")
    return result.returncode


def copy_pngs():
    for fname in ["comparison_params.png", "comparison_curves.png"]:
        src = PROJECT_ROOT / "experiments" / fname
        if src.exists():
            dst = ONEDRIVE_DESKTOP / fname
            shutil.copy2(src, dst)
            log(f"Copied {fname} -> {dst}")
        else:
            log(f"WARNING: {fname} not found, skipping copy")


def main():
    wait_for_optuna()

    # 1. Generate comparison plots
    run(
        [sys.executable, "-u", "scripts/compare_runs.py"],
        "Generating comparison plots",
    )
    copy_pngs()

    # 2. Train Ayush's transformer
    run(
        [
            sys.executable, "-u", "scripts/train_e1.py",
            "--model", "transformer",
            "--zarr-path", "data/processed/oisst_coralsea.zarr",
            "--output-dir", "experiments/results/ayush_transformer_e1",
            "--tf-d-model", "128",
            "--tf-nhead", "8",
            "--tf-layers", "4",
            "--tf-ffn-dim", "256",
            "--dropout", "0.1",
            "--lr", "7e-4",
            "--lr-factor", "0.5",
            "--lr-patience", "3",
            "--batch-size", "8",
            "--max-epochs", "25",
            "--patience", "5",
            "--seed", "42",
        ],
        "Training Ayush SpatialFlatTransformer (25 epochs, B=8)",
    )

    log("\n=== All done! ===")
    log("  - Optuna complete, plots saved + copied to OneDrive Desktop")
    log("  - Transformer results in experiments/results/ayush_transformer_e1/")


if __name__ == "__main__":
    main()
