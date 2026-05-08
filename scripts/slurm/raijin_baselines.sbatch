#!/bin/bash
#SBATCH --job-name=sstf_baselines
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --requeue
#SBATCH --output=experiments/results/%x-%j/slurm.out
#SBATCH --error=experiments/results/%x-%j/slurm.err
# Optional: pin to a specific 64-GB node by uncommenting one of these
# #SBATCH --nodelist=hpc-02
# #SBATCH --exclude=hpc-03,hpc-06   # reserve 128 GB nodes for DL training
#
# ─────────────────────────────────────────────────────────────────────────────
# E0 baselines on Raijin (Intel Xeon E5-2670, Sandy Bridge, no GPU).
#
# PREREQUISITES (one-time per cluster):
#   1. /home/hpc NFS-exported rw from hpc-01 to 192.168.2.0/24, mounted on
#      hpc-02..07 at /home/hpc.  Until that is set up this job WILL fail with
#      "No such file or directory" on the conda activation step — run E0 on
#      hpc-01 directly via:
#        conda activate sst && python scripts/run_baselines.py …
#   2. Conda env `sst` populated on hpc-01.
#   3. Zarr store at $HOME/sst-forecasting/data/processed/oisst_coralsea.zarr.
#
# Submit with:
#   sbatch scripts/slurm/raijin_baselines.sbatch
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# --- Trap a graceful USR1 signal sent 60 s before timeout --------------------
graceful_exit() {
    echo "[$(date -Is)] caught USR1 — flushing logs and exiting cleanly"
    exit 0
}
trap graceful_exit USR1

# --- Paths -------------------------------------------------------------------
REPO="${SLURM_SUBMIT_DIR:-$HOME/sst-forecasting}"
RUN_ID="${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
OUT_DIR="$REPO/experiments/results/$RUN_ID"
mkdir -p "$OUT_DIR"
ln -sfn "$RUN_ID" "$REPO/experiments/results/latest_baselines"

cd "$REPO"

# --- Activate Intel oneAPI for MKL/MPI runtime libs -------------------------
# (does NOT replace torch's bundled MKL — just makes MPI/ICX libs visible if
#  the user later runs Intel-MPI jobs from this env)
if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
    # setvars.sh calls `exit` internally — source it in a subshell and capture
    # the resulting env so the exit does NOT kill this script under set -e.
    oneapi_env=$(bash -c 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 && env' 2>/dev/null) || true
    if [[ -n "$oneapi_env" ]]; then
        while IFS='=' read -r key value; do
            [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && export "$key=$value"
        done <<< "$oneapi_env"
    fi
fi

# --- Activate conda env ------------------------------------------------------
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sst

# --- Threading / NUMA pinning ------------------------------------------------
# 16 physical cores total = 8 per socket × 2 sockets.  --hint=nomultithread
# already removes the HT siblings from the cgroup, so we let OpenMP bind one
# thread per physical core spread across BOTH NUMA nodes (close + cores).
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export MKL_DYNAMIC=FALSE
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
unset KMP_AFFINITY                          # avoid Intel OpenMP overriding OMP_PLACES
export MKL_ENABLE_INSTRUCTIONS=AVX        # Sandy Bridge has no AVX2/FMA
export PYTHONHASHSEED=42

# --- Provenance dump ---------------------------------------------------------
{
    echo "=== provenance dump $(date -Is) ==="
    echo "host:       $(hostname)"
    echo "slurm:      job=$SLURM_JOB_ID node=$SLURMD_NODENAME"
    echo "user:       $(whoami)"
    echo "cwd:        $(pwd)"
    echo "git sha:    $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
    echo "git status:"; git status --short || true
    echo "python:     $(which python) ($(python --version 2>&1))"
    echo "torch:      $(python -c 'import torch; print(torch.__version__)' 2>&1)"
    echo "numpy:      $(python -c 'import numpy; print(numpy.__version__)' 2>&1)"
    echo "OMP/MKL:    OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS KMP_AFFINITY=${KMP_AFFINITY:-}"
    echo "==="
} | tee "$OUT_DIR/env.log"

python -m pip freeze > "$OUT_DIR/pip-freeze.txt" 2>/dev/null || true

# --- Run baselines, both sockets, interleaved memory ------------------------
# Use BOTH NUMA nodes (16 phys cores total).  --interleave=all spreads pages
# across nodes so a thread on socket 1 doesn't pay 2× latency for memory
# allocated on socket 0.  CPU placement is handled by OMP_PLACES=cores above.
srun --kill-on-bad-exit=1 \
    numactl --interleave=all \
    python scripts/run_baselines.py \
        --zarr-path  data/processed/oisst_coralsea.zarr \
        --output-dir "$OUT_DIR" \
        --horizons  1 7 30 \
        --ar-context 30 \
        --bootstrap 1000 \
        --seed 42

echo "[$(date -Is)] baseline job complete → $OUT_DIR"
