#!/bin/bash
#SBATCH --job-name=sstf_e1
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodelist=hpc-03
# hpc-03 has 128 GB — use for initial E1 runs until memory footprint is confirmed
# Once footprint is confirmed < 32 GB, switch to any 64-GB node:
# #SBATCH --nodelist=hpc-02,hpc-04,hpc-05,hpc-07
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --output=experiments/results/%x-%j/slurm.out
#SBATCH --error=experiments/results/%x-%j/slurm.err
#
# ─────────────────────────────────────────────────────────────────────────────
# E1 MVE: train LSTM or Transformer at h=7, L=90 on Coral Sea OISST.
#
# Usage
# -----
# Train LSTM (default):
#   sbatch scripts/slurm/raijin_e1.sbatch
#
# Train Transformer:
#   sbatch --job-name=sstf_e1_transformer \
#          --export=ALL,MODEL=transformer \
#          scripts/slurm/raijin_e1.sbatch
#
# PREREQUISITES:
#   1. /home/hpc NFS-mounted on worker nodes (hpc-02..07) — done ✓
#   2. conda env `sst` with all requirements — on hpc-01, shared via NFS ✓
#   3. Zarr store at ~/sst-forecasting/data/processed/oisst_coralsea.zarr ✓
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Graceful USR1 trap (sent 120 s before walltime) ──────────────────────────
graceful_exit() {
    echo "[$(date -Is)] caught USR1 — saving checkpoint and exiting cleanly"
    # The training script saves 'last_model.pt' at every epoch, so partial
    # progress is always preserved.  No additional action needed.
    exit 0
}
trap graceful_exit USR1

# ── Resolve model from env or default ────────────────────────────────────────
MODEL="${MODEL:-lstm}"          # override with --export=ALL,MODEL=transformer
HORIZON="${HORIZON:-7}"
CONTEXT_LEN="${CONTEXT_LEN:-90}"

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO="${SLURM_SUBMIT_DIR:-$HOME/sst-forecasting}"
RUN_ID="${SLURM_JOB_NAME:-sstf_e1_${MODEL}}-${SLURM_JOB_ID:-local}"
OUT_DIR="$REPO/experiments/results/$RUN_ID"
ZARR="$REPO/data/processed/oisst_coralsea.zarr"
mkdir -p "$OUT_DIR"
ln -sfn "$RUN_ID" "$REPO/experiments/results/latest_e1_${MODEL}"

cd "$REPO"

# ── Activate Intel oneAPI (provides MKL runtime, does not replace torch MKL) ─
if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
    oneapi_env=$(bash -c 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 && env' 2>/dev/null) || true
    if [[ -n "$oneapi_env" ]]; then
        while IFS='=' read -r key value; do
            [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && export "$key=$value"
        done <<< "$oneapi_env"
    fi
fi

# ── Activate conda env ────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sst

# ── Threading / NUMA pinning ─────────────────────────────────────────────────
# 16 physical cores (8 per socket × 2 sockets).
# --hint=nomultithread removes HT siblings from the cgroup.
# OMP_PLACES=cores + OMP_PROC_BIND=close = one thread per physical core.
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export MKL_DYNAMIC=FALSE
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
unset KMP_AFFINITY                       # let portable OMP_PLACES take priority
export MKL_ENABLE_INSTRUCTIONS=AVX      # Sandy Bridge: no AVX2/FMA
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1              # flush stdout immediately (needed when piped to tee)

echo "[raijin_e1] MODEL=$MODEL  HORIZON=$HORIZON  CONTEXT_LEN=$CONTEXT_LEN"
echo "[raijin_e1] OMP=$OMP_NUM_THREADS  MKL=$MKL_NUM_THREADS"
echo "[raijin_e1] numactl --interleave=all"
echo "[raijin_e1] OUT_DIR=$OUT_DIR"

# ── Provenance snapshot ───────────────────────────────────────────────────────
{
    echo "=== provenance $(date -Is) ==="
    echo "host:      $(hostname)"
    echo "slurm:     job=${SLURM_JOB_ID:-none}  node=${SLURMD_NODENAME:-none}"
    echo "git sha:   $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "python:    $(which python) ($(python --version 2>&1))"
    echo "torch:     $(python -c 'import torch; print(torch.__version__)' 2>&1)"
    echo "numpy:     $(python -c 'import numpy; print(numpy.__version__)' 2>&1)"
    echo "model:     $MODEL"
    echo "horizon:   $HORIZON"
    echo "context:   $CONTEXT_LEN"
    echo "==="
} | tee "$OUT_DIR/env.log"

python -m pip freeze > "$OUT_DIR/pip-freeze.txt" 2>/dev/null || true

# ── Run training ──────────────────────────────────────────────────────────────
/usr/bin/time -v \
numactl --interleave=all \
    python scripts/train_e1.py \
        --model "$MODEL" \
        --horizon "$HORIZON" \
        --context-len "$CONTEXT_LEN" \
        --zarr-path "$ZARR" \
        --output-dir "$OUT_DIR" \
        --max-epochs 50 \
        --batch-size 16 \
        --num-workers 0 \
        --patience 10 \
        --seed 42 \
        2>&1 | tee "$OUT_DIR/train.log"

echo "[raijin_e1] done. Results in $OUT_DIR"
