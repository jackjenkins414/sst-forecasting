#!/bin/bash
#SBATCH --job-name=sstf_e2_convlstm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodelist=hpc-03
# hpc-03 / hpc-06 have 128 GB — required for ConvLSTM BPTT over L=90 steps
# Estimated activation memory: ~2.4 GB at B=4; headroom needed for MKL workspace.
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --output=experiments/results/%x-%j/slurm.out
#SBATCH --error=experiments/results/%x-%j/slurm.err
#
# ─────────────────────────────────────────────────────────────────────────────
# E2: train SpatialConvLSTM at h=7, L=90, seed=42 on Coral Sea OISST.
#     Uses Intel oneAPI MKL (ICX-compiled) for BLAS ops.
#
# Usage
# -----
#   sbatch scripts/slurm/raijin_e2_convlstm.sbatch
#
# Overrides via --export (all space-separated channel widths must be quoted):
#   sbatch --export=ALL,HIDDEN="64 128",LR=0.0002 scripts/slurm/raijin_e2_convlstm.sbatch
#
# PREREQUISITES:
#   1. /home/hpc NFS-mounted on worker nodes (hpc-02..07) — done ✓
#   2. conda env `sst` with all requirements — shared via NFS ✓
#   3. Zarr store at ~/sst-forecasting/data/processed/oisst_coralsea.zarr ✓
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Graceful USR1 trap (sent 120 s before walltime) ──────────────────────────
graceful_exit() {
    echo "[$(date -Is)] caught USR1 — last_model.pt already saved; exiting cleanly"
    exit 0
}
trap graceful_exit USR1

# ── Hyperparameters (override via --export) ───────────────────────────────────
HIDDEN="${HIDDEN:-32 64}"       # space-separated channel widths, e.g. "32 64" or "64 128"
KERNEL="${KERNEL:-3}"
LR="${LR:-1e-3}"              # matches LSTM lr for controlled comparison (job 117)
BATCH="${BATCH:-16}"            # matches LSTM bs for controlled comparison (job 117)
HORIZON="${HORIZON:-7}"
CONTEXT_LEN="${CONTEXT_LEN:-90}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
SEED="${SEED:-42}"

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO="${SLURM_SUBMIT_DIR:-$HOME/sst-forecasting}"
RUN_ID="${SLURM_JOB_NAME:-sstf_e2_convlstm}-${SLURM_JOB_ID:-local}"
OUT_DIR="$REPO/experiments/results/$RUN_ID"
ZARR="$REPO/data/processed/oisst_coralsea.zarr"
mkdir -p "$OUT_DIR"
ln -sfn "$RUN_ID" "$REPO/experiments/results/latest_e2_convlstm"

cd "$REPO"

# ── Activate Intel oneAPI (provides ICX-compiled MKL runtime) ─────────────────
# setvars.sh calls 'exit' internally so we capture its env in a subshell.
if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
    oneapi_env=$(bash -c 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 && env' 2>/dev/null) || true
    if [[ -n "$oneapi_env" ]]; then
        while IFS='=' read -r key value; do
            [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && export "$key=$value"
        done <<< "$oneapi_env"
    fi
    echo "[raijin_e2] Intel oneAPI sourced — MKL_ROOT=${MKLROOT:-not set}"
fi

# ── Activate conda env ────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sst

# ── Threading / NUMA pinning ──────────────────────────────────────────────────
# Sandy Bridge: 16 physical cores (2 sockets × 8), no AVX2/FMA.
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
unset KMP_AFFINITY                      # let portable OMP_PLACES take priority
export MKL_ENABLE_INSTRUCTIONS=AVX     # Sandy Bridge: no AVX2/FMA — suppress warnings
export PYTHONHASHSEED="${SEED}"
export PYTHONUNBUFFERED=1

echo "[raijin_e2] model=convlstm  hidden=${HIDDEN}  kernel=${KERNEL}  lr=${LR}  batch=${BATCH}"
echo "[raijin_e2] horizon=${HORIZON}  context=${CONTEXT_LEN}  max_epochs=${MAX_EPOCHS}  seed=${SEED}"
echo "[raijin_e2] OMP=${OMP_NUM_THREADS}  MKL=${MKL_NUM_THREADS}  numactl --interleave=all"
echo "[raijin_e2] OUT_DIR=${OUT_DIR}"

# ── Provenance snapshot ───────────────────────────────────────────────────────
{
    echo "=== provenance $(date -Is) ==="
    echo "host:        $(hostname)"
    echo "slurm:       job=${SLURM_JOB_ID:-none}  node=${SLURMD_NODENAME:-none}"
    echo "git sha:     $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "python:      $(which python) ($(python --version 2>&1))"
    echo "torch:       $(python -c 'import torch; print(torch.__version__)' 2>&1)"
    echo "numpy:       $(python -c 'import numpy; print(numpy.__version__)' 2>&1)"
    echo "model:       convlstm"
    echo "hidden:      ${HIDDEN}"
    echo "kernel:      ${KERNEL}"
    echo "lr:          ${LR}"
    echo "batch:       ${BATCH}"
    echo "horizon:     ${HORIZON}"
    echo "context:     ${CONTEXT_LEN}"
    echo "max_epochs:  ${MAX_EPOCHS}"
    echo "seed:        ${SEED}"
    echo "MKL_ROOT:    ${MKLROOT:-not set}"
    echo "ICX:         $(icx --version 2>/dev/null | head -1 || echo not in PATH)"
    echo "==="
} | tee "$OUT_DIR/env.log"

python -m pip freeze > "$OUT_DIR/pip-freeze.txt" 2>/dev/null || true

# ── Run training ──────────────────────────────────────────────────────────────
# shellcheck disable=SC2086
/usr/bin/time -v \
numactl --interleave=all \
    python scripts/train_e1.py \
        --model convlstm \
        --convlstm-hidden ${HIDDEN} \
        --convlstm-kernel "${KERNEL}" \
        --horizon "${HORIZON}" \
        --context-len "${CONTEXT_LEN}" \
        --zarr-path "${ZARR}" \
        --output-dir "${OUT_DIR}" \
        --max-epochs "${MAX_EPOCHS}" \
        --batch-size "${BATCH}" \
        --lr "${LR}" \
        --num-workers 0 \
        --patience 10 \
        --seed "${SEED}" \
        2>&1 | tee "$OUT_DIR/train.log"

echo "[raijin_e2] done. Results in ${OUT_DIR}"
