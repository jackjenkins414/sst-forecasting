#!/bin/bash
#SBATCH --job-name=sstf_e2_convlstm_ddp
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --nodelist=hpc-03,hpc-06
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=0
# hpc-03 + hpc-06 both have 128 GB RAM — required for ConvLSTM BPTT over L=90.
# Estimated per-rank peak: ~38 GB activations (BPTT L=90) + 270 MB preloaded zarr.
#SBATCH --time=36:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --output=experiments/results/%x-%j/slurm.out
#SBATCH --error=experiments/results/%x-%j/slurm.err
#
# ─────────────────────────────────────────────────────────────────────────────
# E2: DDP training of SpatialConvLSTM at h=7, L=90, seed=42 on Coral Sea OISST.
#
# Topology
# --------
#   rank 0 → hpc-03  (IPoIB: 10.0.0.3)  — rendezvous host, writes all outputs
#   rank 1 → hpc-06  (IPoIB: 10.0.0.6)  — participant
#   backend = gloo over IPoIB (ibp176s0, 56 Gb/s FDR)
#   effective global batch size = BATCH × 2
#
# Usage
# -----
#   cd ~/sst-forecasting
#   sbatch scripts/slurm/raijin_e2_convlstm_ddp.sbatch
#
# Overrides via --export (all space-separated channel widths must be quoted):
#   sbatch --export=ALL,HIDDEN="64 128",LR=5e-4 \
#       scripts/slurm/raijin_e2_convlstm_ddp.sbatch
#
# PREREQUISITES:
#   1. /home/hpc NFS-mounted rw on hpc-03 and hpc-06 ✓
#   2. conda env `sst` installed (shared via NFS) ✓
#   3. Zarr store at ~/sst-forecasting/data/processed/oisst_coralsea.zarr ✓
#   4. IPoIB ibp176s0 UP on both nodes with IPs 10.0.0.3, 10.0.0.6 ✓
#   5. Port 29500 free on hpc-03 at job start ✓
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Graceful USR1 trap (sent 120 s before walltime by Slurm) ──────────────────
# The Python training script also handles USR1 (if the signal reaches it) by
# completing the current epoch then saving last_model.pt before exiting.
graceful_exit() {
    echo "[$(date -Is)] caught USR1 — awaiting training script checkpoint save; exiting"
    exit 0
}
trap graceful_exit USR1

# ── Hyperparameters (override via --export=ALL,KEY=value) ────────────────────
HIDDEN="${HIDDEN:-32 64}"       # space-separated channel widths, e.g. "32 64"
KERNEL="${KERNEL:-3}"
LR="${LR:-1e-3}"
BATCH="${BATCH:-16}"            # per-rank; effective global = BATCH × 2
HORIZON="${HORIZON:-7}"
CONTEXT_LEN="${CONTEXT_LEN:-90}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
SEED="${SEED:-42}"

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO="${SLURM_SUBMIT_DIR:-$HOME/sst-forecasting}"
RUN_ID="${SLURM_JOB_NAME:-sstf_e2_convlstm_ddp}-${SLURM_JOB_ID:-local}"
OUT_DIR="$REPO/experiments/results/$RUN_ID"
ZARR="$REPO/data/processed/oisst_coralsea.zarr"
mkdir -p "$OUT_DIR"
ln -sfn "$RUN_ID" "$REPO/experiments/results/latest_e2_convlstm_ddp"

cd "$REPO"

# ── Activate Intel oneAPI (provides ICX-compiled MKL runtime) ─────────────────
# setvars.sh calls 'exit' internally; capture its environment in a subshell.
if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
    oneapi_env=$(bash -c 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 && env' 2>/dev/null) || true
    if [[ -n "$oneapi_env" ]]; then
        while IFS='=' read -r key value; do
            [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && export "$key=$value"
        done <<< "$oneapi_env"
    fi
    echo "[ddp_sbatch] Intel oneAPI sourced — MKL_ROOT=${MKLROOT:-not set}"
fi

# ── Activate conda env ────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sst

# ── Threading (1 process per node, 16 physical cores per node) ───────────────
# Sandy Bridge: 16 physical cores (2 sockets × 8), no AVX2/FMA.
# --hint=nomultithread removes HT siblings; OMP_PLACES=cores pins one
# thread per physical core.  numactl --interleave=all spreads memory
# allocations across both NUMA nodes within each node.
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export MKL_DYNAMIC=FALSE
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
unset KMP_AFFINITY                      # let portable OMP_PLACES take priority
export MKL_ENABLE_INSTRUCTIONS=AVX     # Sandy Bridge: no AVX2/FMA

# ── DDP / gloo network configuration ─────────────────────────────────────────
# IPoIB interface name is ibp176s0 on ALL nodes (confirmed).
# hpc-03-ib = 10.0.0.3  (rendezvous host; TCPStore binds here on port 29500)
# hpc-06-ib = 10.0.0.6  (connects to rendezvous)
# srun propagates these exports to both tasks on both nodes.
export GLOO_SOCKET_IFNAME=ibp176s0
export GLOO_TIMEOUT_SECONDS=1800       # 30-min collective timeout (long epochs)

# ── Misc ──────────────────────────────────────────────────────────────────────
export PYTHONHASHSEED="${SEED}"
export PYTHONUNBUFFERED=1

# ── Rendezvous configuration ──────────────────────────────────────────────────
TORCHRUN="$HOME/miniconda3/envs/sst/bin/torchrun"
RDZV_HOST="hpc-03"      # hpc-03 hostname — must match socket.gethostname() on that node
#                         so _matches_machine_hostname() returns True and torchrun
#                         starts the TCPStore server there.  TCPStore binds on
#                         ethernet (192.168.2.103); gloo collectives go over IB
#                         via GLOO_SOCKET_IFNAME=ibp176s0 (separate path).
RDZV_PORT="29500"        # confirmed free; not bound by any other service

echo "[ddp_sbatch] model=convlstm_ddp  hidden=${HIDDEN}  kernel=${KERNEL}  lr=${LR}  batch=${BATCH} (eff. $((BATCH * 2)))"
echo "[ddp_sbatch] horizon=${HORIZON}  context=${CONTEXT_LEN}  max_epochs=${MAX_EPOCHS}  seed=${SEED}"
echo "[ddp_sbatch] nodes=${SLURM_NODELIST}  world_size=2  rdzv=${RDZV_HOST}:${RDZV_PORT} (TCPStore over eth; gloo collectives over IB)"
echo "[ddp_sbatch] OMP=${OMP_NUM_THREADS}  GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}  timeout=${GLOO_TIMEOUT_SECONDS}s"
echo "[ddp_sbatch] OUT_DIR=${OUT_DIR}"

# ── Provenance snapshot (on the batch-script node, i.e. hpc-03) ───────────────
{
    echo "=== provenance $(date -Is) ==="
    echo "host:          $(hostname)"
    echo "slurm:         job=${SLURM_JOB_ID:-none}  nodes=${SLURM_NODELIST:-none}"
    echo "git sha:       $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "python:        $(which python) ($(python --version 2>&1))"
    echo "torch:         $(python -c 'import torch; print(torch.__version__)' 2>&1)"
    echo "numpy:         $(python -c 'import numpy; print(numpy.__version__)' 2>&1)"
    echo "torchrun:      ${TORCHRUN}"
    echo "rdzv:          ${RDZV_HOST}:${RDZV_PORT}"
    echo "gloo_iface:    ${GLOO_SOCKET_IFNAME}"
    echo "model:         convlstm_ddp"
    echo "hidden:        ${HIDDEN}"
    echo "kernel:        ${KERNEL}"
    echo "lr:            ${LR}"
    echo "batch:         ${BATCH} per-rank  (effective global $((BATCH * 2)))"
    echo "horizon:       ${HORIZON}"
    echo "context:       ${CONTEXT_LEN}"
    echo "max_epochs:    ${MAX_EPOCHS}"
    echo "seed:          ${SEED}"
    echo "MKL_ROOT:      ${MKLROOT:-not set}"
    echo "ICX:           $(icx --version 2>/dev/null | head -1 || echo not in PATH)"
    echo "==="
} | tee "$OUT_DIR/env.log"

python -m pip freeze > "$OUT_DIR/pip-freeze.txt" 2>/dev/null || true

# ── Launch DDP training ───────────────────────────────────────────────────────
#
# srun starts 1 task on each of the 2 allocated nodes (hpc-03, hpc-06).
# Each task runs:  numactl --interleave=all torchrun [options] train_e2_ddp.py
# torchrun (--nproc_per_node=1) spawns exactly 1 Python process per node.
#
# c10d rendezvous: torchrun on hpc-03 starts a TCPStore on RDZV_HOST:RDZV_PORT;
# torchrun on hpc-06 connects to it.  Both register and receive their ranks.
# GLOO_SOCKET_IFNAME=ibp176s0 routes all collective traffic over 56 Gb/s IPoIB.
#
# --kill-on-bad-exit=1: if either rank exits non-zero, srun kills the other.
#
# shellcheck disable=SC2086
srun \
  --kill-on-bad-exit=1 \
  --ntasks=2 \
  --ntasks-per-node=1 \
  numactl --interleave=all \
  "$TORCHRUN" \
    --nproc_per_node=1 \
    --nnodes=2 \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${RDZV_HOST}:${RDZV_PORT}" \
    --rdzv-id="${SLURM_JOB_ID}" \
  scripts/train_e2_ddp.py \
    --convlstm-hidden ${HIDDEN} \
    --convlstm-kernel "${KERNEL}" \
    --horizon "${HORIZON}" \
    --context-len "${CONTEXT_LEN}" \
    --zarr-path "${ZARR}" \
    --output-dir "${OUT_DIR}" \
    --max-epochs "${MAX_EPOCHS}" \
    --batch-size "${BATCH}" \
    --lr "${LR}" \
    --patience 10 \
    --lr-patience 5 \
    --lr-factor 0.5 \
    --grad-clip 1.0 \
    --weight-decay 1e-4 \
    --seed "${SEED}" \
    2>&1 | tee "$OUT_DIR/train.log"

echo "[ddp_sbatch] done. Results in ${OUT_DIR}"
