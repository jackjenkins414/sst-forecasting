#!/bin/bash
# Run E0 baselines locally on hpc-01 using all 16 physical cores (both sockets).
# Usage: bash scripts/run_e0_local.sh [extra args passed to run_baselines.py]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# ── Conda env ────────────────────────────────────────────────────────────────
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sst

# ── Intel oneAPI (optional — makes MKL/MPI libs visible) ────────────────────
# setvars.sh can call `exit` internally; we capture its env exports instead of
# sourcing it directly so an error there doesn't kill this script.
if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
    oneapi_env=$(bash -c 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 && env' 2>/dev/null) || true
    if [[ -n "$oneapi_env" ]]; then
        while IFS='=' read -r key value; do
            [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && export "$key=$value"
        done <<< "$oneapi_env"
    fi
fi

# ── Threading — use both NUMA sockets, no hyperthreading ────────────────────
# 16 physical cores = 8 per socket × 2 sockets.
# OMP_PLACES=cores + OMP_PROC_BIND=close keeps one thread per physical core.
# KMP_AFFINITY is unset to let the portable OMP_PLACES take priority.
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export MKL_DYNAMIC=FALSE
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
unset KMP_AFFINITY 2>/dev/null || true
export MKL_ENABLE_INSTRUCTIONS=AVX    # Sandy Bridge — no AVX2/FMA
export PYTHONHASHSEED=42

echo "[run_e0_local] OMP=$OMP_NUM_THREADS  MKL=$MKL_NUM_THREADS  PLACES=$OMP_PLACES"
echo "[run_e0_local] numactl --interleave=all  (both sockets)"

# ── Execute ──────────────────────────────────────────────────────────────────
/usr/bin/time -v \
numactl --interleave=all \
    python scripts/run_baselines.py \
        --zarr-path  data/processed/oisst_coralsea.zarr \
        --output-dir experiments/results/e0_local \
        --horizons 1 7 30 \
        --ar-context 30 \
        --bootstrap 1000 \
        --seed 42 \
        "$@"
