#!/usr/bin/env python3
"""Run E0 statistical baselines (persistence, climatology, linear AR) on the
test split of the Coral Sea OISST Zarr store.

Outputs
-------
``<output-dir>/baselines.json``    aggregate metrics for every (model, horizon)
``<output-dir>/run.yaml``          full provenance: git SHA, hostname, env, args
``<output-dir>/<model>_h<h>.npz``  raw predictions + truth (optional)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import zarr

from sst_forecasting.data.splits import SPLITS, date_mask
from sst_forecasting.models.baselines import Climatology, LinearAR, Persistence
from sst_forecasting.utils.metrics import summarise

# ---------------------------------------------------------------------------


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _load_zarr(zarr_path: Path) -> dict:
    root = zarr.open_group(str(zarr_path), mode="r")
    time_days = root["time"][:]
    epoch = pd.Timestamp("1970-01-01")
    time_idx = pd.DatetimeIndex(
        [epoch + pd.Timedelta(days=int(d)) for d in time_days]
    )
    return {
        "sst":         np.asarray(root["sst"][:], dtype=np.float32),       # °C, NaN=land
        "climatology": np.asarray(root["climatology"][:], dtype=np.float32),
        "land_mask":   np.asarray(root["land_mask"][:], dtype=bool),
        "time":        time_idx,
        "attrs":       dict(root.attrs),
    }


def _eval_for_horizon(
    sst: np.ndarray,
    time: pd.DatetimeIndex,
    climatology: np.ndarray,
    horizon: int,
    ar_context: int,
    seed: int,
    bootstrap: int,
) -> dict:
    """Evaluate all three baselines at a single horizon on the test split."""
    train_idx = np.where(date_mask(time, "train"))[0]
    test_idx  = np.where(date_mask(time, "test"))[0]

    sst_train  = sst[train_idx]               # (T_tr, H, W)
    time_train = time[train_idx]

    print(f"\n========== horizon h={horizon} ==========", flush=True)
    print(f"  train: {len(train_idx)} days   test: {len(test_idx)} days", flush=True)

    # Forecast origins t such that t and t+horizon are both in the test split.
    # We additionally need at least L past days for LinearAR — use the full
    # ``sst`` array so origins early in the test split can borrow from the
    # tail of the val split for their lag context.  (Persistence/climatology
    # don't need that, but it doesn't hurt them.)
    origins_in_test = test_idx[: -horizon]    # all t with t+h still in test
    L_max = ar_context
    origins = origins_in_test[origins_in_test - L_max + 1 >= 0]
    n = len(origins)
    print(f"  forecast origins after L={L_max} guard: {n}", flush=True)

    H, W = sst.shape[1:]

    # ── Fit baselines ──────────────────────────────────────────────────────
    fit_t = {}
    pers = Persistence().fit(sst_train, time_train, horizon)
    clim = Climatology(climatology).fit(sst_train, time_train, horizon)

    t0 = _time.perf_counter()
    ar = LinearAR(context_len=ar_context, alpha=1.0).fit(sst_train, time_train, horizon)
    fit_t["linear_ar"] = _time.perf_counter() - t0
    print(f"  LinearAR fit: {fit_t['linear_ar']:.2f}s", flush=True)

    # ── Predict for every origin (vectorised) ─────────────────────────────
    # Build truth + climatology lookup vector-wise; LinearAR uses one batched
    # GEMM via predict_batch so MKL can use all cores.
    t1 = _time.perf_counter()
    truth     = sst[origins + horizon].astype(np.float32, copy=False)            # (n, H, W)
    pred_pers = sst[origins].astype(np.float32, copy=False)                      # (n, H, W)

    doy = (time[origins + horizon].dayofyear.to_numpy() - 1).astype(np.int64)
    pred_clim = climatology[doy].astype(np.float32, copy=False)                  # (n, H, W)
    clim_truth = pred_clim                                                       # for ACC reference

    pred_ar = ar.predict_batch(sst, origins, time, horizon)                      # (n, H, W)
    print(f"  predict (batched): {_time.perf_counter()-t1:.2f}s", flush=True)

    # ── Metrics ────────────────────────────────────────────────────────────
    rng_seed = seed + horizon
    res_pers = summarise(pred_pers, truth, climatology=clim_truth,
                         n_boot=bootstrap, seed=rng_seed)
    rmse_pers = res_pers["rmse_C"]

    res_clim = summarise(pred_clim, truth, climatology=clim_truth,
                         rmse_persistence=rmse_pers,
                         n_boot=bootstrap, seed=rng_seed)
    res_ar   = summarise(pred_ar,   truth, climatology=clim_truth,
                         rmse_persistence=rmse_pers,
                         n_boot=bootstrap, seed=rng_seed)

    return {
        "horizon":     horizon,
        "n_windows":   int(n),
        "ar_context":  ar_context,
        "persistence": res_pers,
        "climatology": res_clim,
        "linear_ar":   res_ar,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description="E0 baselines for SST forecasting.")
    p.add_argument("--zarr-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 7, 30])
    p.add_argument("--ar-context", type=int, default=30,
                   help="Context length L for the LinearAR baseline.")
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="Number of bootstrap resamples for 95% CIs.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]

    # ── Provenance ─────────────────────────────────────────────────────────
    np.random.seed(args.seed)
    provenance = {
        "git_sha":        _git_sha(repo_root),
        "hostname":       socket.gethostname(),
        "platform":       platform.platform(),
        "python":         sys.version.split()[0],
        "numpy":          np.__version__,
        "pandas":         pd.__version__,
        "zarr":           zarr.__version__,
        "slurm_job_id":   os.environ.get("SLURM_JOB_ID", ""),
        "slurm_node":     os.environ.get("SLURMD_NODENAME", ""),
        "omp_threads":    os.environ.get("OMP_NUM_THREADS", ""),
        "mkl_threads":    os.environ.get("MKL_NUM_THREADS", ""),
        "args":           {k: (str(v) if isinstance(v, Path) else v)
                           for k, v in vars(args).items()},
        "splits":         SPLITS,
    }
    with open(args.output_dir / "run.yaml", "w") as f:
        yaml.safe_dump(provenance, f, sort_keys=False)
    print(f"[provenance] git={provenance['git_sha'][:8]}  host={provenance['hostname']}  "
          f"OMP={provenance['omp_threads']}  MKL={provenance['mkl_threads']}", flush=True)

    # ── Load data ──────────────────────────────────────────────────────────
    t0 = _time.perf_counter()
    print(f"[data] loading {args.zarr_path} …", flush=True)
    data = _load_zarr(args.zarr_path)
    print(f"[data] sst={data['sst'].shape} time=[{data['time'][0].date()}..."
          f"{data['time'][-1].date()}] loaded in {_time.perf_counter()-t0:.1f}s",
          flush=True)

    # ── Per-horizon evaluation ─────────────────────────────────────────────
    all_results = []
    for h in args.horizons:
        r = _eval_for_horizon(
            data["sst"], data["time"], data["climatology"],
            horizon=h, ar_context=args.ar_context,
            seed=args.seed, bootstrap=args.bootstrap,
        )
        all_results.append(r)
        # Pretty print
        for m in ("persistence", "climatology", "linear_ar"):
            row = r[m]
            ci = row["rmse_C_ci95"]
            sk = row.get("skill_vs_persistence")
            sk_s = f"  SS={sk:+.3f}" if sk is not None else ""
            print(f"  {m:>11s}  RMSE={row['rmse_C']:.4f} °C "
                  f"[{ci[0]:.4f}, {ci[1]:.4f}]  ACC={row['acc']:.3f}{sk_s}",
                  flush=True)

    # ── Persist metrics ────────────────────────────────────────────────────
    out_path = args.output_dir / "baselines.json"
    payload = {"provenance": provenance, "results": all_results}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n[done] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
