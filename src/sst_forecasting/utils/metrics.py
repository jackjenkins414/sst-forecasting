"""Forecast metrics with bootstrap 95% CIs.

All functions accept *prediction* and *truth* arrays in **physical SST °C
space** of shape ``(N, H, W)`` where ``N`` is the number of forecast windows
and ``(H, W)`` is the spatial grid.  Land cells (NaN in *truth*) are masked
out automatically; predictions at land cells are ignored.

The bootstrap is over the *N* axis (block bootstrap of independent forecast
origins is acceptable here because we report aggregate skill — for serial-
correlation aware CIs use a moving block bootstrap, future work).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

__all__ = [
    "rmse",
    "mae",
    "anomaly_correlation",
    "skill_score",
    "bootstrap_ci",
    "summarise",
]


# ---------------------------------------------------------------------------
# Point metrics (return scalar over all ocean cells × all windows)
# ---------------------------------------------------------------------------


def _ocean_mask(truth: np.ndarray) -> np.ndarray:
    """Boolean mask of ocean cells in ``truth`` (True = ocean, finite)."""
    return np.isfinite(truth)


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    """Root-mean-square error in °C over ocean cells."""
    m = _ocean_mask(truth)
    diff = pred[m] - truth[m]
    return float(np.sqrt(np.mean(diff * diff)))


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    """Mean absolute error in °C over ocean cells."""
    m = _ocean_mask(truth)
    return float(np.mean(np.abs(pred[m] - truth[m])))


def anomaly_correlation(
    pred: np.ndarray,
    truth: np.ndarray,
    climatology: np.ndarray,
) -> float:
    """Anomaly Correlation Coefficient against the supplied climatology.

    ``pred``, ``truth``, ``climatology`` all have shape ``(N, H, W)`` and are in
    the same units (°C).  ACC is computed pooled over all ocean cells × windows.
    """
    m = _ocean_mask(truth)
    p = pred[m] - climatology[m]
    t = truth[m] - climatology[m]
    p_mean = p.mean()
    t_mean = t.mean()
    num = np.sum((p - p_mean) * (t - t_mean))
    den = np.sqrt(np.sum((p - p_mean) ** 2) * np.sum((t - t_mean) ** 2))
    return float(num / den) if den > 0 else float("nan")


def skill_score(rmse_model: float, rmse_reference: float) -> float:
    """``1 - RMSE_model / RMSE_reference`` (positive = better than reference)."""
    if rmse_reference <= 0:
        return float("nan")
    return float(1.0 - rmse_model / rmse_reference)


# ---------------------------------------------------------------------------
# Bootstrap CI — resample over forecast windows (axis 0)
# ---------------------------------------------------------------------------


def bootstrap_ci(
    pred: np.ndarray,
    truth: np.ndarray,
    metric: str = "rmse",
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    climatology: np.ndarray | None = None,
) -> tuple[float, float]:
    """Two-sided ``(1-alpha)`` percentile CI for a metric.

    Resamples the *window* axis (axis 0) with replacement.  Returns the
    ``(alpha/2, 1-alpha/2)`` percentiles of the bootstrap distribution.

    Strategy for low memory use:
      RMSE / MAE — pre-reduce spatial dimension to per-window scalars (N,);
                   bootstrap only needs a (n_boot, N) index array.
      ACC        — pre-extract flat (N, P) ocean arrays; loop over n_boot with
                   pure NumPy ops (fast; avoids (n_boot, N, P) tensor).
    """
    if metric == "acc" and climatology is None:
        raise ValueError("climatology required for ACC bootstrap")

    n = pred.shape[0]
    rng = np.random.default_rng(seed)

    # Pre-extract ocean cells once: (N, P)
    mask = np.isfinite(truth[0]).ravel()
    p_flat = pred.reshape(n, -1)[:, mask].astype(np.float32)    # (N, P)
    t_flat = truth.reshape(n, -1)[:, mask].astype(np.float32)   # (N, P)

    if metric == "rmse":
        # Per-window MSE scalar — (N,); bootstrap reduces to mean over (n_boot, N)
        per_win = np.mean((p_flat - t_flat) ** 2, axis=1)        # (N,)
        idx = rng.integers(0, n, size=(n_boot, n), dtype=np.int32)
        samples = np.sqrt(per_win[idx].mean(axis=1))             # (n_boot,)

    elif metric == "mae":
        per_win = np.mean(np.abs(p_flat - t_flat), axis=1)       # (N,)
        idx = rng.integers(0, n, size=(n_boot, n), dtype=np.int32)
        samples = per_win[idx].mean(axis=1)                      # (n_boot,)

    else:  # acc — decomposed into 5 per-window scalars so bootstrap is O(N)
        c_flat = climatology.reshape(n, -1)[:, mask].astype(np.float64)
        p_anom = (p_flat - c_flat).astype(np.float64)            # (N, P)
        t_anom = (t_flat - c_flat).astype(np.float64)            # (N, P)
        P = p_anom.shape[1]
        # Per-window summary stats — one scalar per window
        ps = p_anom.sum(axis=1)                                  # (N,)
        ts = t_anom.sum(axis=1)                                  # (N,)
        p2 = (p_anom ** 2).sum(axis=1)                          # (N,)
        t2 = (t_anom ** 2).sum(axis=1)                          # (N,)
        pt = (p_anom * t_anom).sum(axis=1)                      # (N,)
        # Bootstrap: (n_boot, N) index → aggregate the 5 per-window vectors
        idx_all = rng.integers(0, n, size=(n_boot, n), dtype=np.int32)
        SP   = ps[idx_all].sum(axis=1)                           # (n_boot,)
        ST   = ts[idx_all].sum(axis=1)
        SPT  = pt[idx_all].sum(axis=1)
        SP2  = p2[idx_all].sum(axis=1)
        ST2  = t2[idx_all].sum(axis=1)
        NP   = float(n * P)
        num  = SPT - SP * ST / NP
        dp   = SP2 - SP ** 2 / NP
        dt   = ST2 - ST ** 2 / NP
        den  = np.sqrt(np.maximum(dp * dt, 0.0))
        samples = np.where(den > 0, num / den, np.nan)

    lo = float(np.nanquantile(samples, alpha / 2))
    hi = float(np.nanquantile(samples, 1 - alpha / 2))
    return lo, hi


# ---------------------------------------------------------------------------
# One-shot summary
# ---------------------------------------------------------------------------


def summarise(
    pred: np.ndarray,
    truth: np.ndarray,
    *,
    climatology: np.ndarray,
    rmse_persistence: float | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Compute RMSE / MAE / ACC + bootstrap CIs + skill vs persistence.

    Parameters
    ----------
    pred, truth, climatology : ``(N, H, W)`` arrays in °C
    rmse_persistence : optional reference RMSE for the skill score
    """
    out: dict = {
        "n_windows": int(pred.shape[0]),
        "rmse_C": rmse(pred, truth),
        "mae_C":  mae(pred, truth),
        "acc":    anomaly_correlation(pred, truth, climatology),
    }
    out["rmse_C_ci95"] = list(
        bootstrap_ci(pred, truth, "rmse", n_boot=n_boot, seed=seed)
    )
    out["mae_C_ci95"] = list(
        bootstrap_ci(pred, truth, "mae", n_boot=n_boot, seed=seed)
    )
    out["acc_ci95"] = list(
        bootstrap_ci(
            pred, truth, "acc", n_boot=n_boot, seed=seed,
            climatology=climatology,
        )
    )
    if rmse_persistence is not None:
        out["skill_vs_persistence"] = skill_score(out["rmse_C"], rmse_persistence)
    return out
