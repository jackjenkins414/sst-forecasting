"""Smoke tests for the E0 baselines and metric utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sst_forecasting.models.baselines import Climatology, LinearAR, Persistence
from sst_forecasting.utils.metrics import (
    anomaly_correlation,
    bootstrap_ci,
    mae,
    rmse,
    skill_score,
    summarise,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_data():
    """Tiny synthetic SST-like dataset: T=400 days, 8×8 grid, deterministic.

    A single sinusoidal seasonal component + per-cell linear trend + noise.
    Land = (0,0) and (7,7) (NaN); rest is ocean.
    """
    rng = np.random.default_rng(0)
    T, H, W = 400, 8, 8
    time = pd.date_range("1990-01-01", periods=T, freq="D")
    doy = time.dayofyear.values.astype(np.float32)
    seasonal = 25.0 + 5.0 * np.sin(2 * np.pi * doy / 365.0)            # (T,)
    trend = (np.arange(T)[:, None, None] * 1e-4).astype(np.float32)     # (T,1,1)
    cell_offset = rng.normal(0, 0.5, size=(H, W)).astype(np.float32)
    sst = seasonal[:, None, None] + cell_offset[None] + trend
    sst = sst + rng.normal(0, 0.1, size=(T, H, W)).astype(np.float32)
    # Land cells
    sst[:, 0, 0] = np.nan
    sst[:, 7, 7] = np.nan
    # Climatology: mean over training half (T/2 days) by DOY
    train = sst[: T // 2]
    train_doy = time[: T // 2].dayofyear.values
    clim = np.zeros((366, H, W), dtype=np.float32)
    counts = np.zeros((366,), dtype=np.int32)
    for k in range(len(train)):
        d = train_doy[k] - 1
        np.add(clim[d], train[k], out=clim[d], where=np.isfinite(train[k]))
        counts[d] += 1
    counts = np.maximum(counts, 1)[:, None, None]
    clim = clim / counts
    return {"sst": sst, "time": time, "climatology": clim, "T_train": T // 2}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_rmse_and_mae_zero_for_perfect_pred():
    truth = np.array([[1.0, np.nan], [2.0, 3.0]])[None]
    pred = truth.copy()
    assert rmse(pred, truth) == 0.0
    assert mae(pred, truth) == 0.0


def test_rmse_known_value():
    truth = np.array([[0.0, 0.0]])[None]
    pred = np.array([[1.0, 1.0]])[None]
    assert rmse(pred, truth) == pytest.approx(1.0)
    assert mae(pred, truth) == pytest.approx(1.0)


def test_acc_perfect_correlation():
    truth = np.array([[1.0, 2.0, 3.0]])[None].astype(np.float32)
    pred = truth.copy()
    clim = np.zeros_like(truth)
    # When pred == truth, ACC should be 1
    assert anomaly_correlation(pred, truth, clim) == pytest.approx(1.0, abs=1e-6)


def test_skill_score_signs():
    assert skill_score(0.5, 1.0) == pytest.approx(0.5)
    assert skill_score(1.0, 0.5) == pytest.approx(-1.0)


def test_bootstrap_ci_shape_and_order():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(50, 4, 4)).astype(np.float32)
    truth = pred + rng.normal(scale=0.1, size=pred.shape).astype(np.float32)
    lo, hi = bootstrap_ci(pred, truth, "rmse", n_boot=200, seed=1)
    assert lo < hi


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_persistence_returns_last_value(synthetic_data):
    sst = synthetic_data["sst"]
    time = synthetic_data["time"]
    p = Persistence().fit(sst, time, horizon=7)
    yhat = p.predict_at(sst, t_origin=100, time=time, horizon=7)
    assert yhat.shape == sst.shape[1:]
    np.testing.assert_array_equal(
        yhat[np.isfinite(sst[100])], sst[100][np.isfinite(sst[100])],
    )


# ---------------------------------------------------------------------------
# Climatology
# ---------------------------------------------------------------------------


def test_climatology_returns_doy_lookup(synthetic_data):
    clim = Climatology(synthetic_data["climatology"])
    time = synthetic_data["time"]
    yhat = clim.predict_at(synthetic_data["sst"], t_origin=100, time=time, horizon=7)
    expected_doy = int(time[107].dayofyear)
    np.testing.assert_array_equal(yhat, synthetic_data["climatology"][expected_doy - 1])


def test_climatology_rejects_wrong_shape():
    with pytest.raises(ValueError, match="climatology must be"):
        Climatology(np.zeros((365, 8, 8)))


# ---------------------------------------------------------------------------
# LinearAR
# ---------------------------------------------------------------------------


def test_linear_ar_fit_and_predict_shape(synthetic_data):
    sst = synthetic_data["sst"]
    time = synthetic_data["time"]
    T_tr = synthetic_data["T_train"]
    ar = LinearAR(context_len=20).fit(sst[:T_tr], time[:T_tr], horizon=7)
    yhat = ar.predict_at(sst, t_origin=T_tr + 30, time=time, horizon=7)
    assert yhat.shape == sst.shape[1:]
    # Land cells should be NaN
    assert np.isnan(yhat[0, 0])
    assert np.isnan(yhat[7, 7])
    # Ocean cells should be finite
    assert np.isfinite(yhat[3, 4])


def test_linear_ar_horizon_mismatch_raises(synthetic_data):
    sst = synthetic_data["sst"]
    time = synthetic_data["time"]
    T_tr = synthetic_data["T_train"]
    ar = LinearAR(context_len=20).fit(sst[:T_tr], time[:T_tr], horizon=7)
    with pytest.raises(ValueError, match="trained for h=7"):
        ar.predict_at(sst, t_origin=T_tr + 30, time=time, horizon=1)


def test_linear_ar_beats_persistence_at_h1(synthetic_data):
    """On smooth synthetic seasonal data, AR(L) should beat persistence at h=1."""
    sst = synthetic_data["sst"]
    time = synthetic_data["time"]
    T_tr = synthetic_data["T_train"]

    h = 1
    L = 20
    ar = LinearAR(context_len=L).fit(sst[:T_tr], time[:T_tr], horizon=h)
    pers = Persistence()

    origins = np.arange(T_tr + L, len(sst) - h)
    pred_ar = np.stack([ar.predict_at(sst, t, time, h) for t in origins])
    pred_p  = np.stack([pers.predict_at(sst, t, time, h) for t in origins])
    truth   = sst[origins + h]

    assert rmse(pred_ar, truth) <= rmse(pred_p, truth) * 1.10  # within 10%


def test_summarise_includes_all_keys(synthetic_data):
    sst = synthetic_data["sst"]
    time = synthetic_data["time"]
    pers = Persistence()
    clim = Climatology(synthetic_data["climatology"])

    h = 1
    origins = np.arange(50, 100)
    pred = np.stack([pers.predict_at(sst, t, time, h) for t in origins])
    truth = sst[origins + h]
    clim_ref = np.stack([clim.predict_at(sst, t, time, h) for t in origins])

    out = summarise(pred, truth, climatology=clim_ref, n_boot=50, seed=0,
                    rmse_persistence=rmse(pred, truth))
    for k in ("rmse_C", "mae_C", "acc", "rmse_C_ci95", "mae_C_ci95",
              "acc_ci95", "skill_vs_persistence"):
        assert k in out
