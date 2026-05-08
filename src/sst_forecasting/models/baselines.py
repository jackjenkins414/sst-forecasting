"""Statistical baselines for SST forecasting (E0).

All baselines operate in **physical SST °C space** (raw, not normalised) and
expose a uniform single-lead interface:

    model = SomeBaseline(...)
    model.fit(sst_train, time_train, horizon=h)
    yhat = model.predict_at(sst, t_origin, time, horizon=h)   # (H, W) in °C

* ``Persistence`` — ``ŷ_{t+h} = y_t``
* ``Climatology`` — ``ŷ_{t+h} = clim[doy(t+h)-1]``
* ``LinearAR``    — per-grid-cell ridge regression, **direct** h-step

Land cells (NaN in input) are propagated through every baseline; metrics ignore
them via ``np.isfinite`` masks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["Persistence", "Climatology", "LinearAR"]


# ---------------------------------------------------------------------------
# 1. Persistence
# ---------------------------------------------------------------------------


class Persistence:
    """Trivial: forecast = last observed value."""

    name = "persistence"

    def fit(self, sst_train: np.ndarray, time_train: pd.DatetimeIndex,    # noqa: ARG002
            horizon: int) -> "Persistence":                                 # noqa: ARG002
        return self

    def predict_at(
        self,
        sst: np.ndarray,
        t_origin: int,
        time: pd.DatetimeIndex,                # noqa: ARG002
        horizon: int,                           # noqa: ARG002
    ) -> np.ndarray:
        return sst[t_origin].astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Climatology (daily DOY)
# ---------------------------------------------------------------------------


class Climatology:
    """Day-of-year climatology lookup.

    The pre-computed climatology of shape ``(366, H, W)`` is supplied at
    construction (it lives in the Zarr store and was built from training years
    only — no leakage).
    """

    name = "climatology"

    def __init__(self, climatology: np.ndarray) -> None:
        if climatology.ndim != 3 or climatology.shape[0] != 366:
            raise ValueError(
                f"climatology must be (366, H, W); got {climatology.shape}."
            )
        self.climatology = climatology.astype(np.float32)

    def fit(self, sst_train: np.ndarray, time_train: pd.DatetimeIndex,    # noqa: ARG002
            horizon: int) -> "Climatology":                                 # noqa: ARG002
        return self

    def predict_at(
        self,
        sst: np.ndarray,                       # noqa: ARG002
        t_origin: int,
        time: pd.DatetimeIndex,
        horizon: int,
    ) -> np.ndarray:
        doy = int(time[t_origin + horizon].dayofyear)         # 1..366
        return self.climatology[doy - 1]


# ---------------------------------------------------------------------------
# 3. Linear AR(L) — direct h-step ridge regression, vectorised over cells
# ---------------------------------------------------------------------------


class LinearAR:
    """Per-grid-cell ridge regression, **direct** h-step prediction.

    For a fixed lead time *h*, fit one linear model per ocean cell:

        ŷ_{t+h} = b + Σ_{j=0..L-1} w_j · y_{t-j}

    Closed-form ridge solution (batched over cells via ``np.linalg.solve``).

    Notes
    -----
    * Land cells (NaN in *sst_train*) are skipped during fit; their predictions
      are returned as NaN.
    * One ``LinearAR`` instance is fitted per horizon; users typically build a
      ``dict[int, LinearAR]`` keyed on h.
    """

    name = "linear_ar"

    def __init__(self, context_len: int = 30, alpha: float = 1.0) -> None:
        self.L = context_len
        self.alpha = alpha
        self._W: np.ndarray | None = None
        self._b: np.ndarray | None = None
        self._h: int = 0
        self._ocean_idx: np.ndarray | None = None
        self._spatial: tuple[int, int] | None = None

    # ......................................................................

    def fit(
        self,
        sst_train: np.ndarray,                  # (T, H, W) °C, NaN=land
        time_train: pd.DatetimeIndex,           # noqa: ARG002 (kept for API parity)
        horizon: int,
    ) -> "LinearAR":
        from numpy.lib.stride_tricks import sliding_window_view

        if sst_train.ndim != 3:
            raise ValueError(f"sst_train must be (T,H,W); got {sst_train.shape}.")
        T, H, Wd = sst_train.shape
        L, h = self.L, horizon
        if T < L + h:
            raise ValueError(f"Need T ≥ L+h = {L+h}; got T={T}.")

        ocean_mask = np.isfinite(sst_train[0]).reshape(-1)
        ocean_idx = np.where(ocean_mask)[0]
        P = ocean_idx.size
        flat = sst_train.reshape(T, H * Wd)[:, ocean_idx].astype(np.float32)   # (T, P)

        # Sliding-window views.  N origins span t = L-1 .. T-h-1 → N = T - L - h + 1
        N = T - L - h + 1
        X_all = sliding_window_view(flat, window_shape=L, axis=0)             # (T-L+1, P, L)
        X_raw = X_all[:N].transpose(1, 0, 2).astype(np.float32)               # (P, N, L)
        Y_raw = flat[L - 1 + h : L - 1 + h + N, :].T.astype(np.float32)       # (P, N)

        # Center BOTH features and targets per cell — without this the ridge
        # regulariser shrinks the (large positive) intercept toward zero and
        # the model learns essentially the training mean.
        x_mean = X_raw.mean(axis=1, keepdims=True)                            # (P, 1, L)
        y_mean = Y_raw.mean(axis=1, keepdims=True)                            # (P, 1)
        X = X_raw - x_mean
        Yc = Y_raw - y_mean

        # Batched ridge in float64 for stability:  W = (XtX + αI)^-1 XtY
        # Use np.matmul (MKL batched GEMM) instead of einsum so all 16 cores
        # are used.  (P, N, L)^T @ (P, N, L) → (P, L, L)
        Xd = X.astype(np.float64)                                              # (P, N, L)
        Ycd = Yc.astype(np.float64)                                            # (P, N)
        XtX = np.matmul(Xd.transpose(0, 2, 1), Xd)                            # (P, L, L)
        XtY = np.matmul(Xd.transpose(0, 2, 1), Ycd[:, :, None]).squeeze(-1)   # (P, L)
        I_L = np.eye(L, dtype=np.float64)
        Wmat = np.linalg.solve(XtX + self.alpha * I_L, XtY[..., None]).squeeze(-1)  # (P, L)

        # Effective intercept after re-centering:  b_eff = ȳ − w·x̄
        b_eff = y_mean.reshape(-1) - np.einsum("pl,pl->p",
                                                Wmat.astype(np.float32),
                                                x_mean.squeeze(1))            # (P,)

        self._W = Wmat.astype(np.float32)                                      # (P, L)
        self._b = b_eff.astype(np.float32)                                     # (P,)
        self._h = h
        self._ocean_idx = ocean_idx
        self._spatial = (H, Wd)
        return self

    # ......................................................................

    def predict_at(
        self,
        sst: np.ndarray,                       # full (T, H, W)
        t_origin: int,
        time: pd.DatetimeIndex,                # noqa: ARG002
        horizon: int,
    ) -> np.ndarray:
        if self._W is None:
            raise RuntimeError("LinearAR.fit must be called first.")
        if horizon != self._h:
            raise ValueError(
                f"This model was trained for h={self._h}; got predict h={horizon}."
            )
        H, W = self._spatial   # type: ignore[misc]
        L = self.L
        if t_origin - L + 1 < 0:
            raise ValueError(f"t_origin={t_origin} < L-1={L-1}.")

        flat = sst.reshape(sst.shape[0], H * W)[:, self._ocean_idx]            # (T, P)
        x = flat[t_origin - L + 1 : t_origin + 1].astype(np.float32)           # (L, P)

        # ŷ_p = Σ_l x[l, p] · W[p, l]   +  b_p
        yhat = np.einsum("lp,pl->p", x, self._W) + self._b                      # (P,)

        out = np.full((H * W,), np.nan, dtype=np.float32)
        out[self._ocean_idx] = yhat
        return out.reshape(H, W)

    # ......................................................................

    def predict_batch(
        self,
        sst: np.ndarray,                       # (T, H, W)
        origins: np.ndarray,                   # (N,) ints, each ≥ L-1
        time: pd.DatetimeIndex,                # noqa: ARG002
        horizon: int,
    ) -> np.ndarray:
        """Vectorised predict over many origins → (N, H, W).

        Equivalent to looping ``predict_at`` but uses a single batched GEMM so
        MKL can saturate all available cores.
        """
        if self._W is None:
            raise RuntimeError("LinearAR.fit must be called first.")
        if horizon != self._h:
            raise ValueError(
                f"This model was trained for h={self._h}; got predict h={horizon}."
            )
        H, Wd = self._spatial   # type: ignore[misc]
        L = self.L
        origins = np.asarray(origins, dtype=np.int64)
        if origins.size == 0:
            return np.empty((0, H, Wd), dtype=np.float32)
        if int(origins.min()) - L + 1 < 0:
            raise ValueError(f"min(origins)={int(origins.min())} < L-1={L-1}.")

        flat = sst.reshape(sst.shape[0], H * Wd)[:, self._ocean_idx]            # (T, P)

        # Build per-origin lag windows: x_batch[n, l, p] = flat[origins[n] - L + 1 + l, p]
        # Use fancy indexing — memory cost: N * L * P * 4 bytes (e.g. 730*30*5000*4 ≈ 440 MB).
        idx = origins[:, None] - L + 1 + np.arange(L)[None, :]                  # (N, L)
        x_batch = flat[idx]                                                     # (N, L, P)

        # One big GEMM: (N, L, P) · (P, L) → (N, P)
        yhat = np.einsum("nlp,pl->np", x_batch, self._W) + self._b              # (N, P)

        N = origins.size
        out = np.full((N, H * Wd), np.nan, dtype=np.float32)
        out[:, self._ocean_idx] = yhat
        return out.reshape(N, H, Wd)
