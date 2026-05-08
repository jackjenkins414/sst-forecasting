"""Temporal train / val / test split definitions.

All date boundaries are from PLAN.md §4.4. Climatology and normalisation
statistics are computed from the training period only to prevent leakage.

Splits
------
train : 1981-09-01 → 1995-12-31  (~14 years)
val   : 1996-01-01 → 1998-12-31  ( 3 years, model selection / early stopping)
test  : 1999-01-01 → 2000-12-31  ( 2 years, frozen until final evaluation)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical split boundaries
# ---------------------------------------------------------------------------
SPLITS: dict[str, tuple[str, str]] = {
    "train": ("1981-09-01", "1995-12-31"),
    "val":   ("1996-01-01", "1998-12-31"),
    "test":  ("1999-01-01", "2000-12-31"),
}

_SPLIT_NAMES = frozenset(SPLITS)


def _validate_split(split: str) -> None:
    if split not in _SPLIT_NAMES:
        raise ValueError(f"Unknown split {split!r}. Must be one of {sorted(_SPLIT_NAMES)}.")


def date_mask(
    time: np.ndarray | pd.DatetimeIndex,
    split: str,
) -> np.ndarray:
    """Boolean array selecting timesteps that belong to *split*.

    Parameters
    ----------
    time:
        1-D array of datetime-like values (numpy datetime64, pandas Timestamp,
        or anything ``pd.DatetimeIndex`` can parse).
    split:
        One of ``"train"``, ``"val"``, ``"test"``.

    Returns
    -------
    mask : np.ndarray of shape ``(len(time),)``, dtype bool
    """
    _validate_split(split)
    start_str, end_str = SPLITS[split]
    start = pd.Timestamp(start_str)
    end   = pd.Timestamp(end_str)
    if not isinstance(time, pd.DatetimeIndex):
        time = pd.DatetimeIndex(time)
    return np.asarray((time >= start) & (time <= end), dtype=bool)


def split_indices(
    time: np.ndarray | pd.DatetimeIndex,
    split: str,
) -> np.ndarray:
    """Integer positions into *time* that belong to *split*.

    Equivalent to ``np.where(date_mask(time, split))[0]`` but named for
    clarity at call sites.
    """
    return np.where(date_mask(time, split))[0]


def split_date_range(split: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return ``(start, end)`` Timestamps for the given split."""
    _validate_split(split)
    start_str, end_str = SPLITS[split]
    return pd.Timestamp(start_str), pd.Timestamp(end_str)
