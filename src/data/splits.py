import numpy as np
import pandas as pd


SPLITS = {
    "train": ("1981-09-01", "1995-12-31"),
    "val":   ("1996-01-01", "1998-12-31"),
    "test":  ("1999-01-01", "2000-12-31"),
}


def date_mask(time, split: str) -> np.ndarray:
    """
    Boolean mask of timesteps belonging to a given split.

    Parameters
    ----------
    time:
        1-D array of datetime-like values, or a pandas DatetimeIndex.
    split:
        One of "train", "val", "test".

    Returns
    -------
    mask:
        Boolean array with shape:
            num_timesteps
        True where the timestep falls inside the split's date range.
    """
    start_str, end_str = SPLITS[split]
    start = pd.Timestamp(start_str)
    end = pd.Timestamp(end_str)

    if not isinstance(time, pd.DatetimeIndex):
        time = pd.DatetimeIndex(time)

    return np.asarray((time >= start) & (time <= end), dtype=bool)