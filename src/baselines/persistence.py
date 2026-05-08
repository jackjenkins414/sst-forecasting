import numpy as np

# Adapted to predict horizon days 
def persistence_forecast(X: np.ndarray, horizon: int) -> np.ndarray:
    """
    Persistence baseline for SST forecasting.

    For each input sequence, predict every future day as equal to the
    final day in the input window.

    Parameters
    ----------
    X:
        Input samples with shape:
            num_samples x context_len x 1 x H x W
    horizon:
        Number of future timesteps to predict.

    Returns
    -------
    preds:
        Persistence predictions with shape:
            num_samples x horizon x H x W
    """
    last_day = X[:, -1, 0]  # (num_samples, H, W)
    preds = np.broadcast_to(
        last_day[:, np.newaxis, :, :],
        (last_day.shape[0], horizon, last_day.shape[1], last_day.shape[2]),
    )
    return np.ascontiguousarray(preds)