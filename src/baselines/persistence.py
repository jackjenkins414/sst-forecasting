import numpy as np


def persistence_forecast(X: np.ndarray) -> np.ndarray:
    """
    Persistence baseline for SST forecasting.

    For each input sequence, predict the future SST as equal to
    the final day in the input window.

    Parameters
    ----------
    X:
        Input samples with shape:
            num_samples x input_length x num_ocean_points

    Returns
    -------
    preds:
        Persistence predictions with shape:
            num_samples x num_ocean_points
    """

    return X[:, -1, :]