import numpy as np


def create_sliding_windows(
    data: np.ndarray,
    input_length: int,
    forecast_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding-window samples for SST forecasting.

    Parameters
    ----------
    data:
        Array with shape:
            time x num_ocean_points

    input_length:
        Number of past days used as model input.

    forecast_horizon:
        Number of days ahead to predict.

        Example:
            forecast_horizon = 1 predicts the next day.
            forecast_horizon = 7 predicts 7 days ahead.

    Returns
    -------
    X:
        Shape:
            num_samples x input_length x num_ocean_points

    y:
        Shape:
            num_samples x num_ocean_points
    """

    X = []
    y = []

    num_time_steps = data.shape[0]

    for start_idx in range(num_time_steps - input_length - forecast_horizon + 1):
        end_idx = start_idx + input_length
        target_idx = end_idx + forecast_horizon - 1

        X.append(data[start_idx:end_idx])
        y.append(data[target_idx])

    return np.array(X, dtype="float32"), np.array(y, dtype="float32")