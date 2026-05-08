import numpy as np


def rmse(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    land_mask: np.ndarray | None = None,
) -> float:
    """
    Root mean squared error.

    If land_mask is given, the metric is computed only over ocean cells.
    land_mask should be a boolean array broadcastable to y_pred / y_true,
    with True = ocean.
    """
    if land_mask is not None:
        diff = (y_pred - y_true)[..., land_mask]
    else:
        diff = y_pred - y_true
    return float(np.sqrt(np.mean(diff ** 2)))


def mae(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    land_mask: np.ndarray | None = None,
) -> float:
    """
    Mean absolute error.

    If land_mask is given, the metric is computed only over ocean cells.
    """
    if land_mask is not None:
        diff = (y_pred - y_true)[..., land_mask]
    else:
        diff = y_pred - y_true
    return float(np.mean(np.abs(diff)))


def rmse_per_step(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    land_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Per-horizon RMSE for multi-step forecasts.

    Parameters
    ----------
    y_pred, y_true:
        Arrays with shape:
            num_samples x horizon x H x W
    land_mask:
        Optional boolean array of shape:
            H x W
        with True = ocean. If given, RMSE is computed only over ocean cells.

    Returns
    -------
    rmse_per_step:
        Array with shape:
            horizon
        Each entry is the RMSE for that lead time, averaged over all samples.
    """
    horizon = y_pred.shape[1]
    return np.array(
        [rmse(y_pred[:, h], y_true[:, h], land_mask) for h in range(horizon)],
        dtype="float32",
    )


def skill_score(model_error: float, baseline_error: float) -> float:
    """
    Skill score relative to a baseline.

    Positive skill means the model is better than the baseline.
    Zero means equal to the baseline.
    Negative skill means the model is worse than the baseline.
    """
    return float(1 - (model_error / baseline_error))