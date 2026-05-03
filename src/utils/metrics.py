import numpy as np


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Root mean squared error.
    """
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Mean absolute error.
    """
    return float(np.mean(np.abs(y_pred - y_true)))


def skill_score(model_error: float, baseline_error: float) -> float:
    """
    Skill score relative to a baseline.

    Positive skill means the model is better than the baseline.
    Zero means equal to the baseline.
    Negative skill means the model is worse than the baseline.
    """
    return float(1 - (model_error / baseline_error))