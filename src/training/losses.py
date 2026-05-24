import torch
import torch.nn as nn


class AnomalyWeightedMSE(nn.Module):
    """
    MSE loss weighted by the magnitude of the target anomaly.

        loss = mean( (1 + alpha * |y_true|) * (y_pred - y_true)^2 )

    Larger anomalies (warm or cold) are penalised more heavily, pushing the
    model away from hedging toward the mean and toward capturing extremes.
    alpha=0 reduces to standard MSE.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = 1.0 + self.alpha * target.abs()
        return (weight * (pred - target) ** 2).mean()
