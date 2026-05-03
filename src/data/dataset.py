import torch
from torch.utils.data import Dataset


class SstWindowDataset(Dataset):
    """
    PyTorch Dataset for SST sliding-window forecasting.

    Each item contains:
        X[idx] = sequence of past SST maps/vectors
        y[idx] = future SST target vector

    Expected shapes:
        X: num_samples x input_length x num_ocean_points
        y: num_samples x num_ocean_points
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]