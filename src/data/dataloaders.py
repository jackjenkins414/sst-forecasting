from pathlib import Path

from torch.utils.data import DataLoader

from src.data.dataset import SstWindowDataset


def create_dataloaders(
    zarr_path: str | Path,
    context_len: int = 90,
    horizon: int = 7,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """
    Create PyTorch DataLoaders for the train, validation, and test splits.

    Training data is shuffled.
    Validation and test data are not shuffled.

    Parameters
    ----------
    zarr_path:
        Path to the processed Zarr store containing normalised SST anomalies
        and split-date metadata.
    context_len:
        Number of past timesteps fed to the model.
    horizon:
        Number of future timesteps to predict.
    batch_size:
        Number of samples per batch.
    num_workers:
        DataLoader worker processes. >0 overlaps host-side batch assembly /
        H2D copies with GPU compute. The dataset holds its field in RAM, so
        workers fork with copy-on-write (no per-worker reload).
    pin_memory:
        Use pinned host memory for faster asynchronous H2D transfer on CUDA.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    # Prev version built 6 numpy arrays, change it so the Dataset consturcts
    # itself directly from the Zarr store
    train_dataset = SstWindowDataset(zarr_path, "train", context_len, horizon)
    val_dataset = SstWindowDataset(zarr_path, "val", context_len, horizon)
    test_dataset = SstWindowDataset(zarr_path, "test", context_len, horizon)

    # persistent_workers keeps workers alive between epochs (avoids re-fork
    # cost each epoch); only valid when num_workers > 0.
    persistent = num_workers > 0
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    train_loader = DataLoader(train_dataset, shuffle=True,  **common)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **common)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **common)

    return train_loader, val_loader, test_loader