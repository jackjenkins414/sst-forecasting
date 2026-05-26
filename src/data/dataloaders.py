from pathlib import Path

from torch.utils.data import DataLoader

from src.data.dataset import SstWindowDataset


def create_dataloaders(
    zarr_path: str | Path,
    context_len: int = 90,
    horizon: int = 7,
    batch_size: int = 16,
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

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    # Prev version built 6 numpy arrays, change it so the Dataset consturcts
    # itself directly from the Zarr store 
    train_dataset = SstWindowDataset(zarr_path, "train", context_len, horizon)
    val_dataset = SstWindowDataset(zarr_path, "val", context_len, horizon)
    test_dataset = SstWindowDataset(zarr_path, "test", context_len, horizon)

    # num_workers>0 overlaps Zarr decompression with GPU compute; pin_memory
    # speeds host->GPU copies; persistent_workers avoids re-spawn each epoch
    # (important on Windows spawn). Verified pickle-safe with SstWindowDataset.
    loader_kwargs = dict(num_workers=4, pin_memory=True, persistent_workers=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader