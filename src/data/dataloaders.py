from torch.utils.data import DataLoader

from src.data.dataset import SstWindowDataset


def create_dataloaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    batch_size: int = 16,
):
    """
    Create PyTorch DataLoaders for train, validation, and test sets.

    Training data is shuffled after the chronological split.
    Validation and test data are not shuffled.
    """

    train_dataset = SstWindowDataset(X_train, y_train)
    val_dataset = SstWindowDataset(X_val, y_val)
    test_dataset = SstWindowDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader