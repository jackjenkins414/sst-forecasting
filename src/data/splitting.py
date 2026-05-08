def chronological_train_val_test_split(
    X,
    y,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
):
    """
    Split sliding-window samples chronologically into train, validation, and test sets.

    This is important for forecasting because future samples should not be used
    to train models that are evaluated on earlier samples.

    Parameters
    ----------
    X:
        Input samples with shape:
            num_samples x input_length x num_features

    y:
        Target samples with shape:
            num_samples x num_features

    train_fraction:
        Fraction of samples used for training.

    val_fraction:
        Fraction of samples used for validation.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
    """

    num_samples = X.shape[0]

    train_end = int(num_samples * train_fraction)
    val_end = int(num_samples * (train_fraction + val_fraction))

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test