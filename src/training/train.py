import torch


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs: int = 10,
):
    """
    Train a PyTorch model and evaluate validation loss after each epoch.

    Returns
    -------
    train_losses:
        List of average training losses, one per epoch.

    val_losses:
        List of average validation losses, one per epoch.
    """

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            preds = model(batch_X)
            loss = criterion(preds, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                preds = model(batch_X)
                loss = criterion(preds, batch_y)

                val_loss += loss.item() * batch_X.size(0)

        val_loss = val_loss / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1:02d}/{num_epochs} "
            f"| Train Loss: {train_loss:.6f} "
            f"| Val Loss: {val_loss:.6f}"
        )

    return train_losses, val_losses