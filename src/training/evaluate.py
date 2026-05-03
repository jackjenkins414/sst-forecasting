import torch


def predict(model, data_loader, device):
    """
    Run a trained model over a DataLoader and return predictions and targets.

    Returns
    -------
    preds:
        NumPy array of model predictions.

    targets:
        NumPy array of true target values.
    """

    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            preds = model(batch_X)

            all_preds.append(preds.cpu())
            all_targets.append(batch_y.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    return preds, targets