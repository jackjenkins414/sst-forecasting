import torch
import torch.nn as nn


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs: int = 10,
    land_mask: torch.Tensor | None = None,
    grad_clip: float | None = None,
    scheduler=None,
    early_stop_patience: int | None = None,
    epoch_callback=None,
    use_amp: bool = False,
):
    """
    Train a PyTorch model and evaluate validation loss after each epoch.

    Tracks the best val loss checkpoint and restores it before returning,
    so the model always ends in its best-generalising state regardless of
    how many epochs are run.

    If early_stop_patience is set, training stops when val loss has not
    improved for that many consecutive epochs.
    """
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_state    = None
    epochs_no_improve = 0

    # BF16 autocast on CUDA: lets H200 tensor cores run the convs at ~tensor-core
    # throughput. BF16 has the same exponent range as FP32 so no GradScaler is
    # needed (unlike FP16). A no-op when use_amp=False or on CPU.
    amp_enabled = bool(use_amp) and torch.device(device).type == "cuda"

    def _autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled)

    for epoch in range(num_epochs):
        model.train()
        # Accumulate the loss on-device to avoid a host sync (loss.item()) every
        # batch — at B=2 that was thousands of syncs/epoch serialising the pipe.
        train_loss_sum = torch.zeros((), device=device)
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with _autocast():
                preds = model(batch_X)
                if land_mask is not None:
                    mask = land_mask.expand_as(preds)
                    loss = criterion(preds[mask], batch_y[mask])
                else:
                    loss = criterion(preds, batch_y)

            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_loss_sum += loss.detach() * batch_X.size(0)

        train_loss = (train_loss_sum / len(train_loader.dataset)).item()

        model.eval()
        val_loss_sum = torch.zeros((), device=device)
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                with _autocast():
                    preds = model(batch_X)
                    if land_mask is not None:
                        mask = land_mask.expand_as(preds)
                        loss = criterion(preds[mask], batch_y[mask])
                    else:
                        loss = criterion(preds, batch_y)

                val_loss_sum += loss.detach() * batch_X.size(0)

        val_loss = (val_loss_sum / len(val_loader.dataset)).item()

        if scheduler is not None:
            scheduler.step(val_loss)

        # Track best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            improved = " *"
        else:
            epochs_no_improve += 1
            improved = ""

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1:02d}/{num_epochs} "
            f"| Train Loss: {train_loss:.6f} "
            f"| Val Loss: {val_loss:.6f}{improved}"
        )

        if epoch_callback is not None and epoch_callback(epoch, val_loss):
            print(f"Trial pruned at epoch {epoch + 1}")
            break

        if early_stop_patience is not None and epochs_no_improve >= early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1} (no improvement for {early_stop_patience} epochs)")
            break

    # Restore the best weights before returning
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"Restored best checkpoint (val loss {best_val_loss:.6f})")

    return train_losses, val_losses