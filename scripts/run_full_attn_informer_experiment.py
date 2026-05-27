import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import zarr

from src.data.dataloaders import create_dataloaders

from src.models.informer import ProbSparseInformer

from src.training.train import train_model
from src.training.evaluate import predict

from src.baselines.persistence import persistence_forecast
from src.baselines.rnn import RNN
from src.utils.metrics import rmse, rmse_per_step, mae, skill_score


# Configuration

ZARR_PATH = PROJECT_ROOT / "data/processed/oisst_coralsea.zarr"

CONTEXT_LEN = 90
HORIZON = 7
BATCH_SIZE = 16

# Informer Params
D_MODEL = 128
N_HEADS = 4
N_ENCODER_LAYERS = 3
N_DECODER_LAYERS = 2
D_FF = 512
DROPOUT = 0.1
FACTOR = 5

# RNN Params
D_SPATIAL = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 2

# Shared Training Params
NUM_EPOCHS = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

RANDOM_SEED = 42


def main():
    # Seeds
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Load metadata from Zarr (norm stats, land mask, grid shape)
    root = zarr.open_group(str(ZARR_PATH), mode="r")
    norm_mean = float(root.attrs["norm_mean"])
    norm_std = float(root.attrs["norm_std"])
    land_mask_np = np.array(root["land_mask"])  # (H, W) bool, True = ocean
    H, W = land_mask_np.shape

    print(f"Grid: H={H} W={W} ocean cells={int(land_mask_np.sum())}")
    print(f"norm_mean={norm_mean:.5f} norm_std={norm_std:.5f}")

    # Build dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        zarr_path=ZARR_PATH,
        context_len=CONTEXT_LEN,
        horizon=HORIZON,
        batch_size=BATCH_SIZE,
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    land_mask_torch = torch.from_numpy(land_mask_np).to(device)

    print("CUDA available:", torch.cuda.is_available())

    informer_model = ProbSparseInformer(
        height=H,
        width=W,
        context_len=CONTEXT_LEN,
        horizon=HORIZON,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_encoder_layers=N_ENCODER_LAYERS,
        n_decoder_layers=N_DECODER_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        factor=FACTOR,
        attention_type="full"
    ).to(device)
    print("Device:", next(informer_model.parameters()).device)

    informer_params = sum(p.numel() for p in informer_model.parameters())
    print(f"ProbSparse Informer model parameters: {informer_params:,}")

    criterion = nn.MSELoss()
    informer_optimiser = optim.Adam(
        informer_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ProbSparse Informer Train
    print("\nTraining the ProbSparse Informer")
    informer_train_losses, informer_val_losses = train_model(
        model=informer_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=informer_optimiser,
        device=device,
        num_epochs=NUM_EPOCHS,
        land_mask=land_mask_torch,
        grad_clip=GRAD_CLIP,
    )

    # Evaluate ProbSparse Informer on test set
    print("\nEvaluating the ProbSparse Informer")
    informer_preds_norm, informer_targets_norm = predict(
        model=informer_model,
        data_loader=test_loader,
        device=device,
    )

    # Denormalise to °C
    informer_preds_celsius = informer_preds_norm * norm_std + norm_mean
    informer_targets_celsius = informer_targets_norm * norm_std + norm_mean

    informer_rmse_per_step = rmse_per_step(
        informer_preds_celsius, informer_targets_celsius, land_mask=land_mask_np,
    )
    informer_rmse_mean = float(informer_rmse_per_step.mean())
    informer_mae = mae(informer_preds_celsius, informer_targets_celsius, land_mask=land_mask_np)

    # Persistence baseline
    print("\nEvaluating the persistence baseline...")
    test_X_norm = []
    test_y_norm = []
    for batch_X, batch_y in test_loader:
        test_X_norm.append(batch_X.numpy())
        test_y_norm.append(batch_y.numpy())
    test_X_norm = np.concatenate(test_X_norm, axis=0)
    test_y_norm = np.concatenate(test_y_norm, axis=0)

    persistence_preds_norm = persistence_forecast(test_X_norm, horizon=HORIZON)

    persistence_preds_celsius = persistence_preds_norm * norm_std + norm_mean
    persistence_targets_celsius = test_y_norm * norm_std + norm_mean

    persistence_rmse_per_step = rmse_per_step(
        persistence_preds_celsius, persistence_targets_celsius, land_mask=land_mask_np,
    )
    persistence_rmse_mean = float(persistence_rmse_per_step.mean())
    persistence_mae = mae(
        persistence_preds_celsius, persistence_targets_celsius, land_mask=land_mask_np,
    )

    # RNN Baseline
    print("\nPreparing the RNN baseline...")
    rnn_model = RNN(
        H=H,
        W=W,
        context_len=CONTEXT_LEN,
        horizon=HORIZON,
        d_spatial=D_SPATIAL,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    print("Device:", next(rnn_model.parameters()).device)

    n_rnn_params = sum(p.numel() for p in rnn_model.parameters())
    print(f"Model parameters: {n_rnn_params:,}")

    rnn_optimiser = optim.Adam(
        rnn_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Train RNN
    print("\nTraining the RNN Baseline")
    train_model(
        model=rnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=rnn_optimiser,
        device=device,
        num_epochs=NUM_EPOCHS,
        land_mask=land_mask_torch,
        grad_clip=GRAD_CLIP,
    )

    # Evaluate RNN on test set
    print("\nEvaluating the RNN Baseline")
    rnn_preds_norm, rnn_targets_norm = predict(
        model=rnn_model,
        data_loader=test_loader,
        device=device,
    )

    # Denormalise to °C
    rnn_preds_celsius = (rnn_preds_norm * norm_std) + norm_mean
    rnn_targets_celsius = (rnn_targets_norm * norm_std) + norm_mean

    rnn_rmse_per_step = rmse_per_step(
        rnn_preds_celsius, 
        rnn_targets_celsius, 
        land_mask=land_mask_np,
    )
    rnn_rmse_mean = float(rnn_rmse_per_step.mean())
    rnn_mae = mae(
        rnn_preds_celsius, 
        rnn_targets_celsius, 
        land_mask=land_mask_np
    )

    # Skill scores
    rmse_persistence_skill = skill_score(informer_rmse_mean, persistence_rmse_mean)
    mae_persistence_skill = skill_score(informer_mae, persistence_mae)
    rmse_rnn_skill = skill_score(informer_rmse_mean, rnn_rmse_mean)
    mae_rnn_skill = skill_score(informer_mae, rnn_mae)

    # Summary
    print("\nProbSparse Attention Encoder-Decoder Informer Experiment Summary")
    print("-----------------------")
    print(f"Context length: {CONTEXT_LEN}")
    print(f"Forecast horizon: {HORIZON}")
    print(f"Train/Val/Test batches: {len(train_loader)} / {len(val_loader)} / {len(test_loader)}")
    print(f"Device: {device}")

    print("\nProbSparse Informer (test, °C):")
    for h, r in enumerate(informer_rmse_per_step, start=1):
        print(f"  RMSE day {h}: {r:.4f}")
    print(f"  RMSE mean:  {informer_rmse_mean:.4f}")
    print(f"  MAE  mean:  {informer_mae:.4f}")

    print("\nPersistence (test, °C):")
    for h, r in enumerate(persistence_rmse_per_step, start=1):
        print(f"  RMSE day {h}: {r:.4f}")
    print(f"  RMSE mean:  {persistence_rmse_mean:.4f}")
    print(f"  MAE  mean:  {persistence_mae:.4f}")

    print("\nRNN (test, °C):")
    for h, r in enumerate(rnn_rmse_per_step, start=1):
        print(f"  RMSE day {h}: {r:.4f}")
    print(f"  RMSE mean:  {rnn_rmse_mean:.4f}")
    print(f"  MAE  mean:  {rnn_mae:.4f}")

    print("\nSkill vs persistence:")
    print(f"  RMSE skill: {rmse_persistence_skill:.4f}")
    print(f"  MAE  skill: {mae_persistence_skill:.4f}")

    print("\nSkill vs RNN:")
    print(f"  RMSE skill: {rmse_rnn_skill:.4f}")
    print(f"  MAE  skill: {mae_rnn_skill:.4f}")


if __name__ == "__main__":
    main()