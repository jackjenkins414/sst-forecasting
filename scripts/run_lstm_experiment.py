import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr

from src.data.preprocess import (
    create_ocean_mask,
    flatten_to_ocean_points,
    standardise_sst,
)
from src.data.windowing import create_sliding_windows
from src.data.splitting import chronological_train_val_test_split
from src.data.dataloaders import create_dataloaders

from src.models.lstm import SimpleSstLSTM

from src.training.train import train_model
from src.training.evaluate import predict

from src.baselines.persistence import persistence_forecast
from src.utils.metrics import rmse, mae, skill_score


# Configuration

DATA_PATH = PROJECT_ROOT / "data/processed/oisst_australia_2025_lowres.nc"

INPUT_LENGTH = 30
FORECAST_HORIZON = 1
BATCH_SIZE = 16
HIDDEN_SIZE = 128
NUM_LAYERS = 1
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

RANDOM_SEED = 42


def main():
    # Seeds 
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


    # Load data
    ds_lowres = xr.open_dataset(DATA_PATH)
    sst = ds_lowres["sst"]


    # Preprocess
    ocean_mask = create_ocean_mask(sst)
    sst_ocean = flatten_to_ocean_points(sst, ocean_mask)
    sst_scaled, sst_mean, sst_std = standardise_sst(sst_ocean)

    # Create supervised samples
    X, y = create_sliding_windows(
        sst_scaled,
        input_length=INPUT_LENGTH,
        forecast_horizon=FORECAST_HORIZON,
    )

    X_train, y_train, X_val, y_val, X_test, y_test = chronological_train_val_test_split(
        X,
        y,
        train_fraction=0.70,
        val_fraction=0.15,
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        batch_size=BATCH_SIZE,
    )


    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = X_train.shape[2]
    output_size = y_train.shape[1]

    model = SimpleSstLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=output_size,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    # Train
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
    )

    # Evaluate LSTM
    test_preds_scaled, test_targets_scaled = predict(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    test_preds_celsius = test_preds_scaled * sst_std + sst_mean
    test_targets_celsius = test_targets_scaled * sst_std + sst_mean

    lstm_rmse = rmse(test_preds_celsius, test_targets_celsius)
    lstm_mae = mae(test_preds_celsius, test_targets_celsius)

    # Persistence baseline
    print("Evaluating persistence baseline...")
    persistence_preds_scaled = persistence_forecast(X_test)

    persistence_preds_celsius = persistence_preds_scaled * sst_std + sst_mean
    persistence_targets_celsius = y_test * sst_std + sst_mean

    persistence_rmse = rmse(persistence_preds_celsius, persistence_targets_celsius)
    persistence_mae = mae(persistence_preds_celsius, persistence_targets_celsius)

    # Skill scores
    rmse_skill = skill_score(lstm_rmse, persistence_rmse)
    mae_skill = skill_score(lstm_mae, persistence_mae)

    # Summary
    print("\nLSTM Experiment Summary")
    print("-----------------------")
    print(f"Input length: {INPUT_LENGTH}")
    print(f"Forecast horizon: {FORECAST_HORIZON}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Train/Val/Test: {len(X_train)} / {len(X_val)} / {len(X_test)}")
    print(f"Device: {device}")

    print("\nLSTM:")
    print(f"RMSE: {lstm_rmse:.4f} °C")
    print(f"MAE:  {lstm_mae:.4f} °C")

    print("\nPersistence:")
    print(f"RMSE: {persistence_rmse:.4f} °C")
    print(f"MAE:  {persistence_mae:.4f} °C")

    print("\nSkill vs persistence:")
    print(f"RMSE skill: {rmse_skill:.4f}")
    print(f"MAE skill:  {mae_skill:.4f}")


if __name__ == "__main__":
    main()