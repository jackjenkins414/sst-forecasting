#!/usr/bin/env python3
"""Generate per-grid-cell RMSE heatmap for a trained ConvLSTM or LSTM checkpoint."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import zarr
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from sst_forecasting.data.dataset import SSTWindowDataset
from sst_forecasting.models.convlstm import SpatialConvLSTM
from sst_forecasting.models.lstm import SpatialFlatLSTM

RESULTS = Path(__file__).parent.parent / "experiments" / "results"
FIGURES = Path(__file__).parent.parent / "report" / "figures"


def build_model(cfg, H, W):
    model_name = cfg["args"]["model"]
    horizon    = cfg["args"]["horizon"]
    ctx        = cfg["args"]["context_len"]
    dropout    = cfg["args"].get("dropout", 0.1)
    if model_name == "convlstm":
        return SpatialConvLSTM(
            H=H, W=W,
            context_len=ctx,
            horizon=horizon,
            hidden_channels=cfg["args"]["convlstm_hidden"],
            kernel_size=cfg["args"].get("convlstm_kernel", 3),
            dropout=dropout,
            checkpoint_segments=0,
        )
    return SpatialFlatLSTM(
        H=H, W=W,
        context_len=ctx,
        horizon=horizon,
        d_spatial=cfg["args"].get("lstm_d_spatial", 64),
        hidden_size=cfg["args"].get("lstm_hidden", 128),
        num_layers=cfg["args"].get("lstm_layers", 2),
        dropout=dropout,
    )


@torch.no_grad()
def run_inference(model, zarr_path, context_len, horizon, batch_size, device):
    root = zarr.open(zarr_path, mode="r")
    norm_std = float(root.attrs["norm_std"])

    ds = SSTWindowDataset(zarr_path, split="test", context_len=context_len, horizon=horizon)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    land_mask = torch.from_numpy(np.array(root["land_mask"]))  # (H, W) bool
    H, W = land_mask.shape

    sum_sq_err = torch.zeros(horizon, H, W)
    count      = torch.zeros(horizon, H, W)

    model.eval()
    model.to(device)

    for x, y in loader:
        x = x.to(device)
        pred = model(x).cpu()  # (B, h, H, W) normalised
        pred_c = pred * norm_std
        y_c    = y    * norm_std
        err2   = (pred_c - y_c) ** 2
        sum_sq_err += err2.sum(dim=0)
        count += (~torch.isnan(y_c)).float().sum(dim=0)

    rmse_grid = torch.sqrt(sum_sq_err / count.clamp(min=1))  # (h, H, W)
    rmse_mean = rmse_grid.mean(dim=0)                        # (H, W)
    rmse_mean[~land_mask] = float("nan")

    return rmse_mean.numpy(), norm_std, root


def plot_heatmap(rmse_grid, root, title, out_path):
    lat = np.array(root["lat"])
    lon = np.array(root["lon"])

    fig, ax = plt.subplots(figsize=(9, 5))
    img = ax.imshow(
        rmse_grid,
        origin="lower",
        extent=[lon.min(), lon.max(), lat.min(), lat.max()],
        aspect="auto",
        cmap="YlOrRd",
    )
    cbar = fig.colorbar(img, ax=ax, label="RMSE (°C)")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir",  required=True,
                   help="Experiment results dir containing best_model.pt and run.yaml")
    p.add_argument("--zarr-path",  default="data/processed/oisst_coralsea.zarr")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out-dir",    default=str(FIGURES))
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    out       = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "run.yaml") as f:
        cfg = yaml.safe_load(f)

    root = zarr.open(args.zarr_path, mode="r")
    sst  = np.array(root["sst"])
    H, W = sst.shape[1], sst.shape[2]

    model = build_model(cfg, H, W)
    state = torch.load(model_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(state)

    context_len = cfg["args"]["context_len"]
    horizon     = cfg["args"]["horizon"]
    model_name  = cfg["args"]["model"]

    rmse_grid, norm_std, root = run_inference(
        model, args.zarr_path, context_len, horizon,
        args.batch_size, args.device
    )

    out_name  = f"heatmap_{model_name}_rmse_h{horizon}.png"
    title     = f"{model_name.upper()} — per-grid-cell mean RMSE (°C), h={horizon}, test 1999–2000"
    plot_heatmap(rmse_grid, root, title, out / out_name)


if __name__ == "__main__":
    main()
