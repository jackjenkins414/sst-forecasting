"""
Per-model diagnostic report.

Reads artifacts from experiments/best_<model>/ (produced by retrain_best.py)
and generates a single PNG with 8 panels:

  Row 1: loss curves | RMSE/step vs persistence | skill/step | metrics table
  Row 2: true SST    | predicted SST            | error map  | best HP config

Usage
-----
    python scripts/report_model.py --model tubelet
    python scripts/report_model.py --model lstm
    python scripts/report_model.py --all          # generate for all available models
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import zarr

ZARR_PATH   = PROJECT_ROOT / "data/processed/oisst_coralsea.zarr"
BEST_DIR    = PROJECT_ROOT / "experiments"
HORIZON     = 7
DAYS        = np.arange(1, HORIZON + 1)

MODEL_COLOURS = {
    "tubelet":     "#4e79a7",
    "lstm":        "#f28e2b",
    "convlstm":    "#59a14f",
    "informer":    "#76b7b2",
    "transformer": "#b07aa1",
    "patch_transformer": "#edc948",
}

# HP keys to display in the config panel (label, key)
HP_DISPLAY = [
    ("Learning rate",  "learning_rate"),
    ("Dropout",        "dropout"),
    ("Anomaly α",      "anomaly_alpha"),
    ("LR decay",       "lr_factor"),
    ("Epochs",         "num_epochs"),
    # model-specific
    ("d_model",        "d_model"),
    ("n_heads",        "n_heads"),
    ("n_layers",       "n_layers"),
    ("d_ff",           "d_ff"),
    ("t_s",            "t_s"),
    ("Hidden size",    "hidden_size"),
    ("d_spatial",      "d_spatial"),
    ("Hidden dim",     "hidden_dim"),
    ("Kernel size",    "kernel_size"),
    ("Factor",         "factor"),
    ("Label len",      "label_len"),
    ("FFN dim",        "ffn_dim"),
]

def load_artifacts(model_type: str) -> dict:
    d = BEST_DIR / f"best_{model_type}"
    if not d.exists():
        raise FileNotFoundError(
            f"No artifacts found at {d}. Run retrain_best.py --models {model_type} first."
        )
    preds   = np.load(d / "predictions.npy")   # (N, h, H, W) denormalised
    targets = np.load(d / "targets.npy")
    summary = json.load(open(d / "summary.json"))
    curves  = json.load(open(d / "loss_curves.json"))
    config  = json.load(open(d / "config.json"))
    return dict(preds=preds, targets=targets, summary=summary,
                curves=curves, config=config)

def _mask(arr: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    """Set land pixels to NaN for display."""
    out = arr.copy().astype(float)
    out[~land_mask.astype(bool)] = np.nan
    return out

def generate_report(model_type: str, land_mask: np.ndarray,
                    lat: np.ndarray, lon: np.ndarray):
    colour = MODEL_COLOURS.get(model_type, "#9c9c9c")
    art    = load_artifacts(model_type)
    preds, targets = art["preds"], art["targets"]
    s, c           = art["summary"], art["config"]

    example_idx  = s["example_idx"]
    example_date = s.get("example_date", f"test window {example_idx}")
    rmse_steps   = np.array(s["rmse_steps"])
    pers_steps   = np.array(s["pers_rmse_steps"])
    skill_steps  = np.array(s["skill_steps"])
    train_loss   = art["curves"]["train"]
    val_loss     = art["curves"]["val"]

    # Example snapshot: day-7 prediction for fixed test window
    true_d7 = _mask(targets[example_idx, 6], land_mask)   # day 7
    pred_d7 = _mask(preds[example_idx,   6], land_mask)
    err_d7  = _mask(preds[example_idx,   6] - targets[example_idx, 6], land_mask)

    fig = plt.figure(figsize=(22, 11))
    fig.suptitle(
        f"{model_type.upper()} - Best HPO config  "
        f"({s['n_params']:,} params  ·  RMSE {s['mean_rmse']:.4f} °C  ·  "
        f"skill {s['mean_skill']:.4f})",
        fontsize=14, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        hspace=0.40, wspace=0.35,
        left=0.05, right=0.97, top=0.93, bottom=0.05,
    )

    # --- Row 0, Col 0: loss curves ---
    ax_loss = fig.add_subplot(gs[0, 0])
    epochs = np.arange(1, len(train_loss) + 1)
    ax_loss.plot(epochs, train_loss, color=colour, linewidth=1.8, label="train")
    ax_loss.plot(epochs, val_loss,   color=colour, linewidth=1.8,
                 linestyle="--", label="val")
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training curves")
    ax_loss.legend(fontsize=8); ax_loss.grid(alpha=0.3)

    # --- Row 0, Col 1: RMSE per step ---
    ax_rmse = fig.add_subplot(gs[0, 1])
    ax_rmse.plot(DAYS, pers_steps,  "k--", linewidth=1.6, label="Persistence")
    ax_rmse.plot(DAYS, rmse_steps,  color=colour, linewidth=2.2,
                 marker="o", markersize=5, label=model_type)
    ax_rmse.set_xlabel("Forecast day"); ax_rmse.set_ylabel("RMSE (°C)")
    ax_rmse.set_title("RMSE per forecast day")
    ax_rmse.legend(fontsize=8); ax_rmse.grid(alpha=0.3)

    # --- Row 0, Col 2: skill per step ---
    ax_skill = fig.add_subplot(gs[0, 2])
    ax_skill.axhline(0, color="black", linewidth=1.0, linestyle="--")
    ax_skill.plot(DAYS, skill_steps, color=colour, linewidth=2.2,
                  marker="o", markersize=5)
    ax_skill.set_xlabel("Forecast day"); ax_skill.set_ylabel("Skill vs persistence")
    ax_skill.set_title("Skill score per forecast day")
    ax_skill.grid(alpha=0.3)

    # --- Row 0, Col 3: metrics table ---
    ax_tbl = fig.add_subplot(gs[0, 3])
    ax_tbl.axis("off")
    rows = [
        ["Mean RMSE",   f"{s['mean_rmse']:.4f} °C"],
        ["Mean skill",  f"{s['mean_skill']:.4f}"],
        ["BIC",         f"{s['bic']:.0f}"],
        ["Parameters",  f"{s['n_params']:,}"],
        ["Epochs run",  str(s["epochs_trained"])],
        ["Day-1 RMSE",  f"{rmse_steps[0]:.4f} °C"],
        ["Day-7 RMSE",  f"{rmse_steps[6]:.4f} °C"],
        ["Day-1 skill", f"{skill_steps[0]:.4f}"],
        ["Day-7 skill", f"{skill_steps[6]:.4f}"],
    ]
    tbl = ax_tbl.table(
        cellText=rows, colLabels=["Metric", "Value"],
        cellLoc="left", loc="upper center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, _), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(fontweight="bold")
    ax_tbl.set_title("Metrics", fontsize=10)

    # --- Row 1: heatmaps ---
    vmin = np.nanmin([true_d7, pred_d7])
    vmax = np.nanmax([true_d7, pred_d7])
    cmap_sst = "RdYlBu_r"
    err_abs   = np.nanmax(np.abs(err_d7))

    # lat is stored south-to-north (lat[0]=southernmost). Use origin="lower"
    # with extent[bottom,top]=[lat[0], lat[-1]] so north sits at the top.
    extent = [lon[0], lon[-1], lat[0], lat[-1]]

    ax_true = fig.add_subplot(gs[1, 0])
    im1 = ax_true.imshow(true_d7, origin="lower", aspect="auto",
                         extent=extent, cmap=cmap_sst, vmin=vmin, vmax=vmax)
    ax_true.set_title(f"True SST - day 7\n{example_date}", fontsize=9)
    ax_true.set_xlabel("Lon"); ax_true.set_ylabel("Lat")
    plt.colorbar(im1, ax=ax_true, fraction=0.04, pad=0.04).set_label("°C", fontsize=8)

    ax_pred = fig.add_subplot(gs[1, 1])
    im2 = ax_pred.imshow(pred_d7, origin="lower", aspect="auto",
                         extent=extent, cmap=cmap_sst, vmin=vmin, vmax=vmax)
    ax_pred.set_title(f"Predicted SST - day 7\n{model_type}", fontsize=9)
    ax_pred.set_xlabel("Lon"); ax_pred.set_ylabel("Lat")
    plt.colorbar(im2, ax=ax_pred, fraction=0.04, pad=0.04).set_label("°C", fontsize=8)

    ax_err = fig.add_subplot(gs[1, 2])
    im3 = ax_err.imshow(err_d7, origin="lower", aspect="auto",
                        extent=extent, cmap="RdBu_r",
                        vmin=-err_abs, vmax=err_abs)
    ax_err.set_title("Error (Pred − True) - day 7", fontsize=9)
    ax_err.set_xlabel("Lon"); ax_err.set_ylabel("Lat")
    plt.colorbar(im3, ax=ax_err, fraction=0.04, pad=0.04).set_label("°C", fontsize=8)

    # --- Row 1, Col 3: best HP config ---
    ax_hp = fig.add_subplot(gs[1, 3])
    ax_hp.axis("off")
    hp_rows = []
    for label, key in HP_DISPLAY:
        val = c.get(key)
        if val is not None:
            if isinstance(val, float):
                val = f"{val:.4g}"
            hp_rows.append([label, str(val)])
    tbl2 = ax_hp.table(
        cellText=hp_rows, colLabels=["Hyperparameter", "Value"],
        cellLoc="left", loc="upper center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl2.auto_set_font_size(False); tbl2.set_fontsize(9)
    for (r, _), cell in tbl2.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(fontweight="bold")
    ax_hp.set_title("Best HP config", fontsize=10)

    out_path = BEST_DIR / f"report_{model_type}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=list(MODEL_COLOURS),
                       help="Generate report for one model")
    group.add_argument("--all",   action="store_true",
                       help="Generate reports for all models with available artifacts")
    args = parser.parse_args()

    root      = zarr.open_group(str(ZARR_PATH), mode="r")
    land_mask = np.array(root["land_mask"])
    lat       = np.array(root["lat"])
    lon       = np.array(root["lon"])

    models = (list(MODEL_COLOURS) if args.all
              else [args.model])

    for mt in models:
        best_dir = BEST_DIR / f"best_{mt}"
        if args.all and not best_dir.exists():
            print(f"Skipping {mt} - no artifacts (run retrain_best.py first)")
            continue
        print(f"Generating report for {mt}...")
        generate_report(mt, land_mask, lat, lon)

if __name__ == "__main__":
    main()
