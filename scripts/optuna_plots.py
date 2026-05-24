"""
Shared Optuna visualisation helper.

Called at the end of every HPO script to dump study plots to experiments/.
Requires plotly: pip install plotly kaleido
Falls back gracefully if not installed.

Usage
-----
    from scripts.optuna_plots import save_study_plots
    save_study_plots(study, model_name="lstm")
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR      = PROJECT_ROOT / "experiments"


def save_study_plots(study, model_name: str) -> None:
    """Save optimisation history, param importance, and parallel coordinates."""
    try:
        import plotly.io as pio
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
        )
    except ImportError:
        print("plotly / kaleido not installed — skipping Optuna plots. "
              "Run: pip install plotly kaleido")
        return

    completed = [t for t in study.trials
                 if t.state.name == "COMPLETE"]
    if len(completed) < 2:
        print("Not enough completed trials to plot — skipping.")
        return

    plots = {
        f"optuna_{model_name}_history.png":    plot_optimization_history(study),
        f"optuna_{model_name}_importance.png": plot_param_importances(study),
        f"optuna_{model_name}_parallel.png":   plot_parallel_coordinate(study),
    }

    for filename, fig in plots.items():
        path = OUT_DIR / filename
        try:
            pio.write_image(fig, str(path), format="png", width=1200, height=600)
            print(f"Saved: {path}")
        except Exception as e:
            print(f"Could not save {filename}: {e}")
