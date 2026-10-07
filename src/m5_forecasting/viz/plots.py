"""Publication-ready forecast charts for reports / Tableau companion."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_model_comparison(metrics: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9, 5))
    order = metrics.sort_values("rmse")
    sns.barplot(data=order, x="model", y="rmse", hue="model", legend=False, ax=ax, palette="deep")
    ax.set_title("Holdout RMSE by Model")
    ax.set_xlabel("")
    ax.set_ylabel("RMSE")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_forecast_vs_actual(
    forecasts: pd.DataFrame,
    series_id: str,
    out_path: str | Path,
    model: str = "lightgbm",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub = forecasts[(forecasts["series_id"] == series_id) & (forecasts["model"] == model)].copy()
    sub = sub.sort_values("date")
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(sub["date"], sub["y_true"], label="Actual", linewidth=2.2, color="#1f4e79")
    ax.plot(sub["date"], sub["y_pred"], label="Forecast", linewidth=2.2, color="#c55a11")
    ax.set_title(f"{model} — {series_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units sold")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
