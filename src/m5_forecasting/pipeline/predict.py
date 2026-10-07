"""Batch prediction + evaluation + Tableau-ready exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from m5_forecasting.config import ensure_dirs
from m5_forecasting.pipeline.metrics import summarize_forecasts
from m5_forecasting.pipeline.prepare import load_prepared
from m5_forecasting.pipeline.story_dashboard import export_customer_story
from m5_forecasting.viz.plots import plot_forecast_vs_actual, plot_model_comparison


def _load_registry(cfg: dict[str, Any]) -> dict[str, Any]:
    models_dir = Path(cfg["paths"]["models_dir"])
    meta_path = models_dir / "train_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("No trained models found. Run the train step first.")
    meta = json.loads(meta_path.read_text())
    registry: dict[str, Any] = {"_meta": meta}
    for name, info in meta.get("models", {}).items():
        registry[name] = joblib.load(info["path"])
    return registry


def run_batch_predict(cfg: dict[str, Any]) -> dict[str, Path]:
    """Score holdout, write forecasts / metrics / Tableau extracts / figures."""
    ensure_dirs(cfg)
    data = load_prepared(cfg)
    registry = _load_registry(cfg)
    meta = registry["_meta"]

    models_dir = Path(cfg["paths"]["models_dir"])
    test_path = models_dir / "test_series.parquet"
    if not test_path.exists():
        raise FileNotFoundError("Missing test_series.parquet — re-run train.")
    future_series = pd.read_parquet(test_path)
    future_series["y_true"] = future_series["sales"]

    frames: list[pd.DataFrame] = []
    keep = ["series_id", "store_id", "item_id", "cat_id", "date", "y_true", "y_pred", "model"]

    for name in meta.get("enabled", []):
        model = registry.get(name)
        if model is None:
            continue
        pred = model.predict(future_series)
        pred = pred.drop(columns=["y_true"], errors="ignore").merge(
            future_series[["series_id", "date", "y_true"]],
            on=["series_id", "date"],
            how="left",
        )
        frames.append(pred[[c for c in keep if c in pred.columns]])

    forecasts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    metrics = summarize_forecasts(forecasts, history=data["train"])

    # Paths
    forecasts_dir = Path(cfg["paths"]["forecasts_dir"])
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    tableau_dir = Path(cfg["paths"]["tableau_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    for d in (forecasts_dir, metrics_dir, tableau_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    paths = {
        "forecasts": forecasts_dir / "holdout_forecasts.parquet",
        "forecasts_csv": forecasts_dir / "holdout_forecasts.csv",
        "metrics": metrics_dir / "model_comparison.csv",
        "metrics_json": metrics_dir / "model_comparison.json",
        "tableau_forecasts": tableau_dir / "forecasts_daily.csv",
        "tableau_metrics": tableau_dir / "model_metrics.csv",
        "tableau_audit": tableau_dir / "cleaning_audit.csv",
        "tableau_importance": tableau_dir / "feature_importance.csv",
        "fig_rmse": figures_dir / "model_rmse.png",
    }

    forecasts.to_parquet(paths["forecasts"], index=False)
    forecasts.to_csv(paths["forecasts_csv"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    paths["metrics_json"].write_text(metrics.to_json(orient="records", indent=2))

    # Tableau extracts (flat CSVs)
    tab = forecasts.copy()
    tab["date"] = pd.to_datetime(tab["date"]).dt.strftime("%Y-%m-%d")
    tab["error"] = tab["y_true"] - tab["y_pred"]
    tab["abs_error"] = tab["error"].abs()
    tab.to_csv(paths["tableau_forecasts"], index=False)
    metrics.to_csv(paths["tableau_metrics"], index=False)
    data["cleaning_audit"].to_csv(paths["tableau_audit"], index=False)

    fi_src = Path(cfg["paths"]["models_dir"]) / "lightgbm_feature_importance.csv"
    if fi_src.exists():
        fi = pd.read_csv(fi_src)
        fi.to_csv(paths["tableau_importance"], index=False)

    if len(metrics):
        plot_model_comparison(metrics, paths["fig_rmse"])
        # Plot best non-baseline series for lightgbm if present
        if "lightgbm" in forecasts["model"].values:
            top_series = (
                forecasts[forecasts["model"] == "lightgbm"]
                .groupby("series_id")["y_true"]
                .sum()
                .sort_values(ascending=False)
                .index[0]
            )
            paths["fig_forecast"] = figures_dir / "lightgbm_forecast_example.png"
            plot_forecast_vs_actual(
                forecasts, series_id=top_series, out_path=paths["fig_forecast"], model="lightgbm"
            )

    # Merchandising-oriented rollup: next-horizon demand by store × category
    if {"store_id", "model"}.issubset(forecasts.columns):
        rollup = (
            forecasts.groupby(["model", "store_id", "date"], as_index=False)
            .agg(forecast_units=("y_pred", "sum"), actual_units=("y_true", "sum"))
        )
        paths["tableau_store_daily"] = tableau_dir / "store_daily_rollup.csv"
        rollup.to_csv(paths["tableau_store_daily"], index=False)

    summary = {
        "n_forecast_rows": int(len(forecasts)),
        "models": metrics.to_dict(orient="records"),
        "best_model": metrics.iloc[0]["model"] if len(metrics) else None,
    }
    (metrics_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    paths["run_summary"] = metrics_dir / "run_summary.json"

    # Walmart customer storytelling dashboard (champion model only)
    if len(metrics) and len(forecasts):
        story_paths = export_customer_story(cfg, forecasts, metrics, future_series)
        paths.update({f"story_{k}" if not k.startswith("story") else k: v for k, v in story_paths.items()})

    return paths
