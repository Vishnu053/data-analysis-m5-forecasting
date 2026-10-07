"""Train enabled models and persist artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from m5_forecasting.config import ensure_dirs
from m5_forecasting.features.build import aggregate_series
from m5_forecasting.features.series_features import add_series_features
from m5_forecasting.models.arima_model import ArimaModel
from m5_forecasting.models.baseline import SeasonalNaiveModel
from m5_forecasting.models.lightgbm_model import LightGBMForecaster
from m5_forecasting.models.lstm_model import LSTMForecaster
from m5_forecasting.pipeline.prepare import load_prepared
from m5_forecasting.pipeline.splits import time_split


def _series_frame(df: pd.DataFrame, level: str) -> pd.DataFrame:
    if level == "item_store":
        out = df.copy()
        if "series_id" not in out.columns:
            out["series_id"] = out["store_id"] + "__" + out["item_id"]
        return out
    return aggregate_series(df, level)


def train_models(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fit all enabled models; return in-memory registry + written paths."""
    ensure_dirs(cfg)
    data = load_prepared(cfg)
    model_cfg = cfg.get("models", {})
    enabled = model_cfg.get("enabled", ["seasonal_naive", "lightgbm"])
    level = model_cfg.get("series_level", "store_cat")

    # Build a single series-level panel (with features) so all models share a grain
    features_item = data["features"]
    series_all = add_series_features(_series_frame(features_item, level))
    train_series, valid_series, test_series = time_split(series_all, cfg)

    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    registry: dict[str, Any] = {}
    meta: dict[str, Any] = {"enabled": enabled, "series_level": level, "models": {}}

    if "seasonal_naive" in enabled:
        m = SeasonalNaiveModel(season_length=7).fit(train_series)
        path = models_dir / "seasonal_naive.joblib"
        joblib.dump(m, path)
        registry["seasonal_naive"] = m
        meta["models"]["seasonal_naive"] = {"path": str(path)}

    if "arima" in enabled:
        a_cfg = model_cfg.get("arima", {})
        m = ArimaModel(
            order=tuple(a_cfg.get("order", [1, 1, 1])),
            seasonal_order=tuple(a_cfg.get("seasonal_order", [1, 0, 1, 7])),
            max_series=int(a_cfg.get("max_series", 12)),
        ).fit(train_series)
        path = models_dir / "arima.joblib"
        joblib.dump(m, path)
        registry["arima"] = m
        meta["models"]["arima"] = {
            "path": str(path),
            "n_fitted_series": len(m._fitted),
        }

    if "lightgbm" in enabled:
        lgb_cfg = model_cfg.get("lightgbm", {})
        m = LightGBMForecaster(params=lgb_cfg, tune=True).fit(train_series, valid_series)
        path = models_dir / "lightgbm.joblib"
        joblib.dump(m, path)
        registry["lightgbm"] = m
        fi_path = models_dir / "lightgbm_feature_importance.csv"
        if m.feature_importance_ is not None:
            m.feature_importance_.to_csv(fi_path, index=False)
        meta["models"]["lightgbm"] = {
            "path": str(path),
            "best_params": m.best_params_,
            "feature_importance": str(fi_path),
        }

    if "lstm" in enabled:
        m = LSTMForecaster(params=model_cfg.get("lstm", {})).fit(train_series)
        path = models_dir / "lstm.joblib"
        joblib.dump(m, path)
        registry["lstm"] = m
        meta["models"]["lstm"] = {
            "path": str(path),
            "n_fitted_series": len(m._models),
        }

    train_series.to_parquet(models_dir / "train_series.parquet", index=False)
    valid_series.to_parquet(models_dir / "valid_series.parquet", index=False)
    test_series.to_parquet(models_dir / "test_series.parquet", index=False)

    meta_path = models_dir / "train_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    registry["_meta"] = meta
    registry["_train_series"] = train_series
    registry["_valid_series"] = valid_series
    registry["_test_series"] = test_series
    return registry
