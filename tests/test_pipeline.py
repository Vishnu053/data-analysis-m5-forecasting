"""Unit / smoke tests for cleaning, features, and a miniaturized pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from m5_forecasting.config import load_config
from m5_forecasting.data.clean import clean_sales
from m5_forecasting.data.generate_synthetic import generate_synthetic_m5
from m5_forecasting.data.load import melt_sales
from m5_forecasting.pipeline.metrics import mae, mape, rmse
from m5_forecasting.pipeline.prepare import prepare_datasets
from m5_forecasting.pipeline.predict import run_batch_predict
from m5_forecasting.pipeline.train import train_models


@pytest.fixture()
def tiny_cfg(tmp_path: Path) -> dict:
    cfg = load_config()
    # Isolate I/O under tmp
    cfg["paths"]["raw_dir"] = str(tmp_path / "raw")
    cfg["paths"]["sample_dir"] = str(tmp_path / "sample")
    cfg["paths"]["external_dir"] = str(tmp_path / "external")
    cfg["paths"]["processed_dir"] = str(tmp_path / "processed")
    cfg["paths"]["output_dir"] = str(tmp_path / "outputs")
    cfg["paths"]["models_dir"] = str(tmp_path / "outputs" / "models")
    cfg["paths"]["forecasts_dir"] = str(tmp_path / "outputs" / "forecasts")
    cfg["paths"]["metrics_dir"] = str(tmp_path / "outputs" / "metrics")
    cfg["paths"]["tableau_dir"] = str(tmp_path / "outputs" / "tableau")
    cfg["paths"]["figures_dir"] = str(tmp_path / "outputs" / "figures")
    cfg["synthetic"]["n_days"] = 120
    cfg["synthetic"]["n_items"] = 4
    cfg["synthetic"]["n_stores"] = 2
    cfg["models"]["enabled"] = ["seasonal_naive", "lightgbm"]
    cfg["models"]["lightgbm"]["n_estimators"] = 40
    cfg["models"]["lightgbm"]["early_stopping_rounds"] = 10
    # Disable full grid for speed
    return cfg


def test_generate_and_melt(tmp_path: Path):
    paths = generate_synthetic_m5(tmp_path, n_days=60, n_items=3, n_stores=2, seed=0)
    sales = pd.read_csv(paths["sales_train_validation"])
    calendar = pd.read_csv(paths["calendar"])
    long = melt_sales(sales, calendar)
    assert len(long) == 3 * 2 * 60
    assert set(["store_id", "item_id", "date", "sales"]).issubset(long.columns)


def test_clean_flags_negatives_and_outliers():
    dates = pd.date_range("2015-01-01", periods=30, freq="D")
    sales = [5.0] * 30
    sales[5] = -3.0
    sales[10] = 200.0
    df = pd.DataFrame(
        {
            "store_id": ["CA_1"] * 30,
            "item_id": ["FOODS_001"] * 30,
            "date": dates,
            "sales": sales,
            "d": [f"d_{i+1}" for i in range(30)],
        }
    )
    cleaned, audit = clean_sales(df, {"cleaning": {"iqr_multiplier": 3.0, "winsorize": True}})
    assert cleaned["is_negative_fixed"].sum() == 1
    assert cleaned.loc[cleaned["is_negative_fixed"], "sales"].iloc[0] == 0
    assert audit.iloc[0]["n_outliers"] >= 1
    assert cleaned["sales"].max() < 200


def test_metrics():
    y = [1.0, 2.0, 3.0]
    p = [1.0, 2.0, 4.0]
    assert rmse(y, p) == pytest.approx((1 / 3) ** 0.5)
    assert mae(y, p) == pytest.approx(1 / 3)
    assert mape(y, p) > 0


def test_end_to_end_smoke(tiny_cfg: dict):
    # Force lightgbm not to grid-search heavily: patch via few estimators only
    from m5_forecasting.models import lightgbm_model as lgbm_mod

    original = lgbm_mod.LightGBMForecaster._candidate_grid

    def tiny_grid(self):
        return [
            {
                "objective": "regression",
                "n_estimators": 30,
                "learning_rate": 0.1,
                "num_leaves": 15,
                "min_child_samples": 5,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": 42,
                "n_jobs": 1,
                "verbosity": -1,
            }
        ]

    lgbm_mod.LightGBMForecaster._candidate_grid = tiny_grid
    try:
        prepare_datasets(tiny_cfg)
        train_models(tiny_cfg)
        paths = run_batch_predict(tiny_cfg)
    finally:
        lgbm_mod.LightGBMForecaster._candidate_grid = original

    metrics = pd.read_csv(paths["metrics"])
    assert set(metrics["model"]).issuperset({"seasonal_naive", "lightgbm"})
    assert Path(paths["tableau_forecasts"]).exists()
    assert Path(paths["fig_rmse"]).exists()
