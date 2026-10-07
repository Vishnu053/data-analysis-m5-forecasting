"""Ingest → clean → feature → persist processed datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from m5_forecasting.config import ensure_dirs
from m5_forecasting.data.clean import clean_sales
from m5_forecasting.data.load import load_raw_tables, melt_sales
from m5_forecasting.features.build import build_feature_matrix
from m5_forecasting.pipeline.splits import time_split


def prepare_datasets(cfg: dict[str, Any]) -> dict[str, Path]:
    """Run data prep and write parquet artifacts + cleaning audit."""
    ensure_dirs(cfg)
    processed = Path(cfg["paths"]["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)

    tables = load_raw_tables(cfg)
    long = melt_sales(tables["sales"], tables["calendar"])
    cleaned, audit = clean_sales(long, cfg)

    features = build_feature_matrix(
        cleaned,
        tables["calendar"],
        tables["sell_prices"],
        tables["weather"],
        cfg,
    )

    train, valid, test = time_split(features, cfg)

    paths = {
        "sales_long": processed / "sales_long.parquet",
        "features": processed / "features.parquet",
        "train": processed / "train.parquet",
        "valid": processed / "valid.parquet",
        "test": processed / "test.parquet",
        "cleaning_audit": processed / "cleaning_audit.csv",
        "prep_manifest": processed / "prep_manifest.json",
    }

    cleaned.to_parquet(paths["sales_long"], index=False)
    features.to_parquet(paths["features"], index=False)
    train.to_parquet(paths["train"], index=False)
    valid.to_parquet(paths["valid"], index=False)
    test.to_parquet(paths["test"], index=False)
    audit.to_csv(paths["cleaning_audit"], index=False)

    manifest = {
        "n_rows_raw_long": int(len(long)),
        "n_rows_features": int(len(features)),
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "n_test": int(len(test)),
        "n_series": int(features["series_id"].nunique()),
        "date_min": str(features["date"].min().date()),
        "date_max": str(features["date"].max().date()),
        "negatives_fixed": int(audit["n_negative_fixed"].sum()),
        "outliers_flagged": int(audit["n_outliers"].sum()),
        "stockout_days_flagged": int(audit["n_stockout_days"].sum()),
        "weather_joined": tables["weather"] is not None,
    }
    paths["prep_manifest"].write_text(json.dumps(manifest, indent=2))
    return paths


def load_prepared(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    processed = Path(cfg["paths"]["processed_dir"])
    required = ["train.parquet", "valid.parquet", "test.parquet"]
    if not all((processed / name).exists() for name in required):
        prepare_datasets(cfg)
    return {
        "train": pd.read_parquet(processed / "train.parquet"),
        "valid": pd.read_parquet(processed / "valid.parquet"),
        "test": pd.read_parquet(processed / "test.parquet"),
        "features": pd.read_parquet(processed / "features.parquet"),
        "cleaning_audit": pd.read_csv(processed / "cleaning_audit.csv"),
    }
