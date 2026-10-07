"""Load and reshape M5 / synthetic raw tables."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from m5_forecasting.data.generate_synthetic import generate_synthetic_m5


REQUIRED_RAW = ("calendar.csv", "sell_prices.csv", "sales_train_validation.csv")


def _has_raw(raw_dir: Path) -> bool:
    return all((raw_dir / name).exists() for name in REQUIRED_RAW)


def ensure_raw_data(cfg: dict[str, Any]) -> Path:
    """
    Return directory containing raw CSVs.

    Prefer data/raw; if missing and synthetic fallback is enabled, generate
    into data/sample and copy into data/raw.
    """
    raw_dir = Path(cfg["paths"]["raw_dir"])
    sample_dir = Path(cfg["paths"]["sample_dir"])
    external_dir = Path(cfg["paths"]["external_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    external_dir.mkdir(parents=True, exist_ok=True)

    if _has_raw(raw_dir):
        return raw_dir

    if not cfg.get("data", {}).get("use_synthetic_fallback", True):
        missing = [n for n in REQUIRED_RAW if not (raw_dir / n).exists()]
        raise FileNotFoundError(
            f"Missing raw M5 files in {raw_dir}: {missing}. "
            "Download from https://www.kaggle.com/c/m5-forecasting-accuracy "
            "or enable data.use_synthetic_fallback."
        )

    syn = cfg.get("synthetic", {})
    paths = generate_synthetic_m5(
        sample_dir,
        n_days=int(syn.get("n_days", 400)),
        n_items=int(syn.get("n_items", 12)),
        n_stores=int(syn.get("n_stores", 3)),
        states=syn.get("states"),
        categories=syn.get("categories"),
        start_date=syn.get("start_date", "2015-01-01"),
        seed=int(cfg.get("project", {}).get("seed", 42)),
    )
    for name in REQUIRED_RAW:
        shutil.copy2(sample_dir / name, raw_dir / name)
    # Weather is external enrichment
    weather_src = paths.get("weather_daily")
    if weather_src and Path(weather_src).exists():
        shutil.copy2(weather_src, external_dir / "weather_daily.csv")

    return raw_dir


def load_raw_tables(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Load calendar, prices, sales (+ optional weather)."""
    raw_dir = ensure_raw_data(cfg)
    external_dir = Path(cfg["paths"]["external_dir"])

    calendar = pd.read_csv(raw_dir / "calendar.csv")
    sell_prices = pd.read_csv(raw_dir / "sell_prices.csv")
    sales = pd.read_csv(raw_dir / "sales_train_validation.csv")

    weather_path = external_dir / "weather_daily.csv"
    weather = pd.read_csv(weather_path) if weather_path.exists() else None

    return {
        "calendar": calendar,
        "sell_prices": sell_prices,
        "sales": sales,
        "weather": weather,
    }


def melt_sales(sales: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Convert wide M5 sales (d_1…d_N) into a long panel with dates."""
    id_vars = [c for c in sales.columns if not str(c).startswith("d_")]
    d_cols = [c for c in sales.columns if str(c).startswith("d_")]
    long = sales.melt(id_vars=id_vars, value_vars=d_cols, var_name="d", value_name="sales")
    cal = calendar[["d", "date", "wm_yr_wk", "weekday", "wday", "month", "year"]].copy()
    cal["date"] = pd.to_datetime(cal["date"])
    long = long.merge(cal, on="d", how="left")
    long["sales"] = pd.to_numeric(long["sales"], errors="coerce")
    return long.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
