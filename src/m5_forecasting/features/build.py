"""Feature engineering: lags, rolling stats, SNAP, prices, events, weather."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _attach_calendar_events(df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "d",
        "event_name_1",
        "event_type_1",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    present = [c for c in cols if c in calendar.columns]
    cal = calendar[present].copy()
    out = df.merge(cal, on="d", how="left")
    out["has_event"] = out.get("event_name_1", pd.Series(index=out.index)).notna().astype(int)
    snap = np.zeros(len(out), dtype=int)
    for state, col in [("CA", "snap_CA"), ("TX", "snap_TX"), ("WI", "snap_WI")]:
        if col in out.columns and "state_id" in out.columns:
            mask = out["state_id"].eq(state)
            snap = np.where(mask, out[col].fillna(0).astype(int), snap)
    out["snap"] = snap
    return out


def _attach_prices(df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    p = prices.copy()
    out = df.merge(p, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    out["sell_price"] = out.groupby(["store_id", "item_id"])["sell_price"].ffill().bfill()
    out["price_change"] = out.groupby(["store_id", "item_id"])["sell_price"].pct_change()
    out["price_change"] = out["price_change"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _attach_weather(df: pd.DataFrame, weather: pd.DataFrame | None) -> pd.DataFrame:
    if weather is None or weather.empty:
        df = df.copy()
        df["avg_temp_f"] = np.nan
        df["precip_in"] = 0.0
        df["is_severe_weather"] = 0
        return df
    w = weather.copy()
    w["date"] = pd.to_datetime(w["date"])
    return df.merge(w, on=["date", "state_id"], how="left")


def _add_lag_rolling(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    feat_cfg = cfg.get("features", {})
    lags = feat_cfg.get("lags", [1, 7, 14, 28])
    windows = feat_cfg.get("rolling_windows", [7, 14, 28])
    group_cols = ["store_id", "item_id"]

    out = df.sort_values(group_cols + ["date"]).copy()
    g = out.groupby(group_cols, sort=False)["sales"]

    for lag in lags:
        out[f"lag_{lag}"] = g.shift(lag)

    for w in windows:
        shifted = g.shift(1)
        out[f"roll_mean_{w}"] = shifted.transform(lambda s, ww=w: s.rolling(ww, min_periods=1).mean())
        out[f"roll_std_{w}"] = shifted.transform(lambda s, ww=w: s.rolling(ww, min_periods=1).std())
        out[f"roll_std_{w}"] = out[f"roll_std_{w}"].fillna(0.0)

    return out


def _add_calendar_parts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dow"] = out["date"].dt.dayofweek
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype(int)
    out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)
    out["day"] = out["date"].dt.day
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def aggregate_series(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Aggregate item-store panel to a coarser series level for modeling."""
    if level == "item_store":
        out = df.copy()
        if "series_id" not in out.columns:
            out["series_id"] = out["store_id"].astype(str) + "__" + out["item_id"].astype(str)
        return out

    if level == "store_cat":
        group_cols = [
            c
            for c in [
                "store_id",
                "cat_id",
                "state_id",
                "date",
                "d",
                "wm_yr_wk",
                "weekday",
                "wday",
                "month",
                "year",
            ]
            if c in df.columns
        ]
        agg_map: dict[str, tuple[str, str]] = {"sales": ("sales", "sum")}
        for col, how in [
            ("sell_price", "mean"),
            ("snap", "max"),
            ("has_event", "max"),
            ("avg_temp_f", "mean"),
            ("precip_in", "mean"),
            ("is_severe_weather", "max"),
        ]:
            if col in df.columns:
                agg_map[col] = (col, how)
        agg = df.groupby(group_cols, as_index=False).agg(**agg_map)
        agg["item_id"] = agg["cat_id"]
        agg["series_id"] = agg["store_id"].astype(str) + "__" + agg["cat_id"].astype(str)
        return agg

    if level == "store":
        group_cols = [
            c
            for c in [
                "store_id",
                "state_id",
                "date",
                "d",
                "wm_yr_wk",
                "weekday",
                "wday",
                "month",
                "year",
            ]
            if c in df.columns
        ]
        agg = df.groupby(group_cols, as_index=False).agg(sales=("sales", "sum"))
        agg["item_id"] = "ALL"
        agg["cat_id"] = "ALL"
        agg["series_id"] = agg["store_id"].astype(str)
        return agg

    raise ValueError(f"Unknown series_level: {level}")


FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "roll_mean_7",
    "roll_mean_14",
    "roll_mean_28",
    "roll_std_7",
    "roll_std_14",
    "roll_std_28",
    "sell_price",
    "price_change",
    "snap",
    "has_event",
    "avg_temp_f",
    "precip_in",
    "is_severe_weather",
    "dow",
    "weekofyear",
    "is_weekend",
    "day",
    "month",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]


def build_feature_matrix(
    sales_long: pd.DataFrame,
    calendar: pd.DataFrame,
    sell_prices: pd.DataFrame,
    weather: pd.DataFrame | None,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Build model-ready feature matrix at item-store grain."""
    feat_cfg = cfg.get("features", {})
    df = sales_long.copy()
    df["date"] = pd.to_datetime(df["date"])

    if feat_cfg.get("include_events", True) or feat_cfg.get("include_snap", True):
        df = _attach_calendar_events(df, calendar)
    else:
        df["has_event"] = 0
        df["snap"] = 0

    if feat_cfg.get("include_price", True):
        df = _attach_prices(df, sell_prices)
    else:
        df["sell_price"] = 0.0
        df["price_change"] = 0.0

    if feat_cfg.get("include_weather", True):
        df = _attach_weather(df, weather)
    else:
        df["avg_temp_f"] = 0.0
        df["precip_in"] = 0.0
        df["is_severe_weather"] = 0

    if feat_cfg.get("include_calendar", True):
        df = _add_calendar_parts(df)

    df = _add_lag_rolling(df, cfg)
    df["series_id"] = df["store_id"].astype(str) + "__" + df["item_id"].astype(str)

    for col in ("store_id", "item_id", "cat_id", "dept_id", "state_id"):
        if col in df.columns:
            df[f"{col}_code"] = df[col].astype("category").cat.codes

    return df
