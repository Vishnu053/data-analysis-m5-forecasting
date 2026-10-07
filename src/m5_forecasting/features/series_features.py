"""Lag / calendar features on an already-aggregated series panel."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_series_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    """Add causal lag / rolling / calendar features for tree models at series grain."""
    lags = lags or [1, 7, 14, 28]
    out = df.sort_values(["series_id", "date"]).copy()
    out["date"] = pd.to_datetime(out["date"])
    g = out.groupby("series_id", sort=False)["sales"]

    for lag in lags:
        out[f"lag_{lag}"] = g.shift(lag)

    for w in (7, 14, 28):
        shifted = g.shift(1)
        out[f"roll_mean_{w}"] = shifted.transform(lambda s, ww=w: s.rolling(ww, min_periods=1).mean())
        out[f"roll_std_{w}"] = shifted.transform(lambda s, ww=w: s.rolling(ww, min_periods=1).std())
        out[f"roll_std_{w}"] = out[f"roll_std_{w}"].fillna(0.0)

    out["dow"] = out["date"].dt.dayofweek
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype(int)
    out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)
    out["day"] = out["date"].dt.day
    if "month" not in out.columns:
        out["month"] = out["date"].dt.month
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    for col in ("snap", "has_event", "precip_in", "is_severe_weather", "sell_price", "avg_temp_f"):
        if col not in out.columns:
            out[col] = 0.0
        else:
            out[col] = out[col].fillna(0.0)
    if "price_change" not in out.columns:
        out["price_change"] = out.groupby("series_id")["sell_price"].pct_change()
        out["price_change"] = out["price_change"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out["store_id_code"] = out["store_id"].astype("category").cat.codes
    if "item_id" in out.columns:
        out["item_id_code"] = out["item_id"].astype("category").cat.codes
    if "cat_id" in out.columns:
        out["cat_id_code"] = out["cat_id"].astype("category").cat.codes
    if "state_id" in out.columns:
        out["state_id_code"] = out["state_id"].astype("category").cat.codes
    return out
