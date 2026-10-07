"""Data cleaning: negatives, outliers (winsorize), stockout flags."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _zero_streak_flags(sales: pd.Series, min_streak: int) -> pd.Series:
    """Flag observations inside zero-runs of length >= min_streak."""
    is_zero = sales.fillna(0).eq(0).to_numpy()
    flags = np.zeros(len(sales), dtype=bool)
    n = len(is_zero)
    i = 0
    while i < n:
        if not is_zero[i]:
            i += 1
            continue
        j = i
        while j < n and is_zero[j]:
            j += 1
        if j - i >= min_streak:
            flags[i:j] = True
        i = j
    return pd.Series(flags, index=sales.index)


def clean_sales(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean long-format sales panel.

    Returns
    -------
    cleaned : DataFrame
        Sales with negatives fixed and optional winsorization applied.
    audit : DataFrame
        Per series cleaning summary for transparency / Tableau.
    """
    clean_cfg = cfg.get("cleaning", {})
    iqr_mult = float(clean_cfg.get("iqr_multiplier", 3.0))
    winsorize = bool(clean_cfg.get("winsorize", True))
    stockout_streak = int(clean_cfg.get("stockout_zero_streak", 14))
    fill_neg = bool(clean_cfg.get("fill_negative_sales", True))

    out = df.copy()
    out["sales"] = out["sales"].astype(float)
    out["sales_raw"] = out["sales"]
    out["is_negative_fixed"] = False
    out["is_outlier"] = False
    out["is_stockout_flag"] = False

    if fill_neg:
        neg = out["sales"] < 0
        out.loc[neg, "is_negative_fixed"] = True
        out.loc[neg, "sales"] = 0.0

    audit_rows: list[dict] = []
    group_cols = ["store_id", "item_id"]

    cleaned_parts = []
    for keys, grp in out.groupby(group_cols, sort=False):
        g = grp.sort_values("date").copy()
        s = g["sales"].astype(float)

        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower = max(0.0, q1 - iqr_mult * iqr)
        upper = q3 + iqr_mult * iqr
        if iqr == 0:
            # Degenerate series — use robust MAD-style bound
            med = s.median()
            mad = (s - med).abs().median()
            upper = med + iqr_mult * 1.4826 * (mad if mad > 0 else 1.0)
            lower = 0.0

        outlier_mask = (s < lower) | (s > upper)
        g["is_outlier"] = outlier_mask.to_numpy()
        if winsorize and outlier_mask.any():
            g.loc[outlier_mask, "sales"] = s.clip(lower=lower, upper=upper)

        stock_flags = _zero_streak_flags(g["sales"], stockout_streak)
        g["is_stockout_flag"] = stock_flags.to_numpy()

        store_id, item_id = keys if isinstance(keys, tuple) else (keys, None)
        audit_rows.append(
            {
                "store_id": store_id,
                "item_id": item_id,
                "n_obs": len(g),
                "n_negative_fixed": int(g["is_negative_fixed"].sum()),
                "n_outliers": int(g["is_outlier"].sum()),
                "n_stockout_days": int(g["is_stockout_flag"].sum()),
                "winsor_lower": float(lower),
                "winsor_upper": float(upper),
                "mean_sales": float(g["sales"].mean()),
                "max_sales": float(g["sales"].max()),
            }
        )
        cleaned_parts.append(g)

    cleaned = pd.concat(cleaned_parts, ignore_index=True)
    cleaned = cleaned.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    audit = pd.DataFrame(audit_rows)
    return cleaned, audit
