"""Time-based train / validation / test splits."""

from __future__ import annotations

from typing import Any

import pandas as pd


def time_split(
    df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split panel by date: train | validation | test (contiguous holdouts)."""
    data_cfg = cfg.get("data", {})
    val_days = int(data_cfg.get("validation_days", 28))
    test_days = int(data_cfg.get("test_days", 28))

    dates = sorted(df["date"].dropna().unique())
    if len(dates) <= val_days + test_days + 30:
        raise ValueError(
            f"Not enough history ({len(dates)} days) for "
            f"validation_days={val_days}, test_days={test_days}."
        )

    test_start = dates[-test_days]
    val_start = dates[-(test_days + val_days)]

    train = df[df["date"] < val_start].copy()
    valid = df[(df["date"] >= val_start) & (df["date"] < test_start)].copy()
    test = df[df["date"] >= test_start].copy()
    return train, valid, test
