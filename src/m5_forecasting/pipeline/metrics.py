"""Forecast accuracy metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """MAPE with floor on denominator to avoid division by zero on intermittent demand."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def wrmsse_proxy(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """
    Lightweight WRMSSE-inspired score.

    Scales RMSE by the series' naive seasonal (lag-7) scale on the training window.
    Not the official M5 WRMSSE, but useful for model comparison.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) > 7:
        scale = np.mean((y_train[7:] - y_train[:-7]) ** 2)
    else:
        scale = np.mean(np.diff(y_train) ** 2) if len(y_train) > 1 else 1.0
    scale = max(scale, 1e-6)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2) / scale))


def summarize_forecasts(
    forecasts: pd.DataFrame,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute metrics per model (and overall)."""
    rows = []
    for model_name, grp in forecasts.groupby("model"):
        yt = grp["y_true"].to_numpy()
        yp = grp["y_pred"].to_numpy()
        train = None
        if history is not None and "sales" in history.columns:
            train = history["sales"].to_numpy()
        row = {
            "model": model_name,
            "rmse": rmse(yt, yp),
            "mae": mae(yt, yp),
            "mape": mape(yt, yp),
            "n_points": int(len(grp)),
        }
        if train is not None:
            row["wrmsse_proxy"] = wrmsse_proxy(yt, yp, train)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
