"""Per-series SARIMA forecasts (statsmodels)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class ArimaModel:
    name = "arima"

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 7),
        max_series: int = 12,
    ):
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.max_series = max_series
        self._fitted: dict[str, Any] = {}
        self._last_values: dict[str, float] = {}

    def fit(self, train: pd.DataFrame) -> "ArimaModel":
        series_ids = train.groupby("series_id")["sales"].sum().sort_values(ascending=False)
        keep = series_ids.head(self.max_series).index.tolist()
        self._fitted = {}
        self._last_values = {}

        for sid in keep:
            y = (
                train.loc[train["series_id"] == sid]
                .sort_values("date")["sales"]
                .astype(float)
                .to_numpy()
            )
            self._last_values[sid] = float(y[-1]) if len(y) else 0.0
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAX(
                        y,
                        order=self.order,
                        seasonal_order=self.seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    res = model.fit(disp=False, maxiter=50)
                self._fitted[sid] = res
            except Exception:
                # Fallback: store mean as constant predictor
                self._fitted[sid] = float(np.mean(y)) if len(y) else 0.0
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for sid, fut in future.groupby("series_id"):
            horizon = len(fut)
            fitted = self._fitted.get(sid)
            if fitted is None:
                preds = np.full(horizon, self._last_values.get(sid, 0.0))
            elif isinstance(fitted, float):
                preds = np.full(horizon, fitted)
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        preds = np.asarray(fitted.forecast(horizon), dtype=float)
                except Exception:
                    preds = np.full(horizon, self._last_values.get(sid, 0.0))
            preds = np.clip(preds, 0, None)
            out = fut.copy()
            out["y_pred"] = preds
            out["model"] = self.name
            frames.append(out)
        if not frames:
            out = future.copy()
            out["y_pred"] = 0.0
            out["model"] = self.name
            return out
        return pd.concat(frames, ignore_index=True)
