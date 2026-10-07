"""Seasonal naive baseline (lag-7) — strong intermittent-demand benchmark."""

from __future__ import annotations

import pandas as pd


class SeasonalNaiveModel:
    name = "seasonal_naive"

    def __init__(self, season_length: int = 7):
        self.season_length = season_length
        self._history: pd.DataFrame | None = None

    def fit(self, train: pd.DataFrame) -> "SeasonalNaiveModel":
        self._history = train[["series_id", "date", "sales"]].copy()
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        if self._history is None:
            raise RuntimeError("Model not fitted")

        hist = self._history.sort_values(["series_id", "date"])
        frames = []
        for series_id, fut in future.groupby("series_id"):
            h = hist[hist["series_id"] == series_id].sort_values("date")
            sales = h["sales"].tolist()
            preds = []
            # Recursive seasonal naive into the horizon
            buf = list(sales)
            for _ in range(len(fut)):
                if len(buf) >= self.season_length:
                    pred = buf[-self.season_length]
                elif buf:
                    pred = buf[-1]
                else:
                    pred = 0.0
                preds.append(float(pred))
                buf.append(pred)
            out = fut.copy()
            out["y_pred"] = preds
            out["model"] = self.name
            frames.append(out)
        return pd.concat(frames, ignore_index=True)
