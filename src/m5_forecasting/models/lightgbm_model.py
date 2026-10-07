"""LightGBM direct / recursive tabular forecaster with light hyperparameter search."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from m5_forecasting.features.build import FEATURE_COLUMNS


class LightGBMForecaster:
    name = "lightgbm"

    def __init__(self, params: dict[str, Any] | None = None, tune: bool = True):
        self.params = params or {}
        self.tune = tune
        self.model: lgb.LGBMRegressor | None = None
        self.feature_cols: list[str] = []
        self.best_params_: dict[str, Any] = {}
        self.feature_importance_: pd.DataFrame | None = None

    def _available_features(self, df: pd.DataFrame) -> list[str]:
        cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        for extra in ("store_id_code", "item_id_code", "cat_id_code", "state_id_code"):
            if extra in df.columns:
                cols.append(extra)
        # Drop features that are entirely missing (e.g. weather when no external file)
        cols = [c for c in cols if not df[c].isna().all()]
        return cols

    def _candidate_grid(self) -> list[dict[str, Any]]:
        base = {
            "objective": "regression",
            "n_estimators": int(self.params.get("n_estimators", 200)),
            "learning_rate": float(self.params.get("learning_rate", 0.05)),
            "num_leaves": int(self.params.get("num_leaves", 31)),
            "min_child_samples": int(self.params.get("min_child_samples", 20)),
            "subsample": float(self.params.get("subsample", 0.8)),
            "colsample_bytree": float(self.params.get("colsample_bytree", 0.8)),
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }
        if not self.tune:
            return [base]
        # Compact grid — enough to demonstrate tuning without long batch jobs
        grid = []
        for leaves, lr in ((15, 0.05), (31, 0.05), (31, 0.1), (63, 0.03)):
            cfg = dict(base)
            cfg["num_leaves"] = leaves
            cfg["learning_rate"] = lr
            grid.append(cfg)
        return grid

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> "LightGBMForecaster":
        self.feature_cols = self._available_features(train)
        # Only require core lag features for row completeness; fill other NaNs
        required = [c for c in ("lag_1", "lag_7", "lag_14", "lag_28", "sales") if c in self.feature_cols or c == "sales"]
        use = train.dropna(subset=required).copy()
        use[self.feature_cols] = use[self.feature_cols].fillna(0.0)
        X = use[self.feature_cols]
        y = use["sales"].astype(float)
        if len(X) == 0:
            raise ValueError(
                f"LightGBM training set empty after dropna on {required}; "
                f"feature_cols={self.feature_cols}"
            )

        X_val = y_val = None
        if valid is not None and len(valid):
            v = valid.dropna(subset=required).copy()
            if len(v):
                v[self.feature_cols] = v[self.feature_cols].fillna(0.0)
                X_val = v[self.feature_cols]
                y_val = v["sales"].astype(float)

        best_score = np.inf
        best_model = None
        best_params = None
        early = int(self.params.get("early_stopping_rounds", 30))

        for cfg in self._candidate_grid():
            model = lgb.LGBMRegressor(**cfg)
            fit_kwargs: dict[str, Any] = {}
            if X_val is not None:
                fit_kwargs["eval_X"] = X_val
                fit_kwargs["eval_y"] = y_val
                fit_kwargs["callbacks"] = [
                    lgb.early_stopping(early, verbose=False),
                    lgb.log_evaluation(period=0),
                ]
            model.fit(X, y, **fit_kwargs)
            if X_val is not None:
                pred = model.predict(X_val)
                score = float(np.sqrt(np.mean((y_val.to_numpy() - pred) ** 2)))
            else:
                pred = model.predict(X)
                score = float(np.sqrt(np.mean((y.to_numpy() - pred) ** 2)))
            if score < best_score:
                best_score = score
                best_model = model
                best_params = cfg

        self.model = best_model
        self.best_params_ = best_params or {}
        assert self.model is not None
        self.feature_importance_ = (
            pd.DataFrame(
                {
                    "feature": self.feature_cols,
                    "importance": self.model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        out = future.copy()
        X = out[self.feature_cols].fillna(0.0)
        out["y_pred"] = np.clip(self.model.predict(X), 0, None)
        out["model"] = self.name
        return out
