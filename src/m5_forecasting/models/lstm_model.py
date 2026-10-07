"""Lightweight per-series LSTM forecaster (PyTorch)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMForecaster:
    name = "lstm"

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {}
        self.lookback = int(self.params.get("lookback", 28))
        self.hidden_size = int(self.params.get("hidden_size", 32))
        self.num_layers = int(self.params.get("num_layers", 1))
        self.epochs = int(self.params.get("epochs", 15))
        self.batch_size = int(self.params.get("batch_size", 64))
        self.lr = float(self.params.get("learning_rate", 1e-3))
        self.max_series = int(self.params.get("max_series", 12))
        self.device = torch.device("cpu")
        self._models: dict[str, _LSTMNet] = {}
        self._scalers: dict[str, tuple[float, float]] = {}
        self._history: dict[str, np.ndarray] = {}

    @staticmethod
    def _make_windows(y: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for i in range(lookback, len(y)):
            xs.append(y[i - lookback : i])
            ys.append(y[i])
        if not xs:
            return np.empty((0, lookback, 1)), np.empty((0,))
        X = np.asarray(xs, dtype=np.float32)[..., None]
        Y = np.asarray(ys, dtype=np.float32)
        return X, Y

    def fit(self, train: pd.DataFrame) -> "LSTMForecaster":
        torch.manual_seed(42)
        series_ids = train.groupby("series_id")["sales"].sum().sort_values(ascending=False)
        keep = series_ids.head(self.max_series).index.tolist()
        self._models = {}
        self._scalers = {}
        self._history = {}

        for sid in keep:
            y = (
                train.loc[train["series_id"] == sid]
                .sort_values("date")["sales"]
                .astype(float)
                .to_numpy()
            )
            self._history[sid] = y.copy()
            mean, std = float(y.mean()), float(y.std()) if y.std() > 1e-6 else 1.0
            self._scalers[sid] = (mean, std)
            y_norm = (y - mean) / std
            X, Y = self._make_windows(y_norm, self.lookback)
            if len(X) < 8:
                continue

            ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
            loader = DataLoader(ds, batch_size=min(self.batch_size, len(ds)), shuffle=True)
            net = _LSTMNet(1, self.hidden_size, self.num_layers).to(self.device)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            loss_fn = nn.MSELoss()
            net.train()
            for _ in range(self.epochs):
                for xb, yb in loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    opt.zero_grad()
                    pred = net(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    opt.step()
            net.eval()
            self._models[sid] = net
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for sid, fut in future.groupby("series_id"):
            horizon = len(fut)
            hist = self._history.get(sid)
            mean, std = self._scalers.get(sid, (0.0, 1.0))
            net = self._models.get(sid)

            if hist is None or net is None or len(hist) < self.lookback:
                # Cold start: seasonal-ish mean of available history
                base = float(np.mean(hist)) if hist is not None and len(hist) else 0.0
                preds = np.full(horizon, max(base, 0.0))
            else:
                buf = list((hist - mean) / std)
                preds_norm = []
                net.eval()
                with torch.no_grad():
                    for _ in range(horizon):
                        window = np.asarray(buf[-self.lookback :], dtype=np.float32)[None, :, None]
                        x = torch.from_numpy(window).to(self.device)
                        p = float(net(x).cpu().numpy().reshape(-1)[0])
                        preds_norm.append(p)
                        buf.append(p)
                preds = np.clip(np.asarray(preds_norm) * std + mean, 0, None)

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
