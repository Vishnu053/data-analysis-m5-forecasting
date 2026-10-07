"""Generate M5-compatible synthetic Walmart sales tables for local demos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_m5(
    out_dir: str | Path,
    n_days: int = 400,
    n_items: int = 12,
    n_stores: int = 3,
    states: list[str] | None = None,
    categories: list[str] | None = None,
    start_date: str = "2015-01-01",
    seed: int = 42,
) -> dict[str, Path]:
    """
    Write calendar.csv, sell_prices.csv, sales_train_validation.csv,
    and a lightweight weather_daily.csv under out_dir.

    Schema mirrors the Kaggle M5 Forecasting Accuracy competition so the
    rest of the pipeline can swap in real downloads without code changes.
    """
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    states = states or ["CA", "TX", "WI"]
    categories = categories or ["FOODS", "HOBBIES", "HOUSEHOLD"]
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    # --- calendar (SNAP + events) ---
    calendar_rows = []
    events = {
        0: ("SuperBowl", "Sporting"),
        100: ("Easter", "Religious"),
        180: ("IndependenceDay", "National"),
        300: ("Thanksgiving", "National"),
        330: ("Christmas", "National"),
    }
    for i, d in enumerate(dates):
        snap = {
            "snap_CA": int(1 <= d.day <= 10),
            "snap_TX": int(1 <= d.day <= 10 or 15 <= d.day <= 20),
            "snap_WI": int(2 <= d.day <= 15),
        }
        event_name, event_type = events.get(i % 365, (None, None))
        # Sparse event injection
        if i not in events and rng.random() > 0.98:
            event_name, event_type = "PromoDay", "Cultural"
        calendar_rows.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "wm_yr_wk": int(d.strftime("%Y") + f"{d.isocalendar().week:02d}"),
                "weekday": d.day_name(),
                "wday": d.weekday() + 1,
                "month": d.month,
                "year": d.year,
                "d": f"d_{i + 1}",
                "event_name_1": event_name,
                "event_type_1": event_type,
                "event_name_2": None,
                "event_type_2": None,
                **snap,
            }
        )
    calendar = pd.DataFrame(calendar_rows)

    # --- catalog: items × stores ---
    stores = []
    for i in range(n_stores):
        state = states[i % len(states)]
        stores.append({"store_id": f"{state}_{i + 1}", "state_id": state})
    stores_df = pd.DataFrame(stores)

    items = []
    for i in range(n_items):
        cat = categories[i % len(categories)]
        dept = f"{cat}_{1 + (i % 2)}"
        items.append(
            {
                "item_id": f"{cat}_{i + 1:03d}",
                "dept_id": dept,
                "cat_id": cat,
            }
        )
    items_df = pd.DataFrame(items)

    # Cross join for series
    catalog = items_df.assign(_k=1).merge(stores_df.assign(_k=1), on="_k").drop(columns="_k")
    catalog["id"] = catalog["item_id"] + "_" + catalog["store_id"] + "_validation"

    # --- sell prices (weekly) ---
    weeks = sorted(calendar["wm_yr_wk"].unique())
    price_rows = []
    base_price = {
        row.item_id: float(rng.uniform(1.5, 12.0)) for row in items_df.itertuples()
    }
    for _, row in catalog.iterrows():
        p0 = base_price[row.item_id]
        for w_idx, wk in enumerate(weeks):
            # Gentle drift + occasional promo cut
            drift = 1.0 + 0.0005 * w_idx
            promo = 0.85 if rng.random() < 0.08 else 1.0
            price_rows.append(
                {
                    "store_id": row.store_id,
                    "item_id": row.item_id,
                    "wm_yr_wk": wk,
                    "sell_price": round(p0 * drift * promo, 2),
                }
            )
    sell_prices = pd.DataFrame(price_rows)

    # --- daily sales wide matrix ---
    d_cols = [f"d_{i + 1}" for i in range(n_days)]
    sales_wide = catalog.copy()
    cal_snap = calendar.set_index("d")

    for _, row in catalog.iterrows():
        idx = sales_wide.index[sales_wide["id"] == row.id][0]
        # Base demand by category
        cat_base = {"FOODS": 8.0, "HOBBIES": 2.5, "HOUSEHOLD": 4.0}[row.cat_id]
        store_mult = 1.0 + 0.15 * (hash(row.store_id) % 5)
        level = cat_base * store_mult * rng.uniform(0.7, 1.3)

        # Seasonality + trend
        t = np.arange(n_days)
        weekly = 1.0 + 0.25 * np.sin(2 * np.pi * t / 7 + rng.uniform(0, 2))
        yearly = 1.0 + 0.15 * np.sin(2 * np.pi * t / 365)
        trend = 1.0 + 0.0004 * t

        # SNAP lift for FOODS in matching state
        snap_col = f"snap_{row.state_id}"
        snap = cal_snap[snap_col].to_numpy(dtype=float)
        snap_lift = 1.0 + (0.35 if row.cat_id == "FOODS" else 0.05) * snap

        # Event lift
        event_lift = np.ones(n_days)
        for i, ev in enumerate(cal_snap["event_name_1"].tolist()):
            if pd.notna(ev):
                event_lift[i] = 1.4 if row.cat_id == "FOODS" else 1.15

        noise = rng.lognormal(mean=0.0, sigma=0.25, size=n_days)
        mu = level * weekly * yearly * trend * snap_lift * event_lift * noise
        sales = rng.poisson(np.clip(mu, 0.1, None)).astype(float)

        # Inject outliers (~0.5%)
        n_out = max(1, int(0.005 * n_days))
        out_idx = rng.choice(n_days, size=n_out, replace=False)
        sales[out_idx] *= rng.uniform(4, 8, size=n_out)

        # Stockout streaks
        if rng.random() < 0.4:
            start = int(rng.integers(20, n_days - 30))
            length = int(rng.integers(10, 20))
            sales[start : start + length] = 0

        # Rare negative glitch (scanner error) to exercise cleaner
        if rng.random() < 0.3:
            bad = int(rng.integers(0, n_days))
            sales[bad] = -1

        sales_wide.loc[idx, d_cols] = sales

    # --- weather (external enrichment demo) ---
    weather_rows = []
    for state in states:
        base_temp = {"CA": 65, "TX": 72, "WI": 45}[state]
        temps = base_temp + 15 * np.sin(2 * np.pi * np.arange(n_days) / 365) + rng.normal(
            0, 3, n_days
        )
        precip = np.clip(rng.gamma(1.5, 0.4, n_days) - 0.5, 0, None)
        for i, d in enumerate(dates):
            weather_rows.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "state_id": state,
                    "avg_temp_f": round(float(temps[i]), 1),
                    "precip_in": round(float(precip[i]), 2),
                    "is_severe_weather": int(precip[i] > 1.5 or temps[i] < 20),
                }
            )
    weather = pd.DataFrame(weather_rows)

    paths = {
        "calendar": out / "calendar.csv",
        "sell_prices": out / "sell_prices.csv",
        "sales_train_validation": out / "sales_train_validation.csv",
        "weather_daily": out / "weather_daily.csv",
    }
    calendar.to_csv(paths["calendar"], index=False)
    sell_prices.to_csv(paths["sell_prices"], index=False)
    sales_wide.to_csv(paths["sales_train_validation"], index=False)
    weather.to_csv(paths["weather_daily"], index=False)

    # Also copy weather into a sibling external/ if caller uses sample as raw
    return paths


if __name__ == "__main__":
    generate_synthetic_m5("data/sample")
