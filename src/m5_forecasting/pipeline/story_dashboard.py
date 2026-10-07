"""Walmart customer storytelling dashboard — champion-model business insights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from m5_forecasting.viz.story_charts import render_story_charts, write_story_html


INSIGHT_COPY = {
    "title": "Walmart Demand Story — Champion Forecast Insights",
    "subtitle": (
        "Actionable assortment, SNAP/deal, and inventory signals from the "
        "best holdout model — ready for merchandising and marketing reviews."
    ),
}


def _champion_frame(forecasts: pd.DataFrame, metrics: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    if metrics is None or metrics.empty:
        raise ValueError("model metrics required to select champion")
    best = str(metrics.sort_values("rmse").iloc[0]["model"])
    champ = forecasts[forecasts["model"] == best].copy()
    if champ.empty:
        raise ValueError(f"No forecast rows for champion model '{best}'")
    champ["date"] = pd.to_datetime(champ["date"])
    champ["error"] = champ["y_true"] - champ["y_pred"]
    champ["abs_error"] = champ["error"].abs()
    champ["pct_error"] = np.where(
        champ["y_true"].abs() >= 1,
        champ["error"] / champ["y_true"] * 100.0,
        np.nan,
    )
    if "state_id" not in champ.columns and "store_id" in champ.columns:
        champ["state_id"] = champ["store_id"].astype(str).str.split("_").str[0]
    return best, champ


def _attach_drivers(champ: pd.DataFrame, test_series: pd.DataFrame) -> pd.DataFrame:
    """Join SNAP / weekend / weather / price drivers from the test feature panel."""
    driver_cols = [
        c
        for c in [
            "series_id",
            "date",
            "snap",
            "has_event",
            "is_weekend",
            "sell_price",
            "avg_temp_f",
            "precip_in",
            "is_severe_weather",
            "dow",
            "month",
        ]
        if c in test_series.columns
    ]
    drivers = test_series[driver_cols].copy()
    drivers["date"] = pd.to_datetime(drivers["date"])
    out = champ.merge(drivers, on=["series_id", "date"], how="left")
    if "snap" not in out.columns:
        out["snap"] = 0
    if "is_weekend" not in out.columns:
        out["is_weekend"] = out["date"].dt.dayofweek.isin([5, 6]).astype(int)
    if "has_event" not in out.columns:
        out["has_event"] = 0
    out["snap"] = out["snap"].fillna(0).astype(int)
    out["is_weekend"] = out["is_weekend"].fillna(0).astype(int)
    out["has_event"] = out["has_event"].fillna(0).astype(int)
    out["day_type"] = np.where(out["is_weekend"] == 1, "Weekend", "Weekday")
    out["snap_period"] = np.where(out["snap"] == 1, "SNAP window", "Non-SNAP")
    out["event_flag"] = np.where(out["has_event"] == 1, "Event day", "Regular day")
    return out


def build_story_tables(
    champ: pd.DataFrame,
    best_model: str,
    metrics: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Derive merchandising-friendly rollups for Tableau / HTML story."""
    daily = champ.copy()
    daily["date_str"] = daily["date"].dt.strftime("%Y-%m-%d")

    # 1) Executive KPIs
    total_forecast = float(daily["y_pred"].sum())
    total_actual = float(daily["y_true"].sum())
    bias = total_forecast - total_actual
    mape = float(np.mean(np.abs(daily["y_true"] - daily["y_pred"]) / np.maximum(daily["y_true"].abs(), 1.0)) * 100)
    rmse = float(np.sqrt(np.mean((daily["y_true"] - daily["y_pred"]) ** 2)))
    champ_metrics = metrics[metrics["model"] == best_model].iloc[0].to_dict()
    kpi = pd.DataFrame(
        [
            {
                "champion_model": best_model,
                "horizon_days": int(daily["date"].nunique()),
                "n_series": int(daily["series_id"].nunique()),
                "forecast_units": round(total_forecast, 1),
                "actual_units": round(total_actual, 1),
                "forecast_bias_units": round(bias, 1),
                "bias_pct": round(bias / max(total_actual, 1.0) * 100, 2),
                "rmse": round(float(champ_metrics.get("rmse", rmse)), 3),
                "mae": round(float(champ_metrics.get("mae", np.nan)), 3),
                "mape": round(float(champ_metrics.get("mape", mape)), 2),
                "story_headline": (
                    f"{best_model.upper()} is the champion model — "
                    f"{'slight over' if bias > 0 else 'slight under'}-forecast of "
                    f"{abs(bias):.0f} units ({abs(bias)/max(total_actual,1)*100:.1f}%) "
                    f"across the {daily['date'].nunique()}-day planning horizon."
                ),
            }
        ]
    )

    # 2) Store × category demand outlook (assortment planning)
    assortment = (
        daily.groupby(["store_id", "state_id", "cat_id"], as_index=False)
        .agg(
            forecast_units=("y_pred", "sum"),
            actual_units=("y_true", "sum"),
            avg_daily_forecast=("y_pred", "mean"),
            mae=("abs_error", "mean"),
        )
        .sort_values("forecast_units", ascending=False)
    )
    assortment["share_of_forecast_pct"] = (
        assortment["forecast_units"] / assortment["forecast_units"].sum() * 100
    ).round(2)
    assortment["bias_units"] = assortment["forecast_units"] - assortment["actual_units"]
    assortment["action"] = np.select(
        [
            assortment["bias_units"] > assortment["actual_units"] * 0.1,
            assortment["bias_units"] < -assortment["actual_units"] * 0.1,
        ],
        ["Watch overstock risk", "Protect against stockout"],
        default="On plan",
    )

    # 3) Daily demand pulse (trend story)
    demand_pulse = (
        daily.groupby("date", as_index=False)
        .agg(
            forecast_units=("y_pred", "sum"),
            actual_units=("y_true", "sum"),
            snap_days=("snap", "max"),
            event_days=("has_event", "max"),
        )
        .sort_values("date")
    )
    demand_pulse["date_str"] = demand_pulse["date"].dt.strftime("%Y-%m-%d")
    demand_pulse["abs_error"] = (demand_pulse["actual_units"] - demand_pulse["forecast_units"]).abs()

    # 4) SNAP lift — critical for FOODS deal timing
    snap_base = daily.copy()
    snap_lift = (
        snap_base.groupby(["cat_id", "snap_period"], as_index=False)
        .agg(
            avg_daily_demand=("y_true", "mean"),
            avg_daily_forecast=("y_pred", "mean"),
            n_days=("date", "count"),
        )
    )
    pivot = snap_lift.pivot(index="cat_id", columns="snap_period", values="avg_daily_demand")
    lift_rows = []
    for cat in pivot.index:
        non = float(pivot.loc[cat].get("Non-SNAP", np.nan))
        snap = float(pivot.loc[cat].get("SNAP window", np.nan))
        if np.isnan(non) or non == 0 or np.isnan(snap):
            lift_pct = np.nan
        else:
            lift_pct = (snap / non - 1.0) * 100
        lift_rows.append(
            {
                "cat_id": cat,
                "avg_demand_non_snap": round(non, 2) if not np.isnan(non) else np.nan,
                "avg_demand_snap": round(snap, 2) if not np.isnan(snap) else np.nan,
                "snap_lift_pct": round(lift_pct, 1) if lift_pct == lift_pct else np.nan,
                "recommendation": (
                    "Push FOODS promos inside SNAP windows"
                    if cat == "FOODS" and isinstance(lift_pct, float) and lift_pct > 5
                    else (
                        "Modest SNAP sensitivity — keep baseline promos"
                        if isinstance(lift_pct, float) and lift_pct > 0
                        else "No material SNAP lift in this sample"
                    )
                ),
            }
        )
    snap_insight = pd.DataFrame(lift_rows).sort_values("snap_lift_pct", ascending=False)

    # 5) Weekend vs weekday (labor / staging)
    weekend = (
        daily.groupby(["cat_id", "day_type"], as_index=False)
        .agg(avg_demand=("y_true", "mean"), avg_forecast=("y_pred", "mean"), n=("date", "count"))
    )

    # 6) Inventory risk board — under/over forecast series
    risk = (
        daily.groupby(["series_id", "store_id", "state_id", "cat_id"], as_index=False)
        .agg(
            forecast_units=("y_pred", "sum"),
            actual_units=("y_true", "sum"),
            mae=("abs_error", "mean"),
            mean_error=("error", "mean"),
        )
    )
    risk["bias_units"] = risk["forecast_units"] - risk["actual_units"]
    risk["risk_type"] = np.select(
        [risk["bias_units"] < -5, risk["bias_units"] > 5],
        ["Stockout risk (under-forecast)", "Overstock risk (over-forecast)"],
        default="Balanced",
    )
    risk["priority"] = risk["mae"] * (1 + risk["bias_units"].abs() / (risk["actual_units"].abs() + 1))
    risk = risk.sort_values("priority", ascending=False)

    # 7) Store leaderboard
    stores = (
        daily.groupby(["store_id", "state_id"], as_index=False)
        .agg(
            forecast_units=("y_pred", "sum"),
            actual_units=("y_true", "sum"),
            rmse=("error", lambda s: float(np.sqrt(np.mean(np.square(s))))),
        )
        .sort_values("forecast_units", ascending=False)
    )
    stores["accuracy_score"] = (100 - stores["rmse"]).clip(lower=0).round(1)

    # 8) Talking points for the presenter
    top_cat = assortment.groupby("cat_id")["forecast_units"].sum().sort_values(ascending=False)
    top_snap = snap_insight.iloc[0] if len(snap_insight) else None
    top_risk = risk.iloc[0] if len(risk) else None
    talking_points = pd.DataFrame(
        [
            {
                "order": 1,
                "section": "Executive",
                "insight": kpi.iloc[0]["story_headline"],
                "business_ask": "Adopt the champion model as the planning forecast for the next cycle.",
            },
            {
                "order": 2,
                "section": "Assortment",
                "insight": (
                    f"{top_cat.index[0]} drives {top_cat.iloc[0]/total_forecast*100:.0f}% of "
                    f"forecasted units — prioritize shelf space and replenishment there."
                    if len(top_cat)
                    else "Category mix unavailable."
                ),
                "business_ask": "Confirm planogram capacity for the top demand category by store.",
            },
            {
                "order": 3,
                "section": "SNAP / Deals",
                "insight": (
                    f"{top_snap['cat_id']} shows ~{top_snap['snap_lift_pct']}% higher average demand "
                    f"in SNAP windows vs non-SNAP."
                    if top_snap is not None and pd.notna(top_snap.get("snap_lift_pct"))
                    else "SNAP lift not material in this sample."
                ),
                "business_ask": "Time FOODS circulars and digital offers to state SNAP calendars.",
            },
            {
                "order": 4,
                "section": "Inventory risk",
                "insight": (
                    f"{top_risk['series_id']} flagged as {top_risk['risk_type']} "
                    f"(bias {top_risk['bias_units']:.0f} units)."
                    if top_risk is not None
                    else "No high-priority risk series."
                ),
                "business_ask": "Review safety stock / promo depth on flagged store×category cells.",
            },
            {
                "order": 5,
                "section": "Next step",
                "insight": (
                    "Refresh weekly: retrain on latest POS, publish Tableau extracts, "
                    "and review the risk board in merchandising standup."
                ),
                "business_ask": "Schedule a recurring forecast review with Merch + Marketing.",
            },
        ]
    )

    # Flatten daily champion extract for Tableau
    story_daily = daily.copy()
    story_daily["date"] = story_daily["date"].dt.strftime("%Y-%m-%d")
    story_daily["champion_model"] = best_model

    return {
        "story_kpi": kpi,
        "story_daily": story_daily,
        "story_assortment": assortment,
        "story_demand_pulse": demand_pulse.drop(columns=["date"]).rename(columns={"date_str": "date"}),
        "story_snap_lift": snap_insight,
        "story_weekend": weekend,
        "story_inventory_risk": risk,
        "story_store_leaderboard": stores,
        "story_talking_points": talking_points,
    }


def export_customer_story(
    cfg: dict[str, Any],
    forecasts: pd.DataFrame,
    metrics: pd.DataFrame,
    test_series: pd.DataFrame,
) -> dict[str, Path]:
    """
    Build Walmart-customer storytelling artifacts:

    - Tableau CSVs under outputs/tableau/story/
    - Presentation charts under outputs/figures/story/
    - Self-contained HTML dashboard for live demos
    """
    best_model, champ = _champion_frame(forecasts, metrics)
    champ = _attach_drivers(champ, test_series)
    tables = build_story_tables(champ, best_model, metrics)

    tableau_dir = Path(cfg["paths"]["tableau_dir"])
    story_dir = tableau_dir / "story"
    figures_dir = Path(cfg["paths"]["figures_dir"]) / "story"
    story_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {"story_dir": story_dir, "story_figures_dir": figures_dir}
    for name, df in tables.items():
        out = story_dir / f"{name}.csv"
        # Serialize dates cleanly
        export = df.copy()
        for col in export.columns:
            if pd.api.types.is_datetime64_any_dtype(export[col]):
                export[col] = export[col].dt.strftime("%Y-%m-%d")
        export.to_csv(out, index=False)
        paths[name] = out

    # Also drop a single “wide” extract Tableau users can connect once
    wide_path = story_dir / "champion_forecast_enriched.csv"
    tables["story_daily"].to_csv(wide_path, index=False)
    paths["champion_forecast_enriched"] = wide_path

    chart_paths = render_story_charts(tables, best_model, figures_dir)
    paths.update({f"fig_{k}": v for k, v in chart_paths.items()})

    html_path = tableau_dir / "walmart_customer_story.html"
    write_story_html(
        tables=tables,
        best_model=best_model,
        chart_paths=chart_paths,
        out_path=html_path,
        title=INSIGHT_COPY["title"],
        subtitle=INSIGHT_COPY["subtitle"],
    )
    paths["story_html"] = html_path

    manifest = {
        "title": INSIGHT_COPY["title"],
        "champion_model": best_model,
        "extracts": {k: str(v) for k, v in paths.items() if str(v).endswith(".csv")},
        "html_dashboard": str(html_path),
        "tableau_guide": "docs/WALMART_CUSTOMER_STORY.md",
    }
    manifest_path = story_dir / "story_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    paths["story_manifest"] = manifest_path
    return paths
