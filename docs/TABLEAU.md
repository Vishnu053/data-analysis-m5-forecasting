# Tableau dashboard guide

This pipeline writes flat CSVs under `outputs/tableau/` so a merchandiser or
marketing analyst can refresh a workbook without touching Python.

## Extracts

| File | Grain | Use |
|------|-------|-----|
| `forecasts_daily.csv` | model × series × day | Actual vs forecast, error, assortment planning |
| `store_daily_rollup.csv` | model × store × day | Store-level demand outlook |
| `model_metrics.csv` | model | Champion / challenger scorecard |
| `feature_importance.csv` | feature | Explain LightGBM drivers (SNAP, price, lags) |
| `cleaning_audit.csv` | item × store | Data-quality transparency (outliers, stockouts) |

## Suggested sheets

1. **Demand outlook** — line chart of `y_pred` vs `y_true` filtered to the champion model; break by `store_id` / `cat_id`.
2. **Deal planning** — highlight FOODS series on SNAP-heavy weeks (join calendar or use feature importance narrative).
3. **Assortment risk** — series with largest `abs_error` or sustained under-forecast (stockout risk).
4. **Model scorecard** — bar chart of RMSE / MAE from `model_metrics.csv`.
5. **Data quality** — outlier and stockout counts from `cleaning_audit.csv`.

## Refresh workflow

```bash
python scripts/run_pipeline.py run-all
# Point Tableau to outputs/tableau/*.csv (or publish via Hyper extract)
```

For production, schedule the same command in Airflow / cron and overwrite the
extract directory Tableau Server watches.
