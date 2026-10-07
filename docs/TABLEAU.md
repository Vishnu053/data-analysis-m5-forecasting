# Tableau dashboard guide

This pipeline writes flat CSVs under `outputs/tableau/` so a merchandiser or
marketing analyst can refresh a workbook without touching Python.

There are **two** dashboard tracks:

1. **Ops / model monitoring** — all models, data quality, feature importance  
2. **Walmart customer story** — champion-model-only insights for executive storytelling  
   → see [WALMART_CUSTOMER_STORY.md](WALMART_CUSTOMER_STORY.md)

## Ops extracts (`outputs/tableau/`)

| File | Grain | Use |
|------|-------|-----|
| `forecasts_daily.csv` | model × series × day | Actual vs forecast, error, assortment planning |
| `store_daily_rollup.csv` | model × store × day | Store-level demand outlook |
| `model_metrics.csv` | model | Champion / challenger scorecard |
| `feature_importance.csv` | feature | Explain LightGBM drivers (SNAP, price, lags) |
| `cleaning_audit.csv` | item × store | Data-quality transparency (outliers, stockouts) |

## Suggested ops sheets

1. **Demand outlook** — line chart of `y_pred` vs `y_true` filtered to the champion model; break by `store_id` / `cat_id`.
2. **Deal planning** — highlight FOODS series on SNAP-heavy weeks.
3. **Assortment risk** — series with largest `abs_error` or sustained under-forecast.
4. **Model scorecard** — bar chart of RMSE / MAE from `model_metrics.csv`.
5. **Data quality** — outlier and stockout counts from `cleaning_audit.csv`.

## Customer storytelling dashboard

| Artifact | Location |
|----------|----------|
| HTML demo (browser) | `outputs/tableau/walmart_customer_story.html` |
| Tableau story CSVs | `outputs/tableau/story/` |
| Slide charts | `outputs/figures/story/` |
| Storyboard + talk track | [WALMART_CUSTOMER_STORY.md](WALMART_CUSTOMER_STORY.md) |

```bash
python3 scripts/run_pipeline.py predict   # regenerates story with forecasts
python3 scripts/run_pipeline.py story     # rebuild story only from latest outputs
```

## Refresh workflow

```bash
python3 scripts/run_pipeline.py run-all
# Point Tableau to outputs/tableau/*.csv and outputs/tableau/story/*.csv
```

For production, schedule the same command in Airflow / cron and overwrite the
extract directory Tableau Server watches.
