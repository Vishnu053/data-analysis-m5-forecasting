# Walmart Sales Forecasting (M5)

Production-style batch pipeline for the [M5 Forecasting Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy) problem: clean intermittent retail demand, engineer calendar / price / SNAP / weather features, and train a **LightGBM** forecaster to export holdout forecasts for a Tableau merchandising dashboard.

```text
raw M5 CSVs  →  clean  →  features  →  LightGBM train  →  batch predict  →  Tableau extracts
```

## Business-Insights Dashboard

The **[`walmart-m5-tableau/`](walmart-m5-tableau/)** folder contains a ready-to-present Tableau + HTML dashboard package with executive KPIs, LightGBM forecast accuracy, SNAP/promotion lift analysis, and inventory risk views.

| Artifact | Description |
|----------|-------------|
| [`walmart-m5-tableau/Walmart_M5_Executive_Dashboard.html`](walmart-m5-tableau/Walmart_M5_Executive_Dashboard.html) | Standalone HTML dashboard for browser viewing |
| [`walmart-m5-tableau/Walmart_M5_Dashboard.twbx`](walmart-m5-tableau/Walmart_M5_Dashboard.twbx) | Packaged Tableau workbook |
| [`walmart-m5-tableau/PRESENTATION_GUIDE.md`](walmart-m5-tableau/PRESENTATION_GUIDE.md) | Presentation flow and talking points |
| [`walmart-m5-tableau/DATA_DICTIONARY.md`](walmart-m5-tableau/DATA_DICTIONARY.md) | Extract schemas and field definitions |
| [`walmart-m5-tableau/extracts/`](walmart-m5-tableau/extracts/) | CSV/JSON extracts powering the dashboards |

See also [docs/TABLEAU.md](docs/TABLEAU.md) for refresh workflows and suggested sheets.

## Why this project

| Interview theme | What ships here |
|-----------------|-----------------|
| Data cleaning | Negative-sale fixes, IQR winsorization, stockout streak flags + audit table |
| ML forecasting | LightGBM with compact hyperparameter search + feature importance |
| External signals | SNAP by state, events, sell prices, weather |
| Business insights | Executive dashboard with demand outlook, SNAP/deal timing, inventory risk |
| Production | YAML config, Click CLI, parquet artifacts, batch predict, Tableau extracts |

## Quick start

```bash
# from repo root
python3 -m pip install -r requirements.txt
python3 -m pip install -e .

# Full pipeline (generates M5-compatible synthetic data if Kaggle files are absent)
python3 scripts/run_pipeline.py run-all
```

Or step-by-step:

```bash
python3 scripts/run_pipeline.py prepare
python3 scripts/run_pipeline.py train
python3 scripts/run_pipeline.py predict
```

Outputs land under `outputs/`:

- `outputs/metrics/model_comparison.csv` — holdout RMSE / MAE / MAPE
- `outputs/forecasts/holdout_forecasts.csv` — predictions by model
- `outputs/tableau/` — dashboard-ready extracts ([docs/TABLEAU.md](docs/TABLEAU.md))
- `outputs/figures/` — RMSE chart + example forecast plot
- `outputs/models/` — serialized estimators + training metadata

## Using the real Kaggle M5 files

1. Download competition data from [Kaggle M5 Forecasting Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy).
2. Place these files in `data/raw/`:
   - `calendar.csv`
   - `sell_prices.csv`
   - `sales_train_validation.csv`
3. Optionally add `data/external/weather_daily.csv` with columns  
   `date,state_id,avg_temp_f,precip_in,is_severe_weather`.
4. Re-run `prepare → train → predict`.

When `data/raw/` is empty, the pipeline generates a smaller **M5-schema-compatible** sample (configurable in `configs/default.yaml`) so the project runs offline.

## Project layout

```text
configs/default.yaml          # paths, cleaning, features, model hyperparameters
src/m5_forecasting/
  data/                       # synthetic generator, load, clean
  features/                   # lags, SNAP, prices, weather, aggregation
  models/                     # seasonal_naive, arima, lightgbm, lstm
  pipeline/                   # prepare, train, predict, metrics, splits
  viz/                        # matplotlib / seaborn charts
  cli.py                      # click entrypoint
scripts/run_pipeline.py
outputs/tableau/              # CSVs for Tableau
tests/                        # unit + smoke tests
docs/TABLEAU.md
```

## Modeling notes

- **Grain:** default `store_cat` (store × category) so statistical models and DNNs stay tractable; switch `models.series_level` to `item_store` for SKU-level LightGBM work.
- **Horizon:** last 28 days held out as test, prior 28 as validation (M5-style).
- **Champion selection:** lowest holdout RMSE in `outputs/metrics/run_summary.json`.
- **WRMSSE:** a scaled RMSE proxy is reported for ranking; swap in the official M5 WRMSSE if you wire full hierarchy weights.

## Configuration

Edit `configs/default.yaml` to change series count, enabled models, LightGBM / LSTM hyperparameters, and cleaning thresholds. Pass an alternate file with:

```bash
python3 scripts/run_pipeline.py --config configs/default.yaml run-all
```
## Tests

```bash
pytest -q
```

## Tableau

See [docs/TABLEAU.md](docs/TABLEAU.md) for extract schemas and a merchandiser-oriented dashboard outline (assortment planning, SNAP/deal timing, model scorecard).
