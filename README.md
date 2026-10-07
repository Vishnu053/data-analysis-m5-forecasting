# Walmart Sales Forecasting (M5)

Production-style batch pipeline for the [M5 Forecasting Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy) problem: clean intermittent retail demand, engineer calendar / price / SNAP / weather features, train and compare **seasonal naïve → SARIMA → LightGBM → LSTM**, then export holdout forecasts for a Tableau merchandising dashboard.

```text
raw / synthetic M5  →  clean  →  features  →  train  →  batch predict  →  Tableau CSVs
```

## Why this project

| Interview theme | What ships here |
|-----------------|-----------------|
| Data cleaning | Negative-sale fixes, IQR winsorization, stockout streak flags + audit table |
| Classical stats | Per-series SARIMA with weekly seasonality |
| ML | LightGBM with compact hyperparameter search + feature importance |
| Deep learning | Lightweight PyTorch LSTM (recursive multi-step) |
| External signals | SNAP by state, events, sell prices, synthetic weather |
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

- **Ops dashboard** — all models + data quality: [docs/TABLEAU.md](docs/TABLEAU.md)
- **Walmart customer story** — champion-model business insights for executive storytelling: [docs/WALMART_CUSTOMER_STORY.md](docs/WALMART_CUSTOMER_STORY.md)

```bash
python3 scripts/run_pipeline.py story
# open outputs/tableau/walmart_customer_story.html
# or connect Tableau Desktop to outputs/tableau/story/*.csv
```
