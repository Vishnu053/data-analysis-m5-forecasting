# Data Dictionary — Walmart Business Insights Extracts

LightGBM-only demand signal. Model bake-off files removed from this package.

## Scope
Real Kaggle M5 subset: stores CA_1, CA_2, CA_3, TX_1, TX_2, WI_1, WI_2; 60 SKUs; 420 item-store series; full 1913 days. See `SUBSET_NOTES.json`.

## Primary extracts
| File | Use |
|------|-----|
| `forecast_lgbm_daily.csv` | 28d actual vs outlook by store×category |
| `forecast_lgbm_by_state_daily.csv` | State outlook |
| `forecast_lgbm_store_cat_summary.csv` | Bias / gap by store×cat |
| `sales_daily_store_cat.csv` | Historical demand |
| `store_performance_28d.csv` | Rankings + growth |
| `state_performance_28d.csv` / `category_performance_28d.csv` | Regional & assortment growth |
| `heatmap_state_category_28d.csv` | Heatmap |
| `category_intermittency.csv` | Zero-day share by category |
| `snap_lift_foods_by_state.csv` | FOODS SNAP lift |
| `event_lift_top.csv` | Event-type lifts |
| `seasonality_weekday.csv` / `seasonality_month.csv` | Calendar shape |
| `inventory_underforecast_top15.csv` | Stockout-risk gaps |
| `stockout_zero_streak_top20.csv` / `inventory_ops_risk_top15.csv` | Availability risk |
| `deal_lift_by_category.csv` / `price_units_correlation.csv` | Price & deals |
| `executive_kpis.json` | Headline business KPIs |

## Key fields
- **growth_pct**: last 28d units vs prior 28d
- **gap_units / bias**: actual − outlook (positive = under-forecast / stockout risk)
- **lift_pct**: SNAP or event uplift vs baseline
- **deal**: week where sell_price < 90% of item-store median
- **avg_zero_share**: fraction of days with zero sales (intermittency)

## Price response extracts
| File | Meaning |
|------|---------|
| `price_response_by_sku.csv` | Per item-store: units in bottom vs top sell-price quartile weeks |
| `price_response_by_category.csv` | Median / mean lift; % SKUs with positive response |
| `price_cut_wow_response.csv` | When price falls ≥5% week-over-week, median unit change |
| `deal_lift_by_category.csv` | Same as category median lift (Tableau-friendly name) |
