# Walmart customer storytelling dashboard

A second Tableau-oriented workbook designed for **live storytelling** with a Walmart merchandising / marketing customer — powered only by the **champion model** (lowest holdout RMSE).

## What gets generated

After `predict` (or `python3 scripts/run_pipeline.py story`):

| Artifact | Purpose |
|----------|---------|
| `outputs/tableau/walmart_customer_story.html` | Self-contained presentation dashboard (open in a browser) |
| `outputs/tableau/story/*.csv` | Tableau Desktop / Server extracts |
| `outputs/figures/story/*.png` | Slide-ready charts for PowerPoint / PDF leave-behinds |
| `docs/WALMART_CUSTOMER_STORY.md` | This storyboard |

### Story extracts (`outputs/tableau/story/`)

| File | Grain | Story beat |
|------|-------|------------|
| `story_kpi.csv` | 1 row | Executive headline + bias / MAPE |
| `champion_forecast_enriched.csv` | series × day | Master extract (SNAP, weekend, weather joined) |
| `story_demand_pulse.csv` | day | Network actual vs forecast pulse |
| `story_assortment.csv` | store × category | Assortment / shelf priority |
| `story_snap_lift.csv` | category | Deal-timing (SNAP lift %) |
| `story_weekend.csv` | category × day type | Weekend staging |
| `story_inventory_risk.csv` | series | Stockout / overstock exceptions |
| `story_store_leaderboard.csv` | store | Ops attention ranking |
| `story_talking_points.csv` | ordered | Presenter script |

## Tableau Story (recommended build — 6 beats)

Connect Tableau to the folder `outputs/tableau/story/` (Text file / CSV union or separate connections).

### Beat 1 — Title / executive
- **Sheet:** KPI big numbers from `story_kpi`
- **Talk track:** “We selected **{champion_model}** as the planning forecast. Bias is {bias_pct}% over the horizon — accurate enough to drive assortment and promo calendars.”

### Beat 2 — Demand pulse
- **Sheet:** Dual-axis or two-line chart on `story_demand_pulse` (`actual_units`, `forecast_units` by `date`)
- Color SNAP days (`snap_days = 1`) with a reference mark
- **Talk track:** “Here’s the shape of demand we expect network-wide — and where SNAP windows spike volume.”

### Beat 3 — Assortment focus
- **Sheet:** Treemap or bar of `forecast_units` by `cat_id` / `store_id` from `story_assortment`
- Color by `action` (stockout / overstock / on plan)
- **Talk track:** “FOODS (or top category) owns the unit share — protect capacity there first.”

### Beat 4 — SNAP & deals
- **Sheet:** Bar of `snap_lift_pct` by `cat_id` from `story_snap_lift`
- Tooltip shows `recommendation`
- **Talk track:** “Time FOODS circulars and Walmart+ offers to state SNAP calendars — that’s where lift is real.”

### Beat 5 — Store leaderboard
- **Sheet:** Side-by-side bars actual vs forecast from `story_store_leaderboard`
- **Talk track:** “These stores concentrate demand — align labor and inbound DCs accordingly.”

### Beat 6 — Inventory risk (close)
- **Sheet:** Diverging bar of `bias_units` by `series_id` from `story_inventory_risk`, filter top 10 by `priority`
- Color `risk_type`
- **Talk track:** “Here’s this week’s exception list — under-forecast cells need safety stock; over-forecast cells need promo or transfer review.”

Optional dashboard page: place beats 2–4 on one “Planning” dashboard and beats 5–6 on an “Exceptions” dashboard, then assemble into a Tableau **Story** with the talking points from `story_talking_points.csv` in captions.

## Present without Tableau Desktop

```bash
python3 scripts/run_pipeline.py story
# open:
#   outputs/tableau/walmart_customer_story.html
```

The HTML embeds the same champion-model charts and talking points for a customer demo on a laptop.

## Refresh cadence

1. `python3 scripts/run_pipeline.py run-all` (or nightly batch)
2. Tableau Server extract refresh on `outputs/tableau/story/`
3. Merch + Marketing standup reviews the risk board weekly
