# Presentation Guide — Walmart Demand Insights (Business Readout)

## Open this
**`Walmart_M5_Executive_Dashboard.html`** in Chrome/Edge — seven business pages, Walmart blue `#0071CE`.

Tableau: connect to `extracts/*.csv` (LightGBM-only forecasts). Starter `Walmart_M5_Dashboard.twbx` packages the same extracts.

## 15-minute arc
1. **Demand Outlook** — 13,149 units last 28d (+1.9% vs prior); network gap +495 units (actual hotter than outlook).
2. **Store & Region** — Volume leader CA_3 (2,923); rising CA_2 (+22.5%), TX_2 (+11.3%), WI_1 (+5.3%); falling CA_3 (-9.0%), CA_1 (-3.8%), WI_2 (-1.3%).
3. **Category Assortment** — FOODS 45% of units; HOBBIES/HOUSEHOLD far more intermittent (plan shelves differently).
4. **SNAP & Calendar** — FOODS SNAP +9% (CA +7% · TX +10% · WI +14%); weekend +30%.
5. **Inventory Risk** — Under-forecast hotspots + long zero-streak SKUs.
6. **Price & Deals** — Within-SKU low-price weeks: HOBBIES median +12%, FOODS +7%; HOUSEHOLD weak.
7. **Recommendations** — five ops/merch actions.

## Talking points (say these)
1. Protect inventory & weekend labor on **CA_3** and other CA volume stores (2,923 units / 28d). Watch **CA_2 (+23%)** for growth and **CA_3 (-9%)** for cooling volume.
2. **FOODS SNAP +9%** — sync ads, endcaps, and on-hands to **state** SNAP calendars (WI +14% · TX +10% · CA +7%).
3. Network ran **+495 units** hotter than the outlook — raise safety stock on top under-forecast store×category cells.
4. Don’t replenish **HOBBIES** like FOODS — HOBBIES ~76% zero-sale days vs FOODS ~58%; use flexible facings.
5. Fund temporary price investment on **HOBBIES** first (median within-SKU low-vs-high lift +12%, 70% of SKUs respond). Summary: HOBBIES +12% (responders 70%) · FOODS +7% (responders 68%) · HOUSEHOLD -2% (responders 38%).

## What we removed (on purpose)
Model scorecards, RMSE bake-offs, feature-importance pages, and champion-vs-challenger framing. LightGBM is only the demand signal behind the charts — the story is merchandising and operations.
