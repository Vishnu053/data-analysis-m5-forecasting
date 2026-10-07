"""Presentation charts + self-contained HTML story for Walmart customer demos."""

from __future__ import annotations

import base64
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Walmart-adjacent commercial palette (not purple/cream AI defaults)
COLORS = {
    "navy": "#041E42",
    "spark": "#FFC220",
    "blue": "#0071CE",
    "slate": "#5B6770",
    "teal": "#0A7C6B",
    "coral": "#E4572E",
    "bg": "#F7F9FC",
    "grid": "#D9E2EC",
}


def _style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "axes.facecolor": "white",
            "figure.facecolor": COLORS["bg"],
            "axes.edgecolor": COLORS["grid"],
            "grid.color": COLORS["grid"],
            "axes.labelcolor": COLORS["navy"],
            "xtick.color": COLORS["slate"],
            "ytick.color": COLORS["slate"],
            "text.color": COLORS["navy"],
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
        }
    )


def render_story_charts(
    tables: dict[str, pd.DataFrame],
    best_model: str,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Render the five storytelling visuals used in the customer deck / HTML."""
    _style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # 1) Demand pulse — forecast vs actual
    pulse = tables["story_demand_pulse"].copy()
    pulse["date"] = pd.to_datetime(pulse["date"])
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(pulse["date"], pulse["actual_units"], color=COLORS["navy"], lw=2.4, label="Actual")
    ax.plot(
        pulse["date"],
        pulse["forecast_units"],
        color=COLORS["blue"],
        lw=2.4,
        label=f"{best_model} forecast",
    )
    if "snap_days" in pulse.columns:
        snap_days = pulse[pulse["snap_days"] == 1]
        if len(snap_days):
            ax.scatter(
                snap_days["date"],
                snap_days["actual_units"],
                color=COLORS["spark"],
                s=55,
                zorder=5,
                label="SNAP window day",
                edgecolor=COLORS["navy"],
                linewidths=0.6,
            )
    ax.set_title("Network demand pulse — planning horizon")
    ax.set_xlabel("")
    ax.set_ylabel("Units")
    ax.legend(frameon=False, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    paths["demand_pulse"] = out_dir / "01_demand_pulse.png"
    fig.savefig(paths["demand_pulse"], dpi=160)
    plt.close(fig)

    # 2) Assortment mix by category
    assort = tables["story_assortment"].copy()
    cat = (
        assort.groupby("cat_id", as_index=False)["forecast_units"]
        .sum()
        .sort_values("forecast_units", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.barh(cat["cat_id"], cat["forecast_units"], color=[COLORS["blue"], COLORS["teal"], COLORS["spark"]][: len(cat)])
    ax.set_title("Where demand concentrates — category outlook")
    ax.set_xlabel("Forecasted units (horizon)")
    ax.set_ylabel("")
    for i, (v, name) in enumerate(zip(cat["forecast_units"], cat["cat_id"])):
        ax.text(v * 1.01, i, f"{v:,.0f}", va="center", color=COLORS["slate"], fontsize=11)
    fig.tight_layout()
    paths["category_mix"] = out_dir / "02_category_mix.png"
    fig.savefig(paths["category_mix"], dpi=160)
    plt.close(fig)

    # 3) SNAP lift by category
    snap = tables["story_snap_lift"].dropna(subset=["snap_lift_pct"]).copy()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    if len(snap):
        colors = [COLORS["coral"] if v < 0 else COLORS["teal"] for v in snap["snap_lift_pct"]]
        ax.bar(snap["cat_id"], snap["snap_lift_pct"], color=colors, width=0.65)
        ax.axhline(0, color=COLORS["slate"], lw=1)
        ax.set_ylabel("Demand lift vs non-SNAP (%)")
    ax.set_title("SNAP windows — where deals should land")
    ax.set_xlabel("")
    fig.tight_layout()
    paths["snap_lift"] = out_dir / "03_snap_lift.png"
    fig.savefig(paths["snap_lift"], dpi=160)
    plt.close(fig)

    # 4) Store leaderboard
    stores = tables["story_store_leaderboard"].sort_values("forecast_units", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(stores))
    ax.bar(x - 0.18, stores["actual_units"], width=0.36, color=COLORS["navy"], label="Actual")
    ax.bar(x + 0.18, stores["forecast_units"], width=0.36, color=COLORS["blue"], label="Forecast")
    ax.set_xticks(x)
    ax.set_xticklabels(stores["store_id"])
    ax.set_title("Store demand leaderboard")
    ax.set_ylabel("Units (horizon)")
    ax.legend(frameon=False)
    fig.tight_layout()
    paths["store_leaderboard"] = out_dir / "04_store_leaderboard.png"
    fig.savefig(paths["store_leaderboard"], dpi=160)
    plt.close(fig)

    # 5) Inventory risk — top series
    risk = tables["story_inventory_risk"].head(8).copy()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    colors = [
        COLORS["coral"]
        if "Stockout" in t
        else (COLORS["spark"] if "Overstock" in t else COLORS["teal"])
        for t in risk["risk_type"]
    ]
    ax.barh(risk["series_id"][::-1], risk["bias_units"][::-1], color=colors[::-1])
    ax.axvline(0, color=COLORS["slate"], lw=1)
    ax.set_title("Inventory risk board — forecast bias by series")
    ax.set_xlabel("Forecast − actual (units)  ·  negative = under-forecast / stockout risk")
    fig.tight_layout()
    paths["inventory_risk"] = out_dir / "05_inventory_risk.png"
    fig.savefig(paths["inventory_risk"], dpi=160)
    plt.close(fig)

    return paths


def _img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def write_story_html(
    tables: dict[str, pd.DataFrame],
    best_model: str,
    chart_paths: dict[str, Path],
    out_path: str | Path,
    title: str,
    subtitle: str,
) -> Path:
    """Self-contained HTML dashboard for customer storytelling when Tableau Desktop isn't open."""
    out_path = Path(out_path)
    kpi = tables["story_kpi"].iloc[0].to_dict()
    points = tables["story_talking_points"].sort_values("order")

    def cards() -> str:
        items = [
            ("Champion model", str(kpi["champion_model"]).upper()),
            ("Forecast units", f"{kpi['forecast_units']:,.0f}"),
            ("Bias", f"{kpi['forecast_bias_units']:+,.0f} ({kpi['bias_pct']:+.1f}%)"),
            ("MAPE", f"{kpi['mape']:.1f}%"),
            ("RMSE", f"{kpi['rmse']:.2f}"),
            ("Series covered", f"{int(kpi['n_series'])}"),
        ]
        return "".join(
            f'<div class="kpi"><div class="label">{lab}</div><div class="value">{val}</div></div>'
            for lab, val in items
        )

    def talking() -> str:
        blocks = []
        for _, row in points.iterrows():
            blocks.append(
                f"""
                <article class="point">
                  <div class="tag">{row['section']}</div>
                  <p class="insight">{row['insight']}</p>
                  <p class="ask"><strong>Ask:</strong> {row['business_ask']}</p>
                </article>
                """
            )
        return "".join(blocks)

    def fig(key: str, caption: str) -> str:
        p = chart_paths.get(key)
        if not p or not Path(p).exists():
            return ""
        b64 = _img_b64(Path(p))
        return f"""
        <figure class="chart">
          <img src="data:image/png;base64,{b64}" alt="{caption}" />
          <figcaption>{caption}</figcaption>
        </figure>
        """

    # Compact risk + snap tables
    risk_html = tables["story_inventory_risk"].head(6)[
        ["series_id", "cat_id", "bias_units", "risk_type"]
    ].to_html(index=False, classes="data", border=0, float_format=lambda x: f"{x:,.1f}")
    snap_html = tables["story_snap_lift"].to_html(
        index=False, classes="data", border=0, float_format=lambda x: f"{x:,.1f}"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<style>
  :root {{
    --navy: #041E42;
    --blue: #0071CE;
    --spark: #FFC220;
    --slate: #5B6770;
    --bg: #F4F7FB;
    --card: #ffffff;
    --line: #D7E0EA;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    color: var(--navy);
    background:
      radial-gradient(1200px 500px at 10% -10%, #d9e8f8 0%, transparent 55%),
      radial-gradient(900px 400px at 90% 0%, #ffe8a3 0%, transparent 45%),
      var(--bg);
  }}
  .hero {{
    padding: 2.4rem 7vw 1.4rem;
    border-bottom: 1px solid var(--line);
    background: linear-gradient(180deg, rgba(4,30,66,0.04), transparent);
  }}
  .eyebrow {{
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-size: 0.75rem;
    color: var(--blue);
    font-weight: 700;
    margin-bottom: 0.55rem;
  }}
  h1 {{
    margin: 0 0 0.55rem;
    font-size: clamp(1.8rem, 3vw, 2.6rem);
    line-height: 1.15;
  }}
  .subtitle {{ max-width: 62rem; color: var(--slate); font-size: 1.05rem; }}
  .wrap {{ padding: 1.5rem 7vw 3rem; }}
  .kpis {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.8rem;
    margin: 1.2rem 0 1.8rem;
  }}
  .kpi {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 0.95rem 1rem;
    box-shadow: 0 8px 24px rgba(4,30,66,0.04);
  }}
  .kpi .label {{ color: var(--slate); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }}
  .kpi .value {{ margin-top: 0.35rem; font-size: 1.35rem; font-weight: 700; }}
  section {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 1.25rem 1.35rem 1.4rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 10px 28px rgba(4,30,66,0.04);
  }}
  section h2 {{
    margin: 0 0 0.35rem;
    font-size: 1.25rem;
  }}
  section .lead {{ color: var(--slate); margin: 0 0 1rem; }}
  .grid-2 {{
    display: grid;
    grid-template-columns: 1.2fr 0.8fr;
    gap: 1rem;
  }}
  @media (max-width: 960px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
  .chart {{ margin: 0; }}
  .chart img {{ width: 100%; height: auto; border-radius: 10px; border: 1px solid var(--line); }}
  .chart figcaption {{ color: var(--slate); font-size: 0.88rem; margin-top: 0.45rem; }}
  .points {{ display: grid; gap: 0.75rem; }}
  .point {{
    border-left: 4px solid var(--spark);
    background: #fffdf5;
    padding: 0.75rem 0.9rem;
    border-radius: 0 10px 10px 0;
  }}
  .tag {{
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--blue);
    margin-bottom: 0.25rem;
  }}
  .insight {{ margin: 0.15rem 0 0.4rem; }}
  .ask {{ margin: 0; color: var(--slate); font-size: 0.92rem; }}
  table.data {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.92rem;
  }}
  table.data th, table.data td {{
    text-align: left;
    padding: 0.55rem 0.45rem;
    border-bottom: 1px solid var(--line);
  }}
  table.data th {{ color: var(--slate); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  footer {{
    padding: 0 7vw 2.5rem;
    color: var(--slate);
    font-size: 0.85rem;
  }}
</style>
</head>
<body>
  <header class="hero">
    <div class="eyebrow">Walmart customer story · Champion model = {best_model}</div>
    <h1>{title}</h1>
    <p class="subtitle">{subtitle}</p>
  </header>
  <main class="wrap">
    <div class="kpis">{cards()}</div>

    <section>
      <h2>1. Opening narrative</h2>
      <p class="lead">{kpi['story_headline']}</p>
      <div class="points">{talking()}</div>
    </section>

    <section>
      <h2>2. Demand pulse</h2>
      <p class="lead">Actual vs champion forecast across the planning horizon. Yellow markers highlight SNAP-window days.</p>
      {fig("demand_pulse", "Use this chart to open the story: are we tracking the demand shape?")}
    </section>

    <div class="grid-2">
      <section>
        <h2>3. Assortment focus</h2>
        <p class="lead">Category concentration guides shelf space and replenishment priority.</p>
        {fig("category_mix", "Lead with the category that owns the unit share.")}
      </section>
      <section>
        <h2>4. SNAP / deal timing</h2>
        <p class="lead">Lift in SNAP windows vs non-SNAP — primary signal for marketing calendars.</p>
        {fig("snap_lift", "FOODS lift is the talking point for circular timing.")}
        {snap_html}
      </section>
    </div>

    <section>
      <h2>5. Store leaderboard</h2>
      <p class="lead">Where network demand (and forecast attention) concentrates by store.</p>
      {fig("store_leaderboard", "Compare store actuals vs champion forecast for ops staffing.")}
    </section>

    <div class="grid-2">
      <section>
        <h2>6. Inventory risk board</h2>
        <p class="lead">Negative bias = under-forecast / stockout risk. Positive = overstock watchlist.</p>
        {fig("inventory_risk", "Close with the exception list merchandisers can action this week.")}
      </section>
      <section>
        <h2>Priority exceptions</h2>
        <p class="lead">Top series by risk priority from the champion model.</p>
        {risk_html}
      </section>
    </div>
  </main>
  <footer>
    Generated by the M5 forecasting batch pipeline · Tableau extracts live in
    <code>outputs/tableau/story/</code> · Rebuild with
    <code>python3 scripts/run_pipeline.py predict</code>
  </footer>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path
