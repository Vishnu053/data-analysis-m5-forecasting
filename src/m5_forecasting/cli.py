"""Command-line entrypoint for the batch forecasting pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

# Ensure src/ is importable when running as a script
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from m5_forecasting.config import ensure_dirs, load_config
from m5_forecasting.pipeline.predict import run_batch_predict
from m5_forecasting.pipeline.prepare import prepare_datasets
from m5_forecasting.pipeline.train import train_models


@click.group()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to YAML config (default: configs/default.yaml)",
)
@click.pass_context
def main(ctx: click.Context, config_path: str | None) -> None:
    """Walmart M5 sales forecasting — production batch pipeline."""
    cfg = load_config(config_path)
    ensure_dirs(cfg)
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = cfg


@main.command("prepare")
@click.pass_context
def prepare_cmd(ctx: click.Context) -> None:
    """Ingest raw/synthetic data, clean outliers, build features."""
    paths = prepare_datasets(ctx.obj["cfg"])
    click.echo("Prepared datasets:")
    for k, v in paths.items():
        click.echo(f"  {k}: {v}")


@main.command("train")
@click.pass_context
def train_cmd(ctx: click.Context) -> None:
    """Fit enabled models (baseline, ARIMA, LightGBM, LSTM)."""
    registry = train_models(ctx.obj["cfg"])
    meta = registry.get("_meta", {})
    click.echo("Trained models:")
    click.echo(json.dumps(meta.get("models", {}), indent=2, default=str))


@main.command("predict")
@click.pass_context
def predict_cmd(ctx: click.Context) -> None:
    """Batch-score holdout and export Tableau-ready CSVs."""
    paths = run_batch_predict(ctx.obj["cfg"])
    click.echo("Batch prediction outputs:")
    for k, v in paths.items():
        click.echo(f"  {k}: {v}")


@main.command("story")
@click.pass_context
def story_cmd(ctx: click.Context) -> None:
    """Rebuild Walmart customer storytelling dashboard from latest forecasts."""
    import pandas as pd

    from m5_forecasting.pipeline.story_dashboard import export_customer_story

    cfg = ctx.obj["cfg"]
    forecasts_path = Path(cfg["paths"]["forecasts_dir"]) / "holdout_forecasts.parquet"
    metrics_path = Path(cfg["paths"]["metrics_dir"]) / "model_comparison.csv"
    test_path = Path(cfg["paths"]["models_dir"]) / "test_series.parquet"
    if not forecasts_path.exists() or not metrics_path.exists():
        raise click.ClickException("Run predict first so forecasts and metrics exist.")
    forecasts = pd.read_parquet(forecasts_path)
    metrics = pd.read_csv(metrics_path)
    test_series = pd.read_parquet(test_path)
    paths = export_customer_story(cfg, forecasts, metrics, test_series)
    click.echo("Customer story dashboard:")
    click.echo(f"  HTML: {paths.get('story_html')}")
    click.echo(f"  Tableau extracts: {paths.get('story_dir')}")


@main.command("run-all")
@click.pass_context
def run_all_cmd(ctx: click.Context) -> None:
    """Full pipeline: prepare → train → predict (+ customer story dashboard)."""
    cfg = ctx.obj["cfg"]
    click.echo("==> prepare")
    prepare_datasets(cfg)
    click.echo("==> train")
    train_models(cfg)
    click.echo("==> predict + customer story")
    paths = run_batch_predict(cfg)
    summary = Path(cfg["paths"]["metrics_dir"]) / "run_summary.json"
    if summary.exists():
        click.echo(summary.read_text())
    click.echo("Pipeline complete.")
    if "story_html" in paths:
        click.echo(f"  Customer story HTML: {paths['story_html']}")
    for k, v in paths.items():
        click.echo(f"  {k}: {v}")


if __name__ == "__main__":
    main()
