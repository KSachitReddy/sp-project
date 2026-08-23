"""
Unified CLI for the AI Predictive Weather Analysis project.

    uv run weatherai download                              # fetch satellite imagery (resumable)
    uv run weatherai train                                 # train both models
    uv run weatherai train --model rainfall
    uv run weatherai forecast --place Mumbai --date 2026-08-25
    uv run weatherai serve                                 # launch the map + chatbot web app
"""
import json
import subprocess
import sys
from pathlib import Path

import click

BASE_DIR = Path(__file__).parent
SCRIPTS_DIR = BASE_DIR / "scripts"


@click.group()
def cli():
    """AI Predictive Weather Analysis -- data, models, and the web app."""


@cli.command()
def download():
    """Download NASA GIBS satellite imagery into data/satellite/ (resumable -- safe to re-run)."""
    raise SystemExit(subprocess.call([sys.executable, str(SCRIPTS_DIR / "download_satellite_images.py")]))


@cli.command()
@click.option(
    "--model", type=click.Choice(["rainfall", "satellite", "all"]), default="all",
    help="Which model to train.",
)
def train(model):
    """Train the rainfall forecaster (LightGBM) and/or the satellite nowcaster (CNN)."""
    if model in ("rainfall", "all"):
        from weather.forecast_model import train as train_rainfall
        click.echo("Training rainfall forecaster (LightGBM)...")
        _, metrics = train_rainfall()
        click.echo(json.dumps(metrics, indent=2))
    if model in ("satellite", "all"):
        try:
            from weather.nowcast_model import train as train_satellite
        except ImportError:
            click.echo(
                "Skipping satellite nowcaster: torch isn't installed. "
                "Run `uv sync --extra torch` on a 64-bit machine to add it, then retry.",
                err=True,
            )
            if model == "satellite":
                raise SystemExit(1)
        else:
            click.echo("Training satellite nowcaster (CNN)...")
            _, metrics = train_satellite()
            click.echo(json.dumps(metrics, indent=2))


@cli.command()
@click.option("--place", required=True, help="An Indian state, union territory, or city, e.g. 'Kerala'.")
@click.option(
    "--date", "date_str", required=True,
    help="YYYY-MM-DD. Dates already in the historical record (2018-2025) return observed "
         "rainfall; later dates return a forecast.",
)
def forecast(place, date_str):
    """Rainfall (mm) for a place on a given date -- observed if it's in the historical
    record, predicted (ML short-range or climatology) otherwise."""
    from weather import chatbot

    result = chatbot.get_weather(place, date_str)
    if "error" in result:
        click.echo(f"Error: {result['error']}", err=True)
        raise SystemExit(1)

    if "predicted_mm" in result:
        method = "short-range ML forecast" if result["method"] == "ml_short_range" else "climatological average"
        click.echo(f"{result['matched_place']}, {result['target_date']}: {result['predicted_mm']} mm ({method})")
        if result.get("note"):
            click.echo(result["note"])
    else:
        click.echo(f"{result['matched_place']}, {result['start_date']}: {result['total_rainfall_mm']} mm (observed)")


@cli.command()
def serve():
    """Launch the Flask map + chatbot web app at http://127.0.0.1:5050."""
    import app

    app.main()


def main():
    cli()


if __name__ == "__main__":
    main()
