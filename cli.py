"""
Unified CLI for Nimbus -- AI Predictive Weather Analysis.

    uv sync                                                # base install -- works on any hardware
    uv run nimbus setup                                 # detect hardware, configure accordingly
    uv run nimbus download                              # fetch satellite imagery (resumable)
    uv run nimbus train                                 # train both models
    uv run nimbus train --model rainfall
    uv run nimbus forecast --place Mumbai --date 2026-08-25
    uv run nimbus analyse --build                       # precompute real trend/anomaly stats
    uv run nimbus analyse --place Kerala --year 2023
    uv run nimbus sync                                  # pull, fetch new imagery, validate, commit + push
    uv run nimbus serve                                 # launch the map + chatbot web app
"""
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import click

BASE_DIR = Path(__file__).parent
SCRIPTS_DIR = BASE_DIR / "scripts"


@click.group()
def cli():
    """Nimbus -- AI predictive weather analysis: data, models, and the web app."""


@cli.command()
@click.option("--yes", is_flag=True, help="Don't prompt before making changes (non-interactive).")
def setup(yes):
    """Detect this machine's hardware and configure the project to match it.

    Run once after `uv sync`, before `nimbus serve` -- `uv sync` alone
    can't do this part, since it's a fixed dependency resolver, not a
    provisioning tool. This decides whether to install the satellite
    nowcaster's torch dependency, and offers to set up swap on low-RAM
    machines, based on what it actually finds on this machine.
    """
    from weather import hardware

    info = hardware.detect()
    click.echo("=== Hardware detected ===")
    click.echo(hardware.describe(info))
    click.echo()

    if info["tier"] == "minimal":
        click.echo(
            "This looks like ~1GHz-class hardware (e.g. a Raspberry Pi or an old netbook).\n"
            "Recommendation:\n"
            "  - Skip the satellite nowcaster (torch) -- not installed, not worth the RAM/CPU here.\n"
            "  - Don't train models on this machine. Train on a stronger one and copy the\n"
            "    models/ folder over -- it's a few hundred KB, not gigabytes.\n"
            "  - Use this machine only to run `nimbus serve`: reading a small trained\n"
            "    model and a CSV is cheap; training or CNN inference is not."
        )
    elif info["tier"] == "modest":
        click.echo(
            "AVX2 isn't confirmed on this CPU. Recommendation:\n"
            "  - Skip the satellite nowcaster (torch) -- its CPU wheel has a documented\n"
            "    history of crashing with \"Illegal instruction\" on CPUs like this.\n"
            "  - Everything else (map, chatbot, rainfall forecaster) runs fine here."
        )
    else:
        click.echo("AVX2 confirmed -- safe to install the satellite nowcaster if you want it.")
        try:
            import torch  # noqa: F401
            click.echo("torch is already installed.")
        except ImportError:
            if yes or click.confirm("Run `uv sync --extra torch` now to enable it?", default=True):
                subprocess.call(["uv", "sync", "--extra", "torch"])

    click.echo()
    ram, swap = info["ram_gb"], info["swap_gb"] or 0
    if ram is not None and ram <= 8 and swap < 8:
        click.echo(
            f"RAM is {ram} GB with only {swap} GB of swap. A low-RAM machine running Ollama "
            "and Flask together can hit the kernel's out-of-memory killer with no cushion --\n"
            "this happened during this project's own development and took down the IDE, not "
            "just the app."
        )
        size = 12  # GB -- within the 10-15 GB range that's comfortable alongside Ollama
        swap_cmd = (
            f"sudo fallocate -l {size}G /swapfile && sudo chmod 600 /swapfile && "
            f"sudo mkswap /swapfile && sudo swapon /swapfile && "
            f"echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab"
        )
        if not yes and click.confirm(f"Set up a {size} GB swapfile now? (needs sudo -- will prompt for your password)", default=False):
            subprocess.call(swap_cmd, shell=True)
        else:
            click.echo("Skipped. Run this yourself later if you want it:")
            click.echo(f"  {swap_cmd}")

    click.echo()
    click.echo("Setup finished. Start the app with: uv run nimbus serve")


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
        caveat = "  (rough estimate -- might not be accurate)" if result["method"] == "climatology_outlook" else ""
        click.echo(f"{result['matched_place']}, {result['target_date']}: {result['predicted_mm']} mm{caveat}")
    else:
        click.echo(f"{result['matched_place']}, {result['start_date']}: {result['total_rainfall_mm']} mm")


@cli.command()
@click.option("--build", "do_build", is_flag=True, help="(Re)build the analysis database from all rainfall data on disk.")
@click.option("--place", help="An Indian state, union territory, or city, e.g. 'Kerala'.")
@click.option("--year", type=int, help="Year for a single-year summary (with --place).")
@click.option("--trend", "trend_years", type=int, nargs=2, metavar="START END", help="Year-over-year trend for a range (with --place).")
def analyse(do_build, place, year, trend_years):
    """Real computed rainfall trend/anomaly stats -- build once (reads the
    full dataset; do this on a capable machine, not ~1GHz-class hardware),
    then query cheaply anywhere."""
    from weather import analysis

    if do_build:
        click.echo("Building analysis database (reads the full rainfall history)...")
        n = analysis.build()
        click.echo(f"Wrote {n} place/year/month rows to {analysis.DB_PATH}")
        return

    if not place or (not year and not trend_years):
        click.echo("Specify --place with either --year or --trend START END (or --build to precompute).", err=True)
        raise SystemExit(1)

    result = analysis.trend(place, *trend_years) if trend_years else analysis.yearly_summary(place, year)
    if "error" in result:
        click.echo(f"Error: {result['error']}", err=True)
        raise SystemExit(1)
    click.echo(json.dumps(result, indent=2))


@cli.command()
@click.option("--no-push", is_flag=True, help="Commit locally but don't push to the remote.")
def sync(no_push):
    """Pull the latest committed models/analysis state, fetch any new
    satellite imagery, check today's nowcast prediction against real data as
    it becomes available, refresh the analysis database, then commit (and by
    default push) the results back to the repo. Run this whenever you open
    the project.

    Satellite images themselves are never committed (data/satellite/ stays
    gitignored -- multi-GB of regenerable PNGs don't belong in git); only the
    small derived state (models/, data/forecast/) is synced through git.
    """
    from weather import analysis

    click.echo("=== Pulling latest committed state ===")
    subprocess.call(["git", "pull"])

    click.echo("=== Checking for new satellite imagery ===")
    subprocess.call([sys.executable, str(SCRIPTS_DIR / "download_satellite_images.py")])

    click.echo("=== Logging today's nowcast prediction ===")
    image_date = analysis.log_nowcast_prediction()
    click.echo(f"Logged (image {image_date})." if image_date else "Skipped -- torch not installed, or no recent image.")

    click.echo("=== Checking pending nowcasts against real data ===")
    result = analysis.validate_pending_nowcasts()
    if result is None:
        click.echo("Nothing new to validate yet.")
    else:
        click.echo(
            f"Validated {len(result['newly_validated'])} new prediction(s). "
            f"All-time live MAE: {result['live_mae_all_time']} mm."
        )
        drift = result["drift"]
        if drift and drift["significant"]:
            click.echo(
                f"DRIFT DETECTED: live accuracy (MAE {drift['live_mae']} mm) is "
                f"{drift['relative_change'] * 100:+.0f}% off the original training backtest "
                f"(MAE {drift['backtest_mae']} mm) -- consider retraining with `nimbus train --model satellite`."
            )

    click.echo("=== Refreshing the analysis database ===")
    n = analysis.build()
    click.echo(f"{n} place/year/month rows.")

    click.echo("=== Committing ===")
    subprocess.call(["git", "add", "models/", "data/forecast/"])
    commit = subprocess.run(
        ["git", "commit", "-m", f"Automated sync: {date.today().isoformat()}"],
        capture_output=True, text=True,
    )
    click.echo(commit.stdout.strip() or commit.stderr.strip())
    if commit.returncode == 0:
        if no_push:
            click.echo("Skipped push (--no-push).")
        else:
            click.echo("=== Pushing ===")
            subprocess.call(["git", "push"])


@cli.command()
def serve():
    """Launch the Flask map + chatbot web app at http://127.0.0.1:5050."""
    import app

    app.main()


def main():
    cli()


if __name__ == "__main__":
    main()
