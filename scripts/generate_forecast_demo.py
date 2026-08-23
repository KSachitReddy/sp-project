"""
Generate a demo rainfall forecast for the rest of 2026, for every place in
the gazetteer (data/places/india_places.json).

    .venv/bin/python scripts/generate_forecast_demo.py

Writes data/forecast/forecast_demo.csv. Two regimes, both from
weather.forecast_model.predict_rainfall() -- see that module's docstring for
why: the first ~2 weeks past the last available rainfall data get a genuine
short-range ML forecast (method=ml_short_range); everything after that has no
real recent data to build lag features from, so it falls back to the
historical climatological average for that place and time of year
(method=climatology_outlook). The `method` column says which is which --
this file is a demo artifact, not a verified prediction, until IMD releases
actual 2026 rainfall data to backtest against.
"""
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from weather import chatbot
from weather import features as feat
from weather.forecast_model import model_available, predict_rainfall

OUT_DIR = Path(__file__).parent.parent / "data" / "forecast"
OUT_PATH = OUT_DIR / "forecast_demo.csv"


def target_dates(last_available, year_end):
    """Daily for the genuine short-range window, then weekly (climatology
    regime) for the rest of the year -- keeps the file a manageable size
    instead of one row per place per day for months of climatology repeats."""
    daily_end = last_available + timedelta(days=14)
    d = last_available + timedelta(days=1)
    while d <= daily_end:
        yield d
        d += timedelta(days=1)
    d += timedelta(days=6)  # first weekly point a week past the daily window
    while d <= year_end:
        yield d
        d += timedelta(days=7)


def main():
    if not model_available():
        print("Forecast model not trained yet -- run scripts/train_forecast_model.py first.")
        sys.exit(1)

    full = feat.load_rainfall()
    last_available = full["date"].max()
    year_end = pd.Timestamp(year=2026, month=12, day=31)

    places = chatbot._load_places()
    dates = list(target_dates(last_available, year_end))
    print(f"Forecasting {len(dates)} dates x {len(places)} places...")

    rows = []
    for place in places:
        grid_lat, grid_lon = chatbot.nearest_cell(place["lat"], place["lon"])
        for d in dates:
            result = predict_rainfall(grid_lat, grid_lon, d)
            if "error" in result:
                continue
            rows.append({
                "place": place["name"],
                "place_type": place["type"],
                "target_date": result["target_date"],
                "predicted_mm": result["predicted_mm"],
                "method": result["method"],
                "grid_lat": grid_lat,
                "grid_lon": grid_lon,
            })

    OUT_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")
    print(df["method"].value_counts())


if __name__ == "__main__":
    main()
