"""
Downloads elevation (metres) for every IMD RF25 grid cell over India, via
the public Open-Elevation API (open-elevation.com, no API key needed).

Static data -- India's terrain doesn't change -- so this is a one-time
download, unlike the satellite imagery downloader. Written to
data/parameters/elevation.csv (latitude, longitude, elevation_m).

Why this matters: India's rainfall is heavily shaped by terrain -- the
Western Ghats and Himalayas force monsoon winds upward, dumping much more
rain on the windward side than a few hundred km away on the leeward side.
The rainfall forecaster currently has no notion of this at all (only
lat/lon), so this is a cheap, high-value feature to add.

    .venv/bin/python scripts/download_elevation.py
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from weather.chatbot import _valid_cells  # noqa: E402

OUT_PATH = BASE_DIR / "data" / "parameters" / "elevation.csv"
API_URL = "https://api.open-elevation.com/api/v1/lookup"
BATCH_SIZE = 100
REQUEST_DELAY_SEC = 1.0  # be polite to a free public service
REQUEST_TIMEOUT_SEC = 30


def _fetch_batch(cells):
    payload = {"locations": [{"latitude": float(la), "longitude": float(lo)} for la, lo in cells]}
    resp = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT_SEC)
    resp.raise_for_status()
    return resp.json()["results"]


def main():
    cells = _valid_cells()
    if cells is None:
        print("No rainfall data on disk yet -- run `uv run nimbus download` / convert .nc files first.")
        sys.exit(1)

    print(f"Fetching elevation for {len(cells)} grid cells from Open-Elevation (no API key needed)...")

    rows = []
    failed_batches = 0
    for i in tqdm(range(0, len(cells), BATCH_SIZE), unit="batch", desc="Elevation"):
        batch = cells[i:i + BATCH_SIZE]
        try:
            for r in _fetch_batch(batch):
                rows.append({"latitude": r["latitude"], "longitude": r["longitude"], "elevation_m": r["elevation"]})
        except requests.exceptions.RequestException as e:
            failed_batches += 1
            tqdm.write(f"Batch starting at row {i} failed: {e}")
        time.sleep(REQUEST_DELAY_SEC)

    if not rows:
        print("No elevation data retrieved -- the public Open-Elevation API may be down. Try again later.")
        sys.exit(1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).drop_duplicates(subset=["latitude", "longitude"])
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)}/{len(cells)} grid cells to {OUT_PATH}")
    if failed_batches:
        print(f"{failed_batches} batch(es) failed -- re-run later to retry (safe: just overwrites this same file).")


if __name__ == "__main__":
    main()
