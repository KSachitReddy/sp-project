"""
Extracts cold-cloud-fraction features from every downloaded thermal image and
caches them to data/satellite/thermal_features.csv, one row per (date, region).
Decoding a thermal PNG (color-LUT matching over every pixel) is the expensive
part of weather/thermal_features.py, so this is a persistent cache instead of
redoing it every time a model trains -- re-running only processes images
added since the last run (resumable, like the downloader itself).

    .venv/bin/python scripts/extract_thermal_features.py

Run after scripts/download_satellite_images.py (or `nimbus download`) fetches
new thermal imagery -- both `nimbus download` and `nimbus sync` call this
automatically. scripts/calibrate_thermal_rainfall.py reads this cache rather
than decoding images itself.
"""
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from weather import thermal_features as tf  # noqa: E402

FEATURES_PATH = BASE_DIR / "data" / "satellite" / "thermal_features.csv"
COLUMNS = ["date", "region"] + tf.FEATURE_COLS


def main():
    existing = pd.read_csv(FEATURES_PATH) if FEATURES_PATH.exists() else pd.DataFrame(columns=COLUMNS)
    done_dates = set(existing["date"])

    all_dates = sorted(p.stem for p in tf.THERMAL_DIR.glob("*/*.png"))
    todo = [d for d in all_dates if d not in done_dates]

    if not todo:
        print(f"Nothing new -- {len(done_dates)} date(s) already cached in {FEATURES_PATH}.")
        return

    print(f"Extracting features from {len(todo)} new thermal image(s) ({len(done_dates)} already cached)...")
    new_rows = []
    skipped = 0
    for date_str in tqdm(todo):
        by_region = tf.extract_feature_row(date_str)
        if not by_region:
            skipped += 1
            continue
        for region, feats in by_region.items():
            new_rows.append({"date": date_str, "region": region, **feats})

    if not new_rows:
        print(f"No usable images among the {len(todo)} new date(s) (skipped {skipped}).")
        return

    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(FEATURES_PATH, index=False)
    print(f"Wrote {len(combined)} total row(s) ({len(new_rows)} new) to {FEATURES_PATH}.")
    if skipped:
        print(f"Skipped {skipped} date(s) with no usable pixels (fully obscured).")


if __name__ == "__main__":
    main()
