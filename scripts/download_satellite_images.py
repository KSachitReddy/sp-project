"""
Download daily NASA GIBS satellite imagery (cloud cover) over India, 2019-2025.

Uses the public GIBS WMS GetMap endpoint (no API key/login needed) to fetch one
PNG per day into data/satellite/india_cloud/<year>/YYYY-MM-DD.png, alongside
the .nc rainfall data already in this repo.

For each date the primary layer (MODIS Terra true-color) is tried first; if the
response is missing, an error, or an implausibly small/blank tile, the script
falls back through MODIS Aqua then VIIRS SNPP before giving up on that date.

Resumable: re-running skips any date whose PNG already exists on disk, so a
stopped run can just be restarted. See docs/satellite_images.md for details.
"""
import csv
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests

BASE_DIR = Path(__file__).parent.parent
IMG_DIR = BASE_DIR / "data" / "satellite" / "india_cloud"
LOG_DIR = BASE_DIR / "data" / "satellite" / "logs"
LOG_CSV = LOG_DIR / "download_log.csv"
ERROR_LOG = LOG_DIR / "errors.txt"

WMS_ENDPOINT = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
BBOX_INDIA = "68,6,98,38"  # lon_min,lat_min,lon_max,lat_max
WIDTH, HEIGHT = 1200, 1000
SRS = "EPSG:4326"

# Tried in order; the first one that yields a plausible image wins.
LAYERS = [
    "MODIS_Terra_CorrectedReflectance_TrueColor",
    "MODIS_Aqua_CorrectedReflectance_TrueColor",
    "VIIRS_SNPP_CorrectedReflectance_TrueColor",
]

START_DATE = date(2019, 1, 1)
END_DATE = date(2025, 12, 31)

REQUEST_DELAY_SEC = 0.7
REQUEST_TIMEOUT_SEC = 30
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# GIBS returns a small, near-uniform PNG for no-data/blank tiles; a real
# 1200x1000 true-color scene compresses to well above this. No Pillow in this
# project's deps, so file size is the practical stand-in for "is this a real
# image" rather than decoding pixels.
MIN_VALID_BYTES = 5000


def daterange(start, end):
    current = start
    one_day = timedelta(days=1)
    while current <= end:
        yield current
        current += one_day


def build_params(layer, date_str):
    return {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "STYLES": "",
        "FORMAT": "image/png",
        "TRANSPARENT": "FALSE",
        "SRS": SRS,
        "BBOX": BBOX_INDIA,
        "WIDTH": WIDTH,
        "HEIGHT": HEIGHT,
        "TIME": date_str,
    }


def validate_image(response):
    """Return (content_bytes_or_None, reason_if_invalid_or_None)."""
    if response.status_code != 200:
        return None, f"HTTP {response.status_code}"

    content_type = response.headers.get("Content-Type", "")
    if not content_type.startswith("image/png"):
        return None, f"non-image response ({content_type or 'no content-type'})"

    content = response.content
    if not content:
        return None, "empty response body"
    if not content.startswith(PNG_SIGNATURE):
        return None, "response is not a valid PNG"
    if len(content) < MIN_VALID_BYTES:
        return None, f"suspiciously small ({len(content)} bytes) — likely blank/no-data tile"

    return content, None


def fetch_day(date_str, log_writer, log_file):
    """Try each layer in order; return (image_bytes, layer_name) or (None, last_error)."""
    last_error = "no layers attempted"
    for layer in LAYERS:
        try:
            resp = requests.get(
                WMS_ENDPOINT, params=build_params(layer, date_str), timeout=REQUEST_TIMEOUT_SEC
            )
        except requests.exceptions.RequestException as e:
            last_error = f"{layer}: request failed ({e})"
            log_writer.writerow([date_str, layer, "fail", 0])
            log_file.flush()
            time.sleep(REQUEST_DELAY_SEC)
            continue

        content, reason = validate_image(resp)
        if content is not None:
            log_writer.writerow([date_str, layer, "success", len(content)])
            log_file.flush()
            return content, layer

        last_error = f"{layer}: {reason}"
        log_writer.writerow([date_str, layer, "fail", 0])
        log_file.flush()
        time.sleep(REQUEST_DELAY_SEC)

    return None, last_error


def main():
    for year in range(START_DATE.year, END_DATE.year + 1):
        (IMG_DIR / str(year)).mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_is_new = not LOG_CSV.exists()
    total_days = (END_DATE - START_DATE).days + 1

    success_count = 0
    skip_count = 0
    fail_dates = []
    layer_success_counts = {layer: 0 for layer in LAYERS}

    with open(LOG_CSV, "a", newline="") as log_file, open(ERROR_LOG, "a") as error_file:
        log_writer = csv.writer(log_file)
        if log_is_new:
            log_writer.writerow(["date", "layer_used", "status", "filesize"])
            log_file.flush()

        for i, day in enumerate(daterange(START_DATE, END_DATE), start=1):
            date_str = day.isoformat()
            out_path = IMG_DIR / str(day.year) / f"{date_str}.png"

            if out_path.exists():
                skip_count += 1
                continue

            try:
                content, result = fetch_day(date_str, log_writer, log_file)
                if content is not None:
                    out_path.write_bytes(content)
                    success_count += 1
                    layer_success_counts[result] += 1
                    print(f"Downloaded {i}/{total_days} - {date_str}.png ({result})")
                else:
                    fail_dates.append(date_str)
                    error_file.write(f"{date_str}: {result}\n")
                    error_file.flush()
                    print(f"FAILED {i}/{total_days} - {date_str} ({result})")
            except Exception as e:
                fail_dates.append(date_str)
                error_file.write(f"{date_str}: unexpected error ({e})\n")
                error_file.flush()
                print(f"FAILED {i}/{total_days} - {date_str} (unexpected error: {e})")

    print()
    print("=== Summary ===")
    print(f"Total days in range:  {total_days}")
    print(f"Already downloaded:   {skip_count}")
    print(f"Newly downloaded:     {success_count}")
    print(f"Failed (all layers):  {len(fail_dates)}")
    if fail_dates:
        print(f"  See {ERROR_LOG} for details.")
    if success_count:
        best_layer = max(layer_success_counts, key=layer_success_counts.get)
        print(f"Most-used layer this run: {best_layer} ({layer_success_counts[best_layer]} downloads)")


if __name__ == "__main__":
    sys.exit(main())
