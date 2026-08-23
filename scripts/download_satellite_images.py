"""
Downloads daily NASA GIBS satellite imagery over India, from 2019-01-01
through yesterday (GIBS has ~1-day publish latency, unlike the IMD rainfall
data elsewhere in this repo, which lags by roughly a year) -- re-running
periodically keeps both datasets current:

- **Visible** true-color imagery -> data/satellite/india_cloud/ -- what a
  camera sees. Used by weather/nowcast_model.py's CNN. (IMG_DIR/LOG_DIR/
  LOG_CSV/ERROR_LOG below are this dataset's paths specifically, kept as
  plain module-level names for backward compatibility -- verify_satellite_images.py
  and retry_failed_dates.py both import them directly.)
- **Thermal** infrared brightness temperature -> data/satellite/india_thermal/
  -- cloud-top temperature, the actual physical signal that correlates with
  rain (colder cloud tops mean taller, more violent storms). Not yet wired
  into any model; see weather/thermal_features.py for decoding it.

Both use the public GIBS WMS GetMap endpoint (no API key/login needed), the
same fallback-through-layers logic, and the same resumability: re-running
skips any date whose PNG already exists, so a stopped run can just be
restarted. See docs/satellite_images.md for details.
"""
import csv
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent

# --- Visible dataset (original paths, unchanged -- other scripts import these directly) ---
IMG_DIR = BASE_DIR / "data" / "satellite" / "india_cloud"
LOG_DIR = BASE_DIR / "data" / "satellite" / "logs"
LOG_CSV = LOG_DIR / "download_log.csv"
ERROR_LOG = LOG_DIR / "errors.txt"
LAYERS = [
    "MODIS_Terra_CorrectedReflectance_TrueColor",
    "MODIS_Aqua_CorrectedReflectance_TrueColor",
    "VIIRS_SNPP_CorrectedReflectance_TrueColor",
]

# --- Thermal dataset (new, parallel, separate paths -- doesn't touch the above) ---
THERMAL_IMG_DIR = BASE_DIR / "data" / "satellite" / "india_thermal"
THERMAL_LOG_DIR = BASE_DIR / "data" / "satellite" / "logs_thermal"
THERMAL_LOG_CSV = THERMAL_LOG_DIR / "download_log.csv"
THERMAL_ERROR_LOG = THERMAL_LOG_DIR / "errors.txt"
THERMAL_LAYERS = [
    "MODIS_Terra_Brightness_Temp_Band31_Day",
    "MODIS_Aqua_Brightness_Temp_Band31_Day",
    "VIIRS_SNPP_Brightness_Temp_BandI5_Day",
]

WMS_ENDPOINT = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
BBOX_INDIA = "68,6,98,38"  # lon_min,lat_min,lon_max,lat_max
WIDTH, HEIGHT = 1200, 1000
SRS = "EPSG:4326"

START_DATE = date(2019, 1, 1)
END_DATE = date.today() - timedelta(days=1)  # GIBS has ~1-day publish latency

REQUEST_DELAY_SEC = 0.7
REQUEST_TIMEOUT_SEC = 30
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# GIBS returns a small, near-uniform PNG for no-data/blank tiles; a real
# 1200x1000 scene (photographic or a smooth thermal gradient) compresses to
# well above this. No Pillow dependency needed just for this check, so file
# size is the practical stand-in for "is this a real image".
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


def fetch_day(date_str, layers, log_writer, log_file):
    """Try each layer in order; return (image_bytes, layer_name) or (None, last_error)."""
    last_error = "no layers attempted"
    for layer in layers:
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


def run_dataset(name, img_dir, log_dir, log_csv, error_log, layers):
    for year in range(START_DATE.year, END_DATE.year + 1):
        (img_dir / str(year)).mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_is_new = not log_csv.exists()
    total_days = (END_DATE - START_DATE).days + 1

    success_count = 0
    fail_dates = []
    layer_success_counts = {layer: 0 for layer in layers}

    with open(log_csv, "a", newline="") as log_file, open(error_log, "a") as error_file:
        log_writer = csv.writer(log_file)
        if log_is_new:
            log_writer.writerow(["date", "layer_used", "status", "filesize"])
            log_file.flush()

        bar = tqdm(
            daterange(START_DATE, END_DATE), total=total_days, unit="day",
            desc=f"Satellite imagery ({name})", dynamic_ncols=True,
        )
        for day in bar:
            date_str = day.isoformat()
            out_path = img_dir / str(day.year) / f"{date_str}.png"

            if out_path.exists():
                bar.set_postfix(new=success_count, failed=len(fail_dates), refresh=False)
                continue

            try:
                content, result = fetch_day(date_str, layers, log_writer, log_file)
                if content is not None:
                    out_path.write_bytes(content)
                    success_count += 1
                    layer_success_counts[result] += 1
                else:
                    fail_dates.append(date_str)
                    error_file.write(f"{date_str}: {result}\n")
                    error_file.flush()
                    bar.write(f"FAILED {date_str} ({result})")
            except Exception as e:
                fail_dates.append(date_str)
                error_file.write(f"{date_str}: unexpected error ({e})\n")
                error_file.flush()
                bar.write(f"FAILED {date_str} (unexpected error: {e})")

            bar.set_postfix(new=success_count, failed=len(fail_dates), refresh=False)

    print(f"=== {name} summary ===")
    print(f"Total days in range:  {total_days}")
    print(f"Newly downloaded:     {success_count}")
    print(f"Failed (all layers):  {len(fail_dates)}")
    if fail_dates:
        print(f"  See {error_log} for details.")
    if success_count:
        best_layer = max(layer_success_counts, key=layer_success_counts.get)
        print(f"Most-used layer this run: {best_layer} ({layer_success_counts[best_layer]} downloads)")
    print()


def main():
    run_dataset("visible", IMG_DIR, LOG_DIR, LOG_CSV, ERROR_LOG, LAYERS)
    run_dataset("thermal", THERMAL_IMG_DIR, THERMAL_LOG_DIR, THERMAL_LOG_CSV, THERMAL_ERROR_LOG, THERMAL_LAYERS)


if __name__ == "__main__":
    sys.exit(main())
