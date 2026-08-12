# Satellite Imagery Dataset (India, 2019–2025)

Daily true-color satellite snapshots over India, downloaded from NASA's GIBS
service, meant to sit alongside the IMD RF25 rainfall `.nc` data already in
this repo (same country, comparable timeframe — useful for eyeballing cloud
cover against recorded rainfall).

## Data source

- **API**: [NASA GIBS](https://wiki.earthdata.nasa.gov/display/GIBS) WMS
  `GetMap` endpoint — `https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi`
- No API key or login required.
- **Area**: India bounding box, `68,6,98,38` (lon_min,lat_min,lon_max,lat_max), EPSG:4326
- **Image**: 1200x1000 PNG, one per day
- **Layer** (tried in order, first plausible result wins):
  1. `MODIS_Terra_CorrectedReflectance_TrueColor`
  2. `MODIS_Aqua_CorrectedReflectance_TrueColor` (fallback)
  3. `VIIRS_SNPP_CorrectedReflectance_TrueColor` (fallback)

A response is treated as invalid (triggering the next fallback layer) if the
request fails, the content type isn't `image/png`, the body isn't a valid PNG,
or the file is implausibly small (< 5 KB — GIBS returns a small near-uniform
tile for blank/no-data responses, and this project has no image-decoding
dependency to inspect pixels directly).

## Folder structure

```
data/satellite/
  india_cloud/
    2019/
      2019-01-01.png
      2019-01-02.png
      ...
    2020/
    ...
    2025/
  logs/
    download_log.csv        # every attempt: date, layer_used, status, filesize
    errors.txt               # dates that failed on all three layers, with the reason
    verification_report.txt  # written by scripts/verify_satellite_images.py
    retry_dates.txt           # missing/corrupt dates for scripts/retry_failed_dates.py
```

`data/satellite/` is gitignored — it's a few thousand PNGs (multiple GB) and
is fully regenerable from the script, so it isn't committed. The rainfall
`.nc`/`.csv` data stays tracked because it's much smaller.

## Running / resuming

```bash
.venv/bin/pip install -r requirements.txt   # one-time, if not already installed
.venv/bin/python scripts/download_satellite_images.py
```

The script walks every date from 2019-01-01 to 2025-12-31. Before downloading
a date it checks whether `data/satellite/india_cloud/<year>/<date>.png`
already exists and skips it if so — so you can `Ctrl+C` at any point and just
re-run the same command later; it picks up where it left off without
re-downloading anything.

There's a ~0.7s delay between HTTP requests to avoid hammering the GIBS
server, so a full 7-year run (2,557 days) takes a while — expect it to run
for a few hours depending on how many dates need fallback retries.

At the end it prints a summary: total days in range, how many were already
present, how many were newly downloaded, how many failed on every layer, and
which layer was used most often that run. Permanently failed dates (all three
layers exhausted) are appended to `logs/errors.txt` — you can grep those out
and re-run the script later if NASA's servers were just having a bad day.

## Verifying and retrying gaps

```bash
.venv/bin/python scripts/verify_satellite_images.py
```

Checks every expected date for a present, non-corrupt, non-suspiciously-small
PNG; writes `data/satellite/logs/verification_report.txt` (completeness by
year, layer-usage breakdown, final READY TO USE / NEEDS ATTENTION verdict)
and `data/satellite/logs/retry_dates.txt` (any missing/corrupt dates).

If `retry_dates.txt` is non-empty:

```bash
.venv/bin/python scripts/retry_failed_dates.py
```

Re-attempts only those dates (same fallback-layer logic), then re-run the
verify script to confirm the gaps closed. Repeat a couple of times if needed —
most gaps are transient server hiccups rather than permanent failures.
