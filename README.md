# India Rainfall & Satellite Imagery

Tools for exploring India's daily rainfall (IMD RF25 gridded data, 2018–2025)
and daily satellite cloud cover (NASA GIBS, 2019–2025): a Flask app that
renders rainfall on an interactive map, plus batch scripts that convert the
raw NetCDF data and download the satellite imagery.

## Structure

```
app.py                   Flask entrypoint — serves the map page and its API
rainfall/
  nc_convert.py           Shared IMD RF25 NetCDF -> CSV/JSON conversion logic
scripts/
  convert_all.py           Batch-convert every data/nc/*.nc into data/rainfall/*.csv
  extract.py                One-off extractor for the 2023 CSV + JSON
  download_satellite_images.py   Download daily GIBS imagery for 2019-2025
  verify_satellite_images.py     Completeness/integrity check over the download
  retry_failed_dates.py           Re-attempt only the dates verify flagged
web/
  india_rainfall_2023.html  The map page app.py serves
data/
  nc/                       Source IMD RF25 NetCDF files (tracked in git)
  rainfall/                 Generated CSV/JSON rainfall exports (tracked in git)
  satellite/                Downloaded GIBS PNGs + logs (gitignored — regenerable, multi-GB)
docs/
  satellite_images.md       Satellite dataset details: source, format, how to run/resume
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Running the map app

```bash
.venv/bin/python app.py
```

Serves the rainfall map at `http://localhost:5050`, reading years from
`data/nc/` and generated CSVs from `data/rainfall/`. Uploading a new `.nc`
file through the UI validates it, saves it into `data/nc/`, and writes its
CSV into `data/rainfall/`.

## Batch-converting rainfall data

```bash
.venv/bin/python scripts/convert_all.py
```

Converts every `.nc` file in `data/nc/` to `data/rainfall/rainfall_<year>.csv`.

## Satellite imagery

See [docs/satellite_images.md](docs/satellite_images.md) for the GIBS data
source, folder layout, and how to run/resume/verify the download.
