"""
Downloads ERA5 reanalysis data (temperature, humidity, wind, pressure) over
India, via the Copernicus Climate Data Store (CDS) API.

Unlike every other download script in this project, this one NEEDS a
one-time account setup that only you can do -- there's no way to automate
creating an account or generating your personal API key for you:

  1. Register (free): https://cds.climate.copernicus.eu/
  2. Get your API key from your profile page there
  3. Create ~/.cdsapirc with:
       url: https://cds.climate.copernicus.eu/api
       key: <your-personal-key>

This script checks for that file upfront and tells you exactly what's
missing rather than failing with a cryptic auth error partway through.

Why this one matters most: the rainfall forecaster currently only sees
lat/lon, day-of-year, and historical rainfall statistics -- no actual
atmospheric information. ERA5 is the single dataset that fills that whole
gap in one shot: temperature, humidity, wind, and pressure together are
the "core four" inputs any real weather forecast is built on.

    .venv/bin/python scripts/download_era5.py --year 2023
"""
import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUT_DIR = BASE_DIR / "data" / "parameters" / "era5"
CDSAPI_RC = Path.home() / ".cdsapirc"

AREA = [38, 68, 6, 98]  # India bounding box: North, West, South, East -- matches the rest of this project

VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "total_column_water_vapour",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", required=True, type=int, help="Year to download, e.g. 2023.")
    args = parser.parse_args()

    if not CDSAPI_RC.exists():
        print(
            "No ~/.cdsapirc found -- this needs your own free Copernicus Climate Data\n"
            "Store account (nothing here can create one for you):\n"
            "  1. Register: https://cds.climate.copernicus.eu/\n"
            "  2. Get your API key from your profile page there\n"
            "  3. Create ~/.cdsapirc with:\n"
            "       url: https://cds.climate.copernicus.eu/api\n"
            "       key: <your-personal-key>\n"
            "Then re-run this script."
        )
        sys.exit(1)

    try:
        import cdsapi
    except ImportError:
        print("The cdsapi package isn't installed -- run `uv sync` (plain Python package, no special hardware needed).")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"era5_{args.year}.nc"

    print(
        f"Requesting ERA5 data for {args.year} over India -- this is a queued job on Copernicus's "
        f"servers, not a direct download; expect anywhere from a few minutes to an hour depending on "
        f"how busy they are."
    )
    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": VARIABLES,
            "year": str(args.year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "area": AREA,
            "format": "netcdf",
        },
        str(out_path),
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
