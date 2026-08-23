"""
Downloads three large-scale climate indices known to strongly modulate the
Indian monsoon on seasonal/sub-seasonal timescales -- none of this project's
current models use any of them:

  - ONI  (Oceanic Nino Index)              -- ENSO strength/phase, NOAA CPC
  - DMI  (Dipole Mode Index)                -- Indian Ocean Dipole, NOAA PSL
  - RMM  (Real-time Multivariate MJO index) -- Madden-Julian Oscillation,
                                                Australia's Bureau of Meteorology

All three are small, plain-text, publicly published time series -- no
account or API key needed, unlike ERA5 (see download_era5.py). Written to
data/parameters/{oni,iod,mjo}.csv.

    .venv/bin/python scripts/download_climate_indices.py
"""
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).parent.parent
OUT_DIR = BASE_DIR / "data" / "parameters"

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
IOD_URL = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"
MJO_URL = "http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt"
BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0"}  # BOM blocks the default requests/curl user-agent
TIMEOUT_SEC = 20


def _get(url, headers=None):
    resp = requests.get(url, headers=headers, timeout=TIMEOUT_SEC)
    resp.raise_for_status()
    return resp.text


def download_oni():
    """ENSO: one row per 3-month season (DJF, JFM, ...) per year."""
    df = pd.read_csv(StringIO(_get(ONI_URL)), sep=r"\s+")
    out = OUT_DIR / "oni.csv"
    df.to_csv(out, index=False)
    print(f"ONI (ENSO): wrote {len(df)} rows to {out}")


def download_iod():
    """IOD: one row per (year, month); -9999 marks missing/not-yet-observed and is dropped."""
    lines = _get(IOD_URL).splitlines()
    data_lines = []
    for line in lines[1:]:  # line 0 is "start_year end_year", not data
        stripped = line.strip()
        if not stripped or stripped == "-9999":  # sentinel marking end of data, footer follows
            break
        data_lines.append(stripped)

    rows = []
    for line in data_lines:
        parts = line.split()
        year = int(parts[0])
        for month, val in enumerate(parts[1:13], start=1):
            v = float(val)
            if v <= -9998:
                continue
            rows.append({"year": year, "month": month, "dmi": v})

    df = pd.DataFrame(rows)
    out = OUT_DIR / "iod.csv"
    df.to_csv(out, index=False)
    print(f"IOD (Dipole Mode Index): wrote {len(df)} rows to {out}")


def download_mjo():
    """MJO: one row per day. Missing values are documented as 1.E36 or 999."""
    lines = [
        l for l in _get(MJO_URL, headers=BROWSER_HEADERS).splitlines()
        if l.strip() and not l.strip().upper().startswith("RMM") and not l.strip().lower().startswith("year")
    ]

    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            rmm1, rmm2, phase, amplitude = float(parts[3]), float(parts[4]), int(parts[5]), float(parts[6])
        except ValueError:
            continue
        if abs(rmm1) > 1e30 or abs(rmm2) > 1e30 or abs(rmm1) == 999 or abs(rmm2) == 999:
            continue
        rows.append({
            "date": f"{year:04d}-{month:02d}-{day:02d}",
            "rmm1": rmm1, "rmm2": rmm2, "phase": phase, "amplitude": amplitude,
        })

    df = pd.DataFrame(rows)
    out = OUT_DIR / "mjo.csv"
    df.to_csv(out, index=False)
    print(f"MJO (RMM index): wrote {len(df)} rows to {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in [("ONI/ENSO", download_oni), ("IOD", download_iod), ("MJO", download_mjo)]:
        try:
            fn()
        except requests.exceptions.RequestException as e:
            print(f"{name}: download failed ({e}) -- skipping, re-run later to retry.")


if __name__ == "__main__":
    main()
