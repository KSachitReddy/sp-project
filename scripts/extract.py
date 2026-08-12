"""
Extract IMD gridded rainfall NetCDF (RF25_ind2023_rfp25.nc) into:
  1. rainfall_2023.csv  - date, latitude, longitude, rainfall_mm (one row per valid land cell per day)
  2. rainfall_2023.json - compact, grid-indexed JSON for fast browser loading

Land-cell mask = grid points that have at least one non-NaN value across the year.
Ocean / no-data cells are NaN in the source and are excluded entirely.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = Path(__file__).parent.parent
SRC = BASE_DIR / "data" / "nc" / "RF25_ind2023_rfp25.nc"
OUT_DIR = BASE_DIR / "data" / "rainfall"

print("Opening dataset...")
ds = xr.open_dataset(SRC)

lat = ds["LATITUDE"].values.astype(float)   # (129,)
lon = ds["LONGITUDE"].values.astype(float)  # (135,)
time = ds["TIME"].values                    # (365,) datetime64
rainfall = ds["RAINFALL"].values.astype(np.float32)  # (365, 129, 135)

dates = pd.to_datetime(time).strftime("%Y-%m-%d").tolist()

# ---- Build land mask: valid (non-NaN) on at least one day ----
land_mask = ~np.isnan(rainfall).all(axis=0)  # (lat, lon)
lat_idx, lon_idx = np.where(land_mask)
n_cells = len(lat_idx)
print(f"Land cells: {n_cells} / {land_mask.size}")

# Order cells north-to-south is not required; keep natural (lat asc, lon asc) grid order
cell_lats = lat[lat_idx]
cell_lons = lon[lon_idx]

# ---------------------------------------------------------------
# PART 1a: Full long-format CSV
# ---------------------------------------------------------------
print("Building CSV rows...")
# rainfall for just the land cells, shape (365, n_cells)
rf_cells = rainfall[:, lat_idx, lon_idx]  # (365, n_cells)

# Build long-format arrays via broadcasting, then drop NaNs (rare missing-day gaps)
n_days = len(dates)
date_col = np.repeat(np.array(dates, dtype=object), n_cells)
lat_col = np.tile(cell_lats, n_days)
lon_col = np.tile(cell_lons, n_days)
rain_col = rf_cells.reshape(-1)

valid = ~np.isnan(rain_col)
df = pd.DataFrame({
    "date": date_col[valid],
    "latitude": np.round(lat_col[valid], 2),
    "longitude": np.round(lon_col[valid], 2),
    "rainfall_mm": np.round(rain_col[valid], 1),
})

csv_path = f"{OUT_DIR}/rainfall_2023.csv"
df.to_csv(csv_path, index=False)
print(f"CSV written: {csv_path} ({len(df):,} rows)")

# ---------------------------------------------------------------
# PART 1b: Compact grid-indexed JSON
# ---------------------------------------------------------------
print("Building compact JSON...")

cells = [[round(float(la), 2), round(float(lo), 2)] for la, lo in zip(cell_lats, cell_lons)]

# rainfall[day][cell] rounded to 1 decimal, null where missing (rare)
rainfall_by_day = []
daily_avg = []
for d in range(n_days):
    day_vals = rf_cells[d]
    row = [None if np.isnan(v) else round(float(v), 1) for v in day_vals]
    rainfall_by_day.append(row)
    daily_avg.append(round(float(np.nanmean(day_vals)), 2))

payload = {
    "meta": {
        "source": "IMD RF25 gridded daily rainfall, India, 2023",
        "resolution_deg": 0.25,
        "units": "mm",
        "n_days": n_days,
        "n_cells": n_cells,
        "lat_min": float(lat.min()),
        "lat_max": float(lat.max()),
        "lon_min": float(lon.min()),
        "lon_max": float(lon.max()),
    },
    "dates": dates,
    "daily_avg_mm": daily_avg,
    "cells": cells,          # [[lat, lon], ...] length n_cells
    "rainfall": rainfall_by_day,  # [day][cell] length n_days x n_cells
}

json_path = f"{OUT_DIR}/rainfall_2023.json"
with open(json_path, "w") as f:
    json.dump(payload, f, separators=(",", ":"))

import os
print(f"JSON written: {json_path} ({os.path.getsize(json_path)/1e6:.2f} MB)")
print(f"CSV size: {os.path.getsize(csv_path)/1e6:.2f} MB")
