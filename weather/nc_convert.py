"""
Shared IMD RF25 NetCDF -> grid-payload / CSV conversion logic.

build_payload() is used by app.py's year selector to turn a NetCDF file
into the JSON the map frontend renders — read-only, no disk writes.

save_uploaded_nc() backs the .nc upload route: it validates the file first,
then persists it into "nc files/" and writes its CSV into "outputs/".

convert_nc_to_csv() is for the batch CLI (convert_all.py) that exports the
"nc files" folder to CSV once, outside of the running app.
"""
import os
import shutil

import numpy as np
import pandas as pd
import xarray as xr


class NcFormatError(ValueError):
    """Raised when an uploaded/given .nc file doesn't look like an IMD RF25 grid."""


def _find_var(ds, *candidates):
    lookup = {name.lower(): name for name in ds.variables}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _extract(nc_path):
    """Open an IMD RF25 gridded rainfall NetCDF file and pull out the land-cell grid."""
    ds = xr.open_dataset(nc_path)
    try:
        lat_name = _find_var(ds, "LATITUDE", "lat")
        lon_name = _find_var(ds, "LONGITUDE", "lon")
        time_name = _find_var(ds, "TIME", "time")
        rain_name = _find_var(ds, "RAINFALL", "rain", "precip", "precipitation")

        if not all([lat_name, lon_name, time_name, rain_name]):
            raise NcFormatError(
                "Expected LATITUDE/LONGITUDE/TIME/RAINFALL variables "
                "(an IMD RF25 gridded rainfall file) but couldn't find them all."
            )

        lat = ds[lat_name].values.astype(float)
        lon = ds[lon_name].values.astype(float)
        time = ds[time_name].values
        rainfall = ds[rain_name].values.astype(np.float32)
    finally:
        ds.close()

    if rainfall.ndim != 3 or rainfall.shape[0] != len(time):
        raise NcFormatError(
            f"RAINFALL has unexpected shape {rainfall.shape}; expected (TIME, LATITUDE, LONGITUDE)."
        )

    dates = pd.to_datetime(time).strftime("%Y-%m-%d").tolist()
    year = int(pd.to_datetime(time[0]).year)
    res = round(float(abs(lat[1] - lat[0])), 4) if len(lat) > 1 else 0.25

    land_mask = ~np.isnan(rainfall).all(axis=0)
    lat_idx, lon_idx = np.where(land_mask)
    n_cells = len(lat_idx)
    if n_cells == 0:
        raise NcFormatError("No valid (non-NaN) land cells found in this file.")

    return {
        "dates": dates,
        "year": year,
        "res": res,
        "lat": lat,
        "lon": lon,
        "cell_lats": lat[lat_idx],
        "cell_lons": lon[lon_idx],
        "rf_cells": rainfall[:, lat_idx, lon_idx],  # (n_days, n_cells)
        "n_days": len(dates),
        "n_cells": n_cells,
    }


def _payload_from_extract(g):
    rf_cells = g["rf_cells"]
    cells = [[round(float(la), 2), round(float(lo), 2)] for la, lo in zip(g["cell_lats"], g["cell_lons"])]

    rainfall_by_day = []
    daily_avg = []
    for d in range(g["n_days"]):
        day_vals = rf_cells[d]
        row = [None if np.isnan(v) else round(float(v), 1) for v in day_vals]
        rainfall_by_day.append(row)
        daily_avg.append(round(float(np.nanmean(day_vals)), 2))

    return {
        "meta": {
            "source": f"IMD RF25 gridded daily rainfall, India, {g['year']}",
            "resolution_deg": g["res"],
            "units": "mm",
            "n_days": g["n_days"],
            "n_cells": g["n_cells"],
            "lat_min": float(g["lat"].min()),
            "lat_max": float(g["lat"].max()),
            "lon_min": float(g["lon"].min()),
            "lon_max": float(g["lon"].max()),
            "year": g["year"],
        },
        "dates": g["dates"],
        "daily_avg_mm": daily_avg,
        "cells": cells,
        "rainfall": rainfall_by_day,
    }


def build_payload(nc_path):
    """Parse a .nc file straight into the compact grid payload the map renders."""
    return _payload_from_extract(_extract(nc_path))


def _write_csv(g, out_dir):
    rf_cells = g["rf_cells"]
    n_days, n_cells = g["n_days"], g["n_cells"]

    date_col = np.repeat(np.array(g["dates"], dtype=object), n_cells)
    lat_col = np.tile(g["cell_lats"], n_days)
    lon_col = np.tile(g["cell_lons"], n_days)
    rain_col = rf_cells.reshape(-1)

    valid = ~np.isnan(rain_col)
    df = pd.DataFrame({
        "date": date_col[valid],
        "latitude": np.round(lat_col[valid], 2),
        "longitude": np.round(lon_col[valid], 2),
        "rainfall_mm": np.round(rain_col[valid], 1),
    })

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"rainfall_{g['year']}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def convert_nc_to_csv(nc_path, out_dir):
    """Batch/CLI use only (see convert_all.py) — writes a long-format CSV."""
    g = _extract(nc_path)
    csv_path = _write_csv(g, out_dir)
    return csv_path, _payload_from_extract(g)


def save_uploaded_nc(tmp_path, nc_dir, out_dir):
    """Validate an uploaded .nc file, then persist it into nc_dir and write its
    CSV into out_dir. Nothing is written until the file passes validation.

    Returns (nc_path, csv_path, payload).
    """
    g = _extract(tmp_path)  # raises NcFormatError if this isn't a valid RF25 grid

    os.makedirs(nc_dir, exist_ok=True)
    nc_filename = f"RF25_ind{g['year']}_rfp25.nc"
    nc_path = os.path.join(nc_dir, nc_filename)
    shutil.copyfile(tmp_path, nc_path)

    csv_path = _write_csv(g, out_dir)

    return nc_path, csv_path, _payload_from_extract(g)
