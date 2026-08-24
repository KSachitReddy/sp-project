"""
Decodes NASA GIBS' colorized brightness-temperature imagery
(data/satellite/india_thermal/, downloaded by scripts/download_satellite_images.py)
back into actual temperature in Kelvin, and computes the "cold cloud
fraction" per macro-region per day -- the core signal behind a
physically-motivated rainfall estimate (see scripts/calibrate_thermal_rainfall.py):

    colder cloud-top temperature -> taller, more violent storm -> more
    likely raining hard.

This is the same principle behind the decades-old GOES Precipitation Index,
just calibrated against this project's own real IMD rainfall data instead
of generic fixed constants.

GIBS renders this layer as a colour gradient (not raw temperature values),
so decoding needs the same colour legend GIBS itself publishes for it:
https://gibs.earthdata.nasa.gov/legends/MODIS_Brightness_Temp_Band31_H.png
(180K, dark navy -> 340K, cream). That legend was sampled once into
band31_temperature_lut.csv, bundled here so decoding works fully offline --
no network call needed at runtime. Pure-black [0,0,0] pixels are GIBS' own
no-data marker for this layer (satellite swath gaps), verified against a
real downloaded image, and are treated as missing, not as "very cold".
"""
from pathlib import Path

import numpy as np
from PIL import Image

from weather.satellite_features import IMAGE_SIZE, REGIONS, region_for

BASE_DIR = Path(__file__).parent.parent
THERMAL_DIR = BASE_DIR / "data" / "satellite" / "india_thermal"
LUT_PATH = Path(__file__).parent / "band31_temperature_lut.csv"

BBOX = (68, 6, 98, 38)  # lon_min, lat_min, lon_max, lat_max -- matches the downloader's request

# ~-53C -- a genuinely deep convective storm top, not just "cloudy". The
# GOES Precipitation Index's classic threshold is warmer (~-20C to -38C
# depending on the version); calibrate_thermal_rainfall.py checks a few
# thresholds against real rainfall rather than assuming this one is best.
COLD_THRESHOLD_K = 220.0

# Tried as separate features and left for the model to weigh, rather than
# committing to one cutoff up front. Shared by scripts/extract_thermal_features.py
# (which caches these per image) and scripts/calibrate_thermal_rainfall.py
# (which trains against them), so both stay in sync automatically.
THRESHOLDS_K = [200, 210, 220, 230, 240]
FEATURE_COLS = [f"cold_frac_{t}" for t in THRESHOLDS_K]


def _load_lut():
    import pandas as pd

    df = pd.read_csv(LUT_PATH)
    return df[["r", "g", "b"]].to_numpy(dtype=np.float64), df["temp_k"].to_numpy(dtype=np.float64)


_LUT_COLORS, _LUT_TEMPS = _load_lut()


def _build_region_grid():
    """Which macro-region each pixel of the downsampled IMAGE_SIZE grid falls
    in, computed once at import (the lat/lon of each pixel is fixed given a
    constant BBOX + IMAGE_SIZE) rather than recomputed per image."""
    lon_min, lat_min, lon_max, lat_max = BBOX
    w, h = IMAGE_SIZE
    lons = np.linspace(lon_min, lon_max, w)
    lats = np.linspace(lat_max, lat_min, h)  # row 0 = top = max latitude
    grid = np.empty((h, w), dtype=object)
    for yi, lat in enumerate(lats):
        for xi, lon in enumerate(lons):
            grid[yi, xi] = region_for(lat, lon)
    return grid


_REGION_GRID = _build_region_grid()


def decode_temperature(rgb_array):
    """(H, W, 3) uint8/int RGB -> (H, W) float32 Kelvin, NaN where no data."""
    flat = rgb_array.reshape(-1, 3).astype(np.float64)
    no_data = np.all(flat == 0, axis=1)

    dists = np.sum((flat[:, None, :] - _LUT_COLORS[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(dists, axis=1)
    temps = _LUT_TEMPS[nearest]
    temps[no_data] = np.nan
    return temps.reshape(rgb_array.shape[:2]).astype(np.float32)


def load_temperature(date_str):
    """Decoded (H, W) Kelvin grid for a day, downsampled to IMAGE_SIZE (same
    grid the visible-imagery CNN uses), or None if that day has no thermal
    image on disk."""
    path = THERMAL_DIR / date_str[:4] / f"{date_str}.png"
    if not path.exists():
        return None
    img = np.asarray(Image.open(path).convert("RGB").resize(IMAGE_SIZE))
    return decode_temperature(img)


def cold_fractions_multi_threshold(date_str, thresholds_k):
    """{threshold: {region: fraction}} for every threshold at once, decoding
    the image only once -- calibrate_thermal_rainfall.py tries several
    thresholds per day and shouldn't pay the decode cost repeatedly."""
    temps = load_temperature(date_str)
    if temps is None:
        return None

    result = {}
    region_masks = {region: (_REGION_GRID == region) & ~np.isnan(temps) for region in REGIONS}
    for thresh in thresholds_k:
        result[thresh] = {}
        for region, mask in region_masks.items():
            vals = temps[mask]
            result[thresh][region] = float(np.mean(vals < thresh)) if len(vals) else None
    return result


def cold_fraction_by_region(date_str, threshold_k=COLD_THRESHOLD_K):
    """{region: fraction of that region's pixels colder than threshold_k} for
    one day, or None if there's no thermal image for that date. A region
    with no valid (non-NaN) pixels that day maps to None rather than 0.0,
    so a fully-obscured region isn't silently treated as "no cold clouds"."""
    result = cold_fractions_multi_threshold(date_str, [threshold_k])
    return result[threshold_k] if result else None


def extract_feature_row(date_str):
    """{region: {feature_col: value}} at THRESHOLDS_K for one day, skipping
    any region that's fully obscured (rather than imputing it), or None if
    there's no thermal image for that date at all."""
    fractions = cold_fractions_multi_threshold(date_str, THRESHOLDS_K)
    if fractions is None:
        return None
    by_region = {}
    for region in REGIONS:
        if any(fractions[t][region] is None for t in THRESHOLDS_K):
            continue
        by_region[region] = {f"cold_frac_{t}": fractions[t][region] for t in THRESHOLDS_K}
    return by_region
