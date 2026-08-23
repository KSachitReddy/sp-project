"""
Image/label building for the satellite nowcasting model, shared by
scripts/train_nowcast_model.py (training) and weather/nowcast_model.py
(inference).

Six macro-regions of India (North, South, East, West, Central, Northeast),
defined by simple lat/lon bounding boxes over the same 0.25-degree IMD grid
used elsewhere in this project -- coarse by design; the CNN sees the whole
downsampled image and just needs *a* label with real spatial structure to
predict, not a precise administrative boundary.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

BASE_DIR = Path(__file__).parent.parent
SATELLITE_DIR = BASE_DIR / "data" / "satellite" / "india_cloud"
CACHE_PATH = BASE_DIR / "data" / "satellite" / "nowcast_cache.npz"

IMAGE_SIZE = (96, 80)  # (width, height) downsample target -- CNN input, not for viewing
REGIONS = ["north", "south", "east", "west", "central", "northeast"]


def region_for(lat, lon):
    if lat >= 22 and lon >= 88:
        return "northeast"
    if lat >= 26:
        return "north"
    if lat < 15:
        return "south"
    if lon < 77:
        return "west"
    if lon < 85:
        return "central"
    return "east"


def load_image(date_str):
    """Downsampled satellite PNG for a date as a (3, H, W) float32 array in
    [0, 1] (channel-first, ready for PyTorch), or None if not downloaded."""
    path = SATELLITE_DIR / date_str[:4] / f"{date_str}.png"
    if not path.exists():
        return None
    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
    return np.transpose(arr, (2, 0, 1))  # (3, H, W)


def available_image_dates():
    dates = []
    for year_dir in sorted(SATELLITE_DIR.glob("*")):
        if not year_dir.is_dir():
            continue
        dates.extend(p.stem for p in sorted(year_dir.glob("*.png")))
    return dates


def regional_daily_rainfall(rainfall_df):
    """DataFrame indexed by date, one column per region, holding the mean
    rainfall_mm across that region's grid cells that day."""
    df = rainfall_df[["date", "latitude", "longitude", "rainfall_mm"]].copy()
    df["region"] = [region_for(la, lo) for la, lo in zip(df["latitude"], df["longitude"])]
    regional = df.groupby(["date", "region"])["rainfall_mm"].mean().unstack("region")
    return regional.reindex(columns=REGIONS)


def build_dataset(rainfall_df, use_cache=True):
    """X: (N, 3, H, W) float32 -- day d's image. Y: (N, len(REGIONS)) float32
    -- day d+1's mean regional rainfall. dates: DatetimeIndex of day d for
    each row. Decoding/resizing ~2,500 PNGs takes a while, so the result is
    cached to disk and reused as long as IMAGE_SIZE hasn't changed."""
    if use_cache and CACHE_PATH.exists():
        cached = np.load(CACHE_PATH, allow_pickle=True)
        if tuple(cached["image_size"]) == IMAGE_SIZE:
            return cached["X"], cached["Y"], pd.DatetimeIndex(cached["dates"])

    regional = regional_daily_rainfall(rainfall_df)
    regional.index = pd.to_datetime(regional.index)

    xs, ys, ds = [], [], []
    for date_str in available_image_dates():
        d = pd.Timestamp(date_str)
        next_d = d + pd.Timedelta(days=1)
        if next_d not in regional.index:
            continue
        label = regional.loc[next_d].to_numpy(dtype=np.float32)
        if np.isnan(label).any():
            continue
        img = load_image(date_str)
        if img is None:
            continue
        xs.append(img)
        ys.append(label)
        ds.append(d)

    X = np.stack(xs).astype(np.float32)
    Y = np.stack(ys).astype(np.float32)
    dates = pd.DatetimeIndex(ds)

    if use_cache:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            CACHE_PATH, X=X, Y=Y,
            dates=dates.astype(str).to_numpy(),
            image_size=np.array(IMAGE_SIZE),
        )
    return X, Y, dates
