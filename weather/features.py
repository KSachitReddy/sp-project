"""
Feature engineering for the rainfall forecasting model, shared by
scripts/train_forecast_model.py (training) and weather/forecast_model.py
(inference).

Trains on a spatially-uniform subsample of the ~4,964 IMD RF25 grid cells
(all cells x all 8 years x every horizon is too many rows to comfortably
iterate on on a CPU-only machine) -- the model still generalizes to any
lat/lon at inference time via the lat/lon + climatology features rather than
memorizing per-cell values. Climatology itself is computed over *all* cells
(cheap, and needed at inference for whichever exact grid cell a place
resolves to, not just the training subsample).
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
RAINFALL_DIR = BASE_DIR / "data" / "rainfall"

HORIZONS_DAYS = [1, 3, 7, 14]
LAG_DAYS = [1, 3, 7, 14, 30]
ROLL_WINDOWS = [7, 30]
CELL_GRID_STRIDE = 3  # keep 1-in-3 cells in both lat & lon -> ~1/9 of cells
CLIMATOLOGY_WINDOW_DAYS = 3
TRAIN_YEARS = range(2018, 2023)  # 2018-2022
VAL_YEARS = (2023,)
TEST_YEARS = (2024, 2025)

FEATURE_COLS = [
    "latitude", "longitude", "doy_sin", "doy_cos", "month", "year",
    "lag_1", "lag_3", "lag_7", "lag_14", "lag_30",
    "roll_mean_7", "roll_mean_30", "climatology_mm", "horizon_days",
]


def available_years():
    years = []
    for p in RAINFALL_DIR.glob("rainfall_*.csv"):
        try:
            years.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(years)


def _load_years(years):
    frames = []
    for y in years:
        p = RAINFALL_DIR / f"rainfall_{y}.csv"
        if p.exists():
            frames.append(pd.read_csv(p, parse_dates=["date"]))
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["latitude", "longitude", "date"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def _load_all_cached():
    return _load_years(tuple(available_years()))


def load_rainfall(years=None):
    """All rainfall rows (every grid cell) for the given years, sorted so each
    cell's time series is contiguous -- required before any shift/rolling op.
    The full-history (years=None) case is cached: it's ~14.5M rows, reloaded
    from CSV on every call otherwise, which matters once this is called
    per-request from the forecast API/chatbot, not just at training time."""
    if years is None:
        return _load_all_cached()
    return _load_years(tuple(years))


def doy_365(dates):
    """Day-of-year collapsed onto a fixed 1-365 scale (Feb 29 folds into the
    Feb 28 bucket, every later day shifts down by one) so a (lat, lon, doy)
    climatology bucket means the same calendar point every year, leap or not."""
    doy = dates.dt.dayofyear.to_numpy().copy()
    is_leap = dates.dt.is_leap_year.to_numpy()
    past_feb29 = is_leap & (doy > 59)
    doy[past_feb29] -= 1
    return doy


def compute_climatology(df, train_years=TRAIN_YEARS):
    """Mean rainfall per (lat, lon, day-of-year), smoothed with a +-3 day
    window, using only train_years -- this is later joined onto every row
    (train/val/test/inference alike) as a feature, and is the only thing
    this model can honestly rely on far outside the lag window. Edge days
    (near Jan 1 / Dec 31) get a slightly narrower window rather than wrapping
    circularly -- a negligible approximation given nearby-season rainfall is
    already similar."""
    train = df[df["date"].dt.year.isin(train_years)].copy()
    train["doy"] = doy_365(train["date"])
    daily = (
        train.groupby(["latitude", "longitude", "doy"])["rainfall_mm"]
        .mean()
        .reset_index()
        .sort_values(["latitude", "longitude", "doy"])
    )
    window = 2 * CLIMATOLOGY_WINDOW_DAYS + 1
    daily["climatology_mm"] = daily.groupby(["latitude", "longitude"])["rainfall_mm"].transform(
        lambda s: s.rolling(window, center=True, min_periods=1).mean()
    )
    return daily[["latitude", "longitude", "doy", "climatology_mm"]].reset_index(drop=True)


def select_training_cells(df, stride=CELL_GRID_STRIDE):
    """Spatially-uniform subsample of grid cells, using the 0.25-degree grid index."""
    lat_min, lon_min = df["latitude"].min(), df["longitude"].min()
    idx_lat = np.round((df["latitude"] - lat_min) / 0.25).astype(int)
    idx_lon = np.round((df["longitude"] - lon_min) / 0.25).astype(int)
    keep = (idx_lat % stride == 0) & (idx_lon % stride == 0)
    return df.loc[keep].copy()


def add_time_features(df):
    df = df.copy()
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    return df


def add_lag_features(df):
    """Per-cell lag/rolling features. df must already be sorted by
    (latitude, longitude, date) with one row per cell per day (no gaps)."""
    df = df.copy()
    grp = df.groupby(["latitude", "longitude"])["rainfall_mm"]
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = grp.shift(lag)
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = grp.transform(lambda s: s.shift(1).rolling(w).mean())
    return df


def add_climatology(df, climatology, date_col="date"):
    """Joins climatology_mm keyed on `date_col`'s day-of-year -- pass
    target_date here, not the anchor date, so the feature describes what's
    climatologically typical on the day being *forecast* (the strongest
    single signal at longer horizons), not on the day standing in the present."""
    df = df.copy()
    df["doy"] = doy_365(df[date_col])
    df = df.merge(climatology, on=["latitude", "longitude", "doy"], how="left")
    return df


def expand_horizons(df, horizons=HORIZONS_DAYS):
    """One row per (cell, date, horizon); target_mm is the cell's actual
    rainfall `horizon` days after `date`, and target_date is that future date.
    Lag/rolling features describe only what was known as of `date` (the
    anchor), so the model learns to forecast forward, not just memorize the
    target -- target_date exists so climatology can be joined against the day
    actually being forecast."""
    grp = df.groupby(["latitude", "longitude"])["rainfall_mm"]
    frames = []
    for h in horizons:
        target = grp.shift(-h)
        part = df.copy()
        part["horizon_days"] = h
        part["target_mm"] = target.to_numpy()
        part["target_date"] = part["date"] + pd.Timedelta(days=h)
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def build_dataset(stride=CELL_GRID_STRIDE, horizons=HORIZONS_DAYS):
    """Returns (train_df, val_df, test_df, climatology) ready for training/eval.
    Each split has FEATURE_COLS + target_mm + date + latitude/longitude."""
    full = load_rainfall()
    climatology = compute_climatology(full)

    subset = select_training_cells(full, stride=stride)
    subset = add_lag_features(subset)
    subset = add_time_features(subset)
    subset = expand_horizons(subset, horizons=horizons)
    subset = add_climatology(subset, climatology, date_col="target_date")

    required = FEATURE_COLS + ["target_mm"]
    subset = subset.dropna(subset=required).reset_index(drop=True)

    year = subset["date"].dt.year
    train_df = subset[year.isin(TRAIN_YEARS)].reset_index(drop=True)
    val_df = subset[year.isin(VAL_YEARS)].reset_index(drop=True)
    test_df = subset[year.isin(TEST_YEARS)].reset_index(drop=True)
    return train_df, val_df, test_df, climatology
