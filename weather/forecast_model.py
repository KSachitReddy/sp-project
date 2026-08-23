"""
LightGBM rainfall forecasting model: train/save/load/predict.

Trained on features.FEATURE_COLS to predict rainfall_mm `horizon_days` ahead
of an anchor date, for a spatially-uniform subsample of grid cells (see
weather/features.py) -- generalizes to any lat/lon via the lat/lon +
climatology features.

predict_rainfall() is deliberately anchored to the *last real data point*
(the newest date across data/rainfall/*.csv) rather than "today": IMD's
gridded rainfall product lags real time by roughly a year, so there is no
real recent rainfall to build lag/rolling features from for a date like
"today". For target dates within HORIZONS_DAYS' max of that anchor, this
produces a genuine short-range ML forecast built from real recent data. For
anything further out -- in practice most of "the rest of this year" whenever
IMD's release lag is in effect -- there's no way to honestly build lag
features, so it falls back to a climatology-only estimate and says so rather
than feeding the model made-up inputs.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from weather import features as feat

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "forecast_model.txt"
METRICS_PATH = MODELS_DIR / "forecast_model_metrics.json"
CLIMATOLOGY_PATH = MODELS_DIR / "climatology.csv"

MAX_ML_HORIZON_DAYS = max(feat.HORIZONS_DAYS)


def train(stride=feat.CELL_GRID_STRIDE):
    train_df, val_df, test_df, climatology = feat.build_dataset(stride=stride)

    lgb_train = lgb.Dataset(train_df[feat.FEATURE_COLS], label=train_df["target_mm"])
    lgb_val = lgb.Dataset(val_df[feat.FEATURE_COLS], label=val_df["target_mm"], reference=lgb_train)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }
    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )

    metrics = {
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "best_iteration": booster.best_iteration,
        "by_split": _evaluate(booster, {"val_2023": val_df, "test_2024_2025": test_df}),
    }

    MODELS_DIR.mkdir(exist_ok=True)
    booster.save_model(str(MODEL_PATH))
    climatology.to_csv(CLIMATOLOGY_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return booster, metrics


def _evaluate(booster, splits):
    """MAE/RMSE per forecast horizon, plus a skill score against the
    climatology baseline (1 - model_MSE/climatology_MSE): positive means the
    model beats "just predict the historical average for this day", which is
    the bar a forecasting model actually needs to clear to be useful."""
    metrics = {}
    for name, df in splits.items():
        if df.empty:
            continue
        preds = booster.predict(df[feat.FEATURE_COLS])
        by_horizon = {}
        for h in feat.HORIZONS_DAYS:
            mask = (df["horizon_days"] == h).to_numpy()
            if mask.sum() == 0:
                continue
            y = df.loc[mask, "target_mm"].to_numpy()
            p = preds[mask]
            clim = df.loc[mask, "climatology_mm"].to_numpy()
            mse = float(np.mean((y - p) ** 2))
            clim_mse = float(np.mean((y - clim) ** 2))
            by_horizon[str(h)] = {
                "n": int(mask.sum()),
                "mae": round(float(np.mean(np.abs(y - p))), 3),
                "rmse": round(float(np.sqrt(mse)), 3),
                "climatology_rmse": round(float(np.sqrt(clim_mse)), 3),
                "skill_vs_climatology": round(float(1 - mse / clim_mse), 4) if clim_mse > 0 else None,
            }
        metrics[name] = by_horizon
    return metrics


_booster = None
_climatology = None


def _load():
    global _booster, _climatology
    if _booster is None:
        _booster = lgb.Booster(model_file=str(MODEL_PATH))
    if _climatology is None:
        _climatology = pd.read_csv(CLIMATOLOGY_PATH)
    return _booster, _climatology


def model_available():
    return MODEL_PATH.exists() and CLIMATOLOGY_PATH.exists()


def _climatology_lookup(climatology, lat, lon, target_date):
    doy = int(feat.doy_365(pd.Series([target_date]))[0])
    row = climatology[
        (climatology["latitude"] == lat)
        & (climatology["longitude"] == lon)
        & (climatology["doy"] == doy)
    ]
    return float(row["climatology_mm"].iloc[0]) if not row.empty else None


def predict_rainfall(grid_lat, grid_lon, target_date):
    """grid_lat/grid_lon must be an exact IMD RF25 grid cell (0.25 degree),
    e.g. from weather.chatbot.nearest_cell(). target_date: a date/Timestamp.
    Returns a dict with predicted_mm, method ('ml_short_range' or
    'climatology_outlook'), and supporting context -- or an {"error": ...} dict."""
    if not model_available():
        return {"error": "Forecast model not trained yet -- run scripts/train_forecast_model.py."}

    booster, climatology = _load()
    target_date = pd.Timestamp(target_date).normalize()

    full = feat.load_rainfall()
    last_available = full["date"].max()

    if target_date <= last_available:
        return {
            "error": (
                f"{target_date.date()} is within the historical record (through "
                f"{last_available.date()}) -- use the rainfall lookup tool for actual "
                f"observed data instead of a forecast."
            )
        }

    horizon = (target_date - last_available).days
    climatology_mm = _climatology_lookup(climatology, grid_lat, grid_lon, target_date)

    if horizon > MAX_ML_HORIZON_DAYS:
        if climatology_mm is None:
            return {"error": f"No historical data for grid cell ({grid_lat}, {grid_lon})."}
        return _climatology_result(grid_lat, grid_lon, target_date, climatology_mm, horizon, last_available)

    cell = full[(full["latitude"] == grid_lat) & (full["longitude"] == grid_lon)].sort_values("date")
    if cell.empty:
        return {"error": f"No historical data for grid cell ({grid_lat}, {grid_lon})."}

    tail = cell.set_index("date")["rainfall_mm"]
    row = {
        "latitude": grid_lat,
        "longitude": grid_lon,
        "doy_sin": np.sin(2 * np.pi * last_available.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * last_available.dayofyear / 365.25),
        "month": last_available.month,
        "year": last_available.year,
        "horizon_days": horizon,
        "climatology_mm": climatology_mm,
    }
    # lag_k = rainfall k days before the anchor (matches features.add_lag_features's grp.shift(k))
    for lag in feat.LAG_DAYS:
        idx = last_available - pd.Timedelta(days=lag)
        row[f"lag_{lag}"] = tail.get(idx, np.nan)
    # roll_mean_w = mean of the w days ending the day before the anchor (matches shift(1).rolling(w))
    for w in feat.ROLL_WINDOWS:
        window_vals = tail.loc[last_available - pd.Timedelta(days=w): last_available - pd.Timedelta(days=1)]
        row[f"roll_mean_{w}"] = window_vals.mean() if len(window_vals) else np.nan

    x = pd.DataFrame([row])[feat.FEATURE_COLS]
    if x.isna().any(axis=None):
        if climatology_mm is None:
            return {"error": f"Insufficient recent history for grid cell ({grid_lat}, {grid_lon})."}
        result = _climatology_result(grid_lat, grid_lon, target_date, climatology_mm, horizon, last_available)
        result["note"] = "Missing recent history for this cell's short-range features -- using climatology instead."
        return result

    predicted = float(booster.predict(x)[0])
    return {
        "target_date": str(target_date.date()),
        "grid_cell": {"lat": grid_lat, "lon": grid_lon},
        "predicted_mm": round(max(predicted, 0.0), 2),
        "method": "ml_short_range",
        "anchor_date": str(last_available.date()),
        "horizon_days": horizon,
        "climatology_mm": round(climatology_mm, 2) if climatology_mm is not None else None,
    }


def _climatology_result(grid_lat, grid_lon, target_date, climatology_mm, horizon, last_available):
    return {
        "target_date": str(target_date.date()),
        "grid_cell": {"lat": grid_lat, "lon": grid_lon},
        "predicted_mm": round(climatology_mm, 2),
        "method": "climatology_outlook",
        "note": (
            f"{target_date.date()} is {horizon} days past the last available rainfall data "
            f"({last_available.date()}) -- beyond the {MAX_ML_HORIZON_DAYS}-day range a "
            f"short-range model can honestly forecast from real recent data, so this is the "
            f"historical average for this place around this time of year, not a real forecast."
        ),
    }
