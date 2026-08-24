"""
Calibrates a same-day rainfall estimate from thermal satellite imagery
against real IMD rainfall -- the "cold cloud fraction" signal from
weather/thermal_features.py (colder cloud-top temperature correlates with
heavier rain), regressed against actual observed rainfall for every day
both exist, instead of using the classic GOES Precipitation Index's generic,
decades-old fixed constants.

Unlike the CNN nowcaster (predicts TOMORROW from TODAY's image), this
estimates TODAY's rainfall from TODAY's image -- valuable specifically
because it doesn't depend on IMD's ~1-year data lag at all, so it can
produce a same-day estimate for any date with a thermal image on disk,
including all of 2026.

    .venv/bin/python scripts/calibrate_thermal_rainfall.py

Needs scripts/extract_thermal_features.py to have been run first (reads its
cache at data/satellite/thermal_features.csv rather than decoding images
itself; `nimbus download` runs that extraction step automatically after
fetching new thermal imagery). Writes models/thermal_estimate_model.txt
(LightGBM) and models/thermal_estimate_metrics.json (backtested MAE/RMSE vs.
climatology -- same honesty pattern as every other model in this project:
a measurement, not a claim).
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from weather import features as feat  # noqa: E402
from weather import satellite_features as sfeat  # noqa: E402
from weather import thermal_features as tf  # noqa: E402

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "thermal_estimate_model.txt"
METRICS_PATH = MODELS_DIR / "thermal_estimate_metrics.json"
FEATURES_PATH = BASE_DIR / "data" / "satellite" / "thermal_features.csv"

THRESHOLDS_K = tf.THRESHOLDS_K
FEATURE_COLS = tf.FEATURE_COLS

TRAIN_YEARS = range(2019, 2024)
VAL_YEARS = (2024,)
TEST_YEARS = (2025,)


def build_dataset():
    """One row per (date, region) with cold-fraction-at-each-threshold
    features (from the extraction cache) and that region's actual same-day
    rainfall as the target."""
    if not FEATURES_PATH.exists():
        print(f"{FEATURES_PATH} not found -- run scripts/extract_thermal_features.py first "
              "(or `uv run nimbus download`, which runs it automatically).")
        sys.exit(1)

    features = pd.read_csv(FEATURES_PATH, dtype={"date": str})
    regional = sfeat.regional_daily_rainfall(feat.load_rainfall())
    regional.index = pd.to_datetime(regional.index)

    rows = []
    for row in features.itertuples(index=False):
        date = pd.Timestamp(row.date)
        if date not in regional.index:
            continue
        actual = regional.loc[date, row.region]
        if pd.isna(actual):
            continue
        rows.append({
            "date": row.date, "region": row.region, "actual_mm": float(actual),
            **{col: getattr(row, col) for col in FEATURE_COLS},
        })

    return pd.DataFrame(rows)


def _evaluate(booster, df):
    """MAE/RMSE per region, plus skill vs. that region's climatology for the
    same day-of-year -- the same honesty bar every other model here has to clear."""
    preds = booster.predict(df[FEATURE_COLS])
    df = df.assign(pred_mm=np.clip(preds, 0, None))
    doy = pd.to_datetime(df["date"]).dt.dayofyear

    result = {}
    for region in sfeat.REGIONS:
        mask = (df["region"] == region).to_numpy()
        if mask.sum() == 0:
            continue
        y, p = df.loc[mask, "actual_mm"].to_numpy(), df.loc[mask, "pred_mm"].to_numpy()
        # climatology baseline: this region's mean actual rainfall in the training set (a flat baseline,
        # not day-of-year-specific -- the dataset's too small yet to support a finer one honestly)
        clim = float(df.loc[df["region"] == region, "actual_mm"].mean())
        mse = float(np.mean((y - p) ** 2))
        clim_mse = float(np.mean((y - clim) ** 2))
        result[region] = {
            "n": int(mask.sum()),
            "mae": round(float(np.mean(np.abs(y - p))), 3),
            "rmse": round(float(np.sqrt(mse)), 3),
            "climatology_rmse": round(float(np.sqrt(clim_mse)), 3),
            "skill_vs_climatology": round(float(1 - mse / clim_mse), 4) if clim_mse > 0 else None,
        }
    return result


def main():
    print("Building dataset from thermal images + real rainfall (only days with both)...")
    df = build_dataset()
    if df.empty:
        print("No overlapping thermal-image + rainfall days found -- run the thermal backfill first "
              "(scripts/download_satellite_images.py / `uv run nimbus download`).")
        sys.exit(1)

    df["year"] = pd.to_datetime(df["date"]).dt.year
    train_df = df[df["year"].isin(TRAIN_YEARS)]
    val_df = df[df["year"].isin(VAL_YEARS)]
    test_df = df[df["year"].isin(TEST_YEARS)]
    print(f"Rows: {len(df)} total ({len(train_df)} train, {len(val_df)} val, {len(test_df)} test)")

    if len(train_df) < 50:
        print(
            f"Only {len(train_df)} training rows -- the thermal backfill is likely still running. "
            "This will still produce a model, but treat its metrics as provisional until the full "
            "2019-2025 backfill is done."
        )

    lgb_train = lgb.Dataset(train_df[FEATURE_COLS], label=train_df["actual_mm"])
    valid_sets, valid_names = [], []
    if len(val_df):
        valid_sets.append(lgb.Dataset(val_df[FEATURE_COLS], label=val_df["actual_mm"], reference=lgb_train))
        valid_names.append("val")

    params = {
        "objective": "regression", "metric": "rmse", "learning_rate": 0.05,
        "num_leaves": 15, "min_data_in_leaf": 20, "verbose": -1,
    }
    booster = lgb.train(
        params, lgb_train, num_boost_round=500,
        valid_sets=valid_sets or None, valid_names=valid_names or None,
        callbacks=[lgb.early_stopping(30, verbose=False)] if valid_sets else [],
    )

    metrics = {
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "thresholds_k": THRESHOLDS_K,
        "by_split": {k: v for k, v in {
            "val_2024": _evaluate(booster, val_df) if len(val_df) else None,
            "test_2025": _evaluate(booster, test_df) if len(test_df) else None,
        }.items() if v},
    }

    MODELS_DIR.mkdir(exist_ok=True)
    booster.save_model(str(MODEL_PATH))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print()
    print("=== Backtested accuracy ===")
    print(json.dumps(metrics, indent=2))
    print()
    print(f"Model + metrics written to {MODELS_DIR}")


if __name__ == "__main__":
    main()
