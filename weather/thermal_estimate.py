"""
Same-day rainfall estimate from thermal satellite imagery: loads the model
scripts/calibrate_thermal_rainfall.py trained (cold-cloud-fraction ->
real rainfall) and predicts for any date with a thermal image on disk.

Unlike forecast_model.py (place + future date, needs an anchor because IMD's
data lags ~1 year) and nowcast_model.py (predicts TOMORROW from TODAY's
visible image), this estimates TODAY's rainfall from TODAY's thermal image
-- the one prediction in this project that needs no anchor date and isn't
stuck behind any lag, because it never depends on IMD's own rainfall feed
at inference time, only at training time.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from weather import satellite_features as sfeat
from weather import thermal_features as tf

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "thermal_estimate_model.txt"
METRICS_PATH = MODELS_DIR / "thermal_estimate_metrics.json"

_booster = None


def model_available():
    return MODEL_PATH.exists()


def _load():
    global _booster
    if _booster is None:
        _booster = lgb.Booster(model_file=str(MODEL_PATH))
    return _booster


def predict_thermal_rainfall(date_str):
    """Same-day estimated rainfall (mm) per macro-region, from that day's
    thermal satellite image -- or {"error": ...} if there's no trained model
    or no thermal image for that date."""
    if not model_available():
        return {"error": "Thermal rainfall estimate model not trained yet -- run scripts/calibrate_thermal_rainfall.py."}

    # The trained model's feature order is fixed at training time -- read the
    # exact thresholds it was calibrated on from its own metrics file, rather
    # than risk drifting out of sync with a default defined elsewhere.
    thresholds = json.loads(METRICS_PATH.read_text())["thresholds_k"]

    fractions = tf.cold_fractions_multi_threshold(date_str, thresholds)
    if fractions is None:
        return {"error": f"No thermal satellite image on disk for {date_str}."}

    booster = _load()
    predicted_mm_by_region = {}
    for region in sfeat.REGIONS:
        if any(fractions[t][region] is None for t in thresholds):
            continue  # region fully obscured that day
        row = np.array([[fractions[t][region] for t in thresholds]])
        pred = float(booster.predict(row)[0])
        predicted_mm_by_region[region] = round(max(pred, 0.0), 2)

    if not predicted_mm_by_region:
        return {"error": f"Thermal image for {date_str} exists but every region was obscured/no-data."}

    return {"date": date_str, "predicted_mm_by_region": predicted_mm_by_region}
