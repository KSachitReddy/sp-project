"""Every prediction the forecaster returns must say how it was produced, and
every user-facing label must distinguish the methods."""
import numpy as np
import pandas as pd
import pytest

from weather import features as feat
from weather import forecast_model as fm
from weather.fast_intent import method_label

CELL = (19.0, 73.0)  # Mumbai's grid cell -- present in models/climatology.csv
LAST_DAY = pd.Timestamp("2025-12-31")


@pytest.fixture
def short_history(monkeypatch):
    """Replace the 14.5M-row history with 60 days for one real cell, so the
    real booster + climatology run without loading every CSV."""
    dates = pd.date_range(LAST_DAY - pd.Timedelta(days=59), LAST_DAY)
    df = pd.DataFrame({
        "date": dates, "latitude": CELL[0], "longitude": CELL[1],
        "rainfall_mm": np.linspace(0, 5, len(dates)),
    })
    monkeypatch.setattr(feat, "load_rainfall", lambda years=None: df)


pytestmark = pytest.mark.skipif(not fm.model_available(), reason="forecast model not trained")


@pytest.mark.parametrize("days_ahead, method", [(1, "ml_short_range"), (5, "ml_short_range"),
                                                (14, "ml_short_range"), (15, "climatology_outlook"),
                                                (200, "climatology_outlook")])
def test_predictions_carry_method(short_history, days_ahead, method):
    result = fm.predict_rainfall(*CELL, LAST_DAY + pd.Timedelta(days=days_ahead))
    assert "error" not in result
    assert result["method"] == method
    assert result["predicted_mm"] >= 0
    assert method_label(result)


def test_historical_date_is_an_error_not_a_prediction(short_history):
    result = fm.predict_rainfall(*CELL, LAST_DAY)
    assert "error" in result and "predicted_mm" not in result


def test_method_labels_are_distinct_and_name_the_region():
    labels = {m: method_label({"method": m, "region": "west"}) for m in
              ("ml_short_range", "satellite_nowcast", "satellite_thermal", "climatology_outlook")}
    assert len(set(labels.values())) == 4
    assert "west" in labels["satellite_nowcast"] and "west" in labels["satellite_thermal"]
    assert labels["climatology_outlook"] == "historical average"
