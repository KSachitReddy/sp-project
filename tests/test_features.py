"""Leakage and calendar tests for weather/features.py, on small synthetic frames."""
import numpy as np
import pandas as pd
import pytest

from weather import features as feat


def _synthetic(n_days=45, start="2022-01-01"):
    """Two cells, one row per day, rainfall_mm = day index (+100 for the second
    cell) so every value identifies exactly which day it came from."""
    dates = pd.date_range(start, periods=n_days)
    frames = []
    for offset, (lat, lon) in enumerate([(10.0, 75.0), (20.0, 80.0)]):
        frames.append(pd.DataFrame({
            "date": dates, "latitude": lat, "longitude": lon,
            "rainfall_mm": np.arange(n_days, dtype=float) + 100 * offset,
        }))
    return pd.concat(frames).sort_values(["latitude", "longitude", "date"]).reset_index(drop=True)


def test_lag_features_only_use_days_before_the_anchor():
    df = feat.add_lag_features(_synthetic())
    today = df["rainfall_mm"]
    for lag in feat.LAG_DAYS:
        known = df[f"lag_{lag}"].notna()
        assert known.any()
        np.testing.assert_array_equal(df.loc[known, f"lag_{lag}"], today[known] - lag)
    for w in feat.ROLL_WINDOWS:
        known = df[f"roll_mean_{w}"].notna()
        # mean of days t-w .. t-1 of an arithmetic sequence = today - (w+1)/2
        np.testing.assert_allclose(df.loc[known, f"roll_mean_{w}"], today[known] - (w + 1) / 2)


def test_lags_never_cross_between_cells():
    df = feat.add_lag_features(_synthetic())
    second_cell = df[df["latitude"] == 20.0]
    # the second cell's values are all >= 100; a leak from the first cell would be < 100
    assert (second_cell["lag_1"].dropna() >= 100).all()
    assert second_cell["lag_1"].isna().sum() == 1


def test_expand_horizons_targets_are_strictly_future():
    df = feat.expand_horizons(feat.add_lag_features(_synthetic()))
    known = df["target_mm"].notna()
    np.testing.assert_array_equal(
        df.loc[known, "target_mm"], df.loc[known, "rainfall_mm"] + df.loc[known, "horizon_days"]
    )
    assert (df["target_date"] > df["date"]).all()


def test_split_years_disjoint_and_chronological():
    train, val, test = set(feat.TRAIN_YEARS), set(feat.VAL_YEARS), set(feat.TEST_YEARS)
    assert not (train & val) and not (train & test) and not (val & test)
    assert max(train) < min(val) <= max(val) < min(test)


def test_nowcast_split_years_disjoint_and_chronological():
    nm = pytest.importorskip("weather.nowcast_model")  # torch is optional
    train, val, test = set(nm.TRAIN_YEARS), set(nm.VAL_YEARS), set(nm.TEST_YEARS)
    assert not (train & val) and not (train & test) and not (val & test)
    assert max(train) < min(val) <= max(val) < min(test)


@pytest.mark.parametrize("day, expected", [
    ("2024-02-28", 59), ("2024-02-29", 59), ("2024-03-01", 60),
    ("2023-03-01", 60), ("2024-12-31", 365), ("2023-12-31", 365), ("2024-01-01", 1),
])
def test_doy_365_folds_leap_day(day, expected):
    assert feat.doy_365(pd.Series([pd.Timestamp(day)]))[0] == expected


def test_climatology_ignores_non_training_years():
    df = pd.concat([_synthetic(start="2022-01-01"), _synthetic(start="2023-01-01")])
    df.loc[df["date"].dt.year == 2023, "rainfall_mm"] = 1e6  # a val-year leak would be obvious
    clim = feat.compute_climatology(df, train_years=(2022,))
    assert clim["climatology_mm"].max() < 1e3
