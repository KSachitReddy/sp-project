"""The nowcast dataset cache must be rebuilt when the images or labels change,
not only when IMAGE_SIZE does."""
import numpy as np
import pandas as pd
from PIL import Image

from weather import satellite_features as sfeat


def _setup(tmp_path, monkeypatch, image_dates):
    sat = tmp_path / "india_cloud"
    for d in image_dates:
        (sat / d[:4]).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 16), (int(d[-2:]) * 8, 0, 0)).save(sat / d[:4] / f"{d}.png")
    monkeypatch.setattr(sfeat, "SATELLITE_DIR", sat)
    monkeypatch.setattr(sfeat, "CACHE_PATH", tmp_path / "cache.npz")


def _rainfall(last_day="2024-01-10"):
    dates = pd.date_range("2024-01-01", last_day)
    cells = [(30.0, 75.0), (10.0, 77.0), (24.0, 86.0), (20.0, 73.0), (20.0, 80.0), (25.0, 92.0)]  # one per region
    return pd.DataFrame([{"date": d, "latitude": la, "longitude": lo, "rainfall_mm": 1.0}
                         for d in dates for la, lo in cells])


def test_cache_reused_when_nothing_changed(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, ["2024-01-01", "2024-01-02"])
    X1, _, d1 = sfeat.build_dataset(_rainfall())
    mtime = sfeat.CACHE_PATH.stat().st_mtime_ns
    X2, _, d2 = sfeat.build_dataset(_rainfall())
    assert sfeat.CACHE_PATH.stat().st_mtime_ns == mtime
    np.testing.assert_array_equal(X1, X2)
    assert list(d1) == list(d2)


def test_cache_rebuilt_when_new_image_arrives(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, ["2024-01-01", "2024-01-02"])
    _, _, before = sfeat.build_dataset(_rainfall())
    _setup(tmp_path, monkeypatch, ["2024-01-01", "2024-01-02", "2024-01-03"])
    _, _, after = sfeat.build_dataset(_rainfall())
    assert len(before) == 2 and len(after) == 3


def test_cache_rebuilt_when_labels_extend(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, ["2024-01-01", "2024-01-02", "2024-01-03"])
    _, _, before = sfeat.build_dataset(_rainfall(last_day="2024-01-03"))  # 01-03 has no next-day label
    _, _, after = sfeat.build_dataset(_rainfall(last_day="2024-01-04"))
    assert len(before) == 2 and len(after) == 3


def test_old_cache_without_key_fields_is_ignored(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, ["2024-01-01", "2024-01-02"])
    np.savez(sfeat.CACHE_PATH, X=np.zeros((1, 3, 1, 1)), Y=np.zeros((1, 6)),
             dates=np.array(["2020-01-01"]), image_size=np.array(sfeat.IMAGE_SIZE))
    _, _, dates = sfeat.build_dataset(_rainfall())
    assert len(dates) == 2
