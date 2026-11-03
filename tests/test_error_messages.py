"""Errors must say what's wrong and what to do: impossible dates are reported
as invalid (not "no image"), and missing imagery points at `nimbus download`.
Also: the measurement scripts must import on Windows, which has no `resource`."""
import importlib.util
import sys
from pathlib import Path

import pytest

import app as nimbus_app
from weather import chatbot
from weather.thermal_estimate import predict_thermal_rainfall

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
FAR_FUTURE = "2099-01-01"  # never has imagery on disk


def test_thermal_estimate_rejects_impossible_date():
    assert "Invalid date" in predict_thermal_rainfall("2026-02-30")["error"]


def test_thermal_estimate_missing_image_says_how_to_fix():
    assert "nimbus download" in predict_thermal_rainfall(FAR_FUTURE)["error"]


def test_satellite_endpoint_missing_image_says_how_to_fix():
    r = nimbus_app.app.test_client().get(f"/api/satellite/{FAR_FUTURE}")
    assert r.status_code == 404 and "nimbus download" in r.get_json()["error"]


def test_image_lookup_missing_image_says_how_to_fix():
    assert "nimbus download" in chatbot.get_satellite_image(FAR_FUTURE)["error"]


def test_nowcast_missing_image_says_how_to_fix():
    nm = pytest.importorskip("weather.nowcast_model")  # torch is optional
    if not nm.model_available():
        pytest.skip("nowcast model not trained")
    assert "nimbus download" in nm.predict_nowcast(as_of_date=FAR_FUTURE)["error"]


@pytest.mark.parametrize("script", ["benchmark.py", "evaluate_forecast.py"])
def test_scripts_import_without_resource_module(monkeypatch, script):
    monkeypatch.setitem(sys.modules, "resource", None)  # what Windows looks like: import raises ImportError
    spec = importlib.util.spec_from_file_location(f"_nimbus_{script[:-3]}", SCRIPTS_DIR / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # module-level code only; main() isn't run
    assert module.resource is None
