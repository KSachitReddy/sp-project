"""`nimbus serve` warms the forecaster up before serving, and a failed
warm-up never stops the server. Nothing is actually served or loaded here."""
import pytest
from click.testing import CliRunner

import app as nimbus_app
import cli
from weather import chatbot, features, forecast_model


@pytest.fixture
def events(monkeypatch):
    log = []
    monkeypatch.setattr(nimbus_app, "main", lambda: log.append("app.main"))
    monkeypatch.setattr(features, "load_rainfall", lambda *a, **k: log.append("load_rainfall"))
    monkeypatch.setattr(forecast_model, "_load", lambda: log.append("forecast_model._load"))
    monkeypatch.setattr(chatbot, "_last_available_date", lambda: log.append("last_available_date"))
    return log


def test_serve_warms_up_before_serving(events):
    result = CliRunner().invoke(cli.cli, ["serve"])
    assert result.exit_code == 0, result.output
    assert events == ["load_rainfall", "forecast_model._load", "last_available_date", "app.main"]
    assert "Models ready" in result.output


def test_failed_warm_up_still_serves(monkeypatch, events):
    def boom(*a, **k):
        raise MemoryError("simulated")
    monkeypatch.setattr(features, "load_rainfall", boom)
    result = CliRunner().invoke(cli.cli, ["serve"])
    assert result.exit_code == 0
    assert events == ["app.main"]
    assert "warm-up failed" in result.output and "Models ready" not in result.output


def test_importing_app_does_not_warm_up(monkeypatch):
    import importlib
    calls = []
    monkeypatch.setattr(features, "load_rainfall", lambda *a, **k: calls.append("load_rainfall"))
    monkeypatch.setattr(forecast_model, "_load", lambda: calls.append("forecast_model._load"))
    importlib.reload(nimbus_app)  # what tests and WSGI servers do -- must stay cheap
    assert calls == []
