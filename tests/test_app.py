"""Flask error paths: bad input must give a clean 4xx JSON error, never a 500."""
import io

import pytest

import app as nimbus_app


@pytest.fixture
def client():
    return nimbus_app.app.test_client()


@pytest.mark.parametrize("url, status", [
    ("/api/forecast", 400),
    ("/api/forecast?place=Mumbai", 400),
    ("/api/forecast?place=Mumbai&date=notadate", 400),
    ("/api/forecast?place=Mumbai&date=2026-02-30", 400),
    ("/api/forecast?place=Xyzzyville&date=2030-01-05", 400),
    ("/api/analysis?place=Pune", 400),
    ("/api/analysis?place=Pune&year=1900", 400),
    ("/api/satellite/2025-13-45", 400),
    ("/api/satellite/2026-02-30", 400),
    ("/api/satellite/not-a-date", 400),
    ("/api/year/1999", 404),
])
def test_get_error_paths(client, url, status):
    r = client.get(url)
    assert r.status_code == status
    if url.startswith("/api/") and status == 400:
        assert "error" in r.get_json()


@pytest.mark.parametrize("body", [{}, {"messages": []}, {"messages": "hi"},
                                  {"messages": [{"role": "user", "content": "x"}], "model": 3},
                                  {"messages": [{"role": "user", "content": "x"}], "compute": "gpu"}])
def test_chat_rejects_bad_bodies(client, body):
    r = client.post("/api/chat", json=body)
    assert r.status_code == 400 and "error" in r.get_json()


def test_convert_rejects_missing_and_invalid_files(client):
    assert client.post("/api/convert").status_code == 400
    r = client.post("/api/convert", data={"ncfile": (io.BytesIO(b"x"), "rain.txt")})
    assert r.status_code == 400
    r = client.post("/api/convert", data={"ncfile": (io.BytesIO(b"junk"), "rain.nc")})
    assert r.status_code == 400


def test_years_lists_available_data(client):
    years = client.get("/api/years").get_json()["years"]
    assert years == sorted(years) and 2018 in years
