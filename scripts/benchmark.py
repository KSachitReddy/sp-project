"""
Measure the app's inference latency and memory in-process (Flask test client,
no server needed) and write models/benchmark.json.

    .venv/bin/python scripts/benchmark.py

Records: cold-start time of the first forecast request (it loads the full
14.5M-row rainfall history), median/p95 latency per endpoint once warm, direct
CNN nowcast inference latency (if torch is installed), peak RSS, model file
sizes, and the machine it ran on.
"""
import json
import platform
try:
    import resource  # POSIX-only; absent on Windows
except ImportError:
    resource = None
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).parent.parent))

import app as nimbus_app
from weather import hardware

MODELS_DIR = Path(__file__).parent.parent / "models"
OUT_PATH = MODELS_DIR / "benchmark.json"
PLACES = ["Mumbai", "Delhi", "Chennai", "Kolkata", "Pune", "Kerala", "Assam", "Jaipur", "Bhopal", "Patna"]


def _forecast_url(place, date_str):
    return "/api/forecast?" + urlencode({"place": place, "date": date_str})


def _timed(fn):
    t = time.perf_counter()
    result = fn()
    return (time.perf_counter() - t) * 1000, result


def _summary(times_ms, codes):
    times_ms = sorted(times_ms)
    return {
        "n": len(times_ms),
        "median_ms": round(statistics.median(times_ms), 1),
        "p95_ms": round(times_ms[max(0, int(round(0.95 * len(times_ms))) - 1)], 1),
        "status_codes": {str(c): codes.count(c) for c in sorted(set(codes))},
    }


def bench(client, urls):
    times, codes = [], []
    for url in urls:
        ms, r = _timed(lambda: client.get(url))
        times.append(ms)
        codes.append(r.status_code)
    return _summary(times, codes)


def bench_cnn():
    try:
        from weather import satellite_features as sfeat
        from weather.nowcast_model import predict_nowcast
    except ImportError:
        return {"skipped": "torch not installed"}
    dates = sfeat.available_image_dates()[-20:]
    if not dates:
        return {"skipped": "no satellite images on disk"}
    cold_ms, _ = _timed(lambda: predict_nowcast(as_of_date=dates[0]))  # includes loading the model
    times = [_timed(lambda d=d: predict_nowcast(as_of_date=d))[0] for d in dates]
    return {"cold_ms": round(cold_ms, 1), **_summary(times, [200] * len(times))}


def main():
    client = nimbus_app.app.test_client()
    report = {}

    cold_ms, r = _timed(lambda: client.get(_forecast_url("Mumbai", "2026-01-05")))
    report["cold_start_first_forecast_ms"] = round(cold_ms, 1)
    report["cold_start_status"] = r.status_code

    report["endpoints"] = {
        "forecast_ml_short_range": bench(client, [_forecast_url(p, f"2026-01-{d:02d}") for p in PLACES for d in range(1, 15, 3)]),
        "forecast_satellite": bench(client, [_forecast_url(p, f"2026-0{m}-15") for p in PLACES for m in range(3, 8)]),
        "forecast_climatology": bench(client, [_forecast_url(p, f"2030-0{m}-15") for p in PLACES for m in range(1, 6)]),
        "observed_lookup": bench(client, [_forecast_url(p, f"2024-0{m}-10") for p in PLACES for m in range(1, 6)]),
        "analysis": bench(client, ["/api/analysis?" + urlencode({"place": p, "year": y}) for p in PLACES for y in range(2019, 2024)]),
        "year_payload": bench(client, [f"/api/year/{y}" for y in (2023, 2024, 2025)]),
    }
    report["cnn_nowcast_inference"] = bench_cnn()
    report["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024) if resource else None
    report["model_sizes_bytes"] = {p.name: p.stat().st_size for p in sorted(MODELS_DIR.iterdir())
                                   if p.suffix in (".txt", ".pt", ".csv")}
    info = hardware.detect()
    report["machine"] = {
        "python": platform.python_version(), "os": platform.platform(),
        "cores": info.get("cores"), "cpu_mhz": info.get("mhz"), "ram_gb": info.get("ram_gb"), "avx2": info.get("avx2"),
        "tier": info.get("tier"),
    }

    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
