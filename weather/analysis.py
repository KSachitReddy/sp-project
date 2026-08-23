"""
Real computed rainfall analytics: monthly/yearly totals, anomaly vs. each
place's own long-term average, and year-over-year trend -- precomputed once
into SQLite and read back in milliseconds. This exists so "analyse the
rainfall trend in Kerala" runs on actual arithmetic instead of an LLM
eyeballing a JSON blob of daily figures and describing what it sees.

Build (reads the full ~14.5M-row rainfall history -- do this on a capable
machine, not ~1GHz-class hardware, same as training either model):

    uv run nimbus analyse --build

Query (cheap -- fine on any hardware, including where it was never built):

    uv run nimbus analyse --place Kerala --year 2023
    uv run nimbus analyse --place Kerala --trend 2020 2023

The output db (models/analysis.db) is small (a few MB) and, like the trained
model files, meant to be built once and copied to wherever the app runs.
"""
import sqlite3
from pathlib import Path

import pandas as pd

from weather import chatbot
from weather import features as feat

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "models" / "analysis.db"


def build():
    """Precompute monthly rainfall stats for every gazetteer place, across
    every year of data on disk. Returns the number of (place, year, month)
    rows written."""
    full = feat.load_rainfall()
    full = full.assign(year=full["date"].dt.year, month=full["date"].dt.month)

    rows = []
    for place in chatbot._load_places():
        grid_lat, grid_lon = chatbot.nearest_cell(place["lat"], place["lon"])
        cell = full[(full["latitude"] == grid_lat) & (full["longitude"] == grid_lon)]
        if cell.empty:
            continue

        monthly = cell.groupby(["year", "month"])["rainfall_mm"].agg(["sum", "count", "max"])
        # This place's own long-term average for each calendar month, across all years --
        # the baseline "anomaly" is measured against, not a single arbitrary year.
        yearly_by_month = cell.groupby(["year", "month"])["rainfall_mm"].sum()
        climatology_by_month = yearly_by_month.groupby("month").mean()

        for (year, month), row in monthly.iterrows():
            clim = climatology_by_month.get(month)
            anomaly_pct = ((row["sum"] - clim) / clim * 100) if clim else None
            rows.append((
                place["name"], place["type"], int(year), int(month),
                round(float(row["sum"]), 2),
                round(float(row["sum"] / row["count"]), 2),
                round(float(row["max"]), 2),
                round(float(clim), 2) if clim is not None else None,
                round(float(anomaly_pct), 1) if anomaly_pct is not None else None,
            ))

    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS monthly_stats")
    conn.execute("""
        CREATE TABLE monthly_stats (
            place TEXT, place_type TEXT, year INTEGER, month INTEGER,
            total_mm REAL, avg_daily_mm REAL, max_daily_mm REAL,
            climatology_avg_mm REAL, anomaly_pct REAL,
            PRIMARY KEY (place, year, month)
        )
    """)
    # INSERT OR IGNORE: the gazetteer has a small number of duplicate names
    # (e.g. "Delhi" listed as both a state/UT and a city) -- keep whichever
    # resolves first rather than erroring, since queries filter by name alone.
    conn.executemany("INSERT OR IGNORE INTO monthly_stats VALUES (?,?,?,?,?,?,?,?,?)", rows)
    conn.execute("CREATE INDEX idx_place_year ON monthly_stats(place, year)")
    conn.commit()
    conn.close()
    return len(rows)


def available():
    return DB_PATH.exists()


def _connect():
    if not available():
        raise RuntimeError("Analysis database not built yet -- run `uv run nimbus analyse --build`.")
    return sqlite3.connect(DB_PATH)


def yearly_summary(place_name, year):
    """Real computed stats for one place/year: total, wettest month, and
    anomaly vs. that place's own long-term average for those months."""
    conn = _connect()
    df = pd.read_sql(
        "SELECT * FROM monthly_stats WHERE place = ? AND year = ? ORDER BY month",
        conn, params=(place_name, year),
    )
    conn.close()
    if df.empty:
        return {"error": f"No computed stats for '{place_name}' in {year} -- check the place name and year."}

    wettest = df.loc[df["total_mm"].idxmax()]
    return {
        "place": place_name,
        "year": int(year),
        "total_mm": round(float(df["total_mm"].sum()), 2),
        "wettest_month": int(wettest["month"]),
        "wettest_month_mm": round(float(wettest["total_mm"]), 2),
        "avg_anomaly_pct_vs_own_history": round(float(df["anomaly_pct"].dropna().mean()), 1) if df["anomaly_pct"].notna().any() else None,
        "months": df[["month", "total_mm", "anomaly_pct"]].to_dict("records"),
    }


def trend(place_name, start_year, end_year):
    """Year-over-year totals for a place across a range -- an actual computed
    trend, not a paraphrase of raw numbers."""
    conn = _connect()
    df = pd.read_sql(
        "SELECT year, SUM(total_mm) as total_mm FROM monthly_stats "
        "WHERE place = ? AND year BETWEEN ? AND ? GROUP BY year ORDER BY year",
        conn, params=(place_name, start_year, end_year),
    )
    conn.close()
    if df.empty:
        return {"error": f"No computed stats for '{place_name}' between {start_year} and {end_year}."}

    totals = df["total_mm"].tolist()
    direction = "increasing" if totals[-1] > totals[0] else "decreasing" if totals[-1] < totals[0] else "flat"
    pct_change = round((totals[-1] - totals[0]) / totals[0] * 100, 1) if totals[0] else None

    return {
        "place": place_name,
        "start_year": int(start_year), "end_year": int(end_year),
        "yearly_totals_mm": dict(zip(df["year"].tolist(), [round(t, 2) for t in totals])),
        "direction": direction,
        "pct_change_start_to_end": pct_change,
    }


def _ensure_nowcast_log_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nowcast_log (
            image_date TEXT, region TEXT,
            predicted_date TEXT, predicted_mm REAL,
            actual_mm REAL, validated_at TEXT,
            PRIMARY KEY (image_date, region)
        )
    """)


def log_nowcast_prediction():
    """Record today's satellite nowcast prediction (if not already logged for
    today's image) so it can be checked against real rainfall once that
    becomes available -- the only way to build up a live, day-by-day accuracy
    picture instead of only ever backtesting against past years. Returns the
    image_date logged, or None if no prediction could be made (no torch, no
    recent image, etc.)."""
    try:
        from weather.nowcast_model import predict_nowcast
    except ImportError:
        return None

    result = predict_nowcast()
    if "error" in result:
        return None

    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    _ensure_nowcast_log_table(conn)
    rows = [
        (result["image_date"], region, result["predicted_date"], mm, None, None)
        for region, mm in result["predicted_mm_by_region"].items()
    ]
    conn.executemany("INSERT OR IGNORE INTO nowcast_log VALUES (?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return result["image_date"]


def validate_pending_nowcasts():
    """Fill in actual_mm for any logged predictions whose predicted_date now
    has real rainfall data on disk, then compare the newly-validated error
    against the original backtested metrics (models/nowcast_model_metrics.json)
    to flag drift. Returns a summary dict, or None if there was nothing to check."""
    import json as _json

    from weather import satellite_features as sfeat

    if not DB_PATH.exists():
        return None

    conn = sqlite3.connect(DB_PATH)
    _ensure_nowcast_log_table(conn)
    pending = pd.read_sql("SELECT * FROM nowcast_log WHERE actual_mm IS NULL", conn)
    if pending.empty:
        conn.close()
        return None

    regional = sfeat.regional_daily_rainfall(feat.load_rainfall())
    regional.index = pd.to_datetime(regional.index)

    newly_validated = []
    for _, row in pending.iterrows():
        predicted_date = pd.Timestamp(row["predicted_date"])
        if predicted_date not in regional.index or row["region"] not in regional.columns:
            continue
        actual = regional.loc[predicted_date, row["region"]]
        if pd.isna(actual):
            continue
        conn.execute(
            "UPDATE nowcast_log SET actual_mm = ?, validated_at = ? WHERE image_date = ? AND region = ?",
            (float(actual), pd.Timestamp.now().isoformat(), row["image_date"], row["region"]),
        )
        newly_validated.append({
            "image_date": row["image_date"], "region": row["region"],
            "predicted_mm": row["predicted_mm"], "actual_mm": round(float(actual), 2),
        })
    conn.commit()

    if not newly_validated:
        conn.close()
        return None

    all_validated = pd.read_sql("SELECT * FROM nowcast_log WHERE actual_mm IS NOT NULL", conn)
    conn.close()

    live_mae = float((all_validated["predicted_mm"] - all_validated["actual_mm"]).abs().mean())

    drift = None
    metrics_path = BASE_DIR / "models" / "nowcast_model_metrics.json"
    if metrics_path.exists():
        backtest = _json.loads(metrics_path.read_text())
        backtest_maes = [
            r["mae"] for split in backtest.get("by_split", {}).values() for r in split.values()
        ]
        if backtest_maes:
            backtest_mae = sum(backtest_maes) / len(backtest_maes)
            if backtest_mae > 0:
                rel_change = (live_mae - backtest_mae) / backtest_mae
                drift = {
                    "backtest_mae": round(backtest_mae, 3),
                    "live_mae": round(live_mae, 3),
                    "relative_change": round(rel_change, 3),
                    "significant": abs(rel_change) > 0.25,  # >25% off the original backtest
                }

    return {"newly_validated": newly_validated, "live_mae_all_time": round(live_mae, 3), "drift": drift}
