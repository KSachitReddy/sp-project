"""
Rainfall chatbot: lets a user ask things like "how much did it rain in Pune on
15 July 2023" in plain English.

- india_places.json is an offline gazetteer (states/UTs + major cities) used to
  resolve a place name to a lat/lon, without any external geocoding service.
- get_rainfall() finds the nearest IMD RF25 grid cell (0.25 degree resolution)
  to that lat/lon and reads the matching rows out of data/rainfall/rainfall_<year>.csv.
- chat() drives a tool-use turn against a local Ollama model: the model parses
  the free-text question (place, date/date range) via the get_rainfall tool,
  we execute it against the local data, and the model phrases the final answer.
  Nothing leaves the machine — no API key needed.
"""
import difflib
import json
import math
import os
from datetime import date
from functools import lru_cache
from pathlib import Path

import ollama
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
PLACES_PATH = BASE_DIR / "data" / "places" / "india_places.json"
RAINFALL_DIR = BASE_DIR / "data" / "rainfall"

MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

GET_RAINFALL_TOOL = {
    "type": "function",
    "function": {
        "name": "get_rainfall",
        "description": (
            "Look up observed rainfall (mm) for an Indian state, union territory, or city "
            "on a specific day or over a date range, using the nearest IMD RF25 gridded "
            "rainfall data point (0.25 degree resolution) to that place. "
            "Data is only available for 2018-01-01 through 2025-12-31."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "place": {
                    "type": "string",
                    "description": "Name of an Indian state, union territory, or city, e.g. 'Kerala' or 'Mumbai'.",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format.",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format. Same as start_date for a single day.",
                },
            },
            "required": ["place", "start_date", "end_date"],
        },
    },
}


def _system_prompt():
    return f"""You are a rainfall assistant for India. Today's date is {date.today().isoformat()}.

You answer questions about how much it rained on a given day or over a date range, in a given
state, union territory, or city. You have one tool, get_rainfall, backed by the IMD RF25 gridded
daily rainfall dataset (0.25 degree resolution), covering 2018-01-01 through 2025-12-31 only.
Always call the tool to get real numbers — never guess a rainfall figure yourself.

Always pass both start_date and end_date in YYYY-MM-DD format. For a single day, set them equal.
If a question implies a range ("in July 2023", "monsoon season 2022"), compute concrete
start_date/end_date values yourself. If the user asks about multiple places or periods, call the
tool once per place/period.

If a date falls outside 2018-2025, tell the user the data doesn't cover it rather than guessing.
If the tool returns an error, report it plainly to the user instead of making up a number.

Keep answers short: for a single day, state the mm figure plainly; for a range, give the total
and average. Numbers only need one or two decimal places. Do not mention the tool by name or
describe your process — just answer."""


@lru_cache(maxsize=1)
def _load_places():
    with open(PLACES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["places"]


def _normalize(s):
    return s.strip().lower().replace("&", "and")


def resolve_place(query):
    """Resolve a free-text place name to a gazetteer entry, or None."""
    if not query:
        return None
    places = _load_places()
    q = _normalize(query)

    # Strip common trailing/leading noise words.
    for noise in (" state", " district", " city", " ut", "the ", "in "):
        if q.endswith(noise):
            q = q[: -len(noise)].strip()
        if q.startswith(noise):
            q = q[len(noise):].strip()

    # Exact match on name or alias.
    for place in places:
        if _normalize(place["name"]) == q:
            return place
        if any(_normalize(a) == q for a in place.get("aliases", [])):
            return place

    # Fuzzy match against every name + alias.
    candidates = {}
    for place in places:
        candidates[_normalize(place["name"])] = place
        for a in place.get("aliases", []):
            candidates[_normalize(a)] = place

    close = difflib.get_close_matches(q, candidates.keys(), n=1, cutoff=0.72)
    if close:
        return candidates[close[0]]
    return None


def _nearest_grid_coord(value):
    return round(round(value * 4) / 4, 2)


@lru_cache(maxsize=1)
def _valid_cells():
    """The set of land grid cells that actually carry data (the IMD land mask
    is the same every year, so any single year's file tells us this)."""
    years = available_years()
    if not years:
        return None
    df = pd.read_csv(RAINFALL_DIR / f"rainfall_{years[-1]}.csv", usecols=["latitude", "longitude"])
    return df.drop_duplicates().to_numpy()


def nearest_cell(lat, lon):
    """Nearest grid cell to (lat, lon). Rounding to the 0.25 degree grid can
    land on a coastal point with no data (ocean side of the mask), so fall
    back to the closest cell that actually has data."""
    grid_lat, grid_lon = _nearest_grid_coord(lat), _nearest_grid_coord(lon)
    cells = _valid_cells()
    if cells is None:
        return grid_lat, grid_lon
    if ((cells[:, 0] == grid_lat) & (cells[:, 1] == grid_lon)).any():
        return grid_lat, grid_lon
    d2 = (cells[:, 0] - lat) ** 2 + (cells[:, 1] - lon) ** 2
    idx = int(d2.argmin())
    return float(cells[idx, 0]), float(cells[idx, 1])


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@lru_cache(maxsize=None)
def _load_year_df(year):
    csv_path = RAINFALL_DIR / f"rainfall_{year}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df


def available_years():
    years = []
    for p in RAINFALL_DIR.glob("rainfall_*.csv"):
        try:
            years.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(years)


def get_rainfall(place, start_date, end_date=None):
    """Core lookup used both by the tool call and directly (e.g. for tests)."""
    end_date = end_date or start_date

    resolved = resolve_place(place)
    if resolved is None:
        return {"error": f"Couldn't find an Indian state, union territory, or city matching '{place}'."}

    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
    except (ValueError, TypeError):
        return {"error": f"Couldn't parse the date range '{start_date}' to '{end_date}'."}

    if start > end:
        start, end = end, start

    years_needed = list(range(start.year, end.year + 1))
    yrs_avail = set(available_years())
    missing_years = [y for y in years_needed if y not in yrs_avail]

    frames = [_load_year_df(y) for y in years_needed if y in yrs_avail]
    frames = [f for f in frames if f is not None]

    if not frames:
        return {
            "error": (
                f"No rainfall data available for {start.date()} to {end.date()}. "
                f"Data only covers years: {sorted(yrs_avail)}."
            )
        }

    combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    grid_lat, grid_lon = nearest_cell(resolved["lat"], resolved["lon"])
    mask = (
        (combined["latitude"] == grid_lat)
        & (combined["longitude"] == grid_lon)
        & (combined["date"] >= start)
        & (combined["date"] <= end)
    )
    subset = combined.loc[mask].sort_values("date")

    if subset.empty:
        return {
            "error": (
                f"No rainfall readings for the nearest grid cell ({grid_lat}, {grid_lon}) to "
                f"{resolved['name']} in that date range — it may fall outside the covered years "
                f"or over open water/desert with no land grid cell nearby."
            ),
            "missing_years": missing_years,
        }

    distance_km = round(_haversine_km(resolved["lat"], resolved["lon"], grid_lat, grid_lon), 1)
    total_mm = round(float(subset["rainfall_mm"].sum()), 2)
    avg_mm = round(float(subset["rainfall_mm"].mean()), 2)
    wettest = subset.loc[subset["rainfall_mm"].idxmax()]

    result = {
        "matched_place": resolved["name"],
        "place_type": resolved["type"],
        "query_lat_lon": [resolved["lat"], resolved["lon"]],
        "nearest_grid_cell": {"lat": grid_lat, "lon": grid_lon},
        "distance_from_place_km": distance_km,
        "start_date": str(start.date()),
        "end_date": str(end.date()),
        "days_with_data": int(len(subset)),
        "total_rainfall_mm": total_mm,
        "average_rainfall_mm_per_day": avg_mm,
        "wettest_day": {
            "date": str(wettest["date"].date()),
            "rainfall_mm": round(float(wettest["rainfall_mm"]), 2),
        },
    }
    if missing_years:
        result["note"] = f"No data for years {missing_years}; only counting {sorted(yrs_avail & set(years_needed))}."
    if len(subset) <= 31:
        result["daily_mm"] = [
            {"date": str(r["date"].date()), "rainfall_mm": round(float(r["rainfall_mm"]), 2)}
            for _, r in subset.iterrows()
        ]
    return result


def _execute_tool_call(name, tool_input):
    if name == "get_rainfall":
        return get_rainfall(
            tool_input.get("place", ""),
            tool_input.get("start_date", ""),
            tool_input.get("end_date"),
        )
    return {"error": f"Unknown tool '{name}'."}


def serialize_messages(messages):
    """Turn a conversation (which may hold Ollama SDK model objects from a prior
    assistant turn) into plain JSON so it can round-trip through the browser and
    be sent back unchanged on the next turn."""
    out = []
    for m in messages:
        out.append(m.model_dump() if hasattr(m, "model_dump") else dict(m))
    return out


def chat(messages):
    """Run one assistant turn given the full conversation so far.

    `messages` is a list of {"role": ..., "content": ...} dicts. Returns the
    updated messages list (with the assistant's turn, including any tool
    round-trips, appended) plus the final assistant text reply.
    """
    try:
        ollama.list()
    except Exception as e:
        raise RuntimeError(
            f"Can't reach Ollama at {os.environ.get('OLLAMA_HOST', 'http://localhost:11434')} "
            f"— is `ollama serve` running? ({e})"
        )

    working = [{"role": "system", "content": _system_prompt()}] + list(messages)

    for _ in range(5):  # cap tool round-trips
        try:
            response = ollama.chat(model=MODEL, messages=working, tools=[GET_RAINFALL_TOOL])
        except Exception as e:
            raise RuntimeError(
                f"Ollama model '{MODEL}' isn't available — run `ollama pull {MODEL}`. ({e})"
            )

        assistant_msg = response.message
        working.append(assistant_msg)

        if not assistant_msg.tool_calls:
            reply_text = assistant_msg.content or ""
            # Drop the system prompt we prepended before returning to the caller.
            return working[1:], reply_text

        for call in assistant_msg.tool_calls:
            result = _execute_tool_call(call.function.name, call.function.arguments)
            working.append({"role": "tool", "content": json.dumps(result)})

    return working[1:], "Sorry, I got stuck looking that up — could you rephrase your question?"
