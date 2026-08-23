"""
Rainfall chatbot: lets a user ask things like "how much did it rain in Pune on
15 July 2023" in plain English.

- india_places.json is an offline gazetteer (states/UTs + major cities) used to
  resolve a place name to a lat/lon, without any external geocoding service.
- get_rainfall() finds the nearest IMD RF25 grid cell (0.25 degree resolution)
  to that lat/lon and reads the matching rows out of data/rainfall/rainfall_<year>.csv.
- chat() answers in three tiers, fastest first: a deterministic local parser
  (weather.fast_intent) for common single place/date questions -- no LLM at
  all; the Claude API (Haiku, if ANTHROPIC_API_KEY is set) for anything that
  needs real language understanding; and a local Ollama model as a last-resort
  fallback if the cloud call fails or no API key is configured, so the
  chatbot still works fully offline, just slower. Only the middle tier sends
  anything off the machine, and only when it's configured to.
"""
import difflib
import json
import math
import os
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import httpx
import ollama
import pandas as pd

CLOUD_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")

BASE_DIR = Path(__file__).parent.parent
PLACES_PATH = BASE_DIR / "data" / "places" / "india_places.json"
RAINFALL_DIR = BASE_DIR / "data" / "rainfall"
SATELLITE_DIR = BASE_DIR / "data" / "satellite" / "india_cloud"

MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

# The ollama package has no timeout by default, so if the Ollama server is
# unreachable (not running, wrong OLLAMA_HOST) a call can hang forever --
# and since it happens on the request thread, it freezes the whole page, not
# just the chat. A short connect timeout fails fast with a clear error, while
# leaving read/write untimed so a slow-but-working generation isn't cut off.
_client = ollama.Client(timeout=httpx.Timeout(5.0, read=None, write=None, pool=5.0))

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

GET_RAINFALL_FORECAST_TOOL = {
    "type": "function",
    "function": {
        "name": "get_rainfall_forecast",
        "description": (
            "Predict rainfall (mm) for an Indian state, union territory, or city on a "
            "FUTURE date not covered by the historical record. Uses a real short-range "
            "machine-learning forecast for dates within about two weeks of the last "
            "available rainfall data; beyond that it falls back to a climatological "
            "(historical-average-for-this-time-of-year) estimate, since no genuine "
            "recent data exists to forecast from that far out -- the response's 'method' "
            "field says which one was used. Do not use this for dates already covered by "
            "get_rainfall's 2018-2025 range; use get_rainfall for those instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "place": {
                    "type": "string",
                    "description": "Name of an Indian state, union territory, or city, e.g. 'Kerala' or 'Mumbai'.",
                },
                "date": {
                    "type": "string",
                    "description": "The future date to forecast, in YYYY-MM-DD format.",
                },
            },
            "required": ["place", "date"],
        },
    },
}

GET_SATELLITE_NOWCAST_TOOL = {
    "type": "function",
    "function": {
        "name": "get_satellite_nowcast",
        "description": (
            "Predict tomorrow's mean rainfall (mm) for each macro-region of India "
            "(north, south, east, west, central, northeast) from today's actual satellite "
            "imagery, using a trained CNN -- a genuine nationwide nowcast, not a place-specific "
            "lookup. Use for 'will it rain tomorrow' style questions. Takes no arguments; it "
            "always predicts one day ahead of the most recent satellite image on disk."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

GET_SATELLITE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_satellite_image",
        "description": (
            "Look up the NASA GIBS true-colour satellite image of India for a specific day. "
            "Only available for days that have already been downloaded, roughly "
            "2019-01-01 through 2025-12-31."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format.",
                },
            },
            "required": ["date"],
        },
    },
}

GET_RAINFALL_ANALYSIS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_rainfall_analysis",
        "description": (
            "Real computed rainfall trend/anomaly statistics for a place -- actual arithmetic "
            "over the historical record (2018-2025), not a guess. Use for 'analyse'/'trend'/"
            "'pattern'/'compare over the years' style questions. Two modes: pass 'year' for a "
            "single year's month-by-month totals and anomaly vs. that place's own long-term "
            "average; pass 'start_year' and 'end_year' for a year-over-year trend across a range. "
            "Provide exactly one of these modes, not both."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "place": {
                    "type": "string",
                    "description": "Name of an Indian state, union territory, or city, e.g. 'Kerala'.",
                },
                "year": {"type": "integer", "description": "For a single-year summary."},
                "start_year": {"type": "integer", "description": "For a trend: first year of the range."},
                "end_year": {"type": "integer", "description": "For a trend: last year of the range."},
            },
            "required": ["place"],
        },
    },
}


def _system_prompt():
    today = date.today()
    tomorrow = today + timedelta(days=1)
    next_week = today + timedelta(days=7)
    return f"""You are a rainfall assistant for India. Today's date is {today.isoformat()}.

You answer questions about how much it rained on a given day or over a date range, in a given
state, union territory, or city; you can predict rainfall for future dates, both a place-specific
forecast and a same-day nationwide nowcast; you can pull up the satellite image of India for a
given day; and you can report real computed trend/anomaly statistics for a place. You have five tools:

- get_rainfall, backed by the IMD RF25 gridded daily rainfall dataset (0.25 degree resolution),
  covering 2018-01-01 through 2025-12-31 only. Use this for any date in that range.
- get_rainfall_forecast, for a specific place and a date AFTER 2025-12-31 (i.e. not covered by
  get_rainfall). Gives a real short-range ML forecast for roughly the first two weeks past the
  historical record, and a climatology-based (historical-average-for-this-time-of-year) estimate
  beyond that — its 'method' field says which. Always state plainly which kind of estimate it is:
  a real forecast, or a seasonal/climatological average, never blur the two together.
- get_satellite_nowcast, ONLY for a question with no named place that asks specifically about
  tomorrow (e.g. "will it rain tomorrow?"). Takes no arguments. Predicts tomorrow's mean rainfall
  per macro-region of India (north/south/east/west/central/northeast) from today's actual satellite
  image, not from the rainfall dataset.
- get_satellite_image, which looks up the satellite image already downloaded for a given day
  (roughly 2019-01-01 through 2025-12-31).
- get_rainfall_analysis, for "analyse"/"trend"/"pattern"/"how has X changed" style questions about
  a place. Pass 'year' for one year's month-by-month totals and anomaly vs. that place's own
  long-term average; pass 'start_year' and 'end_year' for a year-over-year trend. This returns
  real computed numbers — always use it for these questions instead of estimating a trend
  yourself from memory or from a get_rainfall lookup's raw daily figures.
Always call the relevant tool to get real data — never guess a rainfall figure yourself, and
never claim you can't show an image without calling get_satellite_image first.

Deciding between get_rainfall_forecast and get_satellite_nowcast for a future-rain question: if
the question names ANY specific place (a state, union territory, or city), always use
get_rainfall_forecast with that place and a concrete YYYY-MM-DD date computed from today's date
above, even if the phrasing is "tomorrow" or "next week" — get_satellite_nowcast is ONLY for a
question about all of India with no place named. Examples, computed from today being
{today.isoformat()}: "will it rain in Kerala next week" -> get_rainfall_forecast(place="Kerala",
date="{next_week.isoformat()}"). "will it rain in Mumbai tomorrow" ->
get_rainfall_forecast(place="Mumbai", date="{tomorrow.isoformat()}"). "will it rain tomorrow" (no
place) -> get_satellite_nowcast(). Always resolve relative phrases ("tomorrow", "next week", "in 3
days") to an actual date yourself before calling get_rainfall_forecast — never pass the words
"tomorrow" or "next week" as the date argument.

For get_rainfall, always pass both start_date and end_date in YYYY-MM-DD format. For a single day,
set them equal. If a question implies a range ("in July 2023", "monsoon season 2022"), compute
concrete start_date/end_date values yourself. If the user asks about multiple places or periods
(e.g. "compare Kerala and Tamil Nadu" or "rainfall each year from 2020 to 2023"), issue ALL of
those tool calls together in a single turn rather than one at a time across several turns —
each turn is expensive to run locally, so batching every call you already know you need saves
real time without changing the answer.

If a date falls before 2018 or is more than a year or so past today, tell the user the data
doesn't cover it rather than guessing. If a tool returns an error, report it plainly to the user
instead of making up a number or image.

When get_satellite_image succeeds, the image itself is shown to the user separately — just
briefly confirm the date it's for, do not describe or repeat the image URL.

Keep answers short: for a single day's rainfall, state the mm figure plainly; for a range, give
the total and average. Numbers only need one or two decimal places. Do not mention tool names or
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


def get_rainfall_forecast(place, date_str):
    """Predict rainfall for a future date, via weather.forecast_model. Reuses
    resolve_place/nearest_cell so a place resolves to the same grid cell as
    get_rainfall would use for the same query."""
    from weather.forecast_model import predict_rainfall  # lazy: avoids importing lightgbm/pandas-heavy model code for callers that never forecast

    resolved = resolve_place(place)
    if resolved is None:
        return {"error": f"Couldn't find an Indian state, union territory, or city matching '{place}'."}

    try:
        target_date = pd.to_datetime(date_str)
    except (ValueError, TypeError):
        return {"error": f"Couldn't parse the date '{date_str}'."}

    grid_lat, grid_lon = nearest_cell(resolved["lat"], resolved["lon"])
    result = predict_rainfall(grid_lat, grid_lon, target_date)
    if "error" not in result:
        result["matched_place"] = resolved["name"]
        result["place_type"] = resolved["type"]
    return result


def _last_available_date():
    """Newest date actually on disk, without loading the full 14.5M-row history --
    reuses _load_year_df's cache, which get_rainfall already pays for."""
    years = available_years()
    if not years:
        return None
    df = _load_year_df(years[-1])
    return None if df is None else df["date"].max()


def get_weather(place, date_str):
    """Single entry point for 'what's the weather in <place> on <date>': observed
    rainfall for a date within the historical record, otherwise a forecast. Powers
    the map page's date lookup and the CLI's `forecast` command, so both agree with
    what the chatbot would say for the same place/date."""
    try:
        target_date = pd.to_datetime(date_str).normalize()
    except (ValueError, TypeError):
        return {"error": f"Couldn't parse the date '{date_str}'."}

    last_available = _last_available_date()
    if last_available is not None and target_date <= last_available:
        return get_rainfall(place, date_str, date_str)
    return get_rainfall_forecast(place, date_str)


def get_satellite_nowcast():
    try:
        from weather.nowcast_model import predict_nowcast  # lazy: avoids importing torch for callers that never nowcast
    except ImportError:
        return {"error": "Satellite nowcasting isn't available -- torch isn't installed on this machine."}

    return predict_nowcast()


def get_satellite_image(date_str):
    try:
        d = pd.to_datetime(date_str)
    except (ValueError, TypeError):
        return {"error": f"Couldn't parse the date '{date_str}'."}

    date_key = str(d.date())
    path = SATELLITE_DIR / date_key[:4] / f"{date_key}.png"
    if not path.exists():
        return {"error": f"No satellite image available for {date_key}."}

    return {"date": date_key, "image_url": f"/api/satellite/{date_key}"}


def _supports_tools(name):
    """True/False if we can tell whether `name` supports tool calling, None if unknown
    (e.g. the model was removed between `ollama list` and this check)."""
    try:
        info = _client.show(name)
    except Exception:
        return None
    return "tools" in (info.capabilities or [])


def list_models():
    """Models already pulled in the local Ollama install, for a model picker. This
    chatbot only works with tool-calling models (it looks up rainfall/satellite data
    via tools rather than letting the model guess), so each entry says whether it
    qualifies."""
    try:
        resp = _client.list()
    except Exception as e:
        raise RuntimeError(
            f"Can't reach Ollama at {os.environ.get('OLLAMA_HOST', 'http://localhost:11434')} "
            f"— is `ollama serve` running? ({e})"
        )

    models = []
    for m in resp.models:
        models.append({
            "name": m.model,
            "size_bytes": m.size,
            "parameter_size": m.details.parameter_size if m.details else None,
            "supports_tools": _supports_tools(m.model),
        })
    return sorted(models, key=lambda m: m["name"])


def _execute_tool_call(name, tool_input):
    """Dispatches to the named tool and always returns a plain dict, even on
    an unexpected failure -- a bug in one tool (missing dependency, bad
    input, whatever) should end that turn with an honest error message, not
    take down the entire chat session with an unhandled traceback."""
    try:
        if name == "get_rainfall":
            return get_rainfall(
                tool_input.get("place", ""),
                tool_input.get("start_date", ""),
                tool_input.get("end_date"),
            )
        if name == "get_rainfall_forecast":
            return get_rainfall_forecast(tool_input.get("place", ""), tool_input.get("date", ""))
        if name == "get_satellite_nowcast":
            return get_satellite_nowcast()
        if name == "get_satellite_image":
            return get_satellite_image(tool_input.get("date", ""))
        if name == "get_rainfall_analysis":
            return get_rainfall_analysis(
                tool_input.get("place", ""),
                tool_input.get("year"),
                tool_input.get("start_year"),
                tool_input.get("end_year"),
            )
        return {"error": f"Unknown tool '{name}'."}
    except Exception as e:
        return {"error": f"'{name}' failed unexpectedly ({e})."}


def get_rainfall_analysis(place, year=None, start_year=None, end_year=None):
    from weather import analysis  # lazy: sqlite3 + the analysis db are only needed here

    resolved = resolve_place(place)
    if resolved is None:
        return {"error": f"Couldn't find an Indian state, union territory, or city matching '{place}'."}
    if not analysis.available():
        return {"error": "Analysis database not built yet -- run `uv run nimbus analyse --build` first."}

    if start_year and end_year:
        result = analysis.trend(resolved["name"], start_year, end_year)
    elif year:
        result = analysis.yearly_summary(resolved["name"], year)
    else:
        return {"error": "Specify either 'year', or both 'start_year' and 'end_year'."}

    if "error" not in result:
        result["matched_place"] = resolved["name"]
    return result


def serialize_messages(messages):
    """Turn a conversation (which may hold Ollama SDK model objects from a prior
    assistant turn) into plain JSON so it can round-trip through the browser and
    be sent back unchanged on the next turn."""
    out = []
    for m in messages:
        out.append(m.model_dump() if hasattr(m, "model_dump") else dict(m))
    return out


def chat(messages, model=None, compute=None):
    """Run one assistant turn given the full conversation so far, fastest tier first:

    1. weather.fast_intent -- deterministic, no LLM at all. Only attempted on a
       single first-turn question (a follow-up needs conversation context this
       parser doesn't track), and only used if it confidently recognizes the
       question; otherwise falls through.
    2. The Claude API (CLOUD_MODEL / ANTHROPIC_API_KEY), if a key is configured --
       real language understanding, answers in a few seconds.
    3. Local Ollama (this function's original implementation) -- always works
       fully offline, but tens of seconds on CPU-only hardware. Used if step 2
       isn't configured or its call fails for any reason (network, auth, etc.),
       so the chatbot never stops working outright.

    `messages` is a list of {"role": ..., "content": ...} dicts. `model` optionally
    overrides the default Ollama model (OLLAMA_MODEL env var) for step 3. `compute`
    optionally forces CPU-only execution ("cpu") for step 3 instead of Ollama's
    default of using a GPU when one's available ("auto", or omitted). Returns the
    updated messages list (with the assistant's turn, including any tool
    round-trips, appended), the final assistant text reply, and a list of any
    satellite images fetched along the way (each {"date": ..., "image_url": ...}).
    """
    if len(messages) == 1 and messages[0].get("role") == "user":
        from weather import fast_intent  # lazy: fast_intent imports this module, avoiding a circular import at load time

        fast = fast_intent.try_fast_reply(messages[0]["content"])
        if fast is not None:
            reply_text, images = fast
            return list(messages) + [{"role": "assistant", "content": reply_text}], reply_text, images

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _cloud_chat(messages)
        except Exception:
            pass  # fall through to local Ollama below

    return _ollama_chat(messages, model=model, compute=compute)


def _cloud_chat(messages):
    """Tool-calling turn against the Claude API. Raises on any failure (missing
    package, bad/missing key, network error, API error) -- chat() catches that
    and falls back to local Ollama, so this never needs to be failure-proof itself."""
    import anthropic

    client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY from the environment
    tools = [GET_RAINFALL_TOOL, GET_RAINFALL_FORECAST_TOOL, GET_SATELLITE_NOWCAST_TOOL, GET_SATELLITE_TOOL, GET_RAINFALL_ANALYSIS_TOOL]
    claude_tools = [
        {"name": t["function"]["name"], "description": t["function"]["description"],
         "input_schema": t["function"]["parameters"]}
        for t in tools
    ]

    working = list(messages)
    claude_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
    images = []

    for _ in range(5):  # cap tool round-trips
        response = client.messages.create(
            model=CLOUD_MODEL,
            max_tokens=512,  # short factual answers only -- see _system_prompt()
            system=_system_prompt(),
            tools=claude_tools,
            messages=claude_messages,
        )

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        reply_text = "".join(b.text for b in response.content if b.type == "text")

        if not tool_use_blocks:
            working.append({"role": "assistant", "content": reply_text})
            return working, reply_text, images

        working.append({
            "role": "assistant",
            "content": reply_text,
            "tool_calls": [{"function": {"name": b.name, "arguments": b.input}} for b in tool_use_blocks],
        })
        claude_messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for b in tool_use_blocks:
            result = _execute_tool_call(b.name, b.input)
            if b.name == "get_satellite_image" and "image_url" in result:
                images.append({"date": result["date"], "image_url": result["image_url"]})
            working.append({"role": "tool", "content": json.dumps(result)})
            tool_results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(result)})
        claude_messages.append({"role": "user", "content": tool_results})

    reply_text = "Sorry, I got stuck looking that up — could you rephrase your question?"
    working.append({"role": "assistant", "content": reply_text})
    return working, reply_text, images


def _ollama_chat(messages, model=None, compute=None):
    """Local Ollama tool-calling loop -- see chat()'s docstring for where this
    fits (last-resort fallback: fully offline, but slow on CPU-only hardware)."""
    chat_model = model or MODEL
    # num_predict caps generation length: on CPU-only inference, an open-ended "analyse
    # this" question can otherwise make the model ramble for a very long response, and
    # decode time (unlike prefill) scales directly with output length -- this bounds the
    # worst case without affecting the short, factual answers this chatbot is meant to give.
    options = {"num_gpu": 0, "num_predict": 300} if compute == "cpu" else {"num_predict": 300}

    try:
        _client.list()
    except Exception as e:
        raise RuntimeError(
            f"Can't reach Ollama at {os.environ.get('OLLAMA_HOST', 'http://localhost:11434')} "
            f"— is `ollama serve` running? ({e})"
        )

    if _supports_tools(chat_model) is False:
        raise RuntimeError(
            f"'{chat_model}' doesn't support tool calling, which this chatbot needs to look up "
            f"real rainfall and satellite data — pick a different model (e.g. llama3.2:3b)."
        )

    working = [{"role": "system", "content": _system_prompt()}] + list(messages)
    images = []
    tools = [GET_RAINFALL_TOOL, GET_RAINFALL_FORECAST_TOOL, GET_SATELLITE_NOWCAST_TOOL, GET_SATELLITE_TOOL, GET_RAINFALL_ANALYSIS_TOOL]

    for _ in range(5):  # cap tool round-trips
        try:
            response = _client.chat(model=chat_model, messages=working, tools=tools, options=options)
        except Exception as e:
            raise RuntimeError(
                f"Ollama model '{chat_model}' isn't available — run `ollama pull {chat_model}`. ({e})"
            )

        assistant_msg = response.message
        working.append(assistant_msg)

        if not assistant_msg.tool_calls:
            reply_text = assistant_msg.content or ""
            # Drop the system prompt we prepended before returning to the caller.
            return working[1:], reply_text, images

        for call in assistant_msg.tool_calls:
            result = _execute_tool_call(call.function.name, call.function.arguments)
            if call.function.name == "get_satellite_image" and "image_url" in result:
                images.append({"date": result["date"], "image_url": result["image_url"]})
            working.append({"role": "tool", "content": json.dumps(result)})

    return working[1:], "Sorry, I got stuck looking that up — could you rephrase your question?", images
