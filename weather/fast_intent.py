"""
Deterministic fast path for the chatbot: recognizes the common question
shapes (single place + single date, observed or forecast) and answers them
directly from local data with zero LLM calls. Anything it can't confidently
parse -- multiple places, date ranges, comparisons, open-ended "analyse"/
"why"/"explain" questions, follow-ups -- returns None so the caller falls
through to a real LLM (cloud, then local Ollama).

This exists because routing every question through a local CPU-only LLM,
even a trivial "how much did it rain in X on Y", costs tens of seconds; that
same question has exactly one correct, unambiguous answer computable in
milliseconds once place and date are extracted.
"""
import re
from datetime import date, timedelta

import pandas as pd

from weather import chatbot

_IMAGE_WORDS = ("image", "picture", "photo", "satellite")
_FORECAST_WORDS = ("will it rain", "forecast", "going to rain", "expect rain")
_ANALYSIS_WORDS = ("analyse", "analyze", "trend", "compare", "why", "explain", "pattern", "history of")

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _find_place(text):
    """The gazetteer place whose name/alias appears in `text` as a whole word,
    preferring the longest match. Returns None if nothing matches.

    Short (<=3 char) *aliases* are excluded even on a whole-word match: several
    are ordinary English words ("as" for Assam, "or"/"up" for Odisha/Uttar
    Pradesh), so word-boundary matching alone isn't enough to avoid false
    positives like "or" inside "will it rain tomorrow". The canonical place
    name itself is always eligible regardless of length (e.g. "Goa"), since a
    real place name isn't a collision risk the way an abbreviation is."""
    text_norm = chatbot._normalize(text)
    best = None
    for place in chatbot._load_places():
        candidates = [(place["name"], True)] + [(a, False) for a in place.get("aliases", [])]
        for candidate, is_canonical_name in candidates:
            c_norm = chatbot._normalize(candidate)
            if not c_norm or (not is_canonical_name and len(c_norm) <= 3):
                continue
            if re.search(r"\b" + re.escape(c_norm) + r"\b", text_norm) and (
                best is None or len(c_norm) > len(best[1])
            ):
                best = (place, c_norm)
    return best[0] if best else None


def _find_dates(text):
    """Explicit YYYY-MM-DD dates, or a recognized relative term, found in
    `text`. Returns a list of date() objects, or None if nothing recognizable
    is present (deliberately doesn't try to parse everything pandas could --
    ambiguous or free-form dates should go through the LLM path instead)."""
    today = date.today()
    lowered = text.lower()

    explicit = [pd.to_datetime(d).date() for d in _DATE_RE.findall(text)]
    if explicit:
        return explicit
    if "tomorrow" in lowered:
        return [today + timedelta(days=1)]
    if "next week" in lowered:
        return [today + timedelta(days=7)]
    if "today" in lowered:
        return [today]
    return None


def try_fast_reply(text):
    """(reply_text, images) for a confidently-recognized simple question, or
    None to defer to the LLM path."""
    lowered = text.lower()
    if any(w in lowered for w in _ANALYSIS_WORDS):
        return None

    place = _find_place(text)
    dates = _find_dates(text)

    if any(w in lowered for w in _IMAGE_WORDS):
        if not (dates and len(dates) == 1):
            return None
        result = chatbot.get_satellite_image(str(dates[0]))
        if "error" in result:
            return None
        return f"Here's the satellite image for {dates[0]}.", [
            {"date": result["date"], "image_url": result["image_url"]}
        ]

    if place is None or not dates or len(dates) != 1:
        return None

    target = dates[0]
    last_available = chatbot._last_available_date()
    is_future = last_available is None or pd.Timestamp(target) > last_available
    is_forecast_phrasing = any(w in lowered for w in _FORECAST_WORDS)

    if is_future or is_forecast_phrasing:
        result = chatbot.get_rainfall_forecast(place["name"], str(target))
        if "error" in result:
            return None  # e.g. a historical date phrased as "will it rain" -- let the LLM sort it out
        caveat = "" if result["method"] == "ml_short_range" else " (historical average, not a real forecast)"
        return f"{result['matched_place']}, {result['target_date']}: {result['predicted_mm']} mm{caveat}", []

    result = chatbot.get_rainfall(place["name"], str(target), str(target))
    if "error" in result:
        return None
    return f"{result['matched_place']}, {target}: {result['total_rainfall_mm']} mm", []
