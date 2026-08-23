# Nimbus

AI predictive weather analysis for India. Tools for exploring and *predicting* India's daily rainfall: a Flask app that
renders historical rainfall (IMD RF25 gridded data, 2018–2025) on an
interactive map, a chatbot answering plain-English rainfall questions, and
two predictive models — a short-range rainfall forecaster and a satellite-
imagery-based nowcaster — plus the batch scripts that build all of it.

## Structure

```
app.py                   Flask entrypoint — serves the map page, chat page, and their APIs
weather/
  nc_convert.py           Shared IMD RF25 NetCDF -> CSV/JSON conversion logic
  chatbot.py               Place gazetteer + rainfall/forecast/nowcast lookup + Ollama tool-use chat loop
  features.py               Feature engineering for the rainfall forecasting model
  forecast_model.py         LightGBM rainfall forecaster: train/save/load/predict
  satellite_features.py     Image/label building for the satellite nowcasting model
  nowcast_model.py          CNN satellite nowcaster: train/save/load/predict
scripts/
  convert_all.py           Batch-convert every data/nc/*.nc into data/rainfall/*.csv
  extract.py                One-off extractor for the 2023 CSV + JSON
  download_satellite_images.py   Download daily GIBS imagery, 2019-01-01 through yesterday
  verify_satellite_images.py     Completeness/integrity check over the download
  retry_failed_dates.py           Re-attempt only the dates verify flagged
  train_forecast_model.py         Train the rainfall forecaster, print backtested accuracy
  train_nowcast_model.py          Train the satellite nowcaster, print backtested accuracy
  generate_forecast_demo.py        Generate a demo rainfall forecast for every gazetteer place
web/
  weather_map.html          The map page app.py serves
  chat.html                 The rainfall chatbot page app.py serves
data/
  nc/                       Source IMD RF25 NetCDF files (tracked in git)
  rainfall/                 Generated CSV/JSON rainfall exports (tracked in git)
  places/                   Offline India states/UTs/cities gazetteer used by the chatbot
  satellite/                Downloaded GIBS PNGs + logs (gitignored — regenerable, multi-GB)
  forecast/                 forecast_demo.csv, the demo forecast output (tracked in git)
models/                    Trained model artifacts + metrics JSON (gitignored — regenerable via the train scripts)
docs/
  satellite_images.md       Satellite dataset details: source, format, how to run/resume
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` intentionally skips torch (see "Hardware requirements"
below) -- the map, chatbot, and rainfall forecaster all run without it. Only
`/api/nowcast` (the satellite CNN) needs it.

## Hardware requirements

Everything except the satellite nowcaster is designed to run on modest,
CPU-only hardware -- it's been verified on a dual-core Intel Ivy Bridge
laptop (3rd-gen Core i3) with 8 GB RAM:

- **CPU**: any x86_64 CPU. No GPU is used anywhere in this project.
- **RAM**: 8 GB is comfortable for the map, chatbot, and rainfall forecaster.
  Budget roughly 3-4 GB for `ollama` running the recommended `llama3.2:3b`
  model if you want the chatbot; the rest (Flask, pandas, the LightGBM model)
  fits in well under 1 GB.
- **Disk**: a few hundred MB for the tracked rainfall data and models; add
  ~7 GB if you download the full satellite imagery archive (optional, only
  needed for the nowcaster).
- **Satellite nowcaster (optional, skip it on older hardware)**: needs
  PyTorch, which has no 32-bit build and whose CPU wheel has a documented
  history of crashing with "Illegal instruction" on CPUs without AVX2 --
  that includes Ivy Bridge and any Intel Core CPU older than 4th-gen
  (Haswell). If your CPU doesn't support AVX2, don't install the `torch`
  extra; the rest of the app works fine without it and `/api/nowcast`
  degrades to a clear 503 instead of crashing anything.

## Running the map app

```bash
.venv/bin/python app.py
```

Serves the rainfall map at `http://localhost:5050`, reading years from
`data/nc/` and generated CSVs from `data/rainfall/`. Uploading a new `.nc`
file through the UI validates it, saves it into `data/nc/`, and writes its
CSV into `data/rainfall/`.

## Rainfall chatbot

`http://localhost:5050/chat` answers plain-English questions like "how much
did it rain in Kerala in August 2022?" — it resolves the place against an
offline gazetteer (`data/places/india_places.json`), finds the nearest IMD
RF25 grid cell, and reads the rainfall figures out of `data/rainfall/`.

Each question is answered by the fastest tier that can handle it:

1. **A deterministic local parser** (`weather/fast_intent.py`) — no LLM at all.
   Handles the common case (a single place + a single date, observed or
   forecast) in milliseconds. Nothing leaves the machine, no setup needed.
2. **The Claude API** (`claude-haiku-4-5` by default), if `ANTHROPIC_API_KEY`
   is set — for anything the fast path can't confidently parse (comparisons,
   date ranges, open-ended "analyse this" questions). Answers in a few
   seconds, but the question text is sent to Anthropic's API. Fully optional —
   set the env var to enable it; override the model with `ANTHROPIC_MODEL`.
3. **A local model via [Ollama](https://ollama.com)** — the original,
   fully-offline path, used automatically if step 2 isn't configured or its
   call fails for any reason. No API key, nothing leaves the machine, but
   tens of seconds to a few minutes on CPU-only hardware, since it's a real
   LLM doing real tool-calling inference. One-time setup:

```bash
# install Ollama (see ollama.com), then:
ollama pull llama3.2:3b
ollama serve   # if it isn't already running as a service
```

The Ollama model is configurable via `OLLAMA_MODEL` (default `llama3.2:3b`)
and the server via `OLLAMA_HOST` (default `http://localhost:11434`, can point
at another machine on the network if this one is slow). Pick a model that
supports tool calling — `llama3.2`, `qwen2.5`, and `mistral-nemo` all do;
`tinyllama` does not. Smaller tool-calling models (`llama3.2:1b`,
`qwen2.5:1.5b`) are noticeably faster but were tested and found unreliable
for this chatbot — they fabricated place names and skipped required
forecast-vs-observed disclosures, so neither is used here despite the speed.
If neither Claude nor Ollama is reachable, `/chat` still loads but
`/api/chat` returns a 503 explaining what to fix.

## Forecasting

Two predictive models, both trained on CPU (no GPU required):

- **Rainfall forecaster** (`weather/forecast_model.py`) — a LightGBM model
  predicting rainfall (mm) for a place + future date, using lat/lon,
  seasonality, recent-rainfall lag/rolling features, and per-cell
  climatology. `GET /api/forecast?place=Mumbai&date=2026-09-01` or ask the
  chatbot "will it rain in Kerala next week?".
- **Satellite nowcaster** (`weather/nowcast_model.py`) — a small CNN
  predicting tomorrow's mean rainfall per macro-region of India (north,
  south, east, west, central, northeast) from today's GIBS satellite image.
  `GET /api/nowcast` or ask the chatbot "will it rain tomorrow?".

Train (or retrain) either:

```bash
.venv/bin/python scripts/train_forecast_model.py    # ~1 min
.venv/bin/python scripts/train_nowcast_model.py    # first run decodes ~2,500 PNGs (a few min); cached after
```

Both print backtested accuracy (MAE/RMSE per horizon or region, plus a skill
score against a baseline — climatology for the rainfall forecaster,
persistence + climatology for the nowcaster) and write their metrics to
`models/*_metrics.json`. Backtesting, not "check it against 2026", is the
real accuracy signal here: IMD's gridded rainfall product lags real time by
roughly a year, so there's no actual 2026 rainfall yet to score a forecast
against. Splits: rainfall forecaster trains on 2018–2022, validates on 2023,
tests on 2024–2025; nowcaster trains on 2019–2023, validates on 2024, tests
on 2025 (no satellite imagery before 2019).

Because of that same IMD lag, `predict_rainfall()` can't honestly build
lag/rolling features for "today" — there's no real recent rainfall to build
them from. It anchors to the *last available* rainfall data instead: dates
within ~2 weeks of that anchor get a genuine ML forecast
(`method: ml_short_range`); anything further out falls back to the
historical average for that place and time of year
(`method: climatology_outlook`) rather than guessing. `scripts/generate_forecast_demo.py`
generates a demo forecast across every gazetteer place using both regimes,
written to `data/forecast/forecast_demo.csv`.

The nowcaster doesn't have this problem at inference time — GIBS imagery is
near-real-time, so re-running `scripts/download_satellite_images.py`
(now open-ended: 2019-01-01 through yesterday, not a fixed year range) keeps
it able to produce a genuinely live "tomorrow" prediction. It's still not
scoreable against real rainfall until IMD's data catches up, same as the
forecaster.

## Batch-converting rainfall data

```bash
.venv/bin/python scripts/convert_all.py
```

Converts every `.nc` file in `data/nc/` to `data/rainfall/rainfall_<year>.csv`.

## Satellite imagery

See [docs/satellite_images.md](docs/satellite_images.md) for the GIBS data
source, folder layout, and how to run/resume/verify the download.
