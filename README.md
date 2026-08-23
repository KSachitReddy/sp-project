# Nimbus

AI predictive weather analysis for India. Tools for exploring and *predicting* India's daily rainfall: a Flask app that
renders historical rainfall (IMD RF25 gridded data, 2018–2025) on an
interactive map, a chatbot answering plain-English rainfall questions, real
computed trend/anomaly analysis, and two predictive models — a short-range
rainfall forecaster and a satellite-imagery-based nowcaster — all reachable
from a single CLI, `nimbus`.

## Structure

```
app.py                   Flask entrypoint — serves the map page, chat page, and their APIs
cli.py                   The `nimbus` CLI — see "Command line" below
bootstrap.sh             One-command setup + launch (Linux/Mac)
bootstrap.ps1            Same, for Windows (PowerShell)
weather/
  nc_convert.py           Shared IMD RF25 NetCDF -> CSV/JSON conversion logic
  chatbot.py               Place gazetteer + rainfall/forecast/nowcast lookup + 3-tier chat pipeline
  fast_intent.py            Deterministic no-LLM parser for common chatbot questions
  features.py               Feature engineering for the rainfall forecasting model
  forecast_model.py         LightGBM rainfall forecaster: train/save/load/predict
  satellite_features.py     Image/label building for the satellite nowcasting model
  nowcast_model.py          CNN satellite nowcaster: train/save/load/predict
  analysis.py                Real computed trend/anomaly stats (SQLite) + live nowcast validation log
  hardware.py                CPU/RAM/AVX2/swap detection, backing `nimbus setup`
scripts/
  convert_all.py           Batch-convert every data/nc/*.nc into data/rainfall/*.csv
  extract.py                One-off extractor for the 2023 CSV + JSON
  download_satellite_images.py   Download daily GIBS imagery, 2019-01-01 through yesterday (progress bar)
  verify_satellite_images.py     Completeness/integrity check over the download
  retry_failed_dates.py           Re-attempt only the dates verify flagged
  train_forecast_model.py         Train the rainfall forecaster, print backtested accuracy
  train_nowcast_model.py          Train the satellite nowcaster, print backtested accuracy
  generate_forecast_demo.py        Generate a demo rainfall forecast for every gazetteer place
  generate_nowcast_report.py       Validate the nowcaster against real 2025 rainfall + generate 2026 predictions
web/
  weather_map.html          The map page app.py serves
  chat.html                 The rainfall chatbot page app.py serves
data/
  nc/                       Source IMD RF25 NetCDF files (tracked in git)
  rainfall/                 Generated CSV/JSON rainfall exports (tracked in git)
  places/                   Offline India states/UTs/cities gazetteer used by the chatbot
  satellite/                Downloaded GIBS PNGs + logs (gitignored — regenerable, multi-GB)
  forecast/                 Demo forecast + nowcast validation/prediction CSVs (tracked in git)
models/                    Trained models, analysis.db, and metrics JSON (tracked in git — small,
                           `nimbus sync` keeps this current instead of everyone regenerating locally)
docs/
  satellite_images.md       Satellite dataset details: source, format, how to run/resume
commands.md                Full CLI reference — every command, every option
```

## Setup

Fastest path -- one command, handles everything below automatically:

```bash
./bootstrap.sh         # Linux / Mac
.\bootstrap.ps1         # Windows (PowerShell -- see commands.md if it's blocked by execution policy)
```

Manual/pip equivalent, if you'd rather not use `uv` or the wrapper script:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` intentionally skips torch (see "Hardware requirements"
below) -- the map, chatbot, and rainfall forecaster all run without it. Only
the satellite nowcaster needs it.

## Command line

Everything is reachable through one CLI, `nimbus` (`uv run nimbus <command>`,
or `.venv/bin/python cli.py <command>` with the pip install). Full reference
with every option and example: **[commands.md](commands.md)**.

| Command | What it does |
|---|---|
| `setup` | Detects this machine's hardware and configures accordingly -- installs the satellite nowcaster's torch dependency only if safe, offers swap/page-file guidance on low-RAM machines. Run once after `uv sync`. |
| `download` | Downloads NASA GIBS satellite imagery (resumable, live progress bar). |
| `train` | Trains the rainfall forecaster and/or satellite nowcaster, printing backtested accuracy. |
| `forecast --place X --date Y` | Rainfall in mm for a place/date -- observed, a real short-range forecast, or an honest climatology estimate, depending how far out. |
| `analyse` | Real computed trend/anomaly statistics (SQLite-backed) -- `--build` to precompute, then `--place X --year Y` or `--place X --trend START END`. |
| `image --date Y` | Views a downloaded satellite image (prints the path, opens it in your default viewer). |
| `nowcast` | Runs the satellite CNN directly -- tomorrow's predicted rainfall per macro-region. |
| `chat` | Talk to the AI assistant right in the terminal -- same pipeline as the web chat page. |
| `sync` | Pulls the latest committed models/analysis, fetches new satellite imagery, validates past nowcast predictions against real data, commits and pushes the result. |
| `serve` | Launches the Flask app (map + chatbot) at `http://127.0.0.1:5050`. |

## Hardware requirements

Everything except the satellite nowcaster is designed to run on modest,
CPU-only hardware, verified across a real range -- from a dual-core Intel
Ivy Bridge laptop (3rd-gen Core i3, 8 GB RAM) down to ~1GHz-class embedded
hardware (Raspberry Pi, an old netbook). `nimbus setup` detects which tier
your machine falls into and configures itself accordingly:

- **CPU**: any x86_64 CPU, or ARM (Raspberry Pi-class). No GPU is used
  anywhere in this project. On ~1GHz-class hardware, train models on a
  stronger machine and copy the small `models/` folder over instead --
  training/CNN inference there is impractical, but serving the already-
  trained models is cheap.
- **RAM**: 8 GB is comfortable for the map, chatbot, and rainfall forecaster.
  Budget roughly 3-4 GB for `ollama` running the recommended `llama3.2:3b`
  model if you want the offline chatbot fallback; the rest (Flask, pandas,
  the LightGBM model) fits in well under 1 GB. On a low-RAM machine, `nimbus
  setup` offers to configure swap (Linux, with your confirmation -- it needs
  `sudo`) or gives page-file guidance (Windows, manual -- the mechanism is
  different enough there that this project won't auto-configure it).
- **Disk**: ~56 MB for the tracked models/analysis database, plus the
  rainfall CSVs/NetCDFs; add ~7 GB if you download the full satellite
  imagery archive (optional, only needed for the nowcaster).
- **Satellite nowcaster (optional, skip it on older/ARM hardware)**: needs
  PyTorch, which has no 32-bit build and whose CPU wheel has a documented
  history of crashing with "Illegal instruction" on CPUs without AVX2 --
  that includes Ivy Bridge and any Intel Core CPU older than 4th-gen
  (Haswell). `nimbus setup` checks for AVX2 and only installs it if safe;
  the rest of the app works fine without it and `/api/nowcast` (or `nimbus
  nowcast`) degrades to a clear error instead of crashing anything.
- **Windows**: fully supported -- `bootstrap.ps1`, hardware detection, and
  every CLI command work the same way. Not yet verified on real Windows
  hardware (built and reviewed carefully, but this project has mostly been
  run on Linux); if something's off, that's the likeliest place.

## Running the app

```bash
uv run nimbus serve
```

Serves the rainfall map at `http://localhost:5050`, reading years from
`data/nc/` and generated CSVs from `data/rainfall/`. Uploading a new `.nc`
file through the UI validates it, saves it into `data/nc/`, and writes its
CSV into `data/rainfall/`.

## Rainfall chatbot

`http://localhost:5050/chat` (or `uv run nimbus chat` for the same thing
right in your terminal) answers plain-English questions like "how much did
it rain in Kerala in August 2022?" -- it resolves the place against an
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

Only the first message in a conversation gets the instant fast-path
treatment -- follow-ups need real conversation context, so they always go
through an LLM.

## Forecasting

Two predictive models, both trained on CPU (no GPU required):

- **Rainfall forecaster** (`weather/forecast_model.py`) — a LightGBM model
  predicting rainfall (mm) for a place + future date, using lat/lon,
  seasonality, recent-rainfall lag/rolling features, and per-cell
  climatology. `uv run nimbus forecast --place Mumbai --date 2026-09-01`,
  `GET /api/forecast?place=Mumbai&date=2026-09-01`, or ask the chatbot
  "will it rain in Kerala next week?".
- **Satellite nowcaster** (`weather/nowcast_model.py`) — a small CNN
  predicting tomorrow's mean rainfall per macro-region of India (north,
  south, east, west, central, northeast) from today's GIBS satellite image.
  `uv run nimbus nowcast`, `GET /api/nowcast`, or ask the chatbot "will it
  rain tomorrow?".

Train (or retrain) either:

```bash
uv run nimbus train --model rainfall     # ~1 min
uv run nimbus train --model satellite    # first run decodes ~2,500 PNGs (a few min); cached after
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
near-real-time, so re-running `uv run nimbus download` keeps it able to
produce a genuinely live "tomorrow" prediction. `scripts/generate_nowcast_report.py`
validates it against real 2025 rainfall and generates its 2026 predictions
from satellite imagery alone; `nimbus sync` extends this into an ongoing
check, logging each day's prediction and flagging drift once real data
catches up to it (see "Keeping everything in sync" below).

## Rainfall analysis

Real computed trend/anomaly statistics, not an LLM paraphrasing raw numbers
-- monthly totals, anomaly vs. each place's own long-term average, and
year-over-year trends, precomputed into SQLite (`models/analysis.db`) and
queried in milliseconds:

```bash
uv run nimbus analyse --build                           # precompute (reads the full history, ~12s)
uv run nimbus analyse --place Kerala --year 2023         # month-by-month + anomaly vs. its own history
uv run nimbus analyse --place Kerala --trend 2018 2025   # year-over-year trend
```

Also available via `GET /api/analysis` and as a chatbot tool
(`get_rainfall_analysis`) for "analyse the trend in Kerala"-style questions.

## Keeping everything in sync

```bash
uv run nimbus sync
```

Pulls the latest committed `models/`/`data/forecast/` state, fetches any new
satellite imagery, logs today's nowcast prediction, checks any
previously-logged predictions against real rainfall now that it's available
(flagging drift if live accuracy diverges from the original training
backtest by more than 25%), refreshes the analysis database, then commits
and pushes the result back to the repo. Satellite images themselves are
never committed -- `data/satellite/` stays gitignored; only the small
derived state goes through git.

## Batch-converting rainfall data

```bash
uv run nimbus download                    # satellite imagery
.venv/bin/python scripts/convert_all.py   # every .nc file in data/nc/ -> data/rainfall/rainfall_<year>.csv
```

## Satellite imagery

See [docs/satellite_images.md](docs/satellite_images.md) for the GIBS data
source, folder layout, and how to run/resume/verify the download.
