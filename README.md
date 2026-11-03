# Nimbus

AI predictive weather analysis for India. Tools for exploring and *predicting* India's daily rainfall: a Flask app that
renders historical rainfall (IMD RF25 gridded data, 2018–2025) on an
interactive map, a chatbot answering plain-English rainfall questions, real
computed trend/anomaly analysis, and two predictive models — a short-range
rainfall forecaster and a satellite-imagery-based nowcaster — all reachable
from a single CLI, `nimbus`.

## Quick start

From the repo root, three commands. Needs **Python ≥ 3.12**; uv fetches a matching
interpreter by itself (`.python-version` pins 3.13, the version everything was tested on).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # 1. install uv  (open a new terminal afterwards)
uv sync --frozen --extra torch                           # 2. install the exact locked dependencies
uv run nimbus forecast --place Mumbai --date 2026-01-05  # 3. first forecast
```

Expected output of step 3:

```
Mumbai, 2026-01-05: 0.87 mm (ML forecast)
```

- **Windows:** install uv with
  `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`.
- **CPU without AVX2** (e.g. Intel 3rd-gen Core or older): drop `--extra torch`.
  Everything except the satellite nowcaster still works.
- **Satellite-based answers need imagery**, which isn't in the repo. Until you run
  `uv run nimbus download` (about 9.6 GB), dates past the short-range ML window fall back
  to the historical average. `nimbus nowcast`, `nimbus image` and
  `nimbus thermal-estimate` say no image is on disk.
- Then `uv run nimbus serve` starts the map and chat at `http://127.0.0.1:5050`. It
  preloads the models for about 10 s, so the first request is fast.

**Check everything in one go:** `uv run python scripts/quickstart.py` runs steps 2–3
plus the tests and prints PASS/FAIL per stage. It never commits, pushes or downloads
imagery.

## Running the tests

```bash
uv run pytest -q
```

The suite in `tests/` covers:
- feature leakage, split boundaries and leap-day handling;
- prediction method labels;
- API error paths;
- safe `nimbus sync` defaults;
- OS hardware detection;
- `requirements.txt` vs `uv.lock` drift.

It runs in about 6 s and needs no imagery or network.

## Project layout

```
app.py, cli.py     Flask app (map + chat + JSON API) and the `nimbus` CLI
weather/           forecaster, nowcaster, thermal estimate, features, chatbot, hardware detection
scripts/           training, evaluation, download and maintenance scripts (listed below)
web/               the map and chat pages
data/              rainfall CSV/NetCDF, places, parameters, forecasts; satellite/ fills in via `nimbus download`
models/            trained models, metrics, forecast_eval_report.json, benchmark.json
tests/, docs/      pytest suite; changelog, CNN seed stability, satellite notes, screenshots
```

### Scripts

| Script | Purpose |
|---|---|
| `quickstart.py` | Check a fresh setup: uv, dependency sync, tests, first forecast (PASS/FAIL per stage) |
| `train_forecast_model.py` | Train the LightGBM rainfall forecaster (~1 min) and print backtested accuracy |
| `train_nowcast_model.py` | Train the satellite CNN nowcaster (needs imagery and torch) |
| `calibrate_thermal_rainfall.py` | Train the thermal same-day rainfall estimate |
| `evaluate_forecast.py` | Extended forecaster evaluation -> `models/forecast_eval_report.json` |
| `benchmark.py` | Latency and memory benchmark -> `models/benchmark.json` |
| `generate_forecast_demo.py` | Demo forecast for every gazetteer place -> `data/forecast/forecast_demo.csv` |
| `generate_nowcast_report.py` | Validate the nowcaster on 2025 and generate 2026 predictions |
| `download_satellite_images.py` | Download daily GIBS true-colour and thermal imagery (what `nimbus download` runs) |
| `extract_thermal_features.py` | Thermal imagery -> `data/satellite/thermal_features.csv` |
| `verify_satellite_images.py` | Completeness/integrity check of the downloaded imagery |
| `retry_failed_dates.py` | Re-download only the dates the verifier flagged |
| `convert_all.py` | Convert every `data/nc/*.nc` into `data/rainfall/*.csv` |
| `download_elevation.py` | Elevation per grid cell (not used by the models yet) |
| `download_climate_indices.py` | ENSO / IOD / MJO indices (not used by the models yet) |
| `download_era5.py` | ERA5 reanalysis (needs your own CDS account; not used by the models yet) |

## Other ways to install

- **One-command setup with imagery.** `./bootstrap.sh` (Linux/macOS) or
  `.\bootstrap.ps1` (Windows) runs four steps, stopping at the first failure:
  `uv sync`, then `nimbus setup --yes` (hardware check, installs torch if safe), then
  `nimbus sync`, then `nimbus serve`. `nimbus sync` does a fast-forward pull and
  downloads about 10 GB of imagery. It never commits or pushes; that needs an explicit
  `nimbus sync --push`, which is for maintainers.
- **pip.** `requirements.txt` is generated from `uv.lock`. It includes pytest but not
  torch (see the note at the bottom of the file).

  ```bash
  python3 -m venv .venv            # Debian/Ubuntu: needs `apt install python3-venv` first
  .venv/bin/pip install -r requirements.txt
  .venv/bin/python cli.py forecast --place Mumbai --date 2026-01-05
  ```

## Command line

Everything is reachable through one CLI, `nimbus` (`uv run nimbus <command>`,
or `.venv/bin/python cli.py <command>` with the pip install). Full reference
with every option and example: **[commands.md](commands.md)**.

| Command | What it does |
|---|---|
| `setup` | Detects this machine's hardware and configures accordingly -- installs the satellite nowcaster's torch dependency only if safe, offers swap/page-file guidance on low-RAM machines. Run once after `uv sync`. |
| `download` | Downloads NASA GIBS satellite imagery (resumable, live progress bar). |
| `params` | Downloads elevation + ENSO/IOD/MJO climate indices (no account needed) -- see "Weather-prediction parameters" below. |
| `train` | Trains the rainfall forecaster and/or satellite nowcaster, printing backtested accuracy. |
| `forecast --place X [--date Y]` | Rainfall in mm for a place/date (default: today) -- observed, ML forecast, satellite nowcast, thermal estimate, or historical average; every answer says which. |
| `analyse` | Real computed trend/anomaly statistics (SQLite-backed) -- `--build` to precompute, then `--place X --year Y` or `--place X --trend START END`. |
| `image --date Y` | Views a downloaded satellite image (prints the path, opens it in your default viewer). |
| `nowcast` | Runs the satellite CNN directly -- tomorrow's predicted rainfall per macro-region. |
| `thermal-estimate --date Y` | Same-day rainfall estimate per macro-region from that day's thermal satellite image (needs `download` first). |
| `chat` | Talk to the AI assistant right in the terminal -- same pipeline as the web chat page. |
| `sync` | Fast-forward pull + download new satellite imagery. Never commits or pushes. `--push` (maintainers) also refreshes derived state (thermal features, nowcast log, analysis DB), commits and pushes. |
| `serve` | Preloads the models (~10 s), then launches the Flask app (map + chatbot) at `http://127.0.0.1:5050`. |

## Hardware requirements

Everything except the satellite nowcaster is designed to run on modest,
CPU-only hardware. The intended targets range from a dual-core Intel Ivy Bridge
laptop (3rd-gen Core i3, 8 GB RAM) down to ~1GHz-class embedded hardware
(Raspberry Pi, an old netbook). These are intended targets, **not verified in
final QA**, which ran on a 4-core Intel i5-6500 (AVX2, 7.7 GB RAM) under Linux.
`nimbus setup` detects which tier your machine falls into and configures itself
accordingly:

- **CPU**: any x86_64 CPU, or ARM (Raspberry Pi-class). No GPU is used
  anywhere in this project. On ~1GHz-class hardware, train models on a
  stronger machine and copy the small `models/` folder over instead --
  training/CNN inference there is impractical, but serving the already-
  trained models is cheap.
- **RAM**: 8 GB is comfortable for the map, chatbot, and rainfall forecaster.
  Measured peak RSS:
  - `nimbus serve` and `nimbus forecast`: ~1.9 GB, mostly the cached 14.5M-row
    rainfall history.
  - Training the forecaster and `scripts/evaluate_forecast.py`: ~3.7 GB.
  - Training the CNN: ~2.4 GB.
  - Budget another 3-4 GB for `ollama` running the recommended `llama3.2:3b`
    model if you want the offline chatbot fallback.
  
  On a 4 GB machine, serving fits only tightly and training needs swap. On a
  low-RAM machine, `nimbus
  setup` offers to configure swap (Linux, with your confirmation -- it needs
  `sudo`) or gives page-file guidance (Windows, manual -- the mechanism is
  different enough there that this project won't auto-configure it).
- **Disk**: ~56 MB for the tracked models/analysis database, plus the
  rainfall CSVs/NetCDFs; add ~9.6 GB if you download the full satellite
  imagery archive (7.1 GB true-colour + 2.3 GB thermal; optional, only needed
  for the nowcaster and thermal estimate).
- **Satellite nowcaster (optional, skip it on older/ARM hardware)**: needs
  PyTorch, which has no 32-bit build and whose CPU wheel has a documented
  history of crashing with "Illegal instruction" on CPUs without AVX2 --
  that includes Ivy Bridge and any Intel Core CPU older than 4th-gen
  (Haswell). `nimbus setup` checks for AVX2 and only installs it if safe;
  the rest of the app works fine without it and `/api/nowcast` (or `nimbus
  nowcast`) degrades to a clear error instead of crashing anything.
- **Windows**: **not verified**. Windows code paths exist (`bootstrap.ps1`, PowerShell
  hardware detection, `os.startfile`) and were reviewed statically, but the project
  has never been run on Windows. See "Limitations & honest results" for the known
  gaps.

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

## Limitations & honest results

All numbers below are measured on held-out data. Rainfall forecaster figures come from
`models/forecast_eval_report.json` (`uv run python scripts/evaluate_forecast.py`,
which first re-checks `forecast_model_metrics.json` exactly). Nowcaster seed figures
are in `docs/cnn_seed_stability.md`.

**Rainfall forecaster (test 2024–2025)**

- **Skill is small and depends on the metric.** Skill vs climatology on squared
  error (the number in `forecast_model_metrics.json`) is +0.034 / +0.021 / +0.015 /
  +0.010 at 1 / 3 / 7 / 14 days. On absolute error it is **negative** at every
  horizon: −0.040 / −0.047 / −0.050 / −0.050. Bootstrap 95% CIs of the squared-error
  skill exclude zero (h1 [0.026, 0.043], h14 [0.003, 0.018]). They resample whole
  dates but ignore day-to-day autocorrelation, so they are somewhat too narrow.
- **Mostly climatology.** Early stopping kept 28 trees, and `climatology_mm`
  carries 71% of feature gain. The model beats persistence comfortably (squared-error
  skill +0.25 at 1 day, +0.48 at 14 days), but persistence is a weak baseline for
  daily rain.
- **Heavy rain is badly under-predicted.** On days with ≥50 mm observed (mean 80.2 mm),
  the 1-day forecast averages 11.6 mm. The largest prediction is 89 mm; the largest
  observation is 389 mm.
- **Drizzle bias on dry days.** 70% of target cell-days are exactly 0 mm. The model
  never predicts below 0.1 mm and averages 2.1 mm on dry days.
- **Rain/no-rain detection is worse than climatology on CSI.** At ≥1 mm the model's
  CSI is 0.320 vs climatology's 0.390. At ≥10 mm it is 0.181 vs 0.205, with POD 0.246
  vs 0.328. Ranking is slightly better (ROC-AUC 0.849 vs 0.831 at ≥1 mm).
- **Negative in winter.** For December–February, 1-day squared-error skill is
  −0.025, with a +0.76 mm bias.
- Rainfall for 2023-10-12 is missing for 79 Gujarat cells. Lag features are
  positional, so those cells' lags are one day off for the rest of 2023 (the
  validation year).

**Satellite nowcaster (CNN, test 2025)**

- **West India loses to persistence on every seed.** Squared-error skill vs persistence
  is −0.385 for the committed seed, and −0.165 and −0.280 for seeds 1 and 2.
- **Northeast correction.** In the committed metrics, northeast skill vs persistence
  is +0.140 on **test**. The −0.077 figure is the 2024 **validation** split, and
  validation northeast is negative on 2 of 3 seeds.
- **Results move with the seed.** East ranges 0.016–0.161 vs persistence and central
  0.024–0.173. The committed metrics are a single seed (0).
- **Climatology is beaten everywhere.** Squared-error skill vs climatology is
  0.22–0.53 in every region, on every seed.
- **Small and unregularised.** It has 260,550 parameters (94% in one dense layer) and
  trains on 1,826 images, with no dropout, batch norm or augmentation. At the best
  epoch, train MSE (log space) is 0.146 vs validation 0.231.

**Thermal estimate.** Its "climatology" baseline is flat: the region's mean rainfall
over the split being evaluated, not a day-of-year climatology. Its skill numbers
therefore aren't comparable to the other two models'.

**Demo forecast.** In `data/forecast/forecast_demo.csv`, 78.1% of rows (5,850 of
7,488) are `climatology_outlook`. ML forecasts cover only 2026-01-01..14. The CSV
was generated before forecasts were routed to the satellite models, so the live
app now returns `satellite_nowcast` for some dates the CSV labels climatology.

**Resources and platforms**

- Peak RAM: the web server and `nimbus forecast` reach about 1.9 GB once forecast
  data is cached. Forecaster retraining and `evaluate_forecast.py` peak at about
  3.7 GB, and CNN training at 2.4 GB.
- Latency: the first forecast request takes about 11 s while 14.5M rows load; after
  that, the median is about 90 ms (see `models/benchmark.json`).
- Without torch, the app degrades cleanly (verified in a torch-less environment):
  `/api/nowcast` returns 503 and forecasts fall back to thermal or climatology.
- **Not verified:** running on Windows, on macOS, and on a real CPU without AVX2.
- **Torch detection on Windows and Apple Silicon is untested on real hardware.**
  `nimbus setup` checks AVX2 on Windows with `IsProcessorFeaturePresent`, and treats
  Apple Silicon as torch-capable (native arm64 wheels). Both paths are unit-tested with
  simulated platforms only.
- **`bootstrap.ps1` exit-code checks are untested.** It now stops at the first failing
  native command (`$LASTEXITCODE` check after each), but it has never been executed:
  no PowerShell was available in QA.

**App behaviour to be aware of**

- `/api/nowcast` (and the map's nowcast card) needs a satellite image from the last
  7 days. Run `uv run nimbus download` before a demo.
- `POST /api/convert` writes the uploaded file to `data/nc/` and **overwrites**
  `data/rainfall/rainfall_<year>.csv` for that year without confirmation, and that
  CSV is training/test data. Only upload years you intend to replace.

## Weather-prediction parameters (not yet used by either model)

Both models above currently see only lat/lon, day-of-year, historical
rainfall, and (for the nowcaster) a satellite image -- no actual atmospheric
data, and no notion of terrain. Real forecasting skill mostly comes from
elsewhere, so this downloads the inputs a future iteration would train on:

```bash
uv run nimbus params
```

Fetches, no account needed:
- **Elevation** for every grid cell -- India's rainfall is heavily shaped by
  terrain (the Western Ghats and Himalayas force monsoon winds upward,
  dumping far more rain on the windward side); the data bears this out
  directly -- coastal cells near Mumbai sit at ~0m, cells 25km inland jump
  past 600m.
- **ENSO, IOD, and MJO** -- three large-scale climate oscillations with a
  well-documented relationship to Indian monsoon strength and timing.

**ERA5 reanalysis** (temperature, humidity, wind, pressure -- the single
most valuable addition, since it's real atmospheric physics rather than
statistics) needs your own free account -- nothing can create one for you:

```bash
# one-time: register at https://cds.climate.copernicus.eu/, put your API
# key in ~/.cdsapirc (the script explains the exact format if it's missing)
.venv/bin/python scripts/download_era5.py --year 2023
```

All four are written to `data/parameters/` (elevation/ENSO/IOD/MJO tracked
in git -- small; ERA5 gitignored -- can be large per year).

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
uv run nimbus sync          # fast-forward pull + download new imagery; never commits or pushes
uv run nimbus sync --push   # maintainers only: also refresh derived state, commit and push
```

The default is safe to run any time: it runs `git pull --ff-only` and fetches any new
satellite imagery, and writes no tracked files.

`--push` additionally:
1. extracts thermal features from the new imagery;
2. logs today's nowcast prediction;
3. checks earlier predictions against real rainfall now that it's available, flagging
   drift if live accuracy diverges from the training backtest by more than 25%;
4. refreshes the analysis database;
5. commits `models/`, `data/forecast/` and the thermal features, and pushes.

Satellite images themselves are never committed.

## Batch-converting rainfall data

```bash
uv run nimbus download                    # satellite imagery
.venv/bin/python scripts/convert_all.py   # every .nc file in data/nc/ -> data/rainfall/rainfall_<year>.csv
```

## Satellite imagery

See [docs/satellite_images.md](docs/satellite_images.md) for the GIBS data
source, folder layout, and how to run/resume/verify the download.
