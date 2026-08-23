# Commands

Every way to run this project, in the order you'd typically need them. For
what each piece actually does, see [README.md](README.md); this file is
just the command reference.

## Fastest path: one command

```bash
./bootstrap.sh        # Linux / Mac
.\bootstrap.ps1        # Windows (PowerShell)
```

Installs dependencies, detects your hardware and configures accordingly,
pulls the latest models/analysis and fetches any new satellite imagery
(with a live progress bar), then starts the app. Safe to re-run any time.
It's a thin wrapper around the steps below -- `uv sync` itself deliberately
can't do this (see "Setup" below for why); these are normal scripts that
just run the steps in order.

If Windows blocks the script with an execution-policy error, run it once
for just that session (doesn't change your system-wide policy):
```powershell
powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1
```

## Setup (once, on a new machine)

```bash
uv sync                    # base install -- map, chatbot, rainfall forecaster. Works on any hardware.
uv run nimbus setup        # detect this machine's hardware, decide whether torch/swap are needed
```

`uv sync` is a dependency resolver, not a provisioning tool -- it can't run
`sudo` commands or start multi-GB downloads on its own (and shouldn't: a
routine "install my dependencies" step silently doing either would be a bad
surprise, and would break in CI/sandboxed environments). `nimbus setup` is
the separate step that actually does that part.

`uv sync` alone deliberately skips the satellite nowcaster's `torch`
dependency — it has no 32-bit build and a documented history of crashing
with "Illegal instruction" on CPUs without AVX2. `nimbus setup` checks for
AVX2 and only installs it if safe:

```bash
uv sync --extra torch      # manual equivalent, only on a 64-bit CPU with AVX2
```

Optional, for the chatbot:

```bash
ollama pull llama3.2:3b    # local model for the chatbot's offline fallback tier
ollama serve               # if it isn't already running as a service
export ANTHROPIC_API_KEY=sk-ant-...   # optional -- enables the faster cloud tier (Claude Haiku)
```

## The `nimbus` CLI

Invoke as `uv run nimbus <command>`, or `.venv/bin/python cli.py <command>`
directly.

### `nimbus setup`

Detects CPU architecture/cores/clock speed, AVX2 support, RAM, and swap;
recommends or installs the satellite nowcaster accordingly; offers to
configure a swapfile on low-RAM machines (needs `sudo`, always asks first).

```bash
uv run nimbus setup
uv run nimbus setup --yes    # skip prompts for the safe parts (never auto-runs the sudo swap step)
```

### `nimbus download`

Downloads NASA GIBS satellite imagery into `data/satellite/`. Resumable —
safe to re-run; only fetches the gap since the last run (2019-01-01 through
yesterday, rolling). Shows a live progress bar (via `tqdm`) with a running
count of new/failed downloads.

```bash
uv run nimbus download
```

### `nimbus params`

Downloads extra weather-prediction parameters neither model currently uses:
**elevation** for every grid cell (terrain shapes India's rainfall a lot --
the Western Ghats/Himalayas force monsoon winds upward, which is exactly
why coastal cells show ~0m and cells 25km inland jump to 600m+) and three
large-scale climate indices known to modulate the Indian monsoon -- **ENSO**
(Oceanic Niño Index), **IOD** (Dipole Mode Index), and **MJO** (Madden-Julian
Oscillation). None of these three need an account or API key. Writes to
`data/parameters/`.

```bash
uv run nimbus params
```

**Not included** (needs your own account -- see below): ERA5 reanalysis
(temperature/humidity/wind/pressure), the single most valuable addition
since it's real atmospheric data rather than statistics, but nothing can
create a Copernicus account for you.

```bash
# one-time: register free at https://cds.climate.copernicus.eu/, put your
# API key in ~/.cdsapirc (the script tells you the exact format if it's missing)
.venv/bin/python scripts/download_era5.py --year 2023
```

### `nimbus train`

Trains the rainfall forecaster (LightGBM) and/or the satellite nowcaster
(CNN), printing backtested MAE/RMSE and skill vs. baseline for each.

```bash
uv run nimbus train                        # both models
uv run nimbus train --model rainfall       # LightGBM only -- no torch needed, ~1 min
uv run nimbus train --model satellite      # CNN only -- needs torch, a few minutes
```

Don't run this on ~1GHz-class hardware (Raspberry Pi, an old netbook) —
train on a stronger machine and copy the `models/` folder over instead
(a few hundred KB, not gigabytes).

### `nimbus forecast`

Rainfall in mm for a place and date — observed if the date's in the
2018–2025 historical record, a real short-range ML forecast within ~2 weeks
past that, or an explicitly-labeled climatology estimate beyond that.

```bash
uv run nimbus forecast --place Mumbai --date 2026-09-01
uv run nimbus forecast --place Kerala --date 2023-08-15   # historical -- returns observed data
```

### `nimbus analyse`

Real computed rainfall trend/anomaly statistics — build once (reads the
full ~14.5M-row history; do this on capable hardware, not ~1GHz-class),
then query cheaply anywhere, including on hardware where it was never built
(copy `models/analysis.db` over, same pattern as the trained models).

```bash
uv run nimbus analyse --build                           # precompute into models/analysis.db (~12s)
uv run nimbus analyse --place Kerala --year 2023         # month-by-month totals + anomaly vs. its own history
uv run nimbus analyse --place Kerala --trend 2018 2025   # year-over-year trend across a range
```

### `nimbus image`

Prints the file path of a downloaded satellite image and opens it in your
default image viewer (a terminal can't render the image itself).

```bash
uv run nimbus image --date 2023-08-15
```

### `nimbus nowcast`

The satellite CNN's prediction for tomorrow's mean rainfall per macro-region
of India, from the most recent downloaded image. Needs torch (`nimbus setup`
installs it if your CPU supports AVX2).

```bash
uv run nimbus nowcast
```

### `nimbus chat`

Talk to the AI assistant right in the terminal — the same three-tier
pipeline (fast local parser → cloud → local Ollama fallback) the web chat
page uses.

```bash
uv run nimbus chat
```

Only the first message in a session gets the instant fast-path treatment;
follow-ups always go through an LLM (cloud, if `ANTHROPIC_API_KEY` is set —
a few seconds; local Ollama otherwise — tens of seconds on CPU-only
hardware, since it's genuine LLM inference).

### `nimbus sync`

Pulls the latest committed `models/`/`data/forecast/` state, fetches any new
satellite imagery (progress bar), logs today's nowcast prediction, validates
any previously-logged predictions against real data now that it's available
(flagging drift from the original training backtest), refreshes the
analysis database, then commits and pushes the result back to the repo.

```bash
uv run nimbus sync              # commits and pushes automatically
uv run nimbus sync --no-push    # commit locally, skip the push
```

Satellite images themselves are never committed — `data/satellite/` stays
gitignored (multi-GB of regenerable PNGs don't belong in git); only the
small derived state goes through git.

### `nimbus serve`

Launches the Flask app — map at `/`, chatbot at `/chat` — on
`http://127.0.0.1:5050`.

```bash
uv run nimbus serve
```

Equivalent without the CLI: `.venv/bin/python app.py`.

## Standalone scripts

Lower-level pieces the CLI doesn't wrap, or that the CLI commands above call
into directly. Run with `.venv/bin/python scripts/<name>.py`.

| Script | What it does |
|---|---|
| `convert_all.py` | Batch-converts every `.nc` file in `data/nc/` to `data/rainfall/rainfall_<year>.csv`. |
| `extract.py` | One-off extractor: `RF25_ind2023_rfp25.nc` → `rainfall_2023.csv` + a compact `rainfall_2023.json` for the map's client-side rendering. |
| `download_satellite_images.py` | What `nimbus download` calls — same resumable GIBS downloader. |
| `verify_satellite_images.py` | Checks every expected satellite image date for a present, non-corrupt PNG; writes a completeness report and a list of gaps to retry. |
| `retry_failed_dates.py` | Re-attempts only the dates `verify_satellite_images.py` flagged as missing/corrupt. |
| `train_forecast_model.py` | What `nimbus train --model rainfall` calls — same LightGBM training. |
| `train_nowcast_model.py` | What `nimbus train --model satellite` calls — same CNN training. |
| `generate_forecast_demo.py` | Generates a demo rainfall forecast for every gazetteer place across the rest of the year, written to `data/forecast/forecast_demo.csv`. |
| `generate_nowcast_report.py` | Validates the satellite nowcaster against real 2025 rainfall (`data/forecast/nowcast_validation_2025.csv`) and generates its 2026 predictions from satellite imagery alone (`data/forecast/nowcast_2026.csv`). |
| `download_elevation.py` | What `nimbus params` calls for elevation — no account needed. |
| `download_climate_indices.py` | What `nimbus params` calls for ENSO/IOD/MJO — no account needed. |
| `download_era5.py --year Y` | ERA5 atmospheric data — needs your own free Copernicus account first (`nimbus params --help` has the setup steps). Not called by `nimbus params`, since it can't succeed without that account. |

## Environment variables

| Variable | Default | Affects |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2:3b` | Which local model the chatbot's offline fallback tier uses. Must support tool calling. |
| `OLLAMA_HOST` | `http://localhost:11434` | Where Ollama is running — point this at a stronger machine on your network if this one is slow. |
| `ANTHROPIC_API_KEY` | unset | Enables the chatbot's cloud tier (Claude Haiku) for questions the fast local parser can't handle. Optional — without it, everything still works, just slower for complex questions. |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5` | Overrides the cloud-tier model. |
