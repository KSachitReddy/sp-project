# Final-submission changelog

Running log of the final 2-hour hardening pass (branch `final-submission`).
No splits, features, or reported model metrics were changed.

## Step 1 — Commit work in progress
- Added `.claude/settings.local.json` to `.gitignore` and untracked it (local editor permissions, not project code).
- Committed the pending satellite-routing / nowcast error-bar work and `docs/screenshots/`.

## Step 2 — Repair local venv
- `uv sync --frozen --extra torch`: installed torch's missing locked deps (sympy 1.14.0, networkx, filelock, fsspec, mpmath, setuptools).
- Check: `torch.optim.Adam(...)` now constructs (previously `ModuleNotFoundError: sympy`, which crashed CNN training).

## Step 3 — Label every prediction; Flask debug opt-in
- `weather/fast_intent.py`: new `method_label()` -> `ML forecast` / `satellite nowcast, <region> avg` / `thermal estimate, <region> avg` / `historical average`; observed values now end in `(observed)`.
- `cli.py` `forecast`: uses the same labels (previously only climatology had a caveat; ML, satellite and observed values printed identically).
- `app.py`: `debug` is now off unless `NIMBUS_DEBUG=1`.
- Check: `nimbus forecast --place Mumbai --date 2026-06-15` -> `1.43 mm (satellite nowcast, west avg)`.

## Step 4 — Extended forecaster evaluation (no retraining)
- New `scripts/evaluate_forecast.py` -> `models/forecast_eval_report.json`: MSE- and MAE-skill vs climatology, skill vs persistence, bias, correlation, POD/FAR/CSI/F1/ROC-AUC at 1/2.5/10 mm (model and climatology), season and region tables, heavy-rain calibration, date-block bootstrap CIs (autocorrelation caveat recorded), feature importance.
- Refuses to write unless it first reproduces `forecast_model_metrics.json` exactly; that check passed. Committed metrics unchanged.
- Check: test h1 MSE skill 0.0341, MAE skill -0.0403, CSI >=10 mm 0.181 vs climatology 0.205. 40 s, peak 3.5 GB.

## Step 5 — Honest limitations in README; thermal comment fix
- README: new "Limitations & honest results" section covering MSE vs MAE skill, heavy-rain under-prediction, CSI below climatology at >=1 and >=10 mm, negative DJF skill, CNN west negative on all 3 seeds, NE val/test correction, seed spread, demo CSV 78.1% climatology, RAM and latency, Windows/AVX2 unverified, /api/convert overwrite, /api/nowcast needing fresh imagery.
- `scripts/calibrate_thermal_rainfall.py`: docstring and comment now say the baseline is the evaluated split's own flat regional mean (they previously claimed day-of-year and training-set). Comments only; no code or metrics change.

## Step 6 — Tests
- Added pytest 9.2.0 as a pinned dev dependency (`[dependency-groups]`, `uv.lock` updated) and `[tool.pytest.ini_options]`.
- `tests/test_features.py`: lag/rolling features use only days before the anchor; no cross-cell leakage; horizon targets strictly future; forecaster and nowcaster split years disjoint and chronological; leap-day `doy_365` folding; climatology ignores non-training years.
- `tests/test_forecast_labels.py`: real booster + climatology on a 60-day synthetic history; every result at 1/5/14/15/200 days carries the right `method` and a label; historical dates return an error, not a number; the four labels are distinct.
- `tests/test_app.py`: Flask test-client 4xx paths for forecast, analysis, satellite, year, chat and convert.
- Check: `uv run pytest -q` -> 37 passed in 5 s. Mutation check: changing lags to `shift(lag - 1)` makes 2 tests fail (then reverted).
- Note: `uv add` performs a plain sync, which uninstalled torch; re-ran `uv sync --frozen --extra torch`. `uv run` does not remove it.

## Step 7 — Benchmark
- New `scripts/benchmark.py` (Flask test client, no server) -> `models/benchmark.json`: cold start, per-endpoint median/p95, direct CNN inference latency, peak RSS, model sizes, machine info.
- Check (this machine: 4 cores, AVX2, 7.7 GB RAM): see `models/benchmark.json`. Peak RSS 1,943 MB; CNN inference median 46.8 ms; `/api/year/<y>` about 3.1 s.

## Step 8 — Nowcast cache key
- `weather/satellite_features.py`: cache key is now image size + image-date count + first/last image date + rainfall row count + last rainfall date (`_cache_key`). Any mismatch, or an old cache missing these fields, triggers a rebuild.
- `tests/test_nowcast_cache.py` (4 tests): cache is reused when nothing changes; rebuilt on a new image, on extended labels, and when the cache is in the old format.
- Check: `uv run pytest -q` -> 41 passed. The existing `data/satellite/nowcast_cache.npz` is old-format, so the next CNN training run rebuilds it once.

## Step 9 — CNN loss history + seed stability (retrain reverted)
- `weather/nowcast_model.py`: `train()` now records per-epoch `{epoch, train_mse_log, val_mse_log}` and saves `history` and `best_epoch` in the metrics JSON. Future runs only; no effect on the model.
- Retrained once (`scripts/train_nowcast_model.py`, 4 min 26 s; this also rebuilt the cache in the new key format). **test_2025 was not identical** (e.g. west −0.385 -> −0.433, east 0.077 -> 0.106). Per the rule, reverted with `git checkout -- models/nowcast_model.pt models/nowcast_model_metrics.json`. The committed model and metrics are unchanged, so the committed JSON has **no** `history` field.
- Diagnosis: the rebuilt cache is byte-identical to the old one. Two further seed-0 runs reading the cache (project venv and fresh-clone venv) reproduced the committed weights and metrics exactly. Only the cache-rebuilding run diverged; cause not isolated.
- New `docs/cnn_seed_stability.md`: seeds 0/1/2 (+ the reverted rerun), test and val skill vs persistence and climatology, the committed model's epoch curve (from the seed-0 run that reproduced it exactly), and the reproducibility note.

## Step 10 — Final verification
- `uv run pytest -q` -> 41 passed.
- `uv run nimbus forecast --place Mumbai --date 2026-06-15` -> `Mumbai, 2026-06-15: 1.43 mm (satellite nowcast, west avg)`.
- `uv run nimbus serve`: "Debug mode: off"; `/api/years` -> 200 `[2018..2025]`; `/api/forecast` carries `"method"`; `/api/nowcast` -> 400 (no image within 7 days; documented in README).
- `scripts/evaluate_forecast.py` rerun: report identical to the first run apart from runtime/RSS; reproduces `forecast_model_metrics.json`.
- Reported metrics (`forecast_model_metrics.json`, `nowcast_model_metrics.json`, `thermal_estimate_metrics.json`) and models are unchanged from the start of this pass.
- Tagged `v1.0-submission`.

## Step 11 — QA fixes (README, Windows import, error messages)
Follow-up to the QA pass on `v1.0-submission`. No retraining; `models/` and all reported metrics unchanged.
1. **README Quick start** at the top of Setup: install uv; `uv sync --frozen --extra torch`; `uv run pytest -q`; `uv run nimbus forecast --place Mumbai --date 2026-01-05`. States Python >= 3.12 and mentions `tests/`, `thermal-estimate`, `scripts/evaluate_forecast.py` and `scripts/benchmark.py`. `bootstrap.sh`/`.ps1` relabelled "Maintainer sync (not for a first run)": they run `nimbus sync` (git pull, ~10 GB download, commit, push). Added a note that fresh clones fall back to the historical average for satellite-based dates until `nimbus download`. The pip path notes the `python3-venv` prerequisite and that pytest isn't included. Added `thermal-estimate` to the command table, and `tests/`, the thermal modules and the 4 missing scripts to the structure list.
2. **`import resource` is optional** in `scripts/benchmark.py` and `scripts/evaluate_forecast.py` (POSIX-only module); `peak_rss_mb` is `None` on Windows.
3. **README claims corrected**: peak RAM ~1.9 GB (serve/forecast) and ~3.7 GB (train/evaluate), replacing "well under 1 GB"; imagery ~9.6 GB, replacing "~7 GB"; Windows now "not verified", replacing "fully supported"; Ivy Bridge / Raspberry Pi reworded to "intended targets, not verified in final QA". Limitations RAM line aligned to the same figures.
4. **`data/satellite/thermal_features.csv` tracked** (`git add -f`, 1.6 MB, 16,977 rows) so the thermal estimate can be recalibrated from a clone. Note: `nimbus sync` appends to it but only stages `models/` and `data/forecast/`, so after a sync this file shows as modified until committed.
5. **Limitations**: Windows and Apple Silicon are never offered torch (`_has_avx2()` has no Windows branch; Apple Silicon has no AVX2), and `bootstrap.ps1` doesn't check `$LASTEXITCODE`.
6. **Error messages**: `thermal-estimate --date 2026-02-30` and `/api/satellite/2026-02-30` now report an invalid date (previously "no image"). `nimbus image`, `nimbus nowcast`, `thermal-estimate` and `/api/satellite` say "run `nimbus download`" when imagery is missing. `/api/satellite/2025-13-45` is now 400 (invalid date), not 404.
- Tests: new `tests/test_error_messages.py` (8 tests: invalid date, `nimbus download` hints, both scripts import with `resource` unavailable); `tests/test_app.py` expectations updated. `uv run pytest -q` -> 49 passed. Mutation check: restoring a bare `import resource` makes the Windows-import test fail (then restored).

## Simplicity pass (code)

Regression guard (scratchpad, not in repo): CLI `forecast` for Mumbai/Chennai 2026-01-05, Mumbai 2024-07-10, Kerala 2030-07-10; the same four via `/api/forecast`; `/api/analysis?place=Kerala&year=2024`; `/api/models`; the chat fast path; sha256 of every file in `models/`. Re-run after every step. The only allowed difference is the additive `"method": "observed"` key from step 2d.

### Step 0 — incident before starting
- At 23:07 something outside this session ran `nimbus sync`. It committed `c25cfa8 "Automated sync: 2026-11-03"` (models/analysis.db) on `final-submission`, downloaded new imagery, and rewrote `data/satellite/thermal_features.csv`. Push almost certainly failed (no upstream; not verified without network).
- Per the owner's decision: saved as branch `sync-2026-11-03` (commit) and `stash@{0}` (CSV), and reset `final-submission` to `c6ea850`. Nothing deleted.

### Step 1 — Safer, simpler sync
- `cli.py` `sync`: the default now runs only `git pull --ff-only` (a failure is reported and the command continues) plus the imagery download. It never writes tracked files, commits, or pushes. New `--push` flag (maintainers) adds the previous behaviour: thermal-feature extraction, nowcast log/validation, analysis DB refresh, commit (`models/`, `data/forecast/`, `data/satellite/thermal_features.csv`) and push. Removed `--no-push`. The interactive menu now asks before pushing, defaulting to No.
- `bootstrap.sh`/`bootstrap.ps1`: no longer push, because their `nimbus sync` call is now the safe default; comments updated. `bootstrap.ps1` gained `Assert-LastExitCode` after each of its 4 native commands (`$ErrorActionPreference` doesn't cover native exit codes). pwsh isn't installed here, so the .ps1 wasn't executed.
- `commands.md` sync section rewritten.
- Tests: `tests/test_sync.py` (7) — default makes no commit/add/push and no analysis writes; `--push` commits and pushes; a failed pull doesn't stop the download; `--no-push` is rejected; neither bootstrap pushes; every native command in the .ps1 is followed by an exit-code check. Pytest 56 passed; guard PASS.

### Step 2 — One-line fixes for common failures
- a. Every user-facing "no imagery" error already said `nimbus download` (step 11); re-verified, no `scripts/download_satellite_images.py` hints remain in user-facing messages.
- b. One date parser: `weather/chatbot.py` `parse_date()` holds the existing `pd.to_datetime` / "Couldn't parse the date '…'" logic, previously copied three times in chatbot.py. It's now used by `get_rainfall_forecast`, `get_weather`, `get_satellite_image`, `thermal_estimate.predict_thermal_rainfall`, and `/api/satellite` (whose regex check stays as a path guard and uses the same wording). Removed the `fromisoformat` checks added in step 11. `cli._prompt_date` (interactive menu re-prompt loop) is unchanged.
- c. `app.py`: 404/405 under `/api/` return JSON (`/api/year/abc`, `/api/unknown`, `GET /api/chat`); page routes keep HTML.
- d. `chatbot.get_weather`: observed results gain `"method": "observed"` (additive; all other keys identical, asserted by test). CLI and map output unchanged.
- e. `nimbus forecast --date` defaults to today.
- Tests: +9 (`test_error_messages.py`: shared-parser equality across 4 callers, satellite bad-date bodies; `test_app.py`: JSON 404/405, HTML 404 for pages, observed additive key, `--date` default). Pytest 65 passed; guard PASS with the one allowed additive change.

### Step 3 — Windows/portability hygiene
- a. Already done in step 11 (`resource` optional in `scripts/benchmark.py` and `scripts/evaluate_forecast.py`, covered by `test_scripts_import_without_resource_module`). `peak_rss_mb = None` serializes as JSON `null`.
- b. `weather/hardware.py` `_has_avx2()`:
  - Windows: `ctypes.windll.kernel32.IsProcessorFeaturePresent(40)` (`PF_AVX2_INSTRUCTIONS_AVAILABLE`), returning `None` on any failure.
  - Apple Silicon (`Darwin` + `arm64`): returns `True`, since the check gates the torch wheel and torch is native on arm64. `describe()` prints "n/a (Apple Silicon -- torch runs natively)" instead of "AVX2: yes".
  - Linux `/proc/cpuinfo` and Intel-Mac `sysctl` paths are unchanged and checked first.
- Tests: `tests/test_hardware.py` (9) — Linux yes/no, Windows yes/no (asserts feature id 40), Windows ctypes failure -> None, Apple Silicon (no sysctl call, describe label), Intel Mac yes/no via sysctl, unknown OS -> None. Not run on real Windows or macOS. Pytest 74 passed; guard PASS.

### Step 4 — Single source of dependencies
- `requirements.txt` regenerated from `uv.lock` via `uv export --frozen --no-hashes --no-emit-project`: all 58 pins including transitive deps and pytest, platform markers kept, header "Generated from uv.lock; do not edit". `pyproject.toml` and `uv.lock` unchanged.
- **Deviation from the request:** exported *without* `--extra torch`. Including torch in the pip install would crash the server ("Illegal instruction", not catchable) on CPUs without AVX2 the first time a forecast routes to the satellite model, which is the stated i3 target. The previous file excluded it for this reason. The pinned torch install line is kept as a comment, at the lock's version. Adding `--extra torch` to the export command is a one-flag change if wanted.
- Tests: `tests/test_requirements.py` (4) — every pin equals uv.lock; core deps present and torch absent; torch hint version equals uv.lock; header present. Pytest 78 passed; guard PASS.

### Step 5 — Remove clutter, proven safe
- Deleted `scripts/extract.py` and `data/rainfall/rainfall_2023.json` (7.5 MB). Evidence: every runtime reader of `data/rainfall/` builds `rainfall_{year}.csv` (`features.py:52`, `chatbot.py:298,328,337`, `evaluate_forecast.py:49`, `nc_convert.py:139`). No `.json` path is built anywhere, and `web/*.html` fetch neither file (2023 is inlined; other years come from `/api/year/<y>` built from `.nc`). The only references were the extractor itself and one self-describing line each in README and `commands.md`; those lines were removed.
- Added `LICENSE` (MIT, 2026, Kommera Sachit Reddy).
- Added `.python-version` = **3.13, not 3.12 as requested**. Only 3.13 is installed here, so a 3.12 pin would make every `uv sync` download Python 3.12 and rebuild the venv (a network download, and a different runtime from the one all tests and QA ran on). 3.13 satisfies `requires-python >=3.12`.
- `data/satellite/thermal_features.csv` was already tracked (step 11).
- Tests: `tests/test_repo_hygiene.py` (4) — MIT license text; pin satisfies requires-python; thermal features tracked; deleted files absent and unreferenced. Pytest 82 passed; guard PASS.
- Note: during this step `uv sync --frozen --extra torch` reported reinstalling torch. It was present for the step-4 run (no skips), no outside process was running, and the reflog showed no new activity, so the cause is unknown. Torch imports fine afterwards.

### Step 6 — Faster first use
- `cli.py` `serve`: `_warm_up()` loads the 14.5M-row rainfall history, the forecaster booster + climatology, and the newest-date cache before `app.main()`, printing "Loading…" then "Models ready (N s)". Any exception becomes a warning and the server still starts (lazy loading, as before). Importing `app` does not warm up, so tests and the test client are unaffected.
- Measured: startup to ready 9.7 s ("Models ready (9 s)"); **first** `/api/forecast` request 0.52 s (was ~12.7 s cold); second 0.10 s.
- Tests: `tests/test_serve.py` (3) — warm-up runs before `app.main` in order; a failing warm-up warns and still serves; reloading `app` triggers no load. Pytest 85 passed; guard PASS.

### Step 7 — README + quick start
- README restructured. The top is now:
  - a 3-command **Quick start** (install uv -> `uv sync --frozen --extra torch` -> `uv run nimbus forecast --place Mumbai --date 2026-01-05`, with expected output `Mumbai, 2026-01-05: 0.87 mm (ML forecast)`);
  - notes on Python ≥ 3.12, Windows uv install, no-AVX2 (drop `--extra torch`), satellite methods needing `nimbus download` (~9.6 GB), and `serve` preloading;
  - **Running the tests**, a **7-line project layout**, a **table of all 16 scripts**, and **Other ways to install** (bootstrap, now non-pushing; pip from the generated requirements).
- This replaced the 45-line structure listing and the step-11 setup block.
- Command table: `forecast` date defaults to today; `sync` documents `--push`; `serve` mentions preloading.
- "Keeping everything in sync" rewritten for `--push`. Limitations: the Windows/Apple Silicon torch bullet and the bootstrap.ps1 bullet updated to the new code; both are still not verified on real hardware.
- Stale claims re-checked: RAM ~1.9/~3.7 GB, imagery ~9.6 GB, Windows "not verified", Ivy Bridge/Raspberry Pi "intended targets, not verified" (from step 11), and no "well under 1 GB", "~7 GB", "fully supported" or `--no-push` left.
- New `scripts/quickstart.py` (stdlib only): uv check -> `uv sync --frozen --extra torch` (retry without torch) -> `uv run pytest -q` -> Mumbai forecast compared to the expected line; PASS/FAIL/SKIP per stage; exit code 0/1. Never runs git, `nimbus sync` or imagery downloads.
- Tests: `tests/test_quickstart.py` (7) — all pass; torch retry; wrong forecast fails; missing uv fails and skips the rest; no git/push/download commands; expected line identical in script and README; stdlib-only imports. `test_repo_hygiene.py` +1 — README script table equals `scripts/*.py` (16) and documents `sync --push` and pytest. Pytest 93 passed; guard PASS.

### Step 8 — Stranger test
- Fresh clone of `final-submission` (9e22d46) into `/var/tmp/nimbus-stranger-GNua`, `VIRTUAL_ENV` unset, following only the README Quick start:
  1. `curl … install.sh | sh`: **skipped** (network installer; uv 0.12.5 already installed).
  2. `uv sync --frozen --extra torch`: 1.0 s (uv cache warm; a cold install downloads ~1 GB of wheels, not measured).
  3. `uv run nimbus forecast --place Mumbai --date 2026-01-05`: 15.5 s -> `Mumbai, 2026-01-05: 0.87 mm (ML forecast)`, **exactly the documented output**.
- Also from the README: `uv run pytest -q` -> 93 passed (8.5 s); `uv run python scripts/quickstart.py` -> 4× PASS (16.6 s).
- Mismatches: none. No tracked file was modified by any command.
- `git diff 19ae3c9 HEAD --stat -- models/` is not empty: it lists the two files *added* in the hardening pass (`models/benchmark.json`, `models/forecast_eval_report.json`). No existing model, metrics, climatology or analysis file differs from 19ae3c9.
