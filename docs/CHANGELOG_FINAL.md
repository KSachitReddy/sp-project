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
