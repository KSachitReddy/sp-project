# Final QA — 2026-11-08

End-to-end check of every feature before writing the final report.
Machine: 4-core Intel i5-6500 (AVX2), 7.7 GB RAM, Debian Linux (kernel 6.12), Python 3.13, uv 0.12.5.
Windows: **Wine 10.0** running the official Windows CPython 3.13.1 (amd64). No real Windows machine was available.

## Linux — fully verified

| Area | What was run | Result |
|---|---|---|
| Install (uv) | `uv sync --frozen --extra torch` | OK |
| Install (pip) | fresh venv + `pip install -r requirements.txt`, then forecast + pytest | OK; 96 passed, 2 skipped (torch tests); nowcast gives a clear "torch isn't installed" message |
| Test suite | `uv run pytest -q` | **98 passed** |
| Quick start | `scripts/quickstart.py` | 4/4 PASS |
| `setup --yes` | hardware detection | 4 cores, AVX2 yes, 7.7 GB, tier `capable` |
| `forecast` | Mumbai 2026-01-05 / Kerala 2023-08-15 / Mumbai 2026-09-01 / Delhi (today) | `0.87 mm (ML forecast)` / `0.0 mm (observed)` / `3.27 mm (satellite nowcast, west avg)` / `0.0 mm (historical average)` |
| `forecast` errors | unknown place, invalid date | clean one-line error, exit 1 |
| `analyse` | `--place Kerala --year 2023`, `--trend 2018 2025` | OK (JSON) |
| `image` | `--date 2023-08-15` | path printed, viewer launched |
| `nowcast` | CNN | per-region mm from the 2026-11-02 image |
| `thermal-estimate` | `--date 2025-08-15` | per-region mm |
| `download` | resumable GIBS download + thermal features | OK; `verify_satellite_images.py`: 100% complete, "READY TO USE" |
| `params` | elevation + ONI/IOD/MJO | indices OK; elevation hit HTTP 429 (see bug 1) |
| `sync` (default) | pull + imagery | OK; no tracked files written, nothing committed |
| Interactive menu | options 4, 5 (summary + trend), 6 (with invalid date re-prompt), 7, 8, invalid choice, 0 | all OK |
| `chat` (CLI) | fast path, then a follow-up via Ollama `llama3.2:3b` | both answered; the LLM follow-up worded a correct 0 mm as "no data available" (LLM wording, value correct) |
| `serve` | `/`, `/chat`, `/api/years`, `/api/year/<y>`, `/api/satellite/<d>`, `/api/forecast`, `/api/analysis` (year + trend), `/api/nowcast`, `/api/models`, `/api/chat` (fast path + Ollama tool call), `/api/convert` | all 200 on valid input, 400/404 JSON errors on bad input; re-uploading the 2018 `.nc` reproduced a byte-identical CSV |
| Retraining | `nimbus train --model rainfall` | **identical** to committed `forecast_model_metrics.json` (63 s) |
| | `nimbus train --model satellite` | reproduces the documented "seed 0 rerun" column of `docs/cnn_seed_stability.md` exactly (249 s); reverted |
| | `calibrate_thermal_rainfall.py` | identical to committed metrics |
| Evaluation | `evaluate_forecast.py`, `benchmark.py` | OK (38 s each); only runtime/RSS fields differ; reverted |

## Windows (Wine) — partially verified

| Area | Result |
|---|---|
| `pip install -r requirements.txt` (Windows wheels, `--only-binary`) | **OK**: every pinned package installs |
| `cli.py --help` | OK |
| `cli.py setup` / `setup --yes` | OK after fix (bug 2): AVX2 via `IsProcessorFeaturePresent`, RAM via `GlobalMemoryStatusEx`, page-file guidance |
| `cli.py sync --push` without git | clean refusal after fix (bug 3) |
| `test_hardware.py`, `test_quickstart.py`, `test_requirements.py` | 9 + 7 + 4 passed |
| `test_repo_hygiene.py` | 3 passed, 2 failed only because Wine has no `git.exe` |
| Text encoding | Wine Python ran with cp1252 (the Windows default); every file read without an explicit encoding decodes under cp1252 |
| Anything importing numpy (forecast, serve, chat, image, most tests) | **not testable**: numpy 2.5.1 aborts on import under Wine 10 (`unimplemented function ucrtbase.dll.crealf`). This is a Wine limitation, not a project bug. Installing Microsoft's VC++ runtime into Wine would likely fix it but wasn't done |
| `os.startfile` (image viewer) | inconclusive under Wine |
| `bootstrap.ps1` | not executed (no PowerShell); reviewed statically: exit code checked after each of the 4 steps |

## Bugs found and fixed in this pass

1. **`nimbus params` could destroy `data/parameters/elevation.csv`.** When the Open-Elevation API rate-limited (HTTP 429), the script overwrote the complete 4,964-cell file with the 500 cells it got. Now it retries 429s with backoff and merges with the existing file, so a partial run never drops cells. Test: `tests/test_download_elevation.py` (fails before the fix).
2. **`nimbus setup --yes` crashed with `FileNotFoundError` when `uv` wasn't on PATH** (pip install route, any OS). Now it prints the matching `pip install torch==2.13.0 --index-url …/cpu` command.
3. **`nimbus sync` crashed when `git` wasn't installed** (e.g. a ZIP download). Default sync now continues with the imagery download; `--push` refuses with a clear error.
4. **`setup` said "only 0 GB of page file" when the size couldn't be detected.** It now says the amount is undetected.
5. Menu item 10's label said "commit+push" though the default never does; relabelled.

Tests for 2–4: `tests/test_missing_tools.py` (4 tests; all fail before the fix).
