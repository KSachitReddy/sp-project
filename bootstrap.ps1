# One command to go from a fresh clone to a running app on Windows:
#
#   .\bootstrap.ps1
#
# If PowerShell blocks this with an execution-policy error, run it once for
# just this session (doesn't change your system-wide policy):
#
#   powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1
#
# This exists because `uv sync` itself deliberately can't do this: it's a
# fixed dependency resolver, not a provisioning tool, and it would be a real
# problem for a routine "install my dependencies" command to silently start
# downloading gigabytes or need elevated permissions -- that's exactly the
# kind of surprising behavior a build step should never have. So this script
# chains the actual steps instead:
#
#   1. uv sync            -- install dependencies (fast, no side effects)
#   2. nimbus setup --yes -- detect hardware, install torch if safe,
#                             print page-file guidance if RAM is low
#                             (Windows manages its page file differently
#                             from Linux swap -- no automatic changes here)
#   3. nimbus sync         -- pull latest models/analysis, fetch new satellite
#                             imagery (progress bar), validate, commit + push
#   4. nimbus serve         -- launch the app
#
# Safe to re-run any time -- every step is idempotent.

$ErrorActionPreference = "Stop"

Write-Host "=== [1/4] Installing dependencies ==="
uv sync

Write-Host ""
Write-Host "=== [2/4] Detecting hardware, configuring accordingly ==="
uv run nimbus setup --yes

Write-Host ""
Write-Host "=== [3/4] Syncing data (satellite imagery, models, analysis) ==="
uv run nimbus sync

Write-Host ""
Write-Host "=== [4/4] Starting the app ==="
uv run nimbus serve
