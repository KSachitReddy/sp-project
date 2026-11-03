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
#   3. nimbus sync         -- fast-forward pull + fetch new satellite imagery
#                             (progress bar, ~10 GB first time); never commits
#                             or pushes (that needs `nimbus sync --push`)
#   4. nimbus serve         -- launch the app
#
# Safe to re-run any time -- every step is idempotent.

$ErrorActionPreference = "Stop"

# $ErrorActionPreference doesn't apply to native commands like uv, so check
# each one's exit code explicitly -- the equivalent of bootstrap.sh's `set -e`.
function Assert-LastExitCode($step) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "$step failed (exit code $LASTEXITCODE) -- stopping."
        exit $LASTEXITCODE
    }
}

Write-Host "=== [1/4] Installing dependencies ==="
uv sync
Assert-LastExitCode "uv sync"

Write-Host ""
Write-Host "=== [2/4] Detecting hardware, configuring accordingly ==="
uv run nimbus setup --yes
Assert-LastExitCode "nimbus setup"

Write-Host ""
Write-Host "=== [3/4] Syncing data (satellite imagery, models, analysis) ==="
uv run nimbus sync
Assert-LastExitCode "nimbus sync"

Write-Host ""
Write-Host "=== [4/4] Starting the app ==="
uv run nimbus serve
Assert-LastExitCode "nimbus serve"
