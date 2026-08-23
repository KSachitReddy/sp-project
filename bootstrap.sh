#!/usr/bin/env bash
# One command to go from a fresh clone to a running app:
#
#   ./bootstrap.sh
#
# This exists because `uv sync` itself deliberately can't do this: it's a
# fixed dependency resolver, not a provisioning tool, and it would be a real
# problem for a routine "install my dependencies" command to silently prompt
# for your sudo password or start downloading gigabytes -- that's exactly
# the kind of surprising behavior a build step should never have, and it
# would break `uv sync` in CI/sandboxed environments where people just want
# the deps installed. So this script chains the actual steps instead:
#
#   1. uv sync            -- install dependencies (fast, no side effects)
#   2. nimbus setup --yes -- detect hardware, install torch if safe, offer swap
#   3. nimbus sync         -- pull latest models/analysis, fetch new satellite
#                             imagery (progress bar), validate, commit + push
#   4. nimbus serve         -- launch the app
#
# Safe to re-run any time -- every step is idempotent.
set -e

echo "=== [1/4] Installing dependencies ==="
uv sync

echo
echo "=== [2/4] Detecting hardware, configuring accordingly ==="
uv run nimbus setup --yes

echo
echo "=== [3/4] Syncing data (satellite imagery, models, analysis) ==="
uv run nimbus sync

echo
echo "=== [4/4] Starting the app ==="
uv run nimbus serve
