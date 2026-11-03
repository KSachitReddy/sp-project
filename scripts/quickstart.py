"""
Check a fresh Nimbus setup end to end and print PASS/FAIL per stage:

    python scripts/quickstart.py        (or: uv run python scripts/quickstart.py)

1. uv is installed
2. `uv sync --frozen --extra torch` (retried without torch if that fails,
   e.g. on a CPU/platform with no suitable torch wheel)
3. `uv run pytest -q`
4. `uv run nimbus forecast --place Mumbai --date 2026-01-05` prints the
   expected answer

Standard library only, so it runs before any dependency is installed. It
never commits, pushes, or downloads satellite imagery (installing the locked
Python packages in stage 2 is the only network use).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_FORECAST = "Mumbai, 2026-01-05: 0.87 mm (ML forecast)"


def _run(cmd):
    """(ok, combined output) for a command run from the repo root."""
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    except FileNotFoundError:
        return False, f"{cmd[0]}: command not found"
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def _last_line(text):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def check_uv():
    ok, out = _run(["uv", "--version"])
    return ok, out if ok else "install uv: https://docs.astral.sh/uv/getting-started/installation/"


def check_sync():
    ok, out = _run(["uv", "sync", "--frozen", "--extra", "torch"])
    if ok:
        return True, "installed (with torch)"
    ok, out = _run(["uv", "sync", "--frozen"])
    return ok, "installed without torch (satellite nowcaster disabled)" if ok else _last_line(out)


def check_tests():
    ok, out = _run(["uv", "run", "pytest", "-q"])
    return ok, _last_line(out)


def check_forecast():
    ok, out = _run(["uv", "run", "nimbus", "forecast", "--place", "Mumbai", "--date", "2026-01-05"])
    got = _last_line(out)
    return ok and got == EXPECTED_FORECAST, got


STAGES = [
    ("uv installed", check_uv),
    ("dependencies (uv sync)", check_sync),
    ("tests (pytest)", check_tests),
    ("first forecast", check_forecast),
]


def main():
    all_ok = True
    for name, check in STAGES:
        if not all_ok:
            print(f"SKIP  {name}")
            continue
        ok, detail = check()
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        all_ok = all_ok and ok
    print("\nAll good -- next: `uv run nimbus serve`." if all_ok else "\nStopped at the first failure above.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
