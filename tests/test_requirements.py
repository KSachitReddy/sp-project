"""requirements.txt is generated from uv.lock and must never drift from it."""
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _lock_versions():
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    return {p["name"].lower(): p["version"] for p in lock["package"] if "version" in p}


def _requirement_pins():
    pins = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if m:
            pins[m.group(1).lower().replace("_", "-")] = m.group(2)
    return pins


def test_every_pin_matches_uv_lock():
    lock, pins = _lock_versions(), _requirement_pins()
    assert len(pins) > 40
    mismatched = {n: (v, lock.get(n)) for n, v in pins.items() if lock.get(n) != v}
    assert not mismatched


def test_core_dependencies_present_and_torch_excluded():
    pins = _requirement_pins()
    for name in ("flask", "lightgbm", "pandas", "numpy", "click", "pytest"):
        assert name in pins
    assert "torch" not in pins  # crashes the server on CPUs without AVX2 -- opt-in only


def test_torch_install_hint_matches_lock():
    hint = re.search(r"pip install torch==(\S+)", (ROOT / "requirements.txt").read_text())
    assert hint and hint.group(1) == _lock_versions()["torch"]


def test_header_says_generated():
    assert (ROOT / "requirements.txt").read_text().startswith("# Generated from uv.lock; do not edit.")
