"""scripts/quickstart.py: stage logic with subprocess replaced by a fake, so
nothing is installed or run. Also: it must never touch git or downloads, and
must stay standard-library only."""
import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "quickstart.py"


@pytest.fixture
def qs():
    spec = importlib.util.spec_from_file_location("_nimbus_quickstart", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake(monkeypatch, qs, outcomes):
    """outcomes: {tuple(cmd): (returncode, stdout)}; anything else succeeds silently."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        rc, out = outcomes.get(tuple(cmd), (0, ""))
        return SimpleNamespace(returncode=rc, stdout=out, stderr="")

    monkeypatch.setattr(qs.subprocess, "run", fake_run)
    return calls


FORECAST = ("uv", "run", "nimbus", "forecast", "--place", "Mumbai", "--date", "2026-01-05")


def test_all_stages_pass(monkeypatch, capsys, qs):
    calls = _fake(monkeypatch, qs, {FORECAST: (0, qs.EXPECTED_FORECAST + "\n"),
                                    ("uv", "run", "pytest", "-q"): (0, "85 passed in 5s\n")})
    assert qs.main() == 0
    out = capsys.readouterr().out
    assert out.count("PASS") == 4 and "FAIL" not in out
    assert ["uv", "sync", "--frozen"] not in calls  # no retry needed


def test_sync_retries_without_torch(monkeypatch, capsys, qs):
    calls = _fake(monkeypatch, qs, {("uv", "sync", "--frozen", "--extra", "torch"): (1, "no wheel"),
                                    FORECAST: (0, qs.EXPECTED_FORECAST)})
    assert qs.main() == 0
    assert ["uv", "sync", "--frozen"] in calls
    assert "without torch" in capsys.readouterr().out


def test_wrong_forecast_fails_and_stops(monkeypatch, capsys, qs):
    _fake(monkeypatch, qs, {FORECAST: (0, "Mumbai, 2026-01-05: 9.99 mm (ML forecast)")})
    assert qs.main() == 1
    assert "FAIL  first forecast" in capsys.readouterr().out


def test_missing_uv_fails_first_and_skips_the_rest(monkeypatch, capsys, qs):
    def no_uv(cmd, **kwargs):
        raise FileNotFoundError(cmd[0])
    monkeypatch.setattr(qs.subprocess, "run", no_uv)
    assert qs.main() == 1
    out = capsys.readouterr().out
    assert "FAIL  uv installed" in out and out.count("SKIP") == 3


def test_never_commits_pushes_or_downloads(monkeypatch, qs):
    calls = _fake(monkeypatch, qs, {FORECAST: (0, qs.EXPECTED_FORECAST)})
    qs.main()
    flat = [" ".join(c) for c in calls]
    assert not any(w in c for c in flat for w in ("git", "download", "nimbus sync", "push"))


def test_expected_output_matches_readme():
    readme = (SCRIPT.parent.parent / "README.md").read_text()
    from_script = ast.literal_eval(next(
        n.value for n in ast.parse(SCRIPT.read_text()).body
        if isinstance(n, ast.Assign) and n.targets[0].id == "EXPECTED_FORECAST"))
    assert from_script in readme


def test_stdlib_only():
    imported = {a.name.split(".")[0] for n in ast.walk(ast.parse(SCRIPT.read_text()))
                if isinstance(n, (ast.Import, ast.ImportFrom)) for a in getattr(n, "names", [])
                if isinstance(n, ast.Import)}
    imported |= {n.module.split(".")[0] for n in ast.walk(ast.parse(SCRIPT.read_text()))
                 if isinstance(n, ast.ImportFrom) and n.module}
    assert imported <= set(sys.stdlib_module_names)
