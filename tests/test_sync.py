"""`nimbus sync` must never commit or push unless --push is given, and the
bootstrap scripts (which call it) must not opt in. subprocess and the
analysis module are replaced with recorders -- nothing really runs."""
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import cli
from weather import analysis

ROOT = Path(__file__).parent.parent


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def fake_call(cmd, *a, **k):
        calls.append(list(cmd))
        return 0

    def fake_run(cmd, *a, **k):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="committed", stderr="")

    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    for name in ("log_nowcast_prediction", "validate_pending_nowcasts", "build"):
        monkeypatch.setattr(analysis, name, lambda *a, **k: calls.append([f"analysis.{name}"]) or None)
    return calls


def _git(calls, sub):
    return [c for c in calls if c[:2] == ["git", sub]]


def test_default_sync_only_pulls_and_downloads(recorded):
    result = CliRunner().invoke(cli.cli, ["sync"])
    assert result.exit_code == 0, result.output
    assert _git(recorded, "pull") == [["git", "pull", "--ff-only"]]
    assert not _git(recorded, "commit") and not _git(recorded, "push") and not _git(recorded, "add")
    scripts = [Path(c[1]).name for c in recorded if len(c) > 1 and c[1].endswith(".py")]
    assert scripts == ["download_satellite_images.py"]
    assert not any(c[0].startswith("analysis.") for c in recorded)  # no tracked-file writes
    assert "Nothing was committed or pushed" in result.output


def test_push_flag_commits_and_pushes(recorded):
    result = CliRunner().invoke(cli.cli, ["sync", "--push"])
    assert result.exit_code == 0, result.output
    assert len(_git(recorded, "commit")) == 1 and _git(recorded, "push") == [["git", "push"]]
    assert ["analysis.build"] in recorded


def test_failed_pull_does_not_stop_download(monkeypatch, recorded):
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd, *a, **k: recorded.append(list(cmd)) or (1 if cmd[:2] == ["git", "pull"] else 0))
    result = CliRunner().invoke(cli.cli, ["sync"])
    assert result.exit_code == 0 and "continuing with the local copy" in result.output
    assert any(c[-1].endswith("download_satellite_images.py") for c in recorded)


def test_old_no_push_flag_is_gone():
    assert CliRunner().invoke(cli.cli, ["sync", "--no-push"]).exit_code != 0


@pytest.mark.parametrize("script", ["bootstrap.sh", "bootstrap.ps1"])
def test_bootstraps_never_push(script):
    text = (ROOT / script).read_text()
    assert "--push" not in re.sub(r"^\s*#.*$", "", text, flags=re.M)  # only mentioned in comments
    assert "git push" not in text


def test_bootstrap_ps1_checks_every_native_exit_code():
    lines = [ln.strip() for ln in (ROOT / "bootstrap.ps1").read_text().splitlines()]
    native = [i for i, ln in enumerate(lines) if ln.startswith("uv ")]
    assert len(native) == 4
    for i in native:
        assert lines[i + 1].startswith("Assert-LastExitCode"), lines[i]
    assert "$LASTEXITCODE -ne 0" in "\n".join(lines)
