"""External programs (uv, git) may be missing -- pip install route, ZIP download,
a fresh Windows machine. Found in final QA under Windows Python: `nimbus setup
--yes` crashed with FileNotFoundError when uv wasn't on PATH."""
import subprocess

from click.testing import CliRunner

import cli
from weather import hardware


def _missing(cmd, *a, **k):
    raise FileNotFoundError(2, "No such file", cmd[0])


def _fake_hw(monkeypatch, swap_gb):
    info = {"tier": "capable", "ram_gb": 7.7, "swap_gb": swap_gb}
    monkeypatch.setattr(hardware, "detect", lambda: info)
    monkeypatch.setattr(hardware, "describe", lambda i: "fake hardware")


def test_setup_without_uv_prints_pip_hint(monkeypatch):
    _fake_hw(monkeypatch, swap_gb=10.0)
    monkeypatch.setattr(cli.subprocess, "call", _missing)
    monkeypatch.setitem(__import__("sys").modules, "torch", None)  # force "torch not installed"
    result = CliRunner().invoke(cli.cli, ["setup", "--yes"])
    assert result.exit_code == 0, result.output
    assert "`uv` was not found on PATH" in result.output
    assert "pip install torch==2.13.0" in result.output


def test_setup_unknown_swap_is_not_reported_as_zero(monkeypatch):
    _fake_hw(monkeypatch, swap_gb=None)
    monkeypatch.setitem(__import__("sys").modules, "torch", subprocess)  # any module: "installed"
    result = CliRunner().invoke(cli.cli, ["setup", "--yes"])
    assert result.exit_code == 0, result.output
    assert "only 0 GB" not in result.output
    assert "an undetected amount of" in result.output


def test_sync_without_git_still_downloads(monkeypatch):
    calls = []

    def fake_call(cmd, *a, **k):
        if cmd[0] == "git":
            raise FileNotFoundError(2, "No such file", "git")
        calls.append(cmd)
        return 0

    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    result = CliRunner().invoke(cli.cli, ["sync"])
    assert result.exit_code == 0, result.output
    assert "continuing with the local copy" in result.output
    assert any(str(c[-1]).endswith("download_satellite_images.py") for c in calls)


def test_sync_push_without_git_refuses_cleanly(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(cli.subprocess, "call", lambda *a, **k: 0)
    result = CliRunner().invoke(cli.cli, ["sync", "--push"])
    assert result.exit_code == 1
    assert "--push needs git" in result.output
