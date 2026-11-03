"""AVX2 / torch-safety detection on each OS, simulated with monkeypatching so
it runs on any machine. Linux and Intel-Mac behaviour must be unchanged."""
import io
from types import SimpleNamespace

import pytest

from weather import hardware


def _no_proc(monkeypatch):
    def fake_open(path, *a, **k):
        raise FileNotFoundError(path)
    monkeypatch.setattr(hardware, "open", fake_open, raising=False)


def _os(monkeypatch, system, machine):
    monkeypatch.setattr(hardware.platform, "system", lambda: system)
    monkeypatch.setattr(hardware.platform, "machine", lambda: machine)


@pytest.mark.parametrize("cpuinfo, expected", [("flags : sse4_2 avx avx2 fma", True), ("flags : sse4_2 avx", False)])
def test_linux_reads_proc_cpuinfo(monkeypatch, cpuinfo, expected):
    monkeypatch.setattr(hardware, "open", lambda *a, **k: io.StringIO(cpuinfo), raising=False)
    assert hardware._has_avx2() is expected


@pytest.mark.parametrize("feature_present, expected", [(1, True), (0, False)])
def test_windows_uses_isprocessorfeaturepresent(monkeypatch, feature_present, expected):
    import ctypes
    _no_proc(monkeypatch)
    _os(monkeypatch, "Windows", "AMD64")
    asked = []
    kernel32 = SimpleNamespace(IsProcessorFeaturePresent=lambda f: asked.append(f) or feature_present)
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=kernel32), raising=False)
    assert hardware._has_avx2() is expected
    assert asked == [40]  # PF_AVX2_INSTRUCTIONS_AVAILABLE


def test_windows_ctypes_failure_is_unknown(monkeypatch):
    import ctypes
    _no_proc(monkeypatch)
    _os(monkeypatch, "Windows", "AMD64")
    monkeypatch.delattr(ctypes, "windll", raising=False)  # e.g. no windll on this interpreter
    assert hardware._has_avx2() is None


def test_apple_silicon_is_torch_capable(monkeypatch):
    _no_proc(monkeypatch)
    _os(monkeypatch, "Darwin", "arm64")
    monkeypatch.setattr(hardware.subprocess, "run", lambda *a, **k: pytest.fail("sysctl shouldn't be needed"))
    assert hardware._has_avx2() is True
    info = {"cores": 8, "arch": "arm64", "mhz": None, "avx2": True, "ram_gb": 16, "swap_gb": 0, "tier": "capable"}
    assert "Apple Silicon" in hardware.describe(info)


@pytest.mark.parametrize("sysctl_out, expected", [("machdep.cpu.leaf7_features: AVX2 BMI1", True), ("machdep.cpu.features: SSE4.2", False)])
def test_intel_mac_still_uses_sysctl(monkeypatch, sysctl_out, expected):
    _no_proc(monkeypatch)
    _os(monkeypatch, "Darwin", "x86_64")
    monkeypatch.setattr(hardware.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=sysctl_out))
    assert hardware._has_avx2() is expected


def test_unknown_os_is_unknown(monkeypatch):
    _no_proc(monkeypatch)
    _os(monkeypatch, "FreeBSD", "amd64")
    assert hardware._has_avx2() is None
