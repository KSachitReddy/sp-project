"""
Hardware detection for running this project on anything from a capable
desktop down to ~1GHz embedded-class hardware (Raspberry Pi, an old
netbook). `uv sync` itself can't do this -- it's a fixed dependency
resolver, not a provisioning tool -- so this backs a separate step:

    uv run nimbus setup

which reads this module's output and decides what to install/configure.
"""
import os
import platform
import re
import subprocess

# Below this clock speed (or with this few cores and no AVX2), local model
# training and the satellite CNN are impractical -- inference/serving only.
MINIMAL_TIER_MHZ = 1400


def _run_powershell(command, timeout=4):
    """Best-effort PowerShell one-liner, or None on any failure. Windows-only
    callers guard this with platform.system() == "Windows" first."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, timeout=timeout,
        )
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
    except Exception:
        return None


def _cpu_mhz():
    """Best-effort peak CPU clock speed in MHz, or None if undeterminable."""
    try:
        with open("/proc/cpuinfo") as f:
            text = f.read()
        values = [float(m) for m in re.findall(r"cpu MHz\s*:\s*([\d.]+)", text)]
        if values:
            return max(values)  # cores can be throttled down; report the fastest seen
    except FileNotFoundError:
        pass
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.cpufrequency_max"], capture_output=True, text=True, timeout=2
            )
            if out.returncode == 0 and out.stdout.strip():
                return int(out.stdout.strip()) / 1_000_000
        except Exception:
            pass
    if platform.system() == "Windows":
        out = _run_powershell("(Get-CimInstance Win32_Processor).MaxClockSpeed")
        if out and out.isdigit():
            return float(out)  # already MHz
    return None


def _has_avx2():
    """True/False if determinable, None if unknown (treated as "not safe to assume")."""
    try:
        with open("/proc/cpuinfo") as f:
            return "avx2" in f.read().lower()
    except FileNotFoundError:
        pass
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["sysctl", "-a"], capture_output=True, text=True, timeout=2)
            if out.returncode == 0:
                return "avx2" in out.stdout.lower()
        except Exception:
            pass
    return None


def _total_ram_gb():
    if platform.system() == "Windows":
        # os.sysconf is POSIX-only -- ask the Win32 API directly via ctypes,
        # no extra dependency needed.
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(status.ullTotalPhys / (1024 ** 3), 1)
        except Exception:
            pass
        return None
    try:
        pages, page_size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / (1024 ** 3), 1)
    except (ValueError, OSError, AttributeError):
        return None


def _total_swap_gb():
    if platform.system() == "Windows":
        # Windows' equivalent of swap is the pagefile, managed very differently
        # (no fallocate/mkswap/fstab) -- see cli.py's `setup` command, which
        # gives pagefile guidance instead of trying to configure it directly.
        out = _run_powershell(
            "(Get-CimInstance Win32_PageFileUsage | Measure-Object -Property AllocatedBaseSize -Sum).Sum"
        )
        if out:
            try:
                return round(float(out) / 1024, 1)  # AllocatedBaseSize is MB
            except ValueError:
                pass
        return None
    try:
        with open("/proc/meminfo") as f:
            text = f.read()
        m = re.search(r"SwapTotal:\s*(\d+)\s*kB", text)
        if m:
            return round(int(m.group(1)) / (1024 ** 2), 1)
    except FileNotFoundError:
        pass
    return None


def detect():
    """A snapshot of this machine's relevant hardware, plus the tier it falls
    into for deciding what this project should do on it.

    - "minimal": ~1GHz-class (Raspberry Pi, old netbook) -- inference/serving
      only; never train here, never install the satellite CNN's torch dependency.
    - "modest": no confirmed AVX2 (e.g. a 3rd-gen-or-older Intel Core) -- fine
      for everything except the satellite nowcaster, which needs AVX2 to
      install safely.
    - "capable": AVX2 confirmed -- safe to install everything.
    """
    arch = platform.machine()
    cores = os.cpu_count() or 1
    mhz = _cpu_mhz()
    avx2 = _has_avx2()
    ram_gb = _total_ram_gb()
    swap_gb = _total_swap_gb()

    if (mhz is not None and mhz < MINIMAL_TIER_MHZ) or (cores <= 2 and not avx2):
        tier = "minimal"
    elif not avx2:
        tier = "modest"
    else:
        tier = "capable"

    return {
        "arch": arch, "cores": cores, "mhz": mhz, "avx2": avx2,
        "ram_gb": ram_gb, "swap_gb": swap_gb, "tier": tier,
    }


def describe(info):
    mhz_str = f"~{info['mhz']:.0f} MHz" if info["mhz"] else "clock speed unknown"
    avx2_str = {True: "yes", False: "no", None: "couldn't determine"}[info["avx2"]]
    ram_str = f"{info['ram_gb']} GB" if info["ram_gb"] is not None else "unknown"
    swap_str = f"{info['swap_gb']} GB" if info["swap_gb"] is not None else "unknown"
    return (
        f"CPU:  {info['cores']} core(s), {info['arch']}, {mhz_str}\n"
        f"AVX2: {avx2_str}\n"
        f"RAM:  {ram_str}\n"
        f"Swap: {swap_str}\n"
        f"Tier: {info['tier']}"
    )
