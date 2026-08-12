"""
Verify the data/satellite/india_cloud dataset downloaded by
download_satellite_images.py: every date from 2019-01-01 to 2025-12-31 should
have a valid PNG on disk.

Checks, per date:
  - file present
  - PNG signature intact and IEND trailer present (catches truncated/corrupt
    downloads without needing an image-decoding dependency)
  - file not suspiciously small (same threshold as the downloader)

Writes a human-readable report to data/satellite/logs/verification_report.txt
and the list of missing/corrupt dates to data/satellite/logs/retry_dates.txt
(one per line; empty file if everything is fine) for retry_failed_dates.py to
pick up.
"""
import csv
import sys
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
IMG_DIR = BASE_DIR / "data" / "satellite" / "india_cloud"
LOG_DIR = BASE_DIR / "data" / "satellite" / "logs"
LOG_CSV = LOG_DIR / "download_log.csv"
ERROR_LOG = LOG_DIR / "errors.txt"
REPORT_PATH = LOG_DIR / "verification_report.txt"
RETRY_PATH = LOG_DIR / "retry_dates.txt"

START_DATE = date(2019, 1, 1)
END_DATE = date(2025, 12, 31)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_IEND_TRAILER = b"IEND\xaeB`\x82"
MIN_VALID_BYTES = 5000


def daterange(start, end):
    current = start
    one_day = timedelta(days=1)
    while current <= end:
        yield current
        current += one_day


def check_file(path):
    if not path.exists():
        return "missing"
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        return "corrupt (bad PNG signature)"
    if data[-8:] != PNG_IEND_TRAILER:
        return "corrupt (truncated / missing IEND)"
    if len(data) < MIN_VALID_BYTES:
        return "suspiciously small"
    return "ok"


def load_layer_by_date():
    """Last logged 'success' layer per date, so retries override earlier attempts."""
    layer_by_date = {}
    if not LOG_CSV.exists():
        return layer_by_date
    with open(LOG_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "success":
                layer_by_date[row["date"]] = row["layer_used"]
    return layer_by_date


def load_permanent_failures():
    """Last logged error line per date from errors.txt (format 'date: reason')."""
    reasons = {}
    if not ERROR_LOG.exists():
        return reasons
    for line in ERROR_LOG.read_text().splitlines():
        if ":" not in line:
            continue
        d, reason = line.split(":", 1)
        reasons[d.strip()] = reason.strip()
    return reasons


def main():
    layer_by_date = load_layer_by_date()
    perm_failures = load_permanent_failures()

    results = {}
    year_stats = {}
    for day in daterange(START_DATE, END_DATE):
        ds = day.isoformat()
        path = IMG_DIR / str(day.year) / f"{ds}.png"
        status = check_file(path)
        results[ds] = status
        stats = year_stats.setdefault(day.year, {"total": 0, "ok": 0})
        stats["total"] += 1
        if status == "ok":
            stats["ok"] += 1

    total = len(results)
    ok_dates = sorted(d for d, s in results.items() if s == "ok")
    bad_dates = sorted(d for d, s in results.items() if s != "ok")

    RETRY_PATH.write_text("\n".join(bad_dates) + ("\n" if bad_dates else ""))

    layer_counts = Counter(layer_by_date.get(d, "unknown") for d in ok_dates)
    status_counts = Counter(results.values())

    lines = []
    lines.append("Satellite Imagery Verification Report")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"Expected days (2019-01-01 to 2025-12-31): {total}")
    lines.append(f"Valid images present:                     {len(ok_dates)} ({len(ok_dates) / total * 100:.2f}%)")
    lines.append(f"Missing/corrupt:                           {len(bad_dates)}")
    lines.append("")
    lines.append("--- Status breakdown ---")
    for status, count in sorted(status_counts.items()):
        lines.append(f"  {status}: {count}")
    lines.append("")
    lines.append("--- Layer usage (valid images) ---")
    for layer, count in layer_counts.most_common():
        lines.append(f"  {layer}: {count}")
    lines.append("")
    lines.append("--- Year-by-year completeness ---")
    for year in sorted(year_stats):
        s = year_stats[year]
        pct = s["ok"] / s["total"] * 100
        lines.append(f"  {year}: {s['ok']}/{s['total']} ({pct:.2f}%)")
    lines.append("")
    if bad_dates:
        lines.append(f"--- {len(bad_dates)} problem dates (full list in retry_dates.txt) ---")
        for d in bad_dates[:50]:
            reason = perm_failures.get(d, "no matching errors.txt entry")
            lines.append(f"  {d}: {results[d]} (last logged error: {reason})")
        if len(bad_dates) > 50:
            lines.append(f"  ... and {len(bad_dates) - 50} more")
    lines.append("")
    verdict = "READY TO USE" if not bad_dates else "NEEDS ATTENTION"
    lines.append(f"VERDICT: {verdict}")

    report = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(report)
    print(report)

    return 0 if not bad_dates else 1


if __name__ == "__main__":
    sys.exit(main())
