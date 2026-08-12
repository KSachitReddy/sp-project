"""
Re-attempt only the dates listed in data/satellite/logs/retry_dates.txt
(produced by verify_satellite_images.py), using the same fallback-layer logic
as download_satellite_images.py.

Existing files for those dates (if any — e.g. a corrupt/truncated PNG that
failed verification) are removed first so the date is always re-fetched
rather than skipped.
"""
import csv
import sys

import download_satellite_images as dsi

RETRY_PATH = dsi.LOG_DIR / "retry_dates.txt"


def main():
    if not RETRY_PATH.exists():
        print(f"No {RETRY_PATH} found — run verify_satellite_images.py first.")
        return 1

    dates = [line.strip() for line in RETRY_PATH.read_text().splitlines() if line.strip()]
    if not dates:
        print("retry_dates.txt is empty — nothing to retry.")
        return 0

    dsi.LOG_DIR.mkdir(parents=True, exist_ok=True)
    total = len(dates)
    recovered = 0
    still_failed = []

    with open(dsi.LOG_CSV, "a", newline="") as log_file, open(dsi.ERROR_LOG, "a") as error_file:
        log_writer = csv.writer(log_file)

        for i, date_str in enumerate(dates, start=1):
            year = date_str[:4]
            out_path = dsi.IMG_DIR / year / f"{date_str}.png"
            if out_path.exists():
                out_path.unlink()

            try:
                content, result = dsi.fetch_day(date_str, log_writer, log_file)
            except Exception as e:
                content, result = None, f"unexpected error ({e})"

            if content is not None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(content)
                recovered += 1
                print(f"Retried {i}/{total} - {date_str}.png ({result})")
            else:
                still_failed.append(date_str)
                error_file.write(f"{date_str}: retry failed - {result}\n")
                error_file.flush()
                print(f"RETRY FAILED {i}/{total} - {date_str} ({result})")

    print()
    print(f"Retry complete: {recovered}/{total} recovered, {len(still_failed)} still failing.")
    return 0 if not still_failed else 1


if __name__ == "__main__":
    sys.exit(main())
