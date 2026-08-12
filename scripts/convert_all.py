"""
Batch-convert every .nc file in data/nc/ into data/rainfall/rainfall_<year>.csv
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
from rainfall.nc_convert import convert_nc_to_csv

NC_DIR = BASE_DIR / "data" / "nc"
OUT_DIR = BASE_DIR / "data" / "rainfall"


def main():
    nc_files = sorted(NC_DIR.glob("*.nc"))
    if not nc_files:
        print(f"No .nc files found in {NC_DIR}")
        sys.exit(1)

    for nc_path in nc_files:
        print(f"Converting {nc_path.name} ...")
        csv_path, payload = convert_nc_to_csv(str(nc_path), str(OUT_DIR))
        meta = payload["meta"]
        print(
            f"  -> {csv_path} "
            f"({meta['n_days']} days x {meta['n_cells']} cells)"
        )


if __name__ == "__main__":
    main()
