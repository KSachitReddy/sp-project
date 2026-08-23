"""
Train the rainfall forecasting model (see weather/forecast_model.py and
weather/features.py) and print its backtested accuracy.

    .venv/bin/python scripts/train_forecast_model.py

Writes models/forecast_model.txt (the LightGBM booster),
models/climatology.csv (per-cell/day-of-year climatology, needed at
inference), and models/forecast_model_metrics.json (the metrics also
printed below).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from weather.forecast_model import train


def main():
    print("Building features and training (this reads all of data/rainfall/*.csv)...")
    _, metrics = train()
    print()
    print("=== Backtested accuracy ===")
    print(json.dumps(metrics, indent=2))
    print()
    print(f"Model + metrics written to {Path(__file__).parent.parent / 'models'}")


if __name__ == "__main__":
    main()
