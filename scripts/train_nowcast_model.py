"""
Train the satellite nowcasting CNN (see weather/nowcast_model.py and
weather/satellite_features.py) and print its backtested accuracy.

    .venv/bin/python scripts/train_nowcast_model.py

Writes models/nowcast_model.pt and models/nowcast_model_metrics.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from weather.nowcast_model import train


def main():
    print("Building the image/label dataset (first run decodes ~2,500 PNGs, cached after that)...")
    _, metrics = train()
    print()
    print("=== Backtested accuracy ===")
    print(json.dumps(metrics, indent=2))
    print()
    print(f"Model + metrics written to {Path(__file__).parent.parent / 'models'}")


if __name__ == "__main__":
    main()
