"""
Compare the satellite nowcaster's predictions against real IMD rainfall for
2025 (the most recent year with both satellite imagery and rainfall data --
never seen during training), then generate its predictions for 2026 from
satellite imagery alone, since IMD hasn't released 2026 rainfall data yet.

    .venv/bin/python scripts/generate_nowcast_report.py

Writes:
  data/forecast/nowcast_validation_2025.csv  -- predicted vs actual, per day/region, 2025
  data/forecast/nowcast_2026.csv              -- predicted only, per day/region, 2026
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from weather import features as feat
from weather import satellite_features as sfeat
from weather.nowcast_model import model_available, predict_nowcast

OUT_DIR = Path(__file__).parent.parent / "data" / "forecast"


def _actual_regional_rainfall():
    regional = sfeat.regional_daily_rainfall(feat.load_rainfall())
    regional.index = pd.to_datetime(regional.index)
    return regional


def validate_2025(regional):
    rows = []
    for date_str in sfeat.available_image_dates():
        if not date_str.startswith("2025"):
            continue
        result = predict_nowcast(as_of_date=date_str)
        if "error" in result:
            continue
        predicted_date = pd.Timestamp(result["predicted_date"])
        if predicted_date not in regional.index:
            continue
        actual_row = regional.loc[predicted_date]
        for region, predicted_mm in result["predicted_mm_by_region"].items():
            actual_mm = actual_row[region]
            if pd.isna(actual_mm):
                continue
            rows.append({
                "date": str(predicted_date.date()),
                "region": region,
                "predicted_mm": predicted_mm,
                "actual_mm": round(float(actual_mm), 2),
                "abs_error_mm": round(abs(predicted_mm - float(actual_mm)), 2),
            })

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "nowcast_validation_2025.csv"
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Overall MAE: {df['abs_error_mm'].mean():.2f} mm")
    print("Per-region MAE:")
    print(df.groupby("region")["abs_error_mm"].mean().round(2).to_string())
    return df


def predict_2026():
    rows = []
    for date_str in sfeat.available_image_dates():
        if not date_str.startswith("2026"):
            continue
        result = predict_nowcast(as_of_date=date_str)
        if "error" in result:
            continue
        for region, predicted_mm in result["predicted_mm_by_region"].items():
            rows.append({
                "image_date": result["image_date"],
                "predicted_date": result["predicted_date"],
                "region": region,
                "predicted_mm": predicted_mm,
            })

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "nowcast_2026.csv"
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Date range: {df['predicted_date'].min()} to {df['predicted_date'].max()}")
    return df


def main():
    if not model_available():
        print("Nowcasting model not trained yet -- run scripts/train_nowcast_model.py first.")
        sys.exit(1)

    regional = _actual_regional_rainfall()

    print("=== Validating against real 2025 rainfall (model never trained on this year) ===")
    validate_2025(regional)
    print()
    print("=== Predicting 2026 from satellite imagery (no IMD rainfall data exists yet) ===")
    predict_2026()


if __name__ == "__main__":
    main()
