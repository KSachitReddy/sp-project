"""
Extended evaluation of the trained rainfall forecaster (models/forecast_model.txt)
on the same val/test rows as weather/forecast_model.train() -- no retraining,
no change to splits or features.

    .venv/bin/python scripts/evaluate_forecast.py

Writes models/forecast_eval_report.json with, per split and horizon: MAE/RMSE,
skill vs climatology on BOTH MSE (the number in forecast_model_metrics.json)
and MAE, skill vs persistence, bias, correlation; plus rain-event metrics
(POD/FAR/CSI/F1/ROC-AUC at 1, 2.5, 10 mm) for model and climatology, season
and region breakdowns, bootstrap CIs, and feature importance.

Rows are rebuilt one year at a time on the training cell subsample (peak RAM
~4 GB instead of loading all 14.5M rows at once) and the script first checks
that it reproduces forecast_model_metrics.json exactly.
"""
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from weather import features as feat
from weather import forecast_model as fm
from weather.satellite_features import region_for

REPORT_PATH = fm.MODELS_DIR / "forecast_eval_report.json"
THRESHOLDS_MM = [1, 2.5, 10]
SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJAS": [6, 7, 8, 9], "ON": [10, 11]}
N_BOOTSTRAP = 1000


def build_rows():
    """Same rows as features.build_dataset(), restricted to val + test years
    (plus train years for the train-vs-test gap), built per year to save RAM."""
    frames = []
    for year in feat.available_years():
        df = pd.read_csv(feat.RAINFALL_DIR / f"rainfall_{year}.csv", parse_dates=["date"])
        frames.append(feat.select_training_cells(df))  # every year has the same cells, so same subsample
    sub = pd.concat(frames).sort_values(["latitude", "longitude", "date"]).reset_index(drop=True)
    sub["rain_anchor_day"] = sub["rainfall_mm"]  # persistence baseline; not a model feature
    sub = feat.add_lag_features(sub)
    sub = feat.add_time_features(sub)
    sub = feat.expand_horizons(sub)
    sub = feat.add_climatology(sub, pd.read_csv(fm.CLIMATOLOGY_PATH), date_col="target_date")
    sub = sub.dropna(subset=feat.FEATURE_COLS + ["target_mm"]).reset_index(drop=True)
    year = sub["date"].dt.year
    sub["split"] = np.select(
        [year.isin(feat.TRAIN_YEARS), year.isin(feat.VAL_YEARS), year.isin(feat.TEST_YEARS)],
        ["train", "val", "test"], "other",
    )
    return sub


def _mse_skill(y, p, ref):
    return float(1 - np.mean((y - p) ** 2) / np.mean((y - ref) ** 2))


def continuous_metrics(d):
    y, p, c = d["target_mm"].to_numpy(), d["pred"].to_numpy(), d["climatology_mm"].to_numpy()
    pers = d["rain_anchor_day"].to_numpy()
    return {
        "n": int(len(d)),
        "mae": round(float(np.mean(np.abs(y - p))), 3),
        "rmse": round(float(np.sqrt(np.mean((y - p) ** 2))), 3),
        "climatology_mae": round(float(np.mean(np.abs(y - c))), 3),
        "climatology_rmse": round(float(np.sqrt(np.mean((y - c) ** 2))), 3),
        "skill_vs_climatology_mse": round(_mse_skill(y, p, c), 4),
        "skill_vs_climatology_mae": round(float(1 - np.mean(np.abs(y - p)) / np.mean(np.abs(y - c))), 4),
        "skill_vs_persistence_mse": round(_mse_skill(y, p, pers), 4),
        "bias_mm": round(float(np.mean(d["pred_clipped"].to_numpy() - y)), 3),
        "corr_model": round(float(np.corrcoef(y, p)[0, 1]), 4),
        "corr_climatology": round(float(np.corrcoef(y, c)[0, 1]), 4),
    }


def event_metrics(d):
    out = {}
    for thr in THRESHOLDS_MM:
        obs = d["target_mm"].to_numpy() >= thr
        per_source = {"base_rate": round(float(obs.mean()), 4)}
        for name, score in (("model", d["pred_clipped"].to_numpy()), ("climatology", d["climatology_mm"].to_numpy())):
            fc = score >= thr
            hit, miss, fa = int((fc & obs).sum()), int((~fc & obs).sum()), int((fc & ~obs).sum())
            per_source[name] = {
                "pod": round(hit / (hit + miss), 4),
                "far": round(fa / (hit + fa), 4) if hit + fa else None,
                "csi": round(hit / (hit + miss + fa), 4),
                "f1": round(2 * hit / (2 * hit + miss + fa), 4),
                "roc_auc": round(float(roc_auc_score(obs, score)), 4),
            }
        out[f">={thr}mm"] = per_source
    return out


def heavy_rain_calibration(d):
    out = {}
    for lo, hi in [(0, 0.1), (0.1, 2.5), (2.5, 10), (10, 50), (50, None)]:
        m = d["target_mm"] >= lo
        if hi is not None:
            m &= d["target_mm"] < hi
        out[f"obs_{lo}_to_{hi if hi is not None else 'inf'}mm"] = {
            "n": int(m.sum()),
            "mean_obs": round(float(d.loc[m, "target_mm"].mean()), 2),
            "mean_pred": round(float(d.loc[m, "pred_clipped"].mean()), 2),
            "mean_climatology": round(float(d.loc[m, "climatology_mm"].mean()), 2),
        }
    out["frac_target_zero"] = round(float((d["target_mm"] == 0).mean()), 4)
    out["frac_pred_below_0.1mm"] = round(float((d["pred_clipped"] < 0.1).mean()), 4)
    out["max_pred"] = round(float(d["pred_clipped"].max()), 1)
    out["max_obs"] = round(float(d["target_mm"].max()), 1)
    return out


def bootstrap_skill_ci(d, rng):
    """Resamples whole anchor dates (all cells for a date move together, which
    keeps the strong spatial correlation). Consecutive dates are still treated as
    independent, so day-to-day autocorrelation makes these CIs somewhat too narrow."""
    g = (
        d.assign(se=(d["target_mm"] - d["pred"]) ** 2, sc=(d["target_mm"] - d["climatology_mm"]) ** 2)
        .groupby("date")[["se", "sc"]].sum().to_numpy()
    )
    samples = []
    for _ in range(N_BOOTSTRAP):
        s = g[rng.integers(0, len(g), len(g))]
        samples.append(1 - s[:, 0].sum() / s[:, 1].sum())
    samples = np.array(samples)
    return {
        "point": round(float(1 - g[:, 0].sum() / g[:, 1].sum()), 4),
        "ci95": [round(float(np.percentile(samples, 2.5)), 4), round(float(np.percentile(samples, 97.5)), 4)],
        "frac_resamples_le_zero": round(float(np.mean(samples <= 0)), 4),
        "n_dates": int(len(g)),
    }


def main():
    t0 = time.time()
    print("Rebuilding val/test rows from data/rainfall/*.csv (about a minute)...")
    rows = build_rows()
    booster = lgb.Booster(model_file=str(fm.MODEL_PATH))
    rows["pred"] = booster.predict(rows[feat.FEATURE_COLS])
    rows["pred_clipped"] = rows["pred"].clip(lower=0)  # what predict_rainfall() actually serves

    # Sanity check: reproduce the committed metrics before reporting anything new.
    committed = json.loads(fm.METRICS_PATH.read_text())["by_split"]
    recomputed = fm._evaluate(booster, {
        "val_2023": rows[rows["split"] == "val"],
        "test_2024_2025": rows[rows["split"] == "test"],
    })
    if recomputed != committed:
        sys.exit("Recomputed metrics differ from forecast_model_metrics.json -- refusing to write a report.")

    report = {
        "reproduces_forecast_model_metrics_json": True,
        "notes": {
            "skill_definitions": "skill = 1 - err_model/err_reference. *_mse uses squared error (the "
                                 "number in forecast_model_metrics.json); *_mae uses absolute error.",
            "persistence": "persistence = rainfall on the anchor day (the newest observation; "
                           "not itself a model feature -- the model's newest input is lag_1).",
            "bias_and_events": "computed on predictions clipped at 0, as served by predict_rainfall().",
            "bootstrap": "date-block bootstrap; ignores day-to-day autocorrelation, so CIs are "
                         "somewhat too narrow.",
        },
        "by_split": {},
    }

    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        d_split = rows[rows["split"] == split]
        report["by_split"][split] = {
            str(h): continuous_metrics(d_split[d_split["horizon_days"] == h]) for h in feat.HORIZONS_DAYS
        }

    test = rows[rows["split"] == "test"].copy()
    cells = test[["latitude", "longitude"]].drop_duplicates()
    cells["region"] = [region_for(la, lo) for la, lo in zip(cells["latitude"], cells["longitude"])]
    test = test.merge(cells, on=["latitude", "longitude"])
    month = test["target_date"].dt.month
    test["season"] = "ON"
    for name, months in SEASONS.items():
        test.loc[month.isin(months), "season"] = name

    report["test"] = {}
    for h in feat.HORIZONS_DAYS:
        d = test[test["horizon_days"] == h]
        report["test"][str(h)] = {
            "events": event_metrics(d),
            "by_season": {s: continuous_metrics(g) for s, g in d.groupby("season")},
            "by_region": {r: continuous_metrics(g) for r, g in d.groupby("region")},
            "bootstrap_skill_vs_climatology_mse": bootstrap_skill_ci(d, rng),
        }
    report["test"]["1"]["heavy_rain_calibration"] = heavy_rain_calibration(test[test["horizon_days"] == 1])

    v = rows[rows["split"] == "val"]
    report["split_boundary"] = {
        "val_rows_with_target_in_test_years": int(v["target_date"].dt.year.isin(feat.TEST_YEARS).sum()),
        "val_rows": int(len(v)),
    }
    gain = booster.feature_importance("gain")
    report["feature_importance_gain"] = {
        f: round(float(g), 4) for f, g in sorted(zip(booster.feature_name(), gain / gain.sum()), key=lambda x: -x[1])
    }
    report["runtime_secs"] = round(time.time() - t0, 1)
    report["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    h1 = report["by_split"]["test"]["1"]
    ev = report["test"]["1"]["events"]
    print(f"Test h1: MSE skill {h1['skill_vs_climatology_mse']}, MAE skill {h1['skill_vs_climatology_mae']}, "
          f"CSI>=10mm model {ev['>=10mm']['model']['csi']} vs climatology {ev['>=10mm']['climatology']['csi']}")
    print(f"Report written to {REPORT_PATH} ({report['runtime_secs']}s, peak {report['peak_rss_mb']} MB)")


if __name__ == "__main__":
    main()
