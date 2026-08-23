"""
Small CNN for next-day regional rainfall nowcasting from satellite imagery.
See weather/satellite_features.py for region definitions and dataset
building (today's image -> tomorrow's mean regional rainfall mm).

Baselines evaluated alongside the CNN, since a nowcasting model isn't useful
unless it beats them: persistence (predict tomorrow = today, per region) and
climatology (historical average for that day-of-year, per region) -- the two
standard nowcasting baselines.

Unlike forecast_model.predict_rainfall(), which is stuck anchored to the last
date IMD's gridded product covers, predict_nowcast() can genuinely run
"live": it just needs today's satellite image, and GIBS imagery is
near-real-time (unlike IMD rainfall, which lags by roughly a year). It still
can't be scored against real rainfall until IMD's gridded data catches up --
but it can produce an honest prediction today, not just in backtests.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from weather import features as feat
from weather import satellite_features as sfeat

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "nowcast_model.pt"
METRICS_PATH = MODELS_DIR / "nowcast_model_metrics.json"

TRAIN_YEARS = range(2019, 2024)  # 2019-2023 (no imagery in 2018)
VAL_YEARS = (2024,)
TEST_YEARS = (2025,)


class NowcastCNN(nn.Module):
    def __init__(self, n_regions=len(sfeat.REGIONS)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        w, h = sfeat.IMAGE_SIZE
        flat = 32 * (h // 8) * (w // 8)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, 64), nn.ReLU(), nn.Linear(64, n_regions),
        )

    def forward(self, x):
        return self.head(self.conv(x))


def _regional_climatology(regional, train_years):
    train = regional[regional.index.year.isin(train_years)]
    doy = feat.doy_365(pd.Series(train.index))
    return train.groupby(doy)[sfeat.REGIONS].mean()


def _evaluate(model, X, Y, dates, regional, climatology):
    model.eval()
    with torch.no_grad():
        pred_log = model(torch.tensor(X)).numpy()
    pred = np.clip(np.expm1(pred_log), 0, None)

    persistence = np.full_like(Y, np.nan)
    for i, d in enumerate(dates):
        if d in regional.index:
            persistence[i] = regional.loc[d].to_numpy()
    valid = ~np.isnan(persistence).any(axis=1)

    target_doy = feat.doy_365(pd.Series(dates[valid] + pd.Timedelta(days=1)))
    clim_pred = climatology.reindex(target_doy).to_numpy()

    Yv, Pv, Persv = Y[valid], pred[valid], persistence[valid]

    result = {}
    for i, region in enumerate(sfeat.REGIONS):
        y, p, pers, cl = Yv[:, i], Pv[:, i], Persv[:, i], clim_pred[:, i]
        mse = float(np.mean((y - p) ** 2))
        pers_mse = float(np.mean((y - pers) ** 2))
        clim_mse = float(np.mean((y - cl) ** 2))
        result[region] = {
            "n": int(len(y)),
            "mae": round(float(np.mean(np.abs(y - p))), 3),
            "rmse": round(float(np.sqrt(mse)), 3),
            "persistence_rmse": round(float(np.sqrt(pers_mse)), 3),
            "skill_vs_persistence": round(float(1 - mse / pers_mse), 4) if pers_mse > 0 else None,
            "climatology_rmse": round(float(np.sqrt(clim_mse)), 3),
            "skill_vs_climatology": round(float(1 - mse / clim_mse), 4) if clim_mse > 0 else None,
        }
    return result


def train(epochs=40, batch_size=64, lr=1e-3, patience=8, seed=0):
    torch.manual_seed(seed)
    rainfall_df = feat.load_rainfall()
    X, Y, dates = sfeat.build_dataset(rainfall_df)

    years = dates.year
    train_mask = np.isin(years, list(TRAIN_YEARS))
    val_mask = np.isin(years, list(VAL_YEARS))
    test_mask = np.isin(years, list(TEST_YEARS))

    # log1p stabilizes training against rainfall's heavy right skew (mostly
    # near-zero, occasional monsoon bursts); undone with expm1 at eval/inference.
    Y_log = np.log1p(np.clip(Y, 0, None))
    Xtr, Ytr = torch.tensor(X[train_mask]), torch.tensor(Y_log[train_mask])
    Xval, Yval = torch.tensor(X[val_mask]), torch.tensor(Y_log[val_mask])

    model = NowcastCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val, best_state, bad_epochs = float("inf"), None, 0
    n = len(Xtr)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xval), Yval).item()
        print(f"epoch {epoch + 1}/{epochs}  train_mse(log)={total_loss / n:.4f}  val_mse(log)={val_loss:.4f}")

        if val_loss < best_val - 1e-5:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)

    regional = sfeat.regional_daily_rainfall(rainfall_df)
    regional.index = pd.to_datetime(regional.index)
    climatology = _regional_climatology(regional, TRAIN_YEARS)

    metrics = {
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "by_split": {
            "val_2024": _evaluate(model, X[val_mask], Y[val_mask], dates[val_mask], regional, climatology),
            "test_2025": _evaluate(model, X[test_mask], Y[test_mask], dates[test_mask], regional, climatology),
        },
    }

    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return model, metrics


_model = None


def _load_model():
    global _model
    if _model is None:
        _model = NowcastCNN()
        _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        _model.eval()
    return _model


def model_available():
    return MODEL_PATH.exists()


def predict_nowcast(as_of_date=None):
    """Predicts tomorrow's mean regional rainfall (mm) from the most recent
    satellite image on disk at or before as_of_date (default: today). Looks
    back up to a week for a downloaded image, since GIBS/the downloader may
    be a day or two behind "today". Returns {"error": ...} if there's no
    trained model or no recent image."""
    if not model_available():
        return {"error": "Nowcasting model not trained yet -- run scripts/train_nowcast_model.py."}

    as_of = pd.Timestamp(as_of_date).normalize() if as_of_date else pd.Timestamp.today().normalize()

    image_date = None
    for back in range(8):
        candidate = as_of - pd.Timedelta(days=back)
        if (sfeat.SATELLITE_DIR / str(candidate.date())[:4] / f"{candidate.date()}.png").exists():
            image_date = candidate
            break
    if image_date is None:
        return {
            "error": (
                f"No satellite image on disk within a week of {as_of.date()} -- run "
                f"scripts/download_satellite_images.py to fetch recent imagery."
            )
        }

    img = sfeat.load_image(str(image_date.date()))
    model = _load_model()
    with torch.no_grad():
        pred_log = model(torch.tensor(img).unsqueeze(0)).numpy()[0]
    pred = np.clip(np.expm1(pred_log), 0, None)

    return {
        "image_date": str(image_date.date()),
        "predicted_date": str((image_date + pd.Timedelta(days=1)).date()),
        "predicted_mm_by_region": {r: round(float(v), 2) for r, v in zip(sfeat.REGIONS, pred)},
    }
