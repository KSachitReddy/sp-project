"""A rate-limited (partial) elevation download must never shrink an existing
complete data/parameters/elevation.csv -- found in final QA, where HTTP 429s
from the public Open-Elevation API cut the file from 4,964 cells to 500."""
import importlib.util
from pathlib import Path

import pandas as pd
import requests

SCRIPT = Path(__file__).parent.parent / "scripts" / "download_elevation.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("download_elevation", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_partial_download_keeps_existing_cells(tmp_path, monkeypatch):
    mod = _load_script()
    out = tmp_path / "elevation.csv"
    cells = [(8.0 + i * 0.25, 77.0) for i in range(4)]
    pd.DataFrame(
        [{"latitude": la, "longitude": lo, "elevation_m": 1.0} for la, lo in cells]
    ).to_csv(out, index=False)

    def fake_fetch(batch):
        if batch[0] == cells[0]:  # first batch succeeds with a new value...
            return [{"latitude": la, "longitude": lo, "elevation": 42.0} for la, lo in batch]
        raise requests.exceptions.HTTPError("429 Client Error: Too Many Requests")  # ...the rest fail

    monkeypatch.setattr(mod, "_valid_cells", lambda: cells)
    monkeypatch.setattr(mod, "_fetch_batch", fake_fetch)
    monkeypatch.setattr(mod, "OUT_PATH", out)
    monkeypatch.setattr(mod, "BATCH_SIZE", 2)
    monkeypatch.setattr(mod, "REQUEST_DELAY_SEC", 0)

    mod.main()

    df = pd.read_csv(out)
    assert len(df) == 4, "failed batches must not drop cells already on disk"
    fresh = df[df.latitude.isin([cells[0][0], cells[1][0]])]
    stale = df[df.latitude.isin([cells[2][0], cells[3][0]])]
    assert (fresh.elevation_m == 42.0).all()
    assert (stale.elevation_m == 1.0).all()
