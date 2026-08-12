"""
Local server for the India Rainfall map.

- Serves the interactive map page.
- GET  /api/years        lists the years available in "nc files/" for the drawer.
- GET  /api/year/<year>  loads one of those years onto the map.
- POST /api/convert      validates an uploaded .nc file, saves it into
                          "nc files/" and its CSV into "outputs/", then
                          loads it onto the map.
"""
import os
import re
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from rainfall.nc_convert import NcFormatError, build_payload, save_uploaded_nc

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "rainfall"
NC_DIR = BASE_DIR / "data" / "nc"
WEB_DIR = BASE_DIR / "web"
MAP_PAGE = "india_rainfall_2023.html"

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB upload cap

YEAR_RE = re.compile(r"(19|20)\d{2}")


def _find_nc_for_year(year):
    matches = sorted(NC_DIR.glob(f"*{year}*.nc"))
    return matches[0] if matches else None


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, MAP_PAGE)


@app.route("/api/years")
def api_years():
    years = set()
    for p in NC_DIR.glob("*.nc"):
        m = YEAR_RE.search(p.stem)
        if m:
            years.add(int(m.group(0)))
    return jsonify({"years": sorted(years)})


@app.route("/api/year/<int:year>")
def api_year(year):
    nc_path = _find_nc_for_year(year)
    if nc_path is None:
        return jsonify({"error": f"No local .nc file found for {year}."}), 404

    try:
        payload = build_payload(str(nc_path))
    except NcFormatError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        app.logger.exception("Failed to load %s", nc_path)
        return jsonify({"error": "Could not read that file."}), 500

    return jsonify(payload)


@app.route("/api/convert", methods=["POST"])
def api_convert():
    uploaded = request.files.get("ncfile")
    if uploaded is None or uploaded.filename == "":
        return jsonify({"error": "No file uploaded."}), 400
    if not uploaded.filename.lower().endswith(".nc"):
        return jsonify({"error": "Please upload a .nc NetCDF file."}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            uploaded.save(tmp.name)
            tmp_path = tmp.name

        nc_path, csv_path, payload = save_uploaded_nc(tmp_path, str(NC_DIR), str(DATA_DIR))
    except NcFormatError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        app.logger.exception("Failed to parse uploaded .nc file")
        return jsonify({"error": "Could not read that file — is it a valid NetCDF (.nc) file?"}), 400
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    payload["nc_filename"] = os.path.basename(nc_path)
    payload["csv_filename"] = os.path.basename(csv_path)
    return jsonify(payload)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
