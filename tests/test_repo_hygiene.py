"""Repo-level guarantees from the simplicity pass: license, interpreter pin,
tracked thermal features, and the removed one-off 2023 extractor staying gone."""
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _tracked(path):
    return subprocess.run(["git", "ls-files", "--error-unmatch", path], cwd=ROOT,
                          capture_output=True).returncode == 0


def test_mit_license():
    text = (ROOT / "LICENSE").read_text()
    assert text.startswith("MIT License") and "Copyright (c) 2026 Kommera Sachit Reddy" in text


def test_python_version_pin_satisfies_requires_python():
    pinned = tuple(int(x) for x in (ROOT / ".python-version").read_text().strip().split("."))
    minimum = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["requires-python"]
    assert minimum == ">=3.12" and pinned >= (3, 12)


def test_thermal_features_are_tracked():
    assert _tracked("data/satellite/thermal_features.csv")


def test_removed_2023_extractor_is_gone_and_unreferenced():
    for gone in ("scripts/extract.py", "data/rainfall/rainfall_2023.json"):
        assert not (ROOT / gone).exists()
    hits = subprocess.run(["git", "grep", "-l", "-e", "rainfall_2023.json", "-e", "scripts/extract.py",
                           "--", ".", ":!docs/CHANGELOG_FINAL.md", ":!tests/test_repo_hygiene.py"],
                          cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert hits == []


def test_readme_lists_every_script_and_the_sync_push_flag():
    import re
    readme = (ROOT / "README.md").read_text()
    listed = set(re.findall(r"^\| `([a-z0-9_]+\.py)` \|", readme, flags=re.M))
    on_disk = {p.name for p in (ROOT / "scripts").glob("*.py")}
    assert listed == on_disk and len(on_disk) == 16
    assert "nimbus sync --push" in readme and "uv run pytest -q" in readme
