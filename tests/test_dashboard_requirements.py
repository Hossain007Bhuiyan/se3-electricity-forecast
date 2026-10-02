# Checks that dashboard/requirements.txt (used by Streamlit Community Cloud) lists exactly the
# package versions in uv.lock. If a package is upgraded in uv.lock but the file is not regenerated,
# this test fails and shows the command that fixes it.

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ["uv", "export", "--only-group", "dashboard", "--no-hashes", "--no-emit-project", "--frozen"]


# Only the package lines, without uv's comment lines (which also name the command that wrote the file)
def packages(text):
    return [line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def test_dashboard_requirements_match_uv_lock():
    expected = subprocess.run(EXPORT, cwd=ROOT, capture_output=True, text=True, check=True).stdout
    actual = (ROOT / "dashboard" / "requirements.txt").read_text()
    assert packages(actual) == packages(expected), (
        "dashboard/requirements.txt is out of date. Regenerate it with:\n"
        "uv export --only-group dashboard --no-hashes --no-emit-project --frozen -o dashboard/requirements.txt"
    )