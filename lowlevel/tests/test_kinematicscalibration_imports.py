"""Smoke test for lowlevel/kinematicscalibration/ after the coord_finder split.

Several of these scripts run PyBullet setup at module scope (no
``if __name__ == "__main__":`` guard), so each is run as its own subprocess
rather than imported in-process -- that also matches how a teammate would
actually invoke them, and keeps one script's PyBullet DIRECT-mode connection
from leaking into the next.

Requires an environment with pybullet, numpy, and Pillow installed (see
lowlevel/kinematicscalibration/requirements.txt), plus the vendored lerobot
package importable (e.g. PYTHONPATH=lowlevel/lerobot/src) -- none of these
scripts touch real hardware, so running them directly is always safe.
"""

import subprocess
import sys
from pathlib import Path

import pytest

KINEMATICSCALIBRATION_DIR = Path(__file__).resolve().parent.parent / "kinematicscalibration"
SCRIPTS = sorted(KINEMATICSCALIBRATION_DIR.glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_runs_cleanly_from_new_home(script):
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=KINEMATICSCALIBRATION_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"{script.name} exited {result.returncode}\n--- stderr ---\n{result.stderr}"
    )
