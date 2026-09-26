"""Shared filesystem paths for the lowlevel/ simulation and calibration code.

Import these instead of hardcoding absolute paths, so scripts work on any
machine the repo is cloned onto.
"""

from pathlib import Path

LOWLEVEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = LOWLEVEL_DIR.parent
SO_ARM100_DIR = REPO_ROOT / "SO-ARM100" / "Simulation" / "SO101"
URDF_PATH = SO_ARM100_DIR / "so101_new_calib.urdf"
ASSETS_DIR = SO_ARM100_DIR / "assets"
MESH_DIR = SO_ARM100_DIR
REAL_WORLD_RUNS_DIR = LOWLEVEL_DIR / "real_world_runs"
