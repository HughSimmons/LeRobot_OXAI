"""Sanity checks for lowlevel/paths.py -- catches a repo that was moved/cloned
somewhere the SO-ARM100 assets can't be found before any real script hits it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paths


def test_urdf_path_exists():
    assert paths.URDF_PATH.exists(), paths.URDF_PATH


def test_assets_dir_exists():
    assert paths.ASSETS_DIR.is_dir(), paths.ASSETS_DIR


def test_real_world_runs_dir_exists():
    assert paths.REAL_WORLD_RUNS_DIR.is_dir(), paths.REAL_WORLD_RUNS_DIR


def test_repo_root_is_parent_of_lowlevel():
    assert paths.REPO_ROOT == paths.LOWLEVEL_DIR.parent
