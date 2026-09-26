"""Regression guard for lowlevel/legacy/ and lowlevel/coord_finder/legacy/.

These scripts are archived, not maintained -- the only bar is that they
(a) still parse as valid Python (a move didn't truncate/corrupt one) and
(b) don't carry a hardcoded path back to whoever cleaned this repo up.
Correctness of their internals is out of scope; several are hardware-shaped
by default and are intentionally not executed here.
"""

import ast
from pathlib import Path

LOWLEVEL_DIR = Path(__file__).resolve().parent.parent
LEGACY_DIRS = [
    LOWLEVEL_DIR / "legacy",
    LOWLEVEL_DIR / "coord_finder" / "legacy",
]


def _legacy_py_files():
    for legacy_dir in LEGACY_DIRS:
        yield from legacy_dir.rglob("*.py")


def test_legacy_files_exist():
    assert list(_legacy_py_files()), "expected legacy/ directories to contain .py files"


def test_legacy_files_parse():
    for f in _legacy_py_files():
        ast.parse(f.read_text(), filename=str(f))


def test_no_hardcoded_personal_paths():
    offenders = [f for f in _legacy_py_files() if "/Users/" in f.read_text()]
    assert not offenders, f"hardcoded /Users/... path(s) found in: {offenders}"
