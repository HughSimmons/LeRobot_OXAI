"""Resolve project paths across the two ``lowlevel/paths.py`` variants.

``find_piece`` ships a full ``paths.py`` (``SO101_ASSETS``, ``resolve_urdf()``,
``make_so101_follower_config()``...), while Oliver's ``oliver/two-camera-capture`` branch ships a
trimmed one (``ASSETS_DIR``, ``URDF_PATH``, ``REAL_WORLD_RUNS_DIR``, ``SO101_PORT``,
``SO101_ID``, ``connect_so101_hardware(calibrate, configure)``). The planner only needs a handful of
facts from either, so this module asks for them by whichever name exists.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
from typing import Any

LOWLEVEL_DIR = Path(__file__).resolve().parents[1]
if str(LOWLEVEL_DIR) not in sys.path:
    sys.path.insert(0, str(LOWLEVEL_DIR))

import paths  # noqa: E402

REPO_ROOT = LOWLEVEL_DIR.parent
FRICTION_FIT_RELATIVE = Path("friction") / "friction_sag_fit.json"


def assets_dir() -> Path:
    """Directory holding the SO-101 STL meshes (``.../SO-ARM100/Simulation/SO101/assets``)."""
    candidates: list[Path] = []
    for name in ("SO101_ASSETS", "ASSETS_DIR"):
        value = getattr(paths, name, None)
        if value is not None:
            candidates.append(Path(value))
    candidates.append(REPO_ROOT / "SO-ARM100" / "Simulation" / "SO101" / "assets")
    for cand in candidates:
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        "SO-101 STL assets directory not found. Tried: " + ", ".join(str(c) for c in candidates)
        + ". Is the SO-ARM100 submodule checked out?"
    )


def standard_urdf() -> Path:
    """Standard SO-101 URDF path (``paths.resolve_urdf()`` or ``paths.URDF_PATH``)."""
    resolver = getattr(paths, "resolve_urdf", None)
    if callable(resolver):
        urdf, _mesh_dir = resolver()
        return Path(urdf)
    urdf_path = getattr(paths, "URDF_PATH", None)
    if urdf_path is not None and Path(urdf_path).is_file():
        return Path(urdf_path)
    raise FileNotFoundError("No SO-101 URDF: paths.resolve_urdf() and paths.URDF_PATH are both unavailable")


def real_world_runs_dir() -> Path:
    """Where run folders are written (``paths.REAL_WORLD_RUNS_DIR``)."""
    value = getattr(paths, "REAL_WORLD_RUNS_DIR", None)
    return Path(value) if value is not None else LOWLEVEL_DIR / "real_world_runs"


def default_friction_fit() -> Path | None:
    """Fitted friction/sag JSON if it exists on this machine, else ``None``."""
    cand = real_world_runs_dir() / FRICTION_FIT_RELATIVE
    return cand if cand.is_file() else None


def so101_port_and_id() -> tuple[str | None, str | None]:
    """Serial port and robot id as ``paths`` knows them (``SO101_PORT`` / ``SO101_ID``), if any."""
    import os

    port = os.environ.get("SO101_PORT", getattr(paths, "SO101_PORT", None))
    robot_id = os.environ.get("SO101_ID", getattr(paths, "SO101_ID", None))
    return port, robot_id


def connect_follower(*, calibrate: bool = True, configure: bool | None = None) -> Any:
    """Connect the SO-101 follower through ``paths.connect_so101_hardware``.

    ``configure`` is forwarded only if that function accepts it (Oliver's does; it controls whether
    lerobot's ``configure()`` cycles torque off/on during connect).
    """
    connector = getattr(paths, "connect_so101_hardware", None)
    if connector is None:
        raise RuntimeError("paths.connect_so101_hardware is missing; cannot connect the SO-101 follower")
    kwargs: dict[str, Any] = {"calibrate": calibrate}
    if configure is not None and "configure" in inspect.signature(connector).parameters:
        kwargs["configure"] = configure
    return connector(**kwargs)
