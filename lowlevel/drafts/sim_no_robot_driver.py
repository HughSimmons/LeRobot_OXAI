"""Keep simulation workers from importing the real SO101 robot driver.

testkinematics.py (imported by chess_traj) does
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    follower = SO101Follower(config)
at import time, even with robotconnected = False. That import chain loads torch, which costs
about 640 MB of RAM and ~3 s per worker process and is never used by the simulation.

install() puts a stand-in `lerobot.robots.so_follower` module in sys.modules *before*
chess_traj is imported. The stand-in classes accept the same constructor arguments; any
attempt to actually talk to a robot (connect, get_observation, send_action, ...) raises.
Kinematics (lerobot.model.kinematics / placo) is untouched, so trajectories and physics are
identical. Call it only in simulation-only processes.
"""

from __future__ import annotations

from pathlib import Path
import sys
import types


class _NoRobotConfig:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs


class _NoRobot:
    def __init__(self, config=None, *args, **kwargs):
        self.config = config

    def __getattr__(self, name):
        def refuse(*args, **kwargs):
            raise RuntimeError(
                f"SO101Follower.{name}() called in a simulation-only process "
                "(sim_no_robot_driver stub is installed)"
            )

        return refuse


def skip_ik_meshes() -> None:
    """Build placo robots from the URDF with its <visual>/<collision> mesh blocks removed.

    lerobot's RobotKinematics creates placo.RobotWrapper(urdf_path), which loads every arm mesh
    into memory (~440 MB per process; Flags.ignore_collisions does not avoid it). FK/IK only
    need the joints, links and frames, so the geometry is never used. Must run before
    testkinematics/chess_traj is imported. Verified to give identical trajectories
    (lowlevel/drafts/check_sim_light_imports.py).
    """
    import re

    import placo

    if getattr(placo.RobotWrapper, "_strips_meshes", False):
        return
    original = placo.RobotWrapper

    def robot_wrapper(path, flags=0, urdf_content=""):
        if not urdf_content and str(path).endswith(".urdf"):
            text = Path(path).read_text(encoding="utf-8")
            urdf_content = re.sub(r"<(visual|collision)\b.*?</\1>", "", text, flags=re.S)
        return original(path, flags, urdf_content)

    robot_wrapper._strips_meshes = True
    placo.RobotWrapper = robot_wrapper


def install_light_sim_imports() -> None:
    """Both savings: no robot driver/torch (~200 MB, ~5 s import), no IK meshes (~440 MB)."""
    install()
    skip_ik_meshes()


def install() -> None:
    # Python imports the parent package (lerobot/robots/__init__.py) before the submodule, and
    # that parent is what pulls in motors -> utils -> torch, so both are replaced.
    if "lerobot.robots" in sys.modules and not getattr(sys.modules["lerobot.robots"], "_is_sim_stub", False):
        return  # the real robot package is already loaded; leave it alone
    import lerobot  # the real top-level package (light); only .robots is replaced

    package = types.ModuleType("lerobot.robots")
    package._is_sim_stub = True
    package.__path__ = []  # a package with no files: submodules resolve only via sys.modules
    driver = types.ModuleType("lerobot.robots.so_follower")
    driver._is_sim_stub = True
    driver.SO101Follower = _NoRobot
    driver.SO101FollowerConfig = _NoRobotConfig
    package.so_follower = driver
    sys.modules["lerobot.robots"] = package
    sys.modules["lerobot.robots.so_follower"] = driver
    lerobot.robots = package
