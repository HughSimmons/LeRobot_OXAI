#!/usr/bin/env python3
"""Print FK xyz for saved current_pose_*.json readings."""

import json
import sys
from pathlib import Path

import numpy as np

from testkinematics import kinematics

LOWLEVEL_DIR = Path(__file__).resolve().parent


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]] or sorted(
        (LOWLEVEL_DIR / "real_world_runs").glob("current_pose*.json")
    )
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        joints_deg = np.array(data["joints_deg"], dtype=float)
        pose = kinematics.forward_kinematics(joints_deg)
        xyz = pose[:3, 3]
        print(f"{path.name:24s} xyz=[{xyz[0]:.5f}, {xyz[1]:.5f}, {xyz[2]:.5f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
