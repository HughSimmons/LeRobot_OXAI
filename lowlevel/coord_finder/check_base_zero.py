import struct
from pathlib import Path
import numpy as np
import pybullet as p

URDF_PATH = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"

p.connect(p.DIRECT)
robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
p.resetJointState(robot_id, 0, 0.0)
aabb_min, aabb_max = p.getAABB(robot_id, -1)  # base_link (base of URDF, link index -1)
print("base_link AABB (world, robot at origin):", aabb_min, aabb_max)
print("lowest point of base_link relative to base_link origin (0,0,0): z =", aabb_min[2])
p.disconnect()
