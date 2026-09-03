import pybullet as p
import numpy as np

URDF_PATH = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
p.connect(p.DIRECT)
robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
link_index_by_name = {}
for j in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, j)
    link_index_by_name[info[12].decode("utf-8")] = j
idx = link_index_by_name["moving_jaw_so101_v1_link"]
print("collision shape data:", p.getCollisionShapeData(robot_id, idx))
print("visual shape data:", p.getVisualShapeData(robot_id)[idx] if idx < len(p.getVisualShapeData(robot_id)) else None)
