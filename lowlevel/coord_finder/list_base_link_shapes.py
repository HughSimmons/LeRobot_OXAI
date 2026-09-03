import pybullet as p
URDF_PATH = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
p.connect(p.DIRECT)
robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
for entry in p.getVisualShapeData(robot_id):
    if entry[1] == -1:
        print(entry)
