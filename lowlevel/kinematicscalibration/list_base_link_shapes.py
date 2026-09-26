from pathlib import Path
import sys
import pybullet as p
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
URDF_PATH = str(paths.URDF_PATH)
p.connect(p.DIRECT)
robot_id = p.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
for entry in p.getVisualShapeData(robot_id):
    if entry[1] == -1:
        print(entry)
