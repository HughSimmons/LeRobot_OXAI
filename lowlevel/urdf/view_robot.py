import time
import pybullet as p
import pybullet_data

URDF = "so101_with_mount.urdf"

p.connect(p.GUI, options="--opengl2")                                      # open a window
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # so it can find plane.urdf
p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")                                # the floor
robot = p.loadURDF(URDF, [0, 0, 0], useFixedBase=True) # bolt the base down at the origin

# list every joint: index, name, whether it moves
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    print(i, info[1].decode(), "fixed" if info[2] == p.JOINT_FIXED else "moves")

# one slider per moving joint, limited to that joint's range from the URDF
movable = [i for i in range(p.getNumJoints(robot)) if p.getJointInfo(robot, i)[2] != p.JOINT_FIXED]
sliders = []
for i in movable:
    info = p.getJointInfo(robot, i)
    sliders.append(p.addUserDebugParameter(info[1].decode(), info[8], info[9], 0))

while True:
    for i, s in zip(movable, sliders):
        target = p.readUserDebugParameter(s)
        p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, targetPosition=target, force=50)
    p.stepSimulation()
    time.sleep(1 / 240)                                 # run at roughly real-time speed
