import inspect
from lerobot.model.kinematics import RobotKinematics
import numpy as np
import sys

# Path to your downloaded URDF
# URDF_PATH = "./SO101/so101_new_calib.urdf"
# URDF_PATH = "/Users/zhg603/Documents/OXAI/so101_new_calib.urdf"
URDF_PATH = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"

# Joint names in the SAME order as the robot motors
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Create kinematics solver
kinematics = RobotKinematics(
    urdf_path=URDF_PATH,
    # target_frame_name="gripper_frame_link",
    target_frame_name="gripper_link",
    # target_frame_name="gripper",
    joint_names=JOINT_NAMES,
)



def move_smooth(target, steps=10, delay=0.05):
    obs = follower.get_observation()
    current = {k: obs[k] for k in target}

    for i in range(steps):
        alpha = (i + 1) / steps
        action = obs.copy()

        for k in target:
            action[k] = current[k] + alpha * (target[k] - current[k])

        follower.send_action(action)
        time.sleep(delay)


def vectodic(vec):
    action = {
    f"{name}.pos": angle
    for name, angle in zip(JOINT_NAMES, vec)
    }
    return(action)




import sys
import time
import math
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

config = SO101FollowerConfig(
    port="/dev/tty.usbmodem5AB90659861",
    # port="/dev/cu.usbmodem5B141140961",
    id="my_awesome_follower_arm"
)
follower = SO101Follower(config)
follower.connect()
# follower.setup_motors()

obs = follower.get_observation()
print("Observation:", obs)
# current_joints = obs

def obstovec(obs):
    vec = np.array([
        obs["shoulder_pan.pos"],
        obs["shoulder_lift.pos"],
        obs["elbow_flex.pos"],
        obs["wrist_flex.pos"],
        obs["wrist_roll.pos"],
        obs["gripper.pos"],
    ])
    return(vec)


def targetcoords(target_pose):
    obs = follower.get_observation()
    current_joints = obstovec(obs)
    # print("Current joints:", current_joints)


    #initialise at current positions
    joint_solution = current_joints

    for _ in range(4):
        joint_solution = kinematics.inverse_kinematics(
            joint_solution,
            target_pose,
            position_weight=100.0,
            orientation_weight=0.0,
        )


    return(joint_solution)



### Example of reasonable target positions
obs = follower.get_observation()
current_joints = obstovec(obs)
current_fk = kinematics.forward_kinematics(current_joints)
target_pose = current_fk.copy()
target_pose[1,3] += 0.02


###Solve for joints to take us there
joint_solution = targetcoords(target_pose)

print("\nJoint solution:")
for name, angle in zip(JOINT_NAMES, joint_solution):
    print(f"{name}: {angle:.3f}")



# ---------------------------
# Forward Kinematics Check
# ---------------------------

fk_pose = kinematics.forward_kinematics(joint_solution)

print("\nFK reconstructed pose:")
print(fk_pose)

# ---------------------------
# Position error
# ---------------------------

target_position = target_pose[:3, 3]
fk_position = fk_pose[:3, 3]

position_error = np.linalg.norm(
    target_position - fk_position
)

print(f"\nPosition error: {position_error:.6f} m")


# action = {
#     f"{name}.pos": angle
#     for name, angle in zip(JOINT_NAMES, joint_solution)
# }

action = vectodic(joint_solution)

# print(action)
# sys.exit()



# move_smooth(action)




#### 4 corners
corner1 = [97.32,0.40,28.40,66.55,177.80,4.95]
corner2 = [38.59,60.88,-58.55,100.48,178.15,4.95]
corner3 = [-52.48,57.98,-56.26,96.26,172.70,4.95]
corner4 = [-108.75, 26.33, 0.79, 81.85, 172.88, 1.4]




corner1ac = vectodic(corner1)
corner2ac = vectodic(corner2)
corner3ac = vectodic(corner3)
corner4ac = vectodic(corner4)

def cornercoords(corner):
    fk_corner = kinematics.forward_kinematics(np.array(corner))
    coords = fk_corner[:3,3]
    return(coords)

coords1 = cornercoords(corner1)
coords2 = cornercoords(corner2)
coords3 = cornercoords(corner3)
coords4 = cornercoords(corner4)


print("Coords 1:")
print(coords1)
print("Coords 2:")
print(coords2)
print("Coords 3:")
print(coords3)
print("Coords 4:")
print(coords4)




testcorners = True

if testcorners:

    move_smooth(corner1ac)
    time.sleep(5)
    move_smooth(corner2ac)
    time.sleep(5)
    move_smooth(corner3ac)
    time.sleep(5)
    move_smooth(corner4ac)



