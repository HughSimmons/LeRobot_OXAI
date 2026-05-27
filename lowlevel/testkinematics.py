import inspect
from lerobot.model.kinematics import RobotKinematics
import numpy as np
import sys


robotconnected = False

# GRASP_OFFSET = np.array([
#     0.0,     # lateral
#     0.01,   # forward between jaws
#     -0.01    # slightly downward
# ])

GRASP_OFFSET = np.array([
    # 0.0,
    -0.025,
    -0.0,
    -0.005
])

# GRASP_OFFSET = np.array([
#     -0.01876807,
#     -0.00805934,
#      -0.01
# ])

# [-0.01193487 -0.02224436  0.02015081]

# Path to your downloaded URDF
# URDF_PATH = "./SO101/so101_new_calib.urdf"
# URDF_PATH = "/Users/zhg603/Documents/OXAI/so101_new_calib.urdf"
URDF_PATH = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
# URDF_PATH = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO100/so100.urdf"

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
    target_frame_name="gripper_frame_link", #works
    # target_frame_name="jaw",
    # target_frame_name="gripper_link",
    # target_frame_name="moving_jaw_so101_v1_link",
    # target_frame_name="gripper",
    joint_names=JOINT_NAMES,
)



def move_smooth(target, steps=10, delay=0.1):
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
    # baudrate=115200,
    # port="/dev/cu.usbmodem5B141140961",
    id="my_awesome_follower_arm"
)
follower = SO101Follower(config)

# follower.bus.port_handler.baudrate = 115200

if robotconnected:
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


def targetcoords(initjnts, target_pose):
    # if robotconnected:
    #     obs = follower.get_observation()
    # else:
    #     obs = homeposition

    # current_joints = obstovec(initjnts)
    current_joints = initjnts
    # print("Current joints:", current_joints)


    #initialise at current positions
    joint_solution = current_joints

    for _ in range(4):
        joint_solution = kinematics.inverse_kinematics(
            joint_solution,
            target_pose,
            position_weight=10.0,
            orientation_weight=0.01,
        )


    return(joint_solution)



homeposition = {'shoulder_pan.pos': 96.92307692307692, 'shoulder_lift.pos': -107.86813186813187, 'elbow_flex.pos': 97.36263736263736, 'wrist_flex.pos': 65.18681318681318, 'wrist_roll.pos': -29.846153846153847, 'gripper.pos': 4.62962962962963}




### Example of reasonable target positions


if robotconnected:
    obs = follower.get_observation()
else:
    obs = homeposition


current_joints = obstovec(obs)
current_fk = kinematics.forward_kinematics(current_joints)
target_pose = current_fk.copy()
target_pose[1,3] += 0.02


###Solve for joints to take us there
# joint_solution = targetcoords(homeposition, target_pose)

print("\nJoint solution:")
# for name, angle in zip(JOINT_NAMES, joint_solution):
#     print(f"{name}: {angle:.3f}")



# ---------------------------
# Forward Kinematics Check
# ---------------------------

# fk_pose = kinematics.forward_kinematics(joint_solution)

# print("\nFK reconstructed pose:")
# print(fk_pose)

# # ---------------------------
# # Position error
# # ---------------------------

# target_position = target_pose[:3, 3]
# fk_position = fk_pose[:3, 3]

# position_error = np.linalg.norm(
#     target_position - fk_position
# )

# print(f"\nPosition error: {position_error:.6f} m")


# action = {
#     f"{name}.pos": angle
#     for name, angle in zip(JOINT_NAMES, joint_solution)
# }

# action = vectodic(joint_solution)

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

if robotconnected:
    move_smooth(corner2ac)
    time.sleep(4)

# sys.exit()

def relativexyz(initjnts, changexyz): 
    fk_init = kinematics.forward_kinematics(initjnts) 
    newpose = fk_init.copy() 

    downorient = True

    if downorient:
        down_orientation = np.array([
            [1, 0,  0],
            [0, 1,  0],
            [0, 0, -1]
        ])
        newpose[2, 2] = -1


    newpose[:3,3] += changexyz 
    # newpose[1,3] += 0.03 
    newjoints = targetcoords(initjnts, newpose) 
    newac = vectodic(newjoints) 

    # print(newac)
    # fk_pose = kinematics.forward_kinematics(newjoints) 
    # # print("new pose") # print(newpose) # print("calculated pose") # print(fk_pose)
    # target_position = newpose[:3, 3]
    # fk_position = fk_init[:3, 3]

    # position_error = np.linalg.norm(target_position - fk_position)
    # print("Position Error: ", position_error)


    target_position = newpose[:3, 3]

    fk_pose = kinematics.forward_kinematics(newjoints)
    fk_position = fk_pose[:3, 3]

    # Gripper/TCP position in world coordinates
    fk_grasp_position = (
        fk_pose[:3,3]
        + fk_pose[:3,:3] @ GRASP_OFFSET
    )

    position_error = np.linalg.norm(
        target_position - fk_grasp_position
    )    

    #when optimising for gripper position directly
    # position_error = np.linalg.norm(
    #     target_position - fk_position
    # )

    if position_error > 0.01:
        print("Position Error:", position_error)



    # move_smooth(newac)
    return(newjoints)

# direc = np.array([0,0.01,0])


# c12 = coords1 - coords2
c12 = coords2 - coords1


c23 = coords3 - coords2


mag = np.linalg.norm(c12)
# direc = 0.01*c12/mag
# direc = c12/10

direc = c23/8

# direc = np.array([0,0,0.03])

# direc = c12
if __name__ == "__main__":
    sys.exit()

    # relativexyz(corner1, direc)
    for _ in range(4):
        if robotconnected:
            update = follower.get_observation()
            updatevec = obstovec(update)
            relativexyz(updatevec, direc)
            time.sleep(2)



    if False:
        ### move up in z
        for _ in range(5):
            update = follower.get_observation()
            updatevec = obstovec(update)
            relativexyz(updatevec, direc)
            time.sleep(2)


        direc = np.array([0.03,0,0])
        ### move up in y
        for _ in range(5):
            update = follower.get_observation()
            updatevec = obstovec(update)
            relativexyz(updatevec, direc)
            time.sleep(2)


    if robotconnected:
        # move_smooth(corner1ac)
        time.sleep(3)
        move_smooth(corner3ac)
        time.sleep(3)


        # np.set_printoptions(precision=3, suppress=True)
        # print(kinematics.forward_kinematics(np.zeros(6)))

        # print(kinematics.forward_kinematics([90,0,0,0,0,0]))
        # print(kinematics.forward_kinematics(np.deg2rad([90,0,0,0,0,0])))

        move_smooth(homeposition)
        follower.disconnect()



    sys.exit()

    print(newjoints)

    sys.exit()

    testcorners = True

    if testcorners:

        move_smooth(corner1ac)

        sys.exit()
        time.sleep(5)
        move_smooth(corner2ac)
        time.sleep(5)
        move_smooth(corner3ac)
        time.sleep(5)
        move_smooth(corner4ac)



