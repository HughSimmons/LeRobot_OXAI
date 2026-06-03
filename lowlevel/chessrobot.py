import inspect
from lerobot.model.kinematics import RobotKinematics
# from lowlevel.chess_traj import pickupmove_traj
import numpy as np
import sys


robotconnected = True

# GRASP_OFFSET = np.array([
#     0.0,     # lateral
#     0.01,   # forward between jaws
#     -0.01    # slightly downward
# ])

# GRASP_OFFSET = np.array([.   ###starting point
#     # 0.0,
#     -0.025,
#     -0.0,
#     -0.005
# ])

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

        # alpha = 0.5 * (
        #     1 - np.cos(
        #         np.pi * (i + 1) / steps
        #     )
        # )
        action = obs.copy()

        for k in target:
            action[k] = current[k] + alpha * (target[k] - current[k])
            # if k==5 and action[k]>7.5:
            #     action[k] = 7.5

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

    # follower.disconnect()
    # sys.exit()
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





homeposition = {'shoulder_pan.pos': 96.92307692307692, 'shoulder_lift.pos': -107.86813186813187, 'elbow_flex.pos': 97.36263736263736, 'wrist_flex.pos': 65.18681318681318, 'wrist_roll.pos': -29.846153846153847, 'gripper.pos': 4.62962962962963}


if robotconnected:
    obs = follower.get_observation()
else:
    obs = homeposition






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

# if robotconnected:
#     move_smooth(corner2ac)
#     time.sleep(4)

# sys.exit()

# def relativexyz(initjnts, changexyz, GRASP_OFFSET, downflag=False): 
#     fk_init = kinematics.forward_kinematics(initjnts) 
#     newpose = fk_init.copy() 

#     downorient = True

#     if downorient:
#         down_orientation = np.array([
#             [1, 0,  0],
#             [0, 1,  0],
#             [0, 0, -1]
#         ])
#         # newpose[2, 2] = -1
#         if downflag:
#             newpose[2, 2] = -1


#     newpose[:3,3] += changexyz 
#     # newpose[1,3] += 0.03 
#     newjoints = targetcoords(initjnts, newpose) 

#     target_position = newpose[:3, 3]

#     fk_pose = kinematics.forward_kinematics(newjoints)
#     # fk_position = fk_pose[:3, 3]

#     # Gripper/TCP position in world coordinates
#     fk_grasp_position = (
#         fk_pose[:3,3]
#         + fk_pose[:3,:3] @ GRASP_OFFSET
#     )

#     position_error = np.linalg.norm(
#         target_position - fk_grasp_position
#     )    

#     #when optimising for gripper position directly
#     # position_error = np.linalg.norm(
#     #     target_position - fk_position
#     # )

#     if position_error > 0.01:
#         print("Position Error:", position_error)



#     # move_smooth(newac)
#     return(newjoints)
target =  {'shoulder_pan.pos': 37.142857142857146, 'shoulder_lift.pos': 60.83516483516483, 'elbow_flex.pos': -56.21978021978022, 'wrist_flex.pos': 101.23076923076923, 'wrist_roll.pos': -30.10989010989011, 'gripper.pos': 4.695767195767195}

# direc = c12
if __name__ == "__main__":
    # board_origin = (0.25, 0, 0)  # Must match the origin used in pybsim_chess.py

    from chess_traj import pickupmove_traj
    # movelist = pickupmove_traj('c1', 'c5', board_origin=board_origin, GRASP_OFFSET=np.array([0,0,0]))  # Example move from e2 to e4
    # movelist = pickupmove_traj('a1', 'a5', board_origin=board_origin, GRASP_OFFSET=np.array([0,0,0]))  # Example move from e2 to e4
    # movelist = pickupmove_traj("e1", "c5", board_origin=(0.25, 0, 0), GRASP_OFFSET=np.array([0,0,0]))

    GRASP_OFFSET = np.array([
        -0.015,
        -0.0,
        -0.005
    ])
    # movelist, idx = pickupmove_traj("a1", "h8", board_origin=(0.25, 0, 0), GRASP_OFFSET=GRASP_OFFSET)
    movelist, idx = pickupmove_traj("b1", "b5", board_origin=(0.3, 0, 0), GRASP_OFFSET=GRASP_OFFSET)
    # movelist, idx = pickupmove_traj("e5", "b1", board_origin=(0.3, 0, 0), GRASP_OFFSET=GRASP_OFFSET)
    # movelist, idx = pickupmove_traj("e5", "c5", board_origin=(0.25, 0, 0), GRASP_OFFSET=GRASP_OFFSET)


    from scipy.interpolate import CubicSpline
    import numpy as np

    # Apply your gripper tweaks first
    traj = []

    closedcnt = 0

    for move in movelist:

        moverl = move - np.array([0,0,-10,0,0,0])

        if moverl[5] == 8:
            closedcnt += 1

            if closedcnt > 3:
                moverl[5] = 7.5

        traj.append(moverl)

    traj = np.array(traj)

    # One time value per waypoint
    t = np.arange(len(traj))

    # Cubic spline for each joint
    splines = [
        CubicSpline(t, traj[:,j], bc_type="natural")
        for j in range(6)
    ]

    # 100 interpolated points between each waypoint
    samples_per_segment = 100

    t_dense = np.linspace(
        0,
        len(traj)-1,
        (len(traj)-1)*samples_per_segment
    )

    traj_dense = np.column_stack([
        spline(t_dense)
        for spline in splines
    ])

    traj_dense[:,5] = np.clip(
        traj_dense[:,5],
        6,
        25
    )

    diffgrip = np.diff(traj_dense[:,5])

    open_indices = np.where(diffgrip > 5)[0]


    # for q in traj_dense:

    #     action = vectodic(q)

    #     follower.send_action(action)

    #     time.sleep(0.01)

    gripper_angle_open = 25
    open_indices = set(open_indices)

    for i, q in enumerate(traj_dense):

        action = vectodic(q)

        follower.send_action(action)

        # Opening transition detected
        if i in open_indices:

            print("Opening gripper...")

            while True:

                obs = follower.get_observation()

                actual = obs["gripper.pos"]

                if actual >= gripper_angle_open - 1:
                    break

                time.sleep(0.02)

            # optional settling time
            time.sleep(0.2)

        time.sleep(0.01)

    # movecnt = 0
    # closedcnt = 0
    # for move in movelist:

    #     moverl = move - np.array([0,0,-10,0,0,0])

    #     if moverl[5] == 8:
    #         closedcnt += 1
    #         if closedcnt>3:
    #             moverl[5] = 7.5

    #     moverl = vectodic(moverl)
    #     print(moverl)
    #     move_smooth(moverl, 100,0.01)



        # movecnt += 1
        # if movecnt == 2:
        #     time.sleep(10)

        # time.sleep(0.02)


    # move_smooth(homeposition)


    follower.disconnect()
    sys.exit()

# {'shoulder_pan.pos': 5.582417582417582, 'shoulder_lift.pos': 2.4175824175824174, 'elbow_flex.pos': -83.38461538461539, 'wrist_flex.pos': 1.3186813186813187, 'wrist_roll.pos': -86.98901098901099, 'gripper.pos': 5.053191489361701}
    # relativexyz(corner1, direc)
    for _ in range(4):
        if robotconnected:
            update = follower.get_observation()
            updatevec = obstovec(update)
            relativexyz(updatevec, direc, base_grasp_offset, downflag=False)
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



