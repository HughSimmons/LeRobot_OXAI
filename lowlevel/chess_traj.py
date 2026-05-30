import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import time
import numpy as np

from testkinematics import relativexyz, vectodic

def smoothmove(pos1, pos2):
    for alpha in np.linspace(0,1,50):

        q = (1-alpha)*np.array(pos1) + alpha*np.array(pos2)

        viz.display(np.deg2rad(q))
        time.sleep(0.01)

def simxyz_old(currentjnt, direcxyz):
    cartcorner1 = relativexyz(np.array(currentjnt), np.array(direcxyz))
    return(cartcorner1)

def simxyz(currentjnt, direcxyz):
    mag = np.linalg.norm(direcxyz)
    if mag > 0.05:
        nsteps = int(mag / 0.05) + 1
        step = direcxyz / nsteps
        current = currentjnt.copy()
        for _ in range(nsteps):
            current = relativexyz(current, step)
        return current
    else:
        cartcorner1 = relativexyz(np.array(currentjnt), np.array(direcxyz))
        return(cartcorner1)


def xyz_homeref_old(xyzcoords, refjnts):
    refxyz = kinematics.forward_kinematics(refjnts)[:3,3]
    refrot = kinematics.forward_kinematics(refjnts)[:3,:3]

    rot_error = abs(-1 -refrot[2, 2])

    direcxyz = xyzcoords - refxyz


    mag = np.linalg.norm(direcxyz) + np.linalg.norm(rot_error)
    min_step = 0.02
    if mag > min_step:
        nsteps = int(mag / min_step) + 1
        step = direcxyz / nsteps
        current = refjnts.copy()
        for _ in range(nsteps):
            current = relativexyz(current, step)
        return current
    else:
        cartcorner1 = relativexyz(np.array(refjnts), np.array(direcxyz))
        return(cartcorner1)

# Offset from gripper frame origin to desired grasp point
# Tune this experimentally
# from testkinematics import GRASP_OFFSET


def xyz_homeref(xyzcoords, refjnts, GRASP_OFFSET):

    # ----------------------------------------
    # Current FK pose
    # ----------------------------------------

    refpose = kinematics.forward_kinematics(refjnts)

    refxyz = refpose[:3,3]
    refrot = refpose[:3,:3]

    # ----------------------------------------
    # Current grasp point in world coordinates
    # ----------------------------------------

    current_grasp_xyz = (
        refxyz
        + refrot @ GRASP_OFFSET
    )

    # ----------------------------------------
    # Cartesian error relative to grasp point
    # ----------------------------------------

    direcxyz = xyzcoords - current_grasp_xyz

    # ----------------------------------------
    # Keep gripper approximately downward
    # ----------------------------------------

    rot_error = abs(-1 - refrot[2,2])

    mag = (
        np.linalg.norm(direcxyz)
        + rot_error
    )

    min_step = 0.02

    # ----------------------------------------
    # Multi-step interpolation
    # ----------------------------------------

    if mag > min_step:

        nsteps = int(mag / min_step) + 1

        step = direcxyz / nsteps

        current = refjnts.copy()

        for _ in range(nsteps):

            current = relativexyz(
                current,
                step,
                GRASP_OFFSET
            )

        return current

    # ----------------------------------------
    # Single-step move
    # ----------------------------------------

    else:

        return relativexyz(
            np.array(refjnts),
            np.array(direcxyz), 
            GRASP_OFFSET
        )

# def xyz_homeref(xyzcoords, refjnts):

#     current = refjnts.copy()

#     min_step = 0.02
#     max_iters = 50

#     for _ in range(max_iters):

#         # ----------------------------------------
#         # Current FK pose
#         # ----------------------------------------

#         pose = kinematics.forward_kinematics(current)

#         current_xyz = pose[:3,3]
#         current_rot = pose[:3,:3]

#         # ----------------------------------------
#         # Current TCP / grasp point
#         # ----------------------------------------

#         current_grasp_xyz = (
#             current_xyz
#             + current_rot @ GRASP_OFFSET
#         )

#         # ----------------------------------------
#         # TCP error
#         # ----------------------------------------

#         direcxyz = xyzcoords - current_grasp_xyz

#         error = np.linalg.norm(direcxyz)

#         # ----------------------------------------
#         # Converged
#         # ----------------------------------------

#         if error < 1e-3:
#             break

#         # ----------------------------------------
#         # Step toward target
#         # ----------------------------------------

#         step_mag = min(error, min_step)

#         step = (
#             direcxyz / error
#         ) * step_mag

#         current = relativexyz(
#             current,
#             step
#         )

#     return current

def chess_to_xy_old(square):
    """
    Convert chess square like 'e4'
    into board coordinates.
    """

    file = FILES.index(square[0].lower())
    rank = int(square[1]) - 1

    return np.array([
        # file * SQUARE_SIZE +0.1,
        file * SQUARE_SIZE +0.3,
        rank * SQUARE_SIZE +0.1,
        0
    ])

def chess_to_xy(square, board_origin=(0.25, 0, 0), square_size=0.04):
    file = FILES.index(square[0].lower())
    rank = int(square[1]) - 1
    board_x, board_y, board_z = board_origin
    board_size = 8 * square_size
    # x = board_x - board_size/2 + (file + 0.5) * square_size
    y = board_y - board_size/2 + (rank + 0.5) * square_size
    x = board_x - board_size/2 + (file + 0.5) * square_size
    # y = board_y - board_size/2 + (rank + 0.85) * square_size
    z = board_z + 0.04  # slightly above the board
    # z = board_z + 0.06  # slightly above the board
    # z = board_z + 0.1  # slightly above the board
    return np.array([x, y, z])

def move_to_squareold(current_joints, from_square, to_square):

    from_xyz = chess_to_xy(from_square)
    to_xyz = chess_to_xy(to_square)

    delta = to_xyz - from_xyz

    current = current_joints.copy()

    steps = 10

    for i in range(steps):

        step_delta = delta / steps

        new = simxyz(current, step_delta)

        smoothmove(current, new)

        current = new

    return current

def move_to_square_v1(current_joints, from_square, to_square):

    from_xyz = chess_to_xy(from_square)
    to_xyz = chess_to_xy(to_square)

    delta = to_xyz - from_xyz

    current = current_joints.copy()

    # ---------------------------
    # 0. Close gripper
    # ---------------------------

    grip_closed = current.copy()
    grip_closed[5] = 0.0   # adjust as needed

    smoothmove(current, grip_closed)

    current = grip_closed.copy()

    # ---------------------------
    # 1. Move vertically 
    # ---------------------------

    lift = np.array([0, 0, 0.10])

    new = simxyz(current, lift)

    smoothmove(current, new)

    current = new.copy()

    # ---------------------------
    # 2. Move across board
    # ---------------------------

    steps = 10

    for _ in range(steps):

        step_delta = delta / steps

        new = simxyz(current, step_delta)

        smoothmove(current, new)

        current = new.copy()

    # ---------------------------
    # 3. Move vertically down
    # ---------------------------

    lower = np.array([0, 0, -0.10])

    new = simxyz(current, lower)

    smoothmove(current, new)

    current = new.copy()

    # ---------------------------
    # 4. Open gripper
    # ---------------------------

    grip_open = current.copy()
    grip_open[5] = 5.0   # adjust as needed

    smoothmove(current, grip_open)

    current = grip_open.copy()


    lift = np.array([0, 0, 0.10])

    new = simxyz(current, lift)

    smoothmove(current, new)

    return current

def move_to_square_v2(current_joints, from_square, to_square):
    """
    Move from current_joints to home, then from home to from_square, pick up piece,
    move to to_square, place piece, and return to home. Returns the final joint position (home).
    """
    global home
    # 1. Move to home
    smoothmove(current_joints, home)
    current = home.copy()

    # 2. Move to from_square (above)
    from_xyz = chess_to_xy(from_square)
    above_from = simxyz(current, from_xyz + np.array([0, 0, 0.10]))
    smoothmove(current, above_from)
    current = above_from.copy()
    # Lower to from_square
    at_from = simxyz(current, np.array([0, 0, -0.10]))
    smoothmove(current, at_from)
    current = at_from.copy()
    # Close gripper
    grip_closed = current.copy()
    grip_closed[5] = 0.0  # adjust as needed
    smoothmove(current, grip_closed)
    current = grip_closed.copy()
    # Lift piece
    lifted = simxyz(current, np.array([0, 0, 0.10]))
    smoothmove(current, lifted)
    current = lifted.copy()

    # 3. Move to to_square (above)
    to_xyz = chess_to_xy(to_square)
    move_delta = to_xyz - from_xyz
    above_to = simxyz(current, move_delta)
    smoothmove(current, above_to)
    current = above_to.copy()
    # Lower to to_square
    at_to = simxyz(current, np.array([0, 0, -0.10]))
    smoothmove(current, at_to)
    current = at_to.copy()
    # Open gripper
    grip_open = current.copy()
    grip_open[5] = 5.0  # adjust as needed
    smoothmove(current, grip_open)
    current = grip_open.copy()
    # Lift up
    lifted = simxyz(current, np.array([0, 0, 0.10]))
    smoothmove(current, lifted)
    current = lifted.copy()

    # 4. Return to home
    smoothmove(current, home)
    current = home.copy()

    return current



urdf_path = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
mesh_dir = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO101"

FILES = "abcdefgh"

SQUARE_SIZE = 0.04   # 4 cm example

corner1 = [97.32,0.40,28.40,66.55,177.80,4.95]
corner2 = [38.59,60.88,-58.55,100.48,178.15,4.95]
corner3 = [-52.48,57.98,-56.26,96.26,172.70,4.95]
corner4 = [-108.75, 26.33, 0.79, 81.85, 172.88, 1.4]

home = np.array([96.92307692307692,  -107.86813186813187,  97.36263736263736, 65.18681318681318,  -29.846153846153847,  4.62962962962963])

from testkinematics import kinematics
homexyz = kinematics.forward_kinematics(home)[:3,3]

def pickupmove_traj(from_square, to_square, board_origin, GRASP_OFFSET):
    """
    Move from current_joints to home, then from home to from_square, pick up piece,
    move to to_square, place piece, and return to home. Returns the final joint position (home).
    """
    global home
    # 1. Move to home
    height = 0.13  # height to lift above squares
    # height = 0.11  # height to lift above squares

    gripper_angle_open = 25
    gripper_angle_closed = 5

    jntslist = []

    current = home.copy()

    # 2. Move to from_square (above)
    from_xyz = chess_to_xy(from_square, board_origin=board_origin)
    above_from = xyz_homeref(from_xyz + np.array([0, 0, height]), current, GRASP_OFFSET)
    above_from[5] = gripper_angle_open  # keep gripper open
    # smoothmove(current, above_from)
    jntslist.append(above_from)


    current = above_from.copy()
    # Lower to from_square
    at_from = xyz_homeref(from_xyz, current, GRASP_OFFSET)
    at_from[5] = gripper_angle_open  # keep gripper open
    # smoothmove(current, at_from)
    jntslist.append(at_from)


    current = at_from.copy()
    # Close gripper
    grip_closed = current.copy()
    grip_closed[5] = gripper_angle_closed  # adjust as needed
    # smoothmove(current, grip_closed)
    jntslist.append(grip_closed)


    current = grip_closed.copy()
    # Lift piece
    lifted = xyz_homeref(from_xyz+np.array([0, 0, height]), current, GRASP_OFFSET)
    lifted[5] = gripper_angle_closed  # keep gripper closed
    #   smoothmove(current, lifted)
    jntslist.append(lifted)


    current = lifted.copy()
    # 3. Move to to_square (above)
    to_xyz = chess_to_xy(to_square, board_origin=board_origin)
    above_to = xyz_homeref(to_xyz + np.array([0, 0, height]), current, GRASP_OFFSET)
    above_to[5] = gripper_angle_closed  # keep gripper closed
    # smoothmove(current, above_to)
    jntslist.append(above_to)

    current = above_to.copy()
    # Lower to to_square
    at_to = xyz_homeref(to_xyz, current, GRASP_OFFSET)
    jntslist.append(at_to)

    #close gripper
    at_to_grip = at_to.copy()
    at_to_grip[5] = gripper_angle_closed  # keep gripper closed
    # smoothmove(current, at_to)
    jntslist.append(at_to_grip)

    current = at_to.copy()
    # Open gripper
    grip_open = current.copy()
    grip_open[5] = gripper_angle_open  # adjust as needed
    # smoothmove(current, grip_open)
    jntslist.append(grip_open)

    current = grip_open.copy()
    # Lift up
    lifted = xyz_homeref(to_xyz + np.array([0, 0, height]), current, GRASP_OFFSET)
    # smoothmove(current, lifted)
    jntslist.append(lifted)
 
    current = lifted.copy()

    # 4. Return to home
    # smoothmove(current, home)
    jntslist.append(home)
    current = home.copy()

    return jntslist




if __name__ == "__main__":

    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        urdf_path,
        mesh_dir
    )

    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    q = pin.neutral(model)


    corner1 = np.array(corner1)

    viz.display(corner1)
    time.sleep(2)

    smoothmove(corner1, corner2)

    time.sleep(3)
    # move_to_square_v2(corner2, "e1", "c5")

    smoothmove(corner2, home)

    time.sleep(5)

    move_to_square_v2(home, "a1", "a5")

