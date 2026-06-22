import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import time
import numpy as np

from testkinematics import relativexyz, relativexyz_with_error, vectodic

B_FILE_PLACE_DOWNFLAG = True

def smoothmove(pos1, pos2):
    for alpha in np.linspace(0,1,50):

        q = (1-alpha)*np.array(pos1) + alpha*np.array(pos2)

        viz.display(np.deg2rad(q))
        time.sleep(0.01)

def smoothjnts(pos1, pos2):
    smoothlist = []
    for alpha in np.linspace(0,1,10):

        q = (1-alpha)*np.array(pos1) + alpha*np.array(pos2)

        smoothlist.append(q)

    return smoothlist

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


def xyz_homeref(xyzcoords, refjnts, GRASP_OFFSET, downflag=True):

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

    if downflag:
        mag = np.linalg.norm(direcxyz) + rot_error
    else:
        mag = np.linalg.norm(direcxyz)

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
                GRASP_OFFSET,
                downflag
            )

        return current

    # ----------------------------------------
    # Single-step move
    # ----------------------------------------

    else:

        return relativexyz(
            np.array(refjnts),
            np.array(direcxyz), 
            GRASP_OFFSET, 
            downflag
        )


def record_fk_error(traj_metrics, stage, target_xyz, fk_error):
    if traj_metrics is None:
        return

    traj_metrics["max_fk_error"] = max(
        traj_metrics["max_fk_error"],
        float(fk_error)
    )

    if fk_error > traj_metrics["event_threshold"]:
        traj_metrics["fk_error_events"].append({
            "stage": stage,
            "target_xyz": np.array(target_xyz).copy(),
            "fk_error": float(fk_error),
        })


def xyz_homeref_with_error(
    xyzcoords,
    refjnts,
    GRASP_OFFSET,
    downflag=True,
    traj_metrics=None,
    stage="xyz_homeref",
):
    refpose = kinematics.forward_kinematics(refjnts)

    refxyz = refpose[:3, 3]
    refrot = refpose[:3, :3]

    current_grasp_xyz = (
        refxyz
        + refrot @ GRASP_OFFSET
    )

    direcxyz = xyzcoords - current_grasp_xyz
    rot_error = abs(-1 - refrot[2, 2])

    if downflag:
        mag = np.linalg.norm(direcxyz) + rot_error
    else:
        mag = np.linalg.norm(direcxyz)

    min_step = 0.02
    max_fk_error = 0.0

    if mag > min_step:
        nsteps = int(mag / min_step) + 1
        step = direcxyz / nsteps
        current = refjnts.copy()

        for _ in range(nsteps):
            current, fk_error = relativexyz_with_error(
                current,
                step,
                GRASP_OFFSET,
                downflag
            )
            max_fk_error = max(max_fk_error, fk_error)

        record_fk_error(traj_metrics, stage, xyzcoords, max_fk_error)
        return current, max_fk_error

    current, fk_error = relativexyz_with_error(
        np.array(refjnts),
        np.array(direcxyz),
        GRASP_OFFSET,
        downflag
    )
    max_fk_error = max(max_fk_error, fk_error)
    record_fk_error(traj_metrics, stage, xyzcoords, max_fk_error)
    return current, max_fk_error


WRIST_ROLL_IDX = 4


def nearest_equivalent_angle_deg(angle, reference):
    return angle + 360.0 * np.round((reference - angle) / 360.0)


def append_joint_interpolation(jntslist, start, end, gripper_angle, nsteps=10):
    start = np.array(start)
    end = np.array(end)
    end[WRIST_ROLL_IDX] = nearest_equivalent_angle_deg(
        end[WRIST_ROLL_IDX],
        start[WRIST_ROLL_IDX]
    )

    for alpha in np.linspace(0, 1, nsteps + 1)[1:]:
        intermediate_joints = (
            (1 - alpha) * start
            + alpha * end
        )

        intermediate_joints[5] = gripper_angle
        jntslist.append(intermediate_joints)

    final_joints = end.copy()
    final_joints[5] = gripper_angle
    return final_joints


def solve_xyz_for_traj(target_xyz, refjnts, frame_offset, downflag, traj_metrics=None, stage="xyz_homeref"):
    if traj_metrics is None:
        return xyz_homeref(
            target_xyz,
            refjnts,
            frame_offset,
            downflag
        )

    solved_joints, _ = xyz_homeref_with_error(
        target_xyz,
        refjnts,
        frame_offset,
        downflag,
        traj_metrics=traj_metrics,
        stage=stage,
    )
    return solved_joints


def solve_seeded_target_xyz(
    target_xyz,
    solve_seed,
    frame_offset,
    gripper_angle,
    downflag,
    traj_metrics=None,
    stage="seeded_target",
):
    seeded_target = solve_xyz_for_traj(
        target_xyz,
        solve_seed,
        frame_offset,
        downflag,
        traj_metrics=traj_metrics,
        stage=stage,
    )
    seeded_target[5] = gripper_angle
    return seeded_target


def grasp_position_error(joints, target_xyz, frame_offset):
    fk_pose = kinematics.forward_kinematics(joints)
    grasp_xyz = fk_pose[:3, 3] + fk_pose[:3, :3] @ frame_offset
    return np.linalg.norm(target_xyz - grasp_xyz)


def downwardness(joints):
    fk_pose = kinematics.forward_kinematics(joints)
    return -fk_pose[2, 2]


def solve_seeded_target_xyz_prefer_down(
    target_xyz,
    solve_seed,
    frame_offset,
    gripper_angle,
    downflag,
    wrist_flex_offsets=(-30, -15, 0, 15, 30),
    wrist_roll_offsets=(-30, 0, 30),
):
    candidates = []

    for wrist_flex_offset in wrist_flex_offsets:
        for wrist_roll_offset in wrist_roll_offsets:
            seed = solve_seed.copy()
            seed[3] += wrist_flex_offset
            seed[4] += wrist_roll_offset

            candidate = solve_seeded_target_xyz(
                target_xyz,
                seed,
                frame_offset,
                gripper_angle,
                downflag
            )

            pos_error = grasp_position_error(candidate, target_xyz, frame_offset)
            down_score = downwardness(candidate)
            # Position dominates; downwardness breaks ties between similarly reachable poses.
            score = pos_error - 0.01 * down_score
            candidates.append((score, pos_error, -down_score, candidate))

    candidates.sort(key=lambda item: item[:3])
    return candidates[0][3]


def mark_segment_start(traj_metrics, name, jntslist):
    if traj_metrics is None:
        return
    segments = traj_metrics.setdefault("segments", {})
    segments[name] = {
        "start": len(jntslist),
        "end": len(jntslist),
    }


def mark_segment_end(traj_metrics, name, jntslist):
    if traj_metrics is None:
        return
    segments = traj_metrics.setdefault("segments", {})
    segment = segments.setdefault(
        name,
        {
            "start": len(jntslist),
            "end": len(jntslist),
        }
    )
    segment["end"] = len(jntslist)


def make_far_square_pickup_seed(square, board_origin, height, reach_pose, GRASP_OFFSET):
    square_xyz = chess_to_xy(square, board_origin=board_origin)
    target_xyz = square_xyz + np.array([0, 0, height])

    return solve_seeded_target_xyz(
        target_xyz,
        reach_pose,
        GRASP_OFFSET,
        gripper_angle_open,
        downflag=False
    )


def make_square_pickup_lift_path(
    square,
    board_origin,
    height,
    reach_pose,
    GRASP_OFFSET,
    downflag,
    traj_metrics=None,
    nsteps=10,
):
    square_xyz = chess_to_xy(square, board_origin=board_origin)
    above_xyz = square_xyz + np.array([0, 0, height])

    above_joints = solve_seeded_target_xyz(
        above_xyz,
        reach_pose,
        GRASP_OFFSET,
        gripper_angle_open,
        downflag,
        traj_metrics=traj_metrics,
        stage=f"{square}_pickup_style_above"
    )

    target_xyz = square_xyz + np.array([0, 0, 0])
    start_xyz = kinematics.forward_kinematics(above_joints)[:3, 3]

    descent_path = []
    current = above_joints.copy()
    for alpha in np.linspace(0, 1, nsteps + 1)[1:]:
        intermediate_xyz = (
            (1 - alpha) * start_xyz
            + alpha * target_xyz
        )

        intermediate_joints = solve_xyz_for_traj(
            intermediate_xyz,
            current,
            GRASP_OFFSET,
            downflag,
            traj_metrics=traj_metrics,
            stage=f"{square}_pickup_style_lower"
        )
        intermediate_joints[5] = gripper_angle_open
        descent_path.append(intermediate_joints.copy())
        current = intermediate_joints.copy()

    lift_path = []
    for joints in descent_path[::-1]:
        lift_joints = joints.copy()
        lift_joints[5] = gripper_angle_closed
        lift_path.append(lift_joints)

    return lift_path



def chess_to_xy(square, board_origin=(0.25, 0, 0), square_size=0.04):
    file = FILES.index(square[0].lower())
    rank = int(square[1]) - 1
    board_x, board_y, board_z = board_origin
    board_size = 8 * square_size
    # x = board_x - board_size/2 + (file + 0.5) * square_size
    y = board_y - board_size/2 + (rank + 0.5) * square_size
    x = board_x - board_size/2 + (file + 0.5) * square_size
    # y = board_y - board_size/2 + (rank + 0.85) * square_size
    z = board_z + 0.04  # best slightly above the board
    # z = board_z + 0.01  # best slightly above the board
    # z = board_z + 0.035  # slightly above the board
    # z = board_z + 0.06  # slightly above the board
    # z = board_z + 0.1  # slightly above the board
    return np.array([x, y, z])


def chess_to_xycalib(square, board_origin=(0.25, 0, 0), square_size=0.04):
    file = FILES.index(square[0].lower())
    rank = int(square[1]) - 1
    board_x, board_y, board_z = board_origin
    board_size = 8 * square_size
    # x = board_x - board_size/2 + (file + 0.5) * square_size
    y = board_y - board_size/2 + (rank ) * square_size
    x = board_x - board_size/2 + (file ) * square_size
    # y = board_y - board_size/2 + (rank + 0.85) * square_size
    z = board_z   # best slightly above the board
    # z = board_z + 0.01  # best slightly above the board
    # z = board_z + 0.035  # slightly above the board
    # z = board_z + 0.06  # slightly above the board
    # z = board_z + 0.1  # slightly above the board
    return np.array([x, y, z])


urdf_path = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
mesh_dir = "/Users/zhg603/Documents/OXAI/SO-ARM100/Simulation/SO101"

FILES = "abcdefgh"

SQUARE_SIZE = 0.04   # 4 cm example

corner1 = [97.32,0.40,28.40,66.55,177.80,4.95]
corner2 = [38.59,60.88,-58.55,100.48,178.15,4.95]
corner3 = [-52.48,57.98,-56.26,96.26,172.70,4.95]
corner4 = [-108.75, 26.33, 0.79, 81.85, 172.88, 1.4]

gripper_angle_open = 25
# gripper_angle_open = 20
gripper_angle_closed = 5
RELEASE_SETTLE_WAYPOINTS = 5


home = np.array([96.92307692307692,  -107.86813186813187,  97.36263736263736, 65.18681318681318,  -29.846153846153847,  4.62962962962963])

from testkinematics import kinematics
homexyz = kinematics.forward_kinematics(home)[:3,3]

def pickupmove_traj_old(from_square, to_square, board_origin, GRASP_OFFSET):
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



# def pickupmove_traj(from_square, to_square, board_origin, GRASP_OFFSET):
#     """
#     Move from current_joints to home, then from home to from_square, pick up piece,
#     move to to_square, place piece, and return to home. Returns the final joint position (home).
#     """
#     global home
#     # 1. Move to home
#     height = 0.13  # height to lift above squares
#     # height = 0.11  # height to lift above squares



#     jntslist = []
#     far_rows = ["f","g", "h"]

#     # current = home.copy()
#     # if "g" in from_square or "h" in from_square:
#     if from_square[0] in far_rows:
#         downflag = False

#         reach_pose = np.array([0.0,70,-90,40,90.0,5.0])
#         current = reach_pose.copy()
#         jntslist.append(current)
#     else:
#         downflag = True 
#         current = home.copy()


#     ###move above square
#     from_xyz = chess_to_xy(
#         from_square,
#         board_origin=board_origin
#     )

#     target_xyz = from_xyz + np.array([0, 0, height])
#     start_xyz = kinematics.forward_kinematics(current)[:3,3]

#     nsteps = 10

#     for alpha in np.linspace(0, 1, nsteps + 1)[1:]:

#         intermediate_xyz = (
#             (1 - alpha) * start_xyz
#             + alpha * target_xyz
#         )

#         intermediate_joints = xyz_homeref(
#             intermediate_xyz,
#             current,
#             GRASP_OFFSET, 
#             downflag
#         )

#         intermediate_joints[5] = gripper_angle_open

#         jntslist.append(intermediate_joints)

#         current = intermediate_joints.copy()
#     ###
#     # aboveboardjnts = current.copy()

#     ### lower to board
#     target_xyz = from_xyz + np.array([0, 0, 0])
#     start_xyz = kinematics.forward_kinematics(current)[:3,3]

#     nsteps = 10
#     downjnts = []
#     stepcnt = 0
#     for alpha in np.linspace(0, 1, nsteps + 1)[1:]:

#         intermediate_xyz = (
#             (1 - alpha) * start_xyz
#             + alpha * target_xyz
#         )

#         intermediate_joints = xyz_homeref(
#             intermediate_xyz,
#             current,
#             GRASP_OFFSET, 
#             downflag
#         )

#         intermediate_joints[5] = gripper_angle_open

#         jntslist.append(intermediate_joints)

#         current = intermediate_joints.copy()

#         if stepcnt == 0:
#             downjnts.append(intermediate_joints.copy())

#         # if stepcnt == 5:
#         #     downjnts.append(intermediate_joints.copy())

#         stepcnt+=1

#     ### close gripper
#     grip_closed = current.copy()
#     grip_closed[5] = gripper_angle_closed  # adjust as needed
#     jntslist.append(grip_closed)
#     current = grip_closed.copy()
#     copy = jntslist.copy()
#     closeidx = len(copy) - 1





#     ###lift back via downjnts
#     for j in downjnts[::-1]:
#         j[5] = gripper_angle_closed
#         jntslist.append(j)
#         # jntslist.extend(smoothjnts(current, j))
#         current = j.copy()

#     # aboveboard_closedjnts = aboveboardjnts.copy()
#     # aboveboard_closedjnts[5] = gripper_angle_closed

#     # jntslist.append(aboveboardjnts)
#     # current = aboveboardjnts.copy()

#     ###lift 
#     # target_xyz = from_xyz + np.array([0, 0, height])
#     # start_xyz = kinematics.forward_kinematics(current)[:3,3]

#     # nsteps = 2

#     # for alpha in np.linspace(0, 1, nsteps + 1)[1:]:

#     #     intermediate_xyz = (
#     #         (1 - alpha) * start_xyz
#     #         + alpha * target_xyz
#     #     )

#     #     intermediate_joints = xyz_homeref(
#     #         intermediate_xyz,
#     #         current,
#     #         GRASP_OFFSET, 
#     #         downflag
#     #     )

#     #     intermediate_joints[5] = gripper_angle_closed

#     #     jntslist.append(intermediate_joints)

#     #     current = intermediate_joints.copy()
    

#     # return jntslist
#     # 3. Move to to_square (above)
#     # downflag = True
#     to_xyz = chess_to_xy(to_square, board_origin=board_origin)
#     # above_to = xyz_homeref(to_xyz + np.array([0, 0, height]), current, GRASP_OFFSET)
#     # above_to[5] = gripper_angle_closed  # keep gripper closed
#     # jntslist.append(above_to)

#     target_xyz = to_xyz + np.array([0, 0, height])
#     start_xyz = kinematics.forward_kinematics(current)[:3,3]

#     nsteps = 5

#     for alpha in np.linspace(0, 1, nsteps + 1)[1:]:

#         intermediate_xyz = (
#             (1 - alpha) * start_xyz
#             + alpha * target_xyz
#         )

#         intermediate_joints = xyz_homeref(
#             intermediate_xyz,
#             current,
#             GRASP_OFFSET, 
#             downflag
#         )

#         intermediate_joints[5] = gripper_angle_closed

#         jntslist.append(intermediate_joints)

#         current = intermediate_joints.copy()
#     ###
 


#     return jntslist, closeidx

#     current = above_to.copy()
#     # Lower to to_square
#     at_to = xyz_homeref(to_xyz, current, GRASP_OFFSET)
#     jntslist.append(at_to)

#     #close gripper
#     at_to_grip = at_to.copy()
#     at_to_grip[5] = gripper_angle_closed  # keep gripper closed
#     # smoothmove(current, at_to)
#     jntslist.append(at_to_grip)

#     current = at_to.copy()
#     # Open gripper
#     grip_open = current.copy()
#     grip_open[5] = gripper_angle_open  # adjust as needed
#     # smoothmove(current, grip_open)
#     jntslist.append(grip_open)

#     current = grip_open.copy()
#     # Lift up
#     lifted = xyz_homeref(to_xyz + np.array([0, 0, height]), current, GRASP_OFFSET)
#     # smoothmove(current, lifted)
#     jntslist.append(lifted)
 
#     current = lifted.copy()

#     # 4. Return to home
#     # smoothmove(current, home)
#     jntslist.append(home)
#     current = home.copy()

#     return jntslist



# def pickupmove_traj(from_square, to_square, board_origin, GRASP_OFFSET):
#     """
#     Move from current_joints to home, then from home to from_square, pick up piece,
#     move to to_square, place piece, and return to home. Returns the final joint position (home).
#     """
#     global home
#     # 1. Move to home
#     # height = 0.23  # height to lift above squares
#     height = 0.13  # height to lift above squares
#     # height = 0.11  # height to lift above squares



#     jntslist = []
#     far_rows = ["f","g", "h"]
#     # far_rows = []

#     # current = home.copy()
#     # if "g" in from_square or "h" in from_square:
#     if from_square[0] in far_rows:
#         downflag = False

#         reach_pose = np.array([0.0,70,-90,40,90.0,5.0])
#         current = reach_pose.copy()
#         jntslist.append(current)
#     else:
#         downflag = True 
#         # downflag = False 
#         current = home.copy()


#     # downflag = True
#     ###move above square
#     from_xyz = chess_to_xy(
#         from_square,
#         board_origin=board_origin
#     )

#     target_xyz = from_xyz + np.array([0, 0, height])
#     start_xyz = kinematics.forward_kinematics(current)[:3,3]

#     nsteps = 10

#     for alpha in np.linspace(0, 1, nsteps + 1)[1:]:

#         intermediate_xyz = (
#             (1 - alpha) * start_xyz
#             + alpha * target_xyz
#         )

#         intermediate_joints = xyz_homeref(
#             intermediate_xyz,
#             current,
#             GRASP_OFFSET, 
#             downflag
#         )

#         intermediate_joints[5] = gripper_angle_open

#         jntslist.append(intermediate_joints)

#         current = intermediate_joints.copy()
#     ###
#     # aboveboardjnts = current.copy()

#     ### lower to board
#     target_xyz = from_xyz + np.array([0, 0, 0])
#     start_xyz = kinematics.forward_kinematics(current)[:3,3]

#     # nsteps = 10
#     nsteps = 10
#     downjnts = []
#     stepcnt = 0
#     for alpha in np.linspace(0, 1, nsteps + 1)[1:]:

#         intermediate_xyz = (
#             (1 - alpha) * start_xyz
#             + alpha * target_xyz
#         )

#         intermediate_joints = xyz_homeref(
#             intermediate_xyz,
#             current,
#             GRASP_OFFSET, 
#             downflag
#         )

#         intermediate_joints[5] = gripper_angle_open

#         jntslist.append(intermediate_joints)

#         current = intermediate_joints.copy()

#         if stepcnt == 0:
#             downjnts.append(intermediate_joints.copy())

#         # if stepcnt == 5:
#         #     downjnts.append(intermediate_joints.copy())

#         stepcnt+=1

#     ### close gripper
#     grip_closed = current.copy()
#     grip_closed[5] = gripper_angle_closed  # adjust as needed
#     jntslist.append(grip_closed)
#     current = grip_closed.copy()
#     copy = jntslist.copy()
#     closeidx = len(copy) - 1






#     ###lift back via downjnts
#     for j in downjnts[::-1]:
#         j[5] = gripper_angle_closed
#         jntslist.append(j)
#         # jntslist.extend(smoothjnts(current, j))
#         current = j.copy()



#     # 3. Move to to_square (above)
#     # downflag = True
#     to_xyz = chess_to_xy(to_square, board_origin=board_origin)
#     # above_to = xyz_homeref(to_xyz + np.array([0, 0, height]), current, GRASP_OFFSET)
#     # above_to[5] = gripper_angle_closed  # keep gripper closed
#     # jntslist.append(above_to)

#     target_xyz = to_xyz + np.array([0, 0, height])
#     start_xyz = kinematics.forward_kinematics(current)[:3,3]

#     nsteps = 5

#     for alpha in np.linspace(0, 1, nsteps + 1)[1:]:

#         intermediate_xyz = (
#             (1 - alpha) * start_xyz
#             + alpha * target_xyz
#         )

#         intermediate_joints = xyz_homeref(
#             intermediate_xyz,
#             current,
#             GRASP_OFFSET, 
#             downflag
#         )

#         intermediate_joints[5] = gripper_angle_closed

#         jntslist.append(intermediate_joints)

#         current = intermediate_joints.copy()
#     ###
 

#     ### lower to board
#     to_xyz = chess_to_xy(to_square, board_origin=board_origin)
#     target_xyz = to_xyz + np.array([0, 0, 0.04]) #wiggle room for drop
#     # target_xyz = to_xyz + np.array([0, 0, 0.0015]) #rook wiggle room for drop
#     # target_xyz = to_xyz + np.array([0, 0, 0]) #wiggle room for drop
#     start_xyz = kinematics.forward_kinematics(current)[:3,3]

#     nsteps = 10
#     downjnts = []
#     stepcnt = 0
#     for alpha in np.linspace(0, 1, nsteps + 1)[1:]:

#         intermediate_xyz = (
#             (1 - alpha) * start_xyz
#             + alpha * target_xyz
#         )

#         intermediate_joints = xyz_homeref(
#             intermediate_xyz,
#             current,
#             GRASP_OFFSET, 
#             downflag
#         )

#         intermediate_joints[5] = gripper_angle_closed

#         jntslist.append(intermediate_joints)

#         current = intermediate_joints.copy()

#         if stepcnt == 0:
#             downjnts.append(intermediate_joints.copy())

#         # if stepcnt == 5:
#         #     downjnts.append(intermediate_joints.copy())

#         stepcnt+=1


#     # return jntslist, closeidx

#     # Open gripper
#     grip_open = current.copy()
#     grip_open[5] = gripper_angle_open  # adjust as needed
#     # smoothmove(current, grip_open)
#     jntslist.append(grip_open)

#     current = grip_open.copy()

#     for j in downjnts[::-1]:
#         j[5] = gripper_angle_closed
#         jntslist.append(j)
#         # jntslist.extend(smoothjnts(current, j))
#         current = j.copy()


#     ###
#     home_xyz = kinematics.forward_kinematics(home)[:3,3]

#     above_home = xyz_homeref(
#         home_xyz + np.array([0, 0, 0.10]),
#         current,
#         GRASP_OFFSET
#     )

#     above_home[5] = current[5]

#     jntslist.append(above_home)

#     current = above_home.copy()

#     jntslist.append(home)

#     current = home.copy()
#     ##
#     jntslist.append(home)
#     current = home.copy()

#     return jntslist, closeidx

def pickupmove_traj(
    from_square,
    to_square,
    board_origin,
    GRASP_OFFSET,
    PLACE_OFFSET,
    traj_metrics=None,
    placement_lower_steps=10,
):
    """
    Move from current_joints to home, then from home to from_square, pick up piece,
    move to to_square, place piece, and return to home. Returns the final joint position (home).
    """
    global home
    # 1. Move to home
    # height = 0.23  # height to lift above squares
    height = 0.13  # height to lift above squares
    # height = 0.03  # height to lift above squares
    # height = 0.11  # height to lift above squares



    jntslist = []
    # Files outside this set, including all a-file destinations, use the
    # near-square downward pose.
    far_rows = ["f","g", "h"]
    # far_rows = []

    # current = home.copy()
    # if "g" in from_square or "h" in from_square:
    if from_square[0] in far_rows:
        downflag = False

        reach_pose = np.array([0.0,70,-90,40,90.0,5.0])
        current = home.copy()
    else:
        downflag = True 
        # downflag = False 
        current = home.copy()


    # downflag = True
    ###move above square
    from_xyz = chess_to_xy(
        from_square,
        board_origin=board_origin
    )

    target_xyz = from_xyz + np.array([0, 0, height])

    if from_square[0] in far_rows:
        above_from = solve_seeded_target_xyz(
            target_xyz,
            reach_pose,
            GRASP_OFFSET,
            gripper_angle_open,
            downflag,
            traj_metrics=traj_metrics,
            stage=f"{from_square}_far_above_pickup"
        )
        current = append_joint_interpolation(
            jntslist,
            current,
            above_from,
            gripper_angle_open,
            nsteps=10
        )
    else:
        start_xyz = kinematics.forward_kinematics(current)[:3,3]

        nsteps = 10

        for alpha in np.linspace(0, 1, nsteps + 1)[1:]:

            intermediate_xyz = (
                (1 - alpha) * start_xyz
                + alpha * target_xyz
            )

            intermediate_joints = solve_xyz_for_traj(
                intermediate_xyz,
                current,
                GRASP_OFFSET, 
                downflag,
                traj_metrics=traj_metrics,
                stage=f"{from_square}_above_pickup"
            )

            intermediate_joints[5] = gripper_angle_open

            jntslist.append(intermediate_joints)

            current = intermediate_joints.copy()
    ###
    # aboveboardjnts = current.copy()

    ### lower to board
    target_xyz = from_xyz + np.array([0, 0, 0])
    start_xyz = kinematics.forward_kinematics(current)[:3,3]

    # nsteps = 10
    nsteps = 10
    downjnts = []
    stepcnt = 0
    for alpha in np.linspace(0, 1, nsteps + 1)[1:]:

        intermediate_xyz = (
            (1 - alpha) * start_xyz
            + alpha * target_xyz
        )

        intermediate_joints = solve_xyz_for_traj(
            intermediate_xyz,
            current,
            GRASP_OFFSET, 
            downflag,
            traj_metrics=traj_metrics,
            stage=f"{from_square}_lower_pickup"
        )

        intermediate_joints[5] = gripper_angle_open

        jntslist.append(intermediate_joints)

        current = intermediate_joints.copy()

        if stepcnt == 0:
            downjnts.append(intermediate_joints.copy())

        # if stepcnt == 5:
        #     downjnts.append(intermediate_joints.copy())

        stepcnt+=1

    ### close gripper
    grip_closed = current.copy()
    grip_closed[5] = gripper_angle_closed  # adjust as needed
    jntslist.append(grip_closed)
    current = grip_closed.copy()
    copy = jntslist.copy()
    closeidx = len(copy) - 1






    ###lift back via downjnts
    for j in downjnts[::-1]:
        j[5] = gripper_angle_closed
        jntslist.append(j)
        # jntslist.extend(smoothjnts(current, j))
        current = j.copy()



    # 3. Move to to_square (above)
    if to_square[0] in far_rows:
        downflag = False

        reach_pose = np.array([0.0,70,-90,40,90.0,5.0])
    elif to_square[0] == "b" and not B_FILE_PLACE_DOWNFLAG:
        downflag = False
    else:
        downflag = True
    to_xyz = chess_to_xy(to_square, board_origin=board_origin)
    # above_to = xyz_homeref(to_xyz + np.array([0, 0, height]), current, GRASP_OFFSET)
    # above_to[5] = gripper_angle_closed  # keep gripper closed
    # jntslist.append(above_to)

    target_xyz = to_xyz + np.array([0, 0, height])

    mark_segment_start(traj_metrics, "destination_above_place", jntslist)
    if to_square[0] in far_rows:
        pickup_like_seed = make_far_square_pickup_seed(
            to_square,
            board_origin,
            height,
            reach_pose,
            GRASP_OFFSET
        )

        above_to = solve_seeded_target_xyz(
            target_xyz,
            pickup_like_seed,
            PLACE_OFFSET,
            gripper_angle_closed,
            downflag,
            traj_metrics=traj_metrics,
            stage=f"{to_square}_far_above_place"
        )
        current = append_joint_interpolation(
            jntslist,
            current,
            above_to,
            gripper_angle_closed,
            nsteps=5
        )
    else:
        start_xyz = kinematics.forward_kinematics(current)[:3,3]

        nsteps = 5

        for alpha in np.linspace(0, 1, nsteps + 1)[1:]:

            intermediate_xyz = (
                (1 - alpha) * start_xyz
                + alpha * target_xyz
            )

            intermediate_joints = solve_xyz_for_traj(
                intermediate_xyz,
                current,
                PLACE_OFFSET,
                downflag,
                traj_metrics=traj_metrics,
                stage=f"{to_square}_above_place"
            )

            intermediate_joints[5] = gripper_angle_closed

            jntslist.append(intermediate_joints)

            current = intermediate_joints.copy()
    mark_segment_end(traj_metrics, "destination_above_place", jntslist)
    ###
 

    ### lower to board
    to_xyz = chess_to_xy(to_square, board_origin=board_origin)
    target_xyz = to_xyz + np.array([0, 0, 0.02]) #wiggle room for drop
    # target_xyz = to_xyz + np.array([0, 0, 0.0015]) #rook wiggle room for drop
    # target_xyz = to_xyz + np.array([0, 0, 0]) #wiggle room for drop
    start_xyz = kinematics.forward_kinematics(current)[:3,3]

    nsteps = placement_lower_steps
    # nsteps = 2
    downjnts = []
    stepcnt = 0
    mark_segment_start(traj_metrics, "destination_lower_place", jntslist)
    if to_square == "f1":
        pickup_lift_path = make_square_pickup_lift_path(
            to_square,
            board_origin,
            height,
            reach_pose,
            GRASP_OFFSET,
            downflag,
            traj_metrics=traj_metrics,
            nsteps=nsteps
        )
        placement_lower_path = pickup_lift_path[::-1]

        for intermediate_joints in placement_lower_path:
            intermediate_joints = intermediate_joints.copy()
            intermediate_joints[5] = gripper_angle_closed
            jntslist.append(intermediate_joints)
            current = intermediate_joints.copy()

        target_xyz = to_xyz + np.array([0, 0, 0])
    else:
        for alpha in np.linspace(0, 1, nsteps + 1)[1:]:

            intermediate_xyz = (
                (1 - alpha) * start_xyz
                + alpha * target_xyz
            )

            intermediate_joints = solve_xyz_for_traj(
                intermediate_xyz,
                current,
                PLACE_OFFSET,
                downflag,
                traj_metrics=traj_metrics,
                stage=f"{to_square}_lower_place"
            )

            intermediate_joints[5] = gripper_angle_closed

            if traj_metrics is not None and placement_lower_steps <= 2:
                traj_metrics.setdefault("slow_waypoint_indices", []).append(
                    len(jntslist)
                )

            jntslist.append(intermediate_joints)

            current = intermediate_joints.copy()

            if stepcnt == 0:
                downjnts.append(intermediate_joints.copy())

            # if stepcnt == 5:
            #     downjnts.append(intermediate_joints.copy())

            stepcnt+=1

    mark_segment_end(traj_metrics, "destination_lower_place", jntslist)

    # return jntslist, closeidx

    if traj_metrics is not None:
        traj_metrics["release_target_xyz"] = target_xyz.copy()
        traj_metrics["release_target_z"] = float(target_xyz[2])

    # Open gripper
    mark_segment_start(traj_metrics, "release_settle", jntslist)
    grip_open = current.copy()
    grip_open[5] = gripper_angle_open  # adjust as needed
    # smoothmove(current, grip_open)
    jntslist.append(grip_open)
    for _ in range(RELEASE_SETTLE_WAYPOINTS):
        jntslist.append(grip_open.copy())
    mark_segment_end(traj_metrics, "release_settle", jntslist)

    current = grip_open.copy()


    # causing problems with far squares for some reason, so skipping for now
    mark_segment_start(traj_metrics, "retreat", jntslist)
    if to_square[0] not in far_rows:
        for j in downjnts[::-1]:
            # Keep the gripper open while retreating from a placed piece.
            j[5] = gripper_angle_open
            jntslist.append(j)
            # jntslist.extend(smoothjnts(current, j))
            current = j.copy()
    else:
        target_xyz = to_xyz + np.array([0, 0, height])
        above_to_open = solve_seeded_target_xyz(
            target_xyz,
            reach_pose,
            GRASP_OFFSET,
            gripper_angle_open,
            downflag,
            traj_metrics=traj_metrics,
            stage=f"{to_square}_far_lift_after_release"
        )
        current = append_joint_interpolation(
            jntslist,
            current,
            above_to_open,
            gripper_angle_open,
            nsteps=10
        )
    mark_segment_end(traj_metrics, "retreat", jntslist)

    # above_to = xyz_homeref(
    #     to_xyz + np.array([0, 0, 0.10]),
    #     current,
    #     GRASP_OFFSET
    # )
    # above_to[5] = current[5]
    # jntslist.append(above_to)
    # current = above_to.copy()
    ###
    home_xyz = kinematics.forward_kinematics(home)[:3,3]

    mark_segment_start(traj_metrics, "return_home", jntslist)
    above_home = solve_xyz_for_traj(
        home_xyz + np.array([0, 0, 0.10]),
        current,
        GRASP_OFFSET,
        True,
        traj_metrics=traj_metrics,
        stage="return_above_home"
    )

    above_home[5] = current[5]

    jntslist.append(above_home)

    current = above_home.copy()

    jntslist.append(home)

    current = home.copy()
    ##
    jntslist.append(home)
    current = home.copy()
    mark_segment_end(traj_metrics, "return_home", jntslist)

    return jntslist, closeidx


def pickupmove_traj_with_metrics(
    from_square,
    to_square,
    board_origin,
    GRASP_OFFSET,
    PLACE_OFFSET,
    placement_lower_steps=10,
):
    traj_metrics = {
        "max_fk_error": 0.0,
        "event_threshold": 0.025,
        "fk_error_events": [],
        "segments": {},
        "release_target_xyz": None,
        "release_target_z": None,
    }

    jntslist, closeidx = pickupmove_traj(
        from_square,
        to_square,
        board_origin,
        GRASP_OFFSET,
        PLACE_OFFSET,
        traj_metrics=traj_metrics,
        placement_lower_steps=placement_lower_steps,
    )

    return jntslist, closeidx, traj_metrics

def calib_board(board_origin, GRASP_OFFSET=np.array([0,0,0]), PLACE_OFFSET=np.array([0,0,0])):
    """
    Calibrate the 4 corners of the board, returning a list of joint trajectories to each corner.
    """
    global home
    # 1. Move to home
    # height = 0.23  # height to lift above squares
    height = 0.13  # height to lift above squares
    # height = 0.11  # height to lift above squares

    sqlist = ["a1", "a8", "h1", "h8"]
    alljntslist = []
    for sq in sqlist:
        print(f"Calibrating square {sq}...")
        jntslist = []
        # far_rows = ["f","g", "h"]
        far_rows = ["e","f","g", "h"]
        # far_rows = []

        # current = home.copy()
        # if "g" in from_square or "h" in from_square:
        if sq[0] in far_rows:
            downflag = False
            reach_pose = np.array([0.0,70,-90,40,90.0,5.0])
            current = reach_pose.copy()
            jntslist.append(current)
        else:
            downflag = True 
            # downflag = False 
            current = home.copy()



        ###move above square
        from_xyz = chess_to_xycalib(
            sq,
            board_origin=board_origin
        )

        target_xyz = from_xyz + np.array([0, 0, height])
        start_xyz = kinematics.forward_kinematics(current)[:3,3]

        nsteps = 10

        for alpha in np.linspace(0, 1, nsteps + 1)[1:]:

            intermediate_xyz = (
                (1 - alpha) * start_xyz
                + alpha * target_xyz
            )

            intermediate_joints = xyz_homeref(
                intermediate_xyz,
                current,
                GRASP_OFFSET, 
                downflag
            )

            intermediate_joints[5] = gripper_angle_open

            jntslist.append(intermediate_joints)

            current = intermediate_joints.copy()
        ###
        # aboveboardjnts = current.copy()

        ### lower to board
        target_xyz = from_xyz + np.array([0, 0, 0])
        start_xyz = kinematics.forward_kinematics(current)[:3,3]

        # nsteps = 10
        nsteps = 10
        downjnts = []
        stepcnt = 0
        for alpha in np.linspace(0, 1, nsteps + 1)[1:]:

            intermediate_xyz = (
                (1 - alpha) * start_xyz
                + alpha * target_xyz
            )

            intermediate_joints = xyz_homeref(
                intermediate_xyz,
                current,
                GRASP_OFFSET, 
                downflag
            )

            intermediate_joints[5] = gripper_angle_open

            jntslist.append(intermediate_joints)

            current = intermediate_joints.copy()

            if stepcnt == 0:
                downjnts.append(intermediate_joints.copy())

            # if stepcnt == 5:
            #     downjnts.append(intermediate_joints.copy())

            stepcnt+=1



        ###lift back via downjnts
        for j in downjnts[::-1]:
            j[5] = gripper_angle_closed
            jntslist.append(j)
            # jntslist.extend(smoothjnts(current, j))
            current = j.copy()


        ###
        home_xyz = kinematics.forward_kinematics(home)[:3,3]

        above_home = xyz_homeref(
            home_xyz + np.array([0, 0, 0.10]),
            current,
            GRASP_OFFSET
        )

        above_home[5] = current[5]

        jntslist.append(above_home)

        current = above_home.copy()

        jntslist.append(home)

        current = home.copy()
        ##
        jntslist.append(home)
        current = home.copy()


        alljntslist.append(jntslist)
    
    return alljntslist

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
