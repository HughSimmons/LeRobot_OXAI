import json
from pathlib import Path

import numpy as np
import pybullet as p

from multisim_chess_fast import (
    PLACE_OFFSET,
    RECORDINGS_DIR,
    append_video_frame,
    build_reverse_trajectory,
    build_source_trajectory,
    close_video_context,
    create_video_context,
    ensure_physics_connected,
    gripper_angle_open,
    home_rad,
    renderfreq,
    run_sim_move,
    runid,
    score_place_result,
    setup_sim_world,
)


SEQUENCE_REPEATS = 4
SEQUENCE_MOVES = (("e1", "f4"), ("f4", "e1"))
RECORD_VIDEO = True
RETURN_HOME_BETWEEN_MOVES = True
HOME_RETURN_STEPS = 120
HOME_SETTLE_STEPS = 80
SIM_JOINT_MAP = (0, 1, 2, 3, 4, 6)
HOME_OPEN_JOINTS = home_rad.copy()
HOME_OPEN_JOINTS[5] = np.deg2rad(gripper_angle_open)

LOOKUP_PATHS = {
    "e1_to_f4": Path(__file__).resolve().parent / "e1_fg_reverse_move_lookup.json",
    "f4_to_e1": Path(__file__).resolve().parent / "f4_e_reverse_move_lookup.json",
}


def move_key(from_square, to_square):
    return f"{from_square}_to_{to_square}"


SEQUENCE_VIDEO_DIR = (
    RECORDINGS_DIR
    / runid
    / "lookup_sequence"
    / f"{'_'.join(move_key(*move) for move in SEQUENCE_MOVES)}_repeats_{SEQUENCE_REPEATS}"
)


def load_lookup_entry(from_square, to_square):
    key = move_key(from_square, to_square)
    lookup_path = LOOKUP_PATHS.get(key)
    if lookup_path is None:
        raise KeyError(f"No lookup path configured for {key}")
    if not lookup_path.exists():
        raise FileNotFoundError(f"Lookup file missing for {key}: {lookup_path}")

    lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
    entry = lookup.get("moves", {}).get(key)
    if entry is None:
        raise KeyError(f"Lookup entry missing: {key} in {lookup_path}")
    if not entry.get("success"):
        raise ValueError(f"Lookup entry is not successful: {key}")
    return entry


def as_array(value):
    return np.array(value, dtype=float)


def build_e1_to_f4_override(entry):
    source_movelist, source_closeidx, source_traj_metrics, _, _ = build_source_trajectory(
        "f4",
        "e1"
    )
    return build_reverse_trajectory(
        source_movelist,
        source_closeidx,
        source_traj_metrics,
        "e1",
        "f4",
        as_array(entry["selected_place_offset"]),
        reversed_grasp_offset=as_array(entry["reversed_grasp_offset"]),
    )


def run_lookup_move(
    world,
    from_square,
    to_square,
    entry,
    repeat_idx,
    restore_state=False,
    video_context=None,
):
    key = move_key(from_square, to_square)
    grasp_offset = as_array(
        entry["reversed_grasp_offset"]
        if key == "e1_to_f4"
        else entry["source_grasp_offset"]
    )
    place_offset = as_array(entry["selected_place_offset"])
    trajectory_override = build_e1_to_f4_override(entry) if key == "e1_to_f4" else None

    result = run_sim_move(
        world,
        from_square,
        to_square,
        grasp_offset,
        place_offset=place_offset,
        return_metrics=True,
        record_video=RECORD_VIDEO,
        video_label=f"sequence_{repeat_idx}_{key}",
        trajectory_override=trajectory_override,
        restore_state=restore_state,
        video_context=video_context,
    )
    result["score"] = score_place_result(result)
    return result


def print_result(repeat_idx, from_square, to_square, result):
    print(
        f"repeat={repeat_idx} | "
        f"move={move_key(from_square, to_square)} | "
        f"pickup={result['pickup_success']} | "
        f"premature_drop={result['premature_drop']} | "
        f"xy_error={result['xy_error']} | "
        f"tilt={result['final_tilt_deg']} | "
        f"euler={result['final_euler_deg']} | "
        f"reject={result['reject_reason']} | "
        f"video={result['video_output_dir']}"
    )


def return_robot_home(world, video_context=None):
    robot_id = world["robot_id"]
    current_joints = np.array([
        p.getJointState(robot_id, sim_idx)[0]
        for sim_idx in SIM_JOINT_MAP
    ])

    for step in range(1, HOME_RETURN_STEPS + 1):
        alpha = step / HOME_RETURN_STEPS
        target_joints = (1.0 - alpha) * current_joints + alpha * HOME_OPEN_JOINTS
        for traj_idx, sim_idx in enumerate(SIM_JOINT_MAP):
            force = 500 if traj_idx == 5 else 50
            p.setJointMotorControl2(
                robot_id,
                sim_idx,
                p.POSITION_CONTROL,
                targetPosition=target_joints[traj_idx],
                force=force,
            )
        p.stepSimulation()
        if video_context is not None and step % renderfreq == 0:
            append_video_frame(video_context)

    for step in range(1, HOME_SETTLE_STEPS + 1):
        p.stepSimulation()
        if video_context is not None and step % renderfreq == 0:
            append_video_frame(video_context)


def main():
    ensure_physics_connected()
    entries = {
        move_key(from_square, to_square): load_lookup_entry(from_square, to_square)
        for from_square, to_square in SEQUENCE_MOVES
    }

    world = setup_sim_world(SEQUENCE_MOVES[0][0])
    video_context = create_video_context(SEQUENCE_VIDEO_DIR) if RECORD_VIDEO else None
    try:
        total_moves = SEQUENCE_REPEATS * len(SEQUENCE_MOVES)
        completed_moves = 0
        for repeat_idx in range(1, SEQUENCE_REPEATS + 1):
            for from_square, to_square in SEQUENCE_MOVES:
                key = move_key(from_square, to_square)
                restore_state = repeat_idx == 1 and (from_square, to_square) == SEQUENCE_MOVES[0]
                result = run_lookup_move(
                    world,
                    from_square,
                    to_square,
                    entries[key],
                    repeat_idx,
                    restore_state=restore_state,
                    video_context=video_context,
                )
                print_result(repeat_idx, from_square, to_square, result)
                if result["reject_reason"] is not None:
                    print(f"Stopping sequence after failed move: {key}")
                    return
                completed_moves += 1
                if RETURN_HOME_BETWEEN_MOVES and completed_moves < total_moves:
                    print("Returning robot to home before next move.")
                    return_robot_home(world, video_context=video_context)
    finally:
        if video_context is not None:
            close_video_context(video_context)
            print(f"Sequence video saved: {SEQUENCE_VIDEO_DIR}")
        p.removeState(world["state_id"])
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
