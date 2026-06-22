import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pybullet as p

from multisim_chess_fast import (
    DEFAULT_MOVE_STEPS_PER_WAYPOINT,
    PLACE_OFFSET,
    run_sim_move,
    setup_sim_world,
)


RANKS = tuple(range(1, 9))
WALK_FILES = ("f", "e", "d", "c", "b", "a")

INITIAL_GRASP_OFFSET = np.array([-0.014, 0.002, -0.003])
LOOKUP_EDGE_SUPPORT_MARGIN = 0.08
MOVE_STEPS_PER_WAYPOINT = DEFAULT_MOVE_STEPS_PER_WAYPOINT
PLACEMENT_LOWER_STEPS = 2

LOCAL_SEARCH_XY_DELTA = 0.001
LOCAL_SEARCH_Z_DELTA = 0.002
LOCAL_SEARCH_XY_STEPS = (-3, -2, -1, 0, 1, 2, 3)
LOCAL_SEARCH_Z_STEPS = (0, -1, 1)

OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "initial_grasp_defaults_across_ranks.json"
)
DIAGNOSTICS_PATH = (
    Path(__file__).resolve().parent
    / "initial_grasp_find_across_ranks_diagnostics.json"
)


class GraspWalkFailure(RuntimeError):
    pass


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def source_square(rank, file_name):
    return f"{file_name}{rank}"


def test_to_square(rank, file_name):
    if file_name == "a":
        return source_square(rank, "b")

    next_file = WALK_FILES[WALK_FILES.index(file_name) + 1]
    return source_square(rank, next_file)


def walk_squares_for_rank(rank):
    return tuple(source_square(rank, file_name) for file_name in WALK_FILES)


def run_pickup_trial(from_square, to_square, grasp_offset):
    world = setup_sim_world(
        from_square,
        edge_support_margin=LOOKUP_EDGE_SUPPORT_MARGIN,
    )
    try:
        return run_sim_move(
            world,
            from_square,
            to_square,
            grasp_offset,
            place_offset=PLACE_OFFSET.copy(),
            return_metrics=True,
            record_video=False,
            move_steps_per_waypoint=MOVE_STEPS_PER_WAYPOINT,
            placement_lower_steps=PLACEMENT_LOWER_STEPS,
        )
    finally:
        p.removeState(world["state_id"])


def result_score(result, delta):
    pickup_failed = not bool(result.get("pickup_success", False))
    rejected = result.get("reject_reason") is not None
    premature_drop = bool(result.get("premature_drop", False))
    fk_error = float(result.get("trajectory_fk_error", np.inf))
    delta_norm = float(np.linalg.norm(delta))
    return (
        int(pickup_failed),
        int(rejected),
        int(premature_drop),
        fk_error,
        delta_norm,
    )


def local_grasp_candidates(base_grasp_offset):
    for dz in LOCAL_SEARCH_Z_STEPS:
        for dx in LOCAL_SEARCH_XY_STEPS:
            for dy in LOCAL_SEARCH_XY_STEPS:
                delta = np.array([
                    dx * LOCAL_SEARCH_XY_DELTA,
                    dy * LOCAL_SEARCH_XY_DELTA,
                    dz * LOCAL_SEARCH_Z_DELTA,
                ])
                yield delta, base_grasp_offset + delta


def local_grasp_candidate_count():
    return (
        len(LOCAL_SEARCH_Z_STEPS)
        * len(LOCAL_SEARCH_XY_STEPS)
        * len(LOCAL_SEARCH_XY_STEPS)
    )


def search_local_grasp(from_square, to_square, base_grasp_offset):
    attempts = []
    total_candidates = local_grasp_candidate_count()
    print(
        f"{from_square}: searching local grasp offsets around "
        f"{base_grasp_offset} ({total_candidates} candidates)"
    )
    for candidate_idx, (delta, grasp_offset) in enumerate(
        local_grasp_candidates(base_grasp_offset),
        start=1,
    ):
        result = run_pickup_trial(from_square, to_square, grasp_offset)
        score = result_score(result, delta)
        attempt = {
            "candidate_index": candidate_idx,
            "candidate_count": total_candidates,
            "delta": delta,
            "grasp_offset": grasp_offset,
            "score": score,
            "pickup_success": bool(result.get("pickup_success", False)),
            "premature_drop": bool(result.get("premature_drop", False)),
            "reject_reason": result.get("reject_reason"),
            "trajectory_fk_error": result.get("trajectory_fk_error"),
        }
        attempts.append(attempt)
        print(
            f"{from_square}->{to_square}: local candidate "
            f"{candidate_idx}/{total_candidates} "
            f"delta={delta} | pickup={attempt['pickup_success']} | "
            f"premature={attempt['premature_drop']} | "
            f"reject={attempt['reject_reason']} | "
            f"fk={attempt['trajectory_fk_error']}"
        )
        if attempt["pickup_success"]:
            print(
                f"{from_square}: local search PASS at delta={delta}; "
                "stopping search early"
            )
            return attempt, attempts

    attempts.sort(key=lambda attempt: attempt["score"])
    return attempts[0], attempts


def write_diagnostics(diagnostics):
    DIAGNOSTICS_PATH.write_text(
        json.dumps(json_safe(diagnostics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote diagnostics: {DIAGNOSTICS_PATH}")


def build_initial_diagnostics():
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ranks": list(RANKS),
        "walk_files": list(WALK_FILES),
        "initial_grasp_offset": INITIAL_GRASP_OFFSET.copy(),
        "local_search_xy_delta": LOCAL_SEARCH_XY_DELTA,
        "local_search_z_delta": LOCAL_SEARCH_Z_DELTA,
        "local_search_xy_steps": LOCAL_SEARCH_XY_STEPS,
        "local_search_z_steps": LOCAL_SEARCH_Z_STEPS,
        "results": [],
        "final_grasps": {},
        "failure": None,
    }


def evaluate_square(rank, from_square, to_square, carried_grasp):
    print(
        f"\n=== Rank {rank} | Source square {from_square} | "
        f"test move {from_square}->{to_square} ==="
    )
    print(f"{from_square}: testing carried grasp {carried_grasp}")
    carried_result = run_pickup_trial(from_square, to_square, carried_grasp)
    carried_success = bool(carried_result.get("pickup_success", False))
    print(
        f"{from_square}: carried grasp "
        f"{'PASS' if carried_success else 'FAIL'} | "
        f"premature={carried_result.get('premature_drop')} | "
        f"reject={carried_result.get('reject_reason')} | "
        f"fk={carried_result.get('trajectory_fk_error')}"
    )

    selected_grasp = carried_grasp.copy()
    selected_source = "carried"
    attempts = []

    if not carried_success:
        print(
            f"{from_square}->{to_square}: carried grasp failed; "
            "running local search around last working grasp"
        )
        selected_attempt, attempts = search_local_grasp(
            from_square,
            to_square,
            carried_grasp,
        )
        if selected_attempt["pickup_success"]:
            selected_grasp = selected_attempt["grasp_offset"].copy()
            selected_source = "local_search"
        else:
            selected_source = "failed_local_search"

    selected_pickup_success = (
        carried_success
        if selected_source == "carried"
        else selected_source == "local_search"
    )
    entry = {
        "rank": rank,
        "from_square": from_square,
        "to_square": to_square,
        "carried_grasp_offset": carried_grasp.copy(),
        "carried_pickup_success": carried_success,
        "selected_grasp_offset": selected_grasp.copy(),
        "selected_source": selected_source,
        "selected_pickup_success": selected_pickup_success,
        "carried_metrics": {
            "pickup_success": carried_result.get("pickup_success"),
            "premature_drop": carried_result.get("premature_drop"),
            "reject_reason": carried_result.get("reject_reason"),
            "trajectory_fk_error": carried_result.get("trajectory_fk_error"),
        },
        "local_search_attempts": attempts,
    }
    print(
        f"{from_square}: selected={selected_grasp} | "
        f"source={selected_source} | "
        f"pickup={'PASS' if selected_pickup_success else 'FAIL'}"
    )
    return entry


def main():
    diagnostics = build_initial_diagnostics()
    final_grasps = {}
    failure = None

    try:
        for rank in RANKS:
            print(f"\n######## Starting rank {rank} ########")
            last_working_grasp = INITIAL_GRASP_OFFSET.copy()
            for from_square in walk_squares_for_rank(rank):
                to_square = test_to_square(rank, from_square[0])
                entry = evaluate_square(
                    rank,
                    from_square,
                    to_square,
                    last_working_grasp.copy(),
                )
                diagnostics["results"].append(entry)

                if not entry["selected_pickup_success"]:
                    failure = {
                        "rank": rank,
                        "from_square": from_square,
                        "to_square": to_square,
                        "message": (
                            f"{from_square}: no pickup-successful grasp found; "
                            "stopping all-ranks initial grasp search"
                        ),
                    }
                    diagnostics["failure"] = failure
                    write_diagnostics(diagnostics)
                    raise GraspWalkFailure(failure["message"])

                last_working_grasp = np.array(
                    entry["selected_grasp_offset"],
                    dtype=float,
                )
                final_grasps[from_square] = last_working_grasp.copy()
                diagnostics["final_grasps"] = final_grasps
    finally:
        if p.isConnected():
            p.disconnect()

    write_diagnostics(diagnostics)
    OUTPUT_PATH.write_text(
        json.dumps(json_safe(final_grasps), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote final initial grasp defaults: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
