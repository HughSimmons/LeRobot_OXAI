#!/usr/bin/env python3
"""Sweep rook start offsets inside one square and record how accurately it lands on a target centre.

Two modes:
  known  - the robot is told the piece's true (offset) position and plans to it.
  blind  - the piece sits at the offset but the robot plans to the source-square
           centre, i.e. the offset is unseen placement noise.

Each sample runs the same pick/place pass as build_general_xy_lookup.run_candidate
(optionally with its sim-feedback placement corrections) but without the builder's
grasp-grid search and recovery fallbacks, so the raw tolerance is visible.

Example:
  python -B lowlevel/drafts/run_rook_offset_bullseye.py --mode blind --from-square e4 --to-square e5
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
LOWLEVEL_DIR = PROJECT_DIR / "lowlevel"
ROOK_MESH = PROJECT_DIR / "rook_kiri2" / "rook2.obj"
ROOK_VISUAL_MESH = PROJECT_DIR / "rook_kiri2" / "rook2_debug_orange_visual.obj"
BAND_COLLISION_DIR = (
    LOWLEVEL_DIR
    / "rook_kiri_lookup"
    / "collision_geometry_preview_20260801_163534"
)
ROOK_ENV = {
    "LOOKUP_PIECE_MODEL": "rook_kiri",
    "ROOK_KIRI_MESH_PATH": str(ROOK_MESH),
    "ROOK_KIRI_VISUAL_MESH_PATH": str(ROOK_VISUAL_MESH),
    "ROOK_KIRI_MESH_UP_AXIS": "y",
    "ROOK_KIRI_COLLISION_MODEL": "banded_hulls",
    "ROOK_KIRI_BAND_COLLISION_MESH_DIR": str(BAND_COLLISION_DIR),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("known", "blind"), required=True)
    parser.add_argument("--from-square", default="e4")
    parser.add_argument("--to-square", default="e5")
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument(
        "--inset",
        type=float,
        default=0.016,
        help="Maximum start offset from the source-square centre in metres (default 16 mm).",
    )
    parser.add_argument(
        "--random-samples",
        type=int,
        default=0,
        help="Instead of a grid, draw this many Gaussian start offsets.",
    )
    parser.add_argument("--noise-sigma", type=float, default=0.004, help="Gaussian sigma per axis (m).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--placement-corrections", type=int, default=0)
    parser.add_argument("--correction-gain", type=float, default=0.5)
    parser.add_argument("--grasp-offset", nargs=3, type=float, default=(-0.014, 0.002, -0.003))
    parser.add_argument("--place-offset", nargs=3, type=float, default=(-0.011, 0.002, -0.003))
    parser.add_argument("--placement-lower-steps", type=int, default=10)
    # Grip overrides; anything left unset keeps the multisim_chess_fast default.
    parser.add_argument("--soften-pinch", choices=("on", "off"), default=None)
    parser.add_argument("--soft-close-force", type=float, default=None)
    parser.add_argument("--soft-hold-force", type=float, default=None)
    parser.add_argument("--close-force", type=float, default=None)
    parser.add_argument("--hold-force", type=float, default=None)
    parser.add_argument("--moving-jaw-friction", type=float, default=None)
    parser.add_argument("--fixed-jaw-friction", type=float, default=None)
    parser.add_argument("--jaw-spinning-friction", type=float, default=None, help="Applied to both jaws.")
    parser.add_argument("--piece-friction", type=float, default=None)
    parser.add_argument("--only", nargs="+", default=None, help="Run only these sample names, e.g. r0_c1 r4_c8.")
    parser.add_argument("--tag", default=None, help="Suffix for the output directory name.")
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def sample_offsets(args: argparse.Namespace) -> list[dict]:
    if args.random_samples > 0:
        rng = np.random.default_rng(args.seed)
        draws = rng.normal(0.0, args.noise_sigma, size=(args.random_samples, 2))
        return [
            {"name": f"rand{i:03d}", "dx": float(dx), "dy": float(dy)}
            for i, (dx, dy) in enumerate(draws)
        ]
    steps = np.linspace(-args.inset, args.inset, args.grid_size)
    return [
        {"name": f"r{row}_c{col}", "row": row, "col": col, "dx": float(dx), "dy": float(dy)}
        for row, dy in enumerate(steps)
        for col, dx in enumerate(steps)
    ]


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def worker_init(log_dir: str) -> None:
    os.environ.update(ROOK_ENV)
    sys.path.insert(0, str(LOWLEVEL_DIR))
    os.chdir(PROJECT_DIR)
    # The sim prints heavily; keep each worker's chatter in its own log.
    log = open(Path(log_dir) / f"worker_{os.getpid()}.log", "w", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)


SIM_FORCE_OVERRIDES = {
    "soft_close_force": "SOFT_GRIPPER_CLOSE_FORCE",
    "soft_hold_force": "SOFT_GRIPPER_TRANSPORT_HOLD_FORCE",
    "close_force": "GRIPPER_CLOSE_FORCE",
    "hold_force": "GRIPPER_TRANSPORT_HOLD_FORCE",
}


def apply_grip_module_overrides(sim, grip: dict) -> None:
    if grip.get("soften_pinch") is not None:
        sim.SOFTEN_GRIPPER_PINCH = grip["soften_pinch"] == "on"
    for key, attr in SIM_FORCE_OVERRIDES.items():
        if grip.get(key) is not None:
            setattr(sim, attr, grip[key])
    if grip.get("piece_friction") is not None:
        # Mutated in place: create_piece reads this dict via active_piece_config().
        sim.ROOK_KIRI_DYNAMICS["lateralFriction"] = grip["piece_friction"]


def apply_jaw_friction(sim, robot_id: int, grip: dict) -> dict:
    """Override jaw contact friction after the world is built; returns the values in effect."""
    links = {
        sim.p.getJointInfo(robot_id, j)[12].decode(): j for j in range(sim.p.getNumJoints(robot_id))
    }
    jaws = {"moving": links["moving_jaw_so101_v1_link"], "fixed": links["gripper_link"]}
    for side, link in jaws.items():
        update = {}
        if grip.get(f"{side}_jaw_friction") is not None:
            update["lateralFriction"] = grip[f"{side}_jaw_friction"]
        if grip.get("jaw_spinning_friction") is not None:
            update["spinningFriction"] = grip["jaw_spinning_friction"]
        if update:
            sim.p.changeDynamics(robot_id, link, **update)
    return {
        side: {
            "lateral": sim.p.getDynamicsInfo(robot_id, link)[1],
            "spinning": sim.p.getDynamicsInfo(robot_id, link)[7],
        }
        for side, link in jaws.items()
    }


def run_sample(job: dict) -> dict:
    from board_coordinates import XYPoint, square_center_world_xy
    import build_general_xy_lookup as builder
    from chess_traj import lift_height_for_square
    import multisim_chess_fast as sim

    cfg = job["config"]
    apply_grip_module_overrides(sim, cfg["grip"])
    src_x, src_y = square_center_world_xy(cfg["from_square"])
    tgt_x, tgt_y = square_center_world_xy(cfg["to_square"])
    piece = XYPoint(src_x + job["dx"], src_y + job["dy"], name=f"piece_{job['name']}")
    if cfg["mode"] == "known":
        commanded = piece
    else:
        commanded = XYPoint(src_x, src_y, name=f"{cfg['from_square']}_center")
    target = XYPoint(tgt_x, tgt_y, name=f"{cfg['to_square']}_center")

    move_steps, _ = builder.default_move_steps_for_points(commanded, target)
    lift_height = float(lift_height_for_square(target))
    run_args = argparse.Namespace(
        placement_corrections=cfg["placement_corrections"],
        correction_gain=cfg["correction_gain"],
        move_steps=move_steps,
        placement_lower_steps=cfg["placement_lower_steps"],
    )

    started = time.time()
    sim.ensure_physics_connected()
    # Spawn the piece where it really is, then label the world with the commanded
    # start so run_sim_move plans its grasp from there and restores this state.
    world = sim.setup_sim_world(piece, edge_support_margin=builder.EDGE_SUPPORT_MARGIN)
    world["from_square"] = commanded
    jaw_friction = apply_jaw_friction(sim, world["robot_id"], cfg["grip"])
    try:
        outcome = builder.run_candidate(
            world,
            commanded,
            target,
            np.array(cfg["grasp_offset"], dtype=float),
            np.array(cfg["place_offset"], dtype=float),
            run_args,
            lift_height,
        )
    except Exception as exc:  # keep the sweep going; record the failure
        return {**job_record(job), "error": f"{type(exc).__name__}: {exc}", "seconds": time.time() - started}

    res = outcome["selected_result"]
    final = np.array(res.get("final_position", [np.nan] * 3), dtype=float)
    landed = (
        bool(res.get("pickup_success"))
        and np.all(np.isfinite(final))
        and float(res.get("final_tilt_deg", np.inf)) < sim.FINAL_TILT_TARGET_DEG
        and final[2] >= sim.active_piece_config()["dropped_z_threshold"] - 0.003
    )
    return {
        **job_record(job),
        "piece_world_xy": piece.as_xy(),
        "commanded_world_xy": commanded.as_xy(),
        "target_world_xy": target.as_xy(),
        "landed_upright": bool(landed),
        "strict_success": bool(outcome["success"]),
        "pickup_success": bool(res.get("pickup_success")),
        "reject_reason": res.get("reject_reason"),
        "premature_drop": bool(res.get("premature_drop")),
        "premature_drop_step": res.get("premature_drop_step"),
        "premature_drop_move_idx": res.get("premature_drop_move_idx"),
        "release_move_idx": res.get("release_move_idx"),
        "release_target_z": res.get("release_target_z"),
        # A rook lying on its side sits near z=0.014; upright rests near 0.026.
        "min_pre_release_piece_z": res.get("min_pre_release_piece_z"),
        "jaw_friction": jaw_friction,
        "final_world_xyz": final,
        "final_error_xy_mm": (final[:2] - np.array([tgt_x, tgt_y])) * 1000.0,
        "xy_error_mm": float(res.get("xy_error", np.inf)) * 1000.0,
        "final_tilt_deg": res.get("final_tilt_deg"),
        "passes_used": len(outcome["placement_passes"]),
        "seconds": time.time() - started,
    }


def job_record(job: dict) -> dict:
    return {k: job[k] for k in ("name", "row", "col", "dx", "dy") if k in job} | {
        "dx_mm": job["dx"] * 1000.0,
        "dy_mm": job["dy"] * 1000.0,
    }


def main() -> int:
    args = parse_args()
    for path in (ROOK_MESH, ROOK_VISUAL_MESH, BAND_COLLISION_DIR):
        if not path.exists():
            raise FileNotFoundError(path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    kind = f"random{args.random_samples}" if args.random_samples else f"grid{args.grid_size}"
    output_dir = (
        args.output_dir
        or LOWLEVEL_DIR
        / "rook_kiri_xy_lookup"
        / f"{args.from_square}_to_{args.to_square}_bullseye_{args.mode}_corr{args.placement_corrections}_{kind}"
        f"{'_' + args.tag if args.tag else ''}_{stamp}"
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    log_dir = output_dir / "logs"
    log_dir.mkdir()

    config = {
        "mode": args.mode,
        "from_square": args.from_square,
        "to_square": args.to_square,
        "grid_size": None if args.random_samples else args.grid_size,
        "inset_m": None if args.random_samples else args.inset,
        "random_samples": args.random_samples,
        "noise_sigma_m": args.noise_sigma if args.random_samples else None,
        "seed": args.seed if args.random_samples else None,
        "placement_corrections": args.placement_corrections,
        "correction_gain": args.correction_gain,
        "grasp_offset": list(args.grasp_offset),
        "place_offset": list(args.place_offset),
        "placement_lower_steps": args.placement_lower_steps,
        "grip": {
            key: getattr(args, key)
            for key in (
                "soften_pinch", *SIM_FORCE_OVERRIDES, "moving_jaw_friction",
                "fixed_jaw_friction", "jaw_spinning_friction", "piece_friction",
            )
        },
        "only": args.only,
        "rook_environment": ROOK_ENV,
    }
    samples = sample_offsets(args)
    if args.only:
        unknown = set(args.only) - {s["name"] for s in samples}
        if unknown:
            raise SystemExit(f"Unknown sample names: {sorted(unknown)}")
        samples = [s for s in samples if s["name"] in args.only]
    jobs = [{**s, "config": config} for s in samples]
    print(f"{len(jobs)} samples, {args.workers} workers -> {output_dir}", flush=True)

    started = time.time()
    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=worker_init, initargs=(str(log_dir),), maxtasksperchild=8) as pool:
        for i, result in enumerate(pool.imap_unordered(run_sample, jobs), start=1):
            results.append(to_jsonable(result))
            err = result.get("error") or (
                f"xy={result['xy_error_mm']:.2f}mm landed={result['landed_upright']}"
            )
            print(
                f"[{i}/{len(jobs)}] {result['name']} offset=({result['dx_mm']:+.1f},{result['dy_mm']:+.1f})mm {err}",
                flush=True,
            )

    results.sort(key=lambda r: r["name"])
    summary = {"config": config, "elapsed_seconds": time.time() - started, "samples": results}
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {output_dir / 'results.json'} ({time.time() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
