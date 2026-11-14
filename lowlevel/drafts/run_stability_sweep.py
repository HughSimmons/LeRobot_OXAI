#!/usr/bin/env python3
"""Pickup / set-down stability sweep (known mode: the robot is told the true start offset).

Phase `pickup`: for every grasp offset on a grid, run every start offset on a grid through grasp,
lift and carry (stops when the arm starts lowering at the target) and score pickup stability.
Phase `setdown`: for every place setting (place-offset z, lowering steps), run every start offset
through the full move and score pickup + set-down stability. Accuracy is recorded, never scored.

Pass rules and fail reasons: `stability_probe.py`. Figures/tables: `stability_report.py`.

Results go to `<out>/results.jsonl`, one line per finished sample, so the run can be stopped and
resumed (re-run the same command with `--output-dir <out>`; finished samples are skipped). New
samples are held back while available RAM is below `--min-free-gb`.

Examples:
  # e4 pickup pilot: grasp x, y +-4 mm in 1 mm steps around (-20, -1, -3) mm; starts 9x9 +-16 mm
  python -B lowlevel/drafts/run_stability_sweep.py --phase pickup --from-square e4 --to-square e5 \
      --grasp-center -0.020 -0.001 -0.003 --grasp-span 0.004 --grasp-step 0.001 --workers 16
  # set-down on e4 -> e5 with a fixed grasp
  python -B lowlevel/drafts/run_stability_sweep.py --phase setdown --from-square e4 --to-square e5 \
      --grasp-center -0.020 -0.001 -0.003 --place-z -0.003 -0.001 0.001 --lower-steps 10 20 30
"""

from __future__ import annotations

import argparse
from datetime import datetime
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_rook_offset_bullseye import (  # noqa: E402
    LOWLEVEL_DIR,
    ROOK_ENV,
    apply_far_place_env,
    apply_grip_module_overrides,
    to_jsonable,
    worker_init,
)
from stability_probe import DEFAULT_THRESHOLDS, StabilityProbe, classify  # noqa: E402

OCT16_PLACE_OFFSET = (-0.00875, 0.00223, -0.00267)
OUT_ROOT = LOWLEVEL_DIR / "rook_kiri_xy_lookup" / "stability"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("pickup", "setdown"), required=True)
    ap.add_argument("--from-square", default="e4")
    ap.add_argument("--to-square", default="e5")
    ap.add_argument("--grasp-center", nargs=3, type=float, default=(-0.020, -0.001, -0.003))
    ap.add_argument("--grasp-span", type=float, default=0.0, help="Half-width of the grasp x/y grid (m); 0 = centre only.")
    ap.add_argument("--grasp-step", type=float, default=0.001)
    ap.add_argument("--grasp-z", nargs="+", type=float, default=None, help="Grasp z values (m); default centre z.")
    ap.add_argument("--place-offset", nargs=3, type=float, default=OCT16_PLACE_OFFSET)
    ap.add_argument("--place-z", nargs="+", type=float, default=None, help="Place z values (m); default place-offset z.")
    ap.add_argument("--lower-steps", nargs="+", type=int, default=[10])
    ap.add_argument("--release-clearance", nargs="+", type=float, default=None,
                    help="Release heights above the pickup height (m), sets CHESS_PLACE_RELEASE_CLEARANCE; "
                         "default: chess_traj default (0.02).")
    ap.add_argument("--grid-size", type=int, default=9)
    ap.add_argument("--inset", type=float, default=0.016, help="Start grid half-width (m).")
    ap.add_argument("--only", nargs="+", default=None, help="Only these start names (e.g. r4_c4).")
    ap.add_argument("--trace-every", type=int, default=0, help="Store a (z, tilt, slip) trace every N steps.")
    ap.add_argument("--far-place-max-tilt", type=float, default=None)
    ap.add_argument("--far-place-style", choices=("waypoint", "pivot"), default="waypoint")
    ap.add_argument("--far-place-pivot-height", type=float, default=0.02)
    ap.add_argument("--far-pickup-max-tilt", type=float, default=None)
    ap.add_argument("--far-pickup-pivot-height", type=float, default=0.02)
    ap.add_argument("--far-pickup-lift", choices=("retrace", "direct"), default="retrace")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-free-gb", type=float, default=1.0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    return ap.parse_args()


def grid_values(center: float, span: float, step: float) -> list[float]:
    if span <= 0:
        return [center]
    n = int(round(span / step))
    return [round(center + k * step, 6) for k in range(-n, n + 1)]


def build_settings(args) -> list[dict]:
    gx = grid_values(args.grasp_center[0], args.grasp_span, args.grasp_step)
    gy = grid_values(args.grasp_center[1], args.grasp_span, args.grasp_step)
    gz = args.grasp_z or [args.grasp_center[2]]
    pz = args.place_z or [args.place_offset[2]]
    settings = []
    clearances = args.release_clearance or [None]
    for x, y, z, plz, ls, cl in itertools.product(gx, gy, gz, pz, args.lower_steps, clearances):
        grasp = [x, y, z]
        place = [args.place_offset[0], args.place_offset[1], plz]
        sid = (
            f"g{x * 1e3:+.1f}_{y * 1e3:+.1f}_{z * 1e3:+.1f}"
            + (f"__p{plz * 1e3:+.2f}_ls{ls}" if args.phase == "setdown" else "")
            + (f"_c{cl * 1e3:.1f}" if cl is not None else "")
        )
        settings.append({"setting": sid, "grasp_offset": grasp, "place_offset": place, "lower_steps": ls,
                         "release_clearance": cl})
    return settings


def start_grid(args) -> list[dict]:
    steps = np.linspace(-args.inset, args.inset, args.grid_size)
    starts = [
        {"name": f"r{r}_c{c}", "row": r, "col": c, "dx": float(dx), "dy": float(dy)}
        for r, dy in enumerate(steps)
        for c, dx in enumerate(steps)
    ]
    if args.only:
        starts = [s for s in starts if s["name"] in set(args.only)]
    return starts


def run_job(job: dict) -> dict:
    from board_coordinates import XYPoint, square_center_world_xy
    import build_general_xy_lookup as builder
    from chess_traj import lift_height_for_square
    import multisim_chess_fast as sim

    cfg = job["config"]
    apply_far_place_env(cfg)
    apply_grip_module_overrides(sim, cfg["grip"])
    src_x, src_y = square_center_world_xy(cfg["from_square"])
    tgt_x, tgt_y = square_center_world_xy(cfg["to_square"])
    piece = XYPoint(src_x + job["dx"], src_y + job["dy"], name=f"piece_{job['name']}")
    target = XYPoint(tgt_x, tgt_y, name=f"{cfg['to_square']}_center")
    grasp = np.array(job["grasp_offset"], dtype=float)
    place = np.array(job["place_offset"], dtype=float)
    if job.get("release_clearance") is None:
        os.environ.pop("CHESS_PLACE_RELEASE_CLEARANCE", None)
    else:
        os.environ["CHESS_PLACE_RELEASE_CLEARANCE"] = str(job["release_clearance"])
    record = {k: job.get(k) for k in ("setting", "name", "row", "col", "dx", "dy", "grasp_offset", "place_offset",
                                      "lower_steps", "release_clearance", "joint_offset_deg", "draw", "k")}
    started = time.time()
    try:
        move_steps, _ = builder.default_move_steps_for_points(piece, target)
        lift_height = float(lift_height_for_square(target))
        sim.ensure_physics_connected()
        world = sim.setup_sim_world(piece, edge_support_margin=builder.EDGE_SUPPORT_MARGIN)
        world["from_square"] = piece  # known mode: plan the grasp at the true position
        # Build the trajectory exactly as run_sim_move would, so the probe knows the waypoint indices.
        with sim.temporary_home_joints(world.get("trajectory_home_joints_deg")):
            movelist, closeidx, traj_metrics = sim.pickupmove_traj_with_metrics(
                piece,
                target,
                board_origin=sim.board_origin,
                GRASP_OFFSET=grasp,
                PLACE_OFFSET=place,
                placement_lower_steps=job["lower_steps"],
                lift_height=lift_height,
                lower_place_path_bias=None,
            )
        probe = StabilityProbe(
            sim,
            world,
            closeidx=closeidx,
            release_idx=sim.find_release_move_index(movelist, closeidx),
            segments=traj_metrics["segments"],
            stop_after_carry=cfg["phase"] == "pickup",
            lifted_z=sim.PIECE_LIFTED_Z_THRESHOLD,
            trace_every=cfg["trace_every"],
        )
        res = sim.run_sim_move(
            world,
            piece,
            target,
            grasp,
            place_offset=place,
            return_metrics=True,
            record_video=False,
            move_steps_per_waypoint=move_steps,
            placement_lower_steps=job["lower_steps"],
            lift_height=lift_height,
            trajectory_override={"movelist": movelist, "closeidx": closeidx, "traj_metrics": traj_metrics},
            step_callback=probe,
            joint_offset_deg=job.get("joint_offset_deg"),
        )
        m = probe.finish()
    except Exception as exc:  # keep the sweep going; record the failure
        return {**record, "error": f"{type(exc).__name__}: {exc}", "seconds": time.time() - started}

    reasons = classify(m, cfg["phase"], cfg["thresholds"])
    out = {
        **record,
        "pass": not reasons,
        "reasons": reasons,
        "trajectory_valid": bool(res.get("trajectory_valid", True)),
        "metrics": m,
        "seconds": time.time() - started,
    }
    if cfg["phase"] == "setdown":  # accuracy: recorded only
        final = np.array(m["final"]["xyz"][:2])
        out["xy_error_mm"] = float(np.linalg.norm(final - np.array([tgt_x, tgt_y])) * 1000.0)
    return out


def mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    return float("inf")


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (
        args.output_dir
        or OUT_ROOT / args.phase / f"{args.from_square}_to_{args.to_square}_{args.phase}{'_' + args.tag if args.tag else ''}_{stamp}"
    ).expanduser().resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)

    config = {
        "phase": args.phase,
        "mode": "known",
        "from_square": args.from_square,
        "to_square": args.to_square,
        "grid_size": args.grid_size,
        "inset_m": args.inset,
        "grasp_center": list(args.grasp_center),
        "grasp_span_m": args.grasp_span,
        "grasp_step_m": args.grasp_step,
        "grasp_z": args.grasp_z,
        "place_offset": list(args.place_offset),
        "place_z": args.place_z,
        "lower_steps": args.lower_steps,
        "release_clearance": args.release_clearance,
        "trace_every": args.trace_every,
        "far_place_max_tilt_deg": args.far_place_max_tilt,
        "far_place_style": args.far_place_style if args.far_place_max_tilt is not None else None,
        "far_place_pivot_height_m": (
            args.far_place_pivot_height
            if args.far_place_max_tilt is not None and args.far_place_style == "pivot" else None
        ),
        "far_pickup_max_tilt_deg": args.far_pickup_max_tilt,
        "far_pickup_pivot_height_m": args.far_pickup_pivot_height if args.far_pickup_max_tilt is not None else None,
        "far_pickup_lift": args.far_pickup_lift if args.far_pickup_max_tilt is not None else None,
        "grip": {},
        "thresholds": DEFAULT_THRESHOLDS,
        "rook_environment": ROOK_ENV,
    }
    cfg_path = out / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        if {k: v for k, v in old.items() if k != "only"} != to_jsonable(config):
            raise SystemExit(f"{cfg_path} differs from this command's config; use a new --output-dir")
    else:
        cfg_path.write_text(json.dumps(to_jsonable(config), indent=2) + "\n")

    settings = build_settings(args)
    starts = start_grid(args)
    results_path = out / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                done.add((r["setting"], r["name"]))
    jobs = [
        {**s, **st, "config": config}
        for s in settings
        for st in starts
        if (s["setting"], st["name"]) not in done
    ]
    total = len(settings) * len(starts)
    print(f"{len(settings)} settings x {len(starts)} starts = {total}; {len(done)} done, {len(jobs)} to run; "
          f"{args.workers} workers -> {out}", flush=True)

    started = time.time()
    n_done = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=worker_init, initargs=(str(out / "logs"), True), maxtasksperchild=8) as pool, \
            results_path.open("a", buffering=1) as fh:
        pending = []
        queue = list(reversed(jobs))
        while queue or pending:
            # Keep at most 2 jobs per worker in flight; hold back while RAM is low.
            while queue and len(pending) < 2 * args.workers:
                if mem_available_gb() < args.min_free_gb:
                    if not pending:
                        time.sleep(5)
                        continue
                    break
                pending.append(pool.apply_async(run_job, (queue.pop(),)))
            still = []
            for ar in pending:
                if ar.ready():
                    r = ar.get()
                    fh.write(json.dumps(to_jsonable(r)) + "\n")
                    n_done += 1
                    status = r.get("error") or ("PASS" if r["pass"] else ",".join(r["reasons"]))
                    rate = n_done / max(1e-9, time.time() - started)
                    eta = (len(jobs) - n_done) / max(rate, 1e-9) / 60
                    print(f"[{n_done}/{len(jobs)}] {r['setting']} {r['name']} {status} "
                          f"({r['seconds']:.0f}s, {rate:.2f}/s, eta {eta:.0f} min, free {mem_available_gb():.1f} GB)",
                          flush=True)
                else:
                    still.append(ar)
            pending = still
            time.sleep(0.2)
    print(f"Done: {results_path} ({time.time() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
