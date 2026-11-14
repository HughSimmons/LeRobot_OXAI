#!/usr/bin/env python3
"""Pickup tolerance to joint execution error (known mode, pickup phase only).

For each grasp offset and each error level k, run the same N paired draws: a joint error
delta = k * sigma * z (z ~ N(0, I), deg, joints pan/lift/elbow/wrist_flex/wrist_roll) and a random
start offset inside the square. The arm is driven to plan + delta (`run_sim_move(joint_offset_deg=...)`)
while the plan is made for the true rook position. Score = pickup only (stops at the end of the carry).

sigma defaults to Oliver's 14 Nov hardware - plan at rest, end of the grasp hold (RMS, URDF deg):
0.80 / 0.62 / 0.22 / 0.27 / 0.70.

Example (e4 grasp search at k = 2):
  python -B lowlevel/drafts/run_pickup_tolerance.py --square e4 --to-square e5 --k 2 --draws 32 \
      --grasp-x -26 -23 -20 -17 -14 --grasp-y -7 -4 -1 2 5 --grasp-z -9 -6 -3 0 3 --tag e4_search
"""

from __future__ import annotations

import argparse
from datetime import datetime
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_rook_offset_bullseye import ROOK_ENV, to_jsonable, worker_init  # noqa: E402
from run_stability_sweep import OCT16_PLACE_OFFSET, OUT_ROOT, mem_available_gb, run_job  # noqa: E402
from stability_probe import DEFAULT_THRESHOLDS  # noqa: E402

SIGMA_T1 = (0.80, 0.62, 0.22, 0.27, 0.70)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--square", required=True)
    ap.add_argument("--to-square", required=True)
    ap.add_argument("--grasp-x", nargs="+", type=float, default=[-20.0], help="mm")
    ap.add_argument("--grasp-y", nargs="+", type=float, default=[-1.0], help="mm")
    ap.add_argument("--grasp-z", nargs="+", type=float, default=[-3.0], help="mm")
    ap.add_argument("--grasp", nargs=3, type=float, action="append", default=None,
                    help="Explicit grasp (mm), repeatable; overrides the x/y/z grid.")
    ap.add_argument("--k", nargs="+", type=float, default=[2.0], help="Error levels (multiples of sigma).")
    ap.add_argument("--sigma", nargs=5, type=float, default=SIGMA_T1)
    ap.add_argument("--draws", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-inset", type=float, default=0.016, help="Random start half-width (m).")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-free-gb", type=float, default=1.0)
    ap.add_argument("--tag", default="tol")
    ap.add_argument("--output-dir", type=Path, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out = (args.output_dir or OUT_ROOT / "pickup_tolerance" /
           f"{args.square}_{args.tag}_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    z = rng.standard_normal((args.draws, 5))
    starts = rng.uniform(-args.start_inset, args.start_inset, (args.draws, 2))
    grasps = args.grasp or list(itertools.product(args.grasp_x, args.grasp_y, args.grasp_z))
    config = {
        "phase": "pickup", "mode": "known_joint_error", "from_square": args.square, "to_square": args.to_square,
        "sigma_deg": list(args.sigma), "k": args.k, "draws": args.draws, "seed": args.seed,
        "start_inset_m": args.start_inset, "grasps_mm": [list(g) for g in grasps],
        "place_offset": list(OCT16_PLACE_OFFSET), "lower_steps": [10], "trace_every": 0, "grip": {},
        "far_place_max_tilt_deg": None, "far_place_style": None, "far_place_pivot_height_m": None,
        "far_pickup_max_tilt_deg": None, "far_pickup_pivot_height_m": None, "far_pickup_lift": None,
        "thresholds": DEFAULT_THRESHOLDS, "rook_environment": ROOK_ENV,
    }
    (out / "config.json").write_text(json.dumps(to_jsonable(config), indent=2) + "\n")

    results_path = out / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                done.add((r["setting"], r["k"], r["draw"]))
    jobs = []
    for (gx, gy, gz), k in itertools.product(grasps, args.k):
        sid = f"g{gx:+.1f}_{gy:+.1f}_{gz:+.1f}"
        for i in range(args.draws):
            if (sid, k, i) in done:
                continue
            delta = (k * np.asarray(args.sigma) * z[i]).tolist()
            jobs.append({
                "setting": sid, "name": f"k{k:g}_d{i}", "row": None, "col": None, "draw": i, "k": k,
                "dx": float(starts[i, 0]), "dy": float(starts[i, 1]),
                "grasp_offset": [gx / 1000, gy / 1000, gz / 1000], "place_offset": list(OCT16_PLACE_OFFSET),
                "lower_steps": 10, "release_clearance": None, "joint_offset_deg": delta, "config": config,
            })
    print(f"{len(grasps)} grasps x {len(args.k)} k x {args.draws} draws; {len(done)} done, {len(jobs)} to run -> {out}",
          flush=True)

    started, n = time.time(), 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=worker_init, initargs=(str(out / "logs"), True), maxtasksperchild=8) as pool, \
            results_path.open("a", buffering=1) as fh:
        pending, queue = [], list(reversed(jobs))
        while queue or pending:
            while queue and len(pending) < 2 * args.workers and mem_available_gb() >= args.min_free_gb:
                pending.append(pool.apply_async(run_job, (queue.pop(),)))
            still = []
            for ar in pending:
                if ar.ready():
                    r = ar.get()
                    fh.write(json.dumps(to_jsonable(r)) + "\n")
                    n += 1
                    if n % 50 == 0 or n == len(jobs):
                        rate = n / (time.time() - started)
                        print(f"[{n}/{len(jobs)}] {rate:.2f}/s eta {(len(jobs) - n) / rate / 60:.0f} min", flush=True)
                else:
                    still.append(ar)
            pending = still
            time.sleep(0.2)
    print(f"Done: {results_path} ({time.time() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
