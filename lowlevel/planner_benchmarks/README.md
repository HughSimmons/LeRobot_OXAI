# `planner_benchmarks` — testing & candidate planners (not for deployment)

Everything here exists to evaluate the production planner in [`../planner`](../planner/README.md).
Nothing in `planner/` imports from this directory.

- `benchmark_crowded_board.py` — 16 crowded 32-piece scenarios x 5 planners, full 240 Hz PyBullet
  rollouts with the V-HACD gripper. Writes `results/benchmark_results.json` (git-ignored) and
  `results/benchmark_report.md`.
- `alternatives/` — rejected candidates: `trajopt_spline` (B-spline + SLSQP), `rrt_connect`
  (bi-directional RRT-Connect) and `hybrid_trajopt_rrt` (TrajOpt with RRT fallback).
- `legacy_chess_traj` (existing Placo + Catmull-Rom pipeline) is used as the baseline.

```bash
python lowlevel/planner_benchmarks/benchmark_crowded_board.py                      # full suite (~3.5 min)
python lowlevel/planner_benchmarks/benchmark_crowded_board.py --scenarios near_pawn_e2_e4 --planners dls_nullspace
python lowlevel/planner_benchmarks/benchmark_crowded_board.py --record-dir lowlevel/planner_benchmarks/videos
```

## Result summary (`results/benchmark_report.md`)

| Planner | Strict pass | Zero-collateral | Median plan | Mean XY err |
|---|---:|---:|---:|---:|
| legacy_chess_traj | 0/16 | 4/16 | 43 ms | 6.9 mm |
| **dls_nullspace (deployed)** | **7/16** | 11/16 | 188 ms | 4.1 mm |
| trajopt_spline | 6/16 | 11/16 | 172 ms | 3.2 mm |
| rrt_connect | 6/16 | 11/16 | 161 ms | 4.1 mm |
| hybrid_trajopt_rrt | 6/16 | 11/16 | 173 ms | 3.2 mm |

All remaining failures are far-file (`g/h`) pickups at the arm-reach ceiling, common to every planner.
