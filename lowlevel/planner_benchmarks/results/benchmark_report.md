# SO-101 Crowded-Board Trajectory Planning Benchmark Report

- **Physics Engine**: PyBullet `DIRECT` @ 240 Hz with 24-hull V-HACD gripper + wrist camera (`so101_with_mount_vhacd_gripper.urdf`)
- **Scenarios Tested**: 16 crowded-board scenarios (32-piece Staunton heights `45–92 mm`, tall blockers, near<->far moves, 2-leg captures)

## 1. Aggregate Planner Comparison

| Planner | Overall Success | Collateral-Free | Upright Landing (<15°) | Median Plan Time (ms) | Mean XY Error on Landed (mm) | Mean Settled Tilt on Landed (°) | Mean Max Collateral Disp (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `legacy_chess_traj` | **0/16 (0%)** | 4/16 | 3/16 | 42.7 ms | 6.93 mm | 14.38° | 27.03 mm |
| `dls_nullspace` | **7/16 (44%)** | 11/16 | 7/16 | 188.3 ms | 4.14 mm | 5.94° | 3.97 mm |
| `trajopt_spline` | **6/16 (38%)** | 11/16 | 6/16 | 172.0 ms | 3.15 mm | 6.00° | 3.62 mm |
| `rrt_connect` | **6/16 (38%)** | 11/16 | 6/16 | 160.8 ms | 4.14 mm | 5.11° | 3.95 mm |
| `hybrid_trajopt_rrt` | **6/16 (38%)** | 11/16 | 6/16 | 172.9 ms | 3.15 mm | 6.00° | 3.62 mm |

## 2. Per-Scenario Breakdown

| Scenario | Category | Planner | Success | Plan (ms) | Pick->Place Tilt (°) | Min SDF (mm) | Collateral Collisions | XY Err (mm) | Final Tilt (°) |
|---|---|---|:---:|---:|---:|---:|---:|---:|---:|
| `near_knight_b1_c3` | `1_near_vertical` | `legacy_chess_traj` | **FAIL** | 46.6 | 10.5->11.0 | -6.5 | 1 (5.0 mm) | 14.1 | 93.7 |
| `near_knight_b1_c3` | `1_near_vertical` | `dls_nullspace` | **PASS** | 143.5 | 0.0->0.0 | +2.1 | 0 (0.0 mm) | 6.1 | 0.0 |
| `near_knight_b1_c3` | `1_near_vertical` | `trajopt_spline` | **PASS** | 145.5 | 0.0->0.0 | +2.1 | 0 (0.0 mm) | 3.8 | 7.3 |
| `near_knight_b1_c3` | `1_near_vertical` | `rrt_connect` | **PASS** | 124.1 | 0.0->0.0 | +2.1 | 0 (0.0 mm) | 5.2 | 0.0 |
| `near_knight_b1_c3` | `1_near_vertical` | `hybrid_trajopt_rrt` | **PASS** | 136.3 | 0.0->0.0 | +2.1 | 0 (0.0 mm) | 3.8 | 7.3 |
| `near_pawn_e2_e4` | `1_near_vertical` | `legacy_chess_traj` | **FAIL** | 66.8 | 10.8->10.9 | -8.2 | 1 (6.2 mm) | 8.5 | 13.2 |
| `near_pawn_e2_e4` | `1_near_vertical` | `dls_nullspace` | **PASS** | 163.8 | 0.0->0.0 | +2.1 | 0 (0.8 mm) | 3.0 | 0.0 |
| `near_pawn_e2_e4` | `1_near_vertical` | `trajopt_spline` | **PASS** | 120.0 | 0.0->0.0 | +2.1 | 0 (0.9 mm) | 1.4 | 5.2 |
| `near_pawn_e2_e4` | `1_near_vertical` | `rrt_connect` | **PASS** | 121.6 | 0.0->0.0 | +2.1 | 0 (1.3 mm) | 6.3 | 0.1 |
| `near_pawn_e2_e4` | `1_near_vertical` | `hybrid_trajopt_rrt` | **PASS** | 111.4 | 0.0->0.0 | +2.1 | 0 (0.9 mm) | 1.4 | 5.2 |
| `near_queen_d1_d4` | `1_near_vertical` | `legacy_chess_traj` | **FAIL** | 48.5 | 10.6->10.8 | -11.5 | 2 (7.2 mm) | 39.6 | 93.5 |
| `near_queen_d1_d4` | `1_near_vertical` | `dls_nullspace` | **PASS** | 159.6 | 0.0->0.0 | -7.0 | 0 (0.0 mm) | 3.1 | 11.4 |
| `near_queen_d1_d4` | `1_near_vertical` | `trajopt_spline` | **PASS** | 206.2 | 0.0->0.0 | -7.0 | 0 (0.0 mm) | 1.8 | 0.0 |
| `near_queen_d1_d4` | `1_near_vertical` | `rrt_connect` | **FAIL** | 216.2 | 0.0->0.0 | -7.0 | 0 (0.0 mm) | 2.6 | 15.3 |
| `near_queen_d1_d4` | `1_near_vertical` | `hybrid_trajopt_rrt` | **PASS** | 411.7 | 0.0->0.0 | -7.0 | 0 (0.0 mm) | 1.8 | 0.0 |
| `near_bishop_c1_e3` | `1_near_vertical` | `legacy_chess_traj` | **FAIL** | 59.8 | 10.4->10.9 | -5.6 | 0 (1.3 mm) | 62.1 | 93.3 |
| `near_bishop_c1_e3` | `1_near_vertical` | `dls_nullspace` | **PASS** | 154.5 | 0.0->0.0 | +2.1 | 0 (0.0 mm) | 3.3 | 9.2 |
| `near_bishop_c1_e3` | `1_near_vertical` | `trajopt_spline` | **PASS** | 150.1 | 0.0->0.0 | +2.1 | 0 (0.0 mm) | 7.0 | 0.0 |
| `near_bishop_c1_e3` | `1_near_vertical` | `rrt_connect` | **PASS** | 124.3 | 0.0->0.0 | +2.1 | 0 (0.0 mm) | 3.6 | 7.7 |
| `near_bishop_c1_e3` | `1_near_vertical` | `hybrid_trajopt_rrt` | **PASS** | 147.4 | 0.0->0.0 | +2.1 | 0 (0.0 mm) | 7.0 | 0.0 |
| `preemptive_queen_d1_h5` | `2_near_to_far_preemptive` | `legacy_chess_traj` | **FAIL** | 35.9 | 9.8->40.3 | -9.1 | 2 (30.3 mm) | 57.7 | 36.3 |
| `preemptive_queen_d1_h5` | `2_near_to_far_preemptive` | `dls_nullspace` | **PASS** | 188.3 | 27.8->27.8 | -0.8 | 0 (0.0 mm) | 4.2 | 10.8 |
| `preemptive_queen_d1_h5` | `2_near_to_far_preemptive` | `trajopt_spline` | **PASS** | 205.7 | 27.8->27.8 | +0.2 | 0 (0.0 mm) | 0.4 | 10.3 |
| `preemptive_queen_d1_h5` | `2_near_to_far_preemptive` | `rrt_connect` | **PASS** | 162.9 | 27.8->27.8 | +0.7 | 0 (0.0 mm) | 7.4 | 0.0 |
| `preemptive_queen_d1_h5` | `2_near_to_far_preemptive` | `hybrid_trajopt_rrt` | **PASS** | 206.4 | 27.8->27.8 | +0.2 | 0 (0.0 mm) | 0.4 | 10.3 |
| `preemptive_pawn_e2_g6` | `2_near_to_far_preemptive` | `legacy_chess_traj` | **FAIL** | 43.8 | 10.8->22.8 | -8.2 | 1 (6.2 mm) | 40.3 | 33.8 |
| `preemptive_pawn_e2_g6` | `2_near_to_far_preemptive` | `dls_nullspace` | **FAIL** | 190.8 | 13.5->13.5 | +2.1 | 1 (12.2 mm) | 5.1 | 37.1 |
| `preemptive_pawn_e2_g6` | `2_near_to_far_preemptive` | `trajopt_spline` | **FAIL** | 165.7 | 13.5->13.5 | +2.1 | 1 (12.3 mm) | 2.5 | 19.1 |
| `preemptive_pawn_e2_g6` | `2_near_to_far_preemptive` | `rrt_connect` | **FAIL** | 160.2 | 13.5->13.5 | +2.1 | 1 (12.2 mm) | 2.1 | 17.7 |
| `preemptive_pawn_e2_g6` | `2_near_to_far_preemptive` | `hybrid_trajopt_rrt` | **FAIL** | 164.8 | 13.5->13.5 | +2.1 | 1 (12.3 mm) | 2.5 | 19.1 |
| `preemptive_bishop_c1_h6` | `2_near_to_far_preemptive` | `legacy_chess_traj` | **FAIL** | 34.6 | 9.3->35.6 | -17.7 | 0 (0.0 mm) | 16.1 | 93.3 |
| `preemptive_bishop_c1_h6` | `2_near_to_far_preemptive` | `dls_nullspace` | **PASS** | 188.3 | 28.8->28.8 | +2.1 | 0 (0.0 mm) | 4.2 | 10.1 |
| `preemptive_bishop_c1_h6` | `2_near_to_far_preemptive` | `trajopt_spline` | **FAIL** | 183.0 | 28.8->28.8 | +2.1 | 0 (0.0 mm) | 420.1 | 71.7 |
| `preemptive_bishop_c1_h6` | `2_near_to_far_preemptive` | `rrt_connect` | **PASS** | 154.5 | 28.8->28.8 | +2.1 | 0 (0.0 mm) | 0.8 | 0.0 |
| `preemptive_bishop_c1_h6` | `2_near_to_far_preemptive` | `hybrid_trajopt_rrt` | **FAIL** | 184.5 | 28.8->28.8 | +2.1 | 0 (0.0 mm) | 420.1 | 71.7 |
| `extreme_corner_a1_h8` | `2_near_to_far_preemptive` | `legacy_chess_traj` | **FAIL** | 37.3 | 9.9->43.9 | -1.6 | 0 (0.0 mm) | 14.3 | 94.0 |
| `extreme_corner_a1_h8` | `2_near_to_far_preemptive` | `dls_nullspace` | **FAIL** | 177.9 | 36.8->36.8 | +2.1 | 0 (0.0 mm) | 403.9 | 94.4 |
| `extreme_corner_a1_h8` | `2_near_to_far_preemptive` | `trajopt_spline` | **FAIL** | 164.5 | 36.8->36.8 | +2.1 | 0 (0.0 mm) | 432.8 | 147.4 |
| `extreme_corner_a1_h8` | `2_near_to_far_preemptive` | `rrt_connect` | **FAIL** | 155.0 | 36.8->36.8 | +2.1 | 0 (0.0 mm) | 403.9 | 94.4 |
| `extreme_corner_a1_h8` | `2_near_to_far_preemptive` | `hybrid_trajopt_rrt` | **FAIL** | 164.8 | 36.8->36.8 | +2.1 | 0 (0.0 mm) | 432.8 | 147.4 |
| `matched_drop_h5_d1` | `3_far_to_near_matched` | `legacy_chess_traj` | **FAIL** | 41.6 | 32.1->11.3 | -2.9 | 1 (28.5 mm) | 219.9 | 16.1 |
| `matched_drop_h5_d1` | `3_far_to_near_matched` | `dls_nullspace` | **FAIL** | 186.5 | 27.7->27.7 | -0.9 | 0 (0.0 mm) | 246.1 | 13.5 |
| `matched_drop_h5_d1` | `3_far_to_near_matched` | `trajopt_spline` | **FAIL** | 198.2 | 27.7->27.7 | +0.3 | 0 (0.0 mm) | 245.8 | 13.2 |
| `matched_drop_h5_d1` | `3_far_to_near_matched` | `rrt_connect` | **FAIL** | 162.7 | 27.7->27.7 | +1.1 | 0 (0.0 mm) | 245.5 | 12.4 |
| `matched_drop_h5_d1` | `3_far_to_near_matched` | `hybrid_trajopt_rrt` | **FAIL** | 199.0 | 27.7->27.7 | +0.3 | 0 (0.0 mm) | 245.8 | 13.2 |
| `matched_drop_g8_f6` | `3_far_to_near_matched` | `legacy_chess_traj` | **FAIL** | 25.1 | 27.7->38.5 | -9.8 | 0 (0.9 mm) | 1.6 | 15.8 |
| `matched_drop_g8_f6` | `3_far_to_near_matched` | `dls_nullspace` | **FAIL** | 228.1 | 22.8->22.8 | +2.1 | 1 (6.5 mm) | 82.7 | 93.7 |
| `matched_drop_g8_f6` | `3_far_to_near_matched` | `trajopt_spline` | **FAIL** | 179.2 | 22.8->22.8 | +2.1 | 1 (6.5 mm) | 82.7 | 93.7 |
| `matched_drop_g8_f6` | `3_far_to_near_matched` | `rrt_connect` | **FAIL** | 174.7 | 22.8->22.8 | +2.1 | 1 (6.5 mm) | 82.7 | 93.7 |
| `matched_drop_g8_f6` | `3_far_to_near_matched` | `hybrid_trajopt_rrt` | **FAIL** | 190.2 | 22.8->22.8 | +2.1 | 1 (6.5 mm) | 82.7 | 93.7 |
| `matched_drop_h8_d5` | `3_far_to_near_matched` | `legacy_chess_traj` | **FAIL** | 45.5 | 42.3->11.0 | -24.1 | 1 (5.0 mm) | 8.7 | 13.9 |
| `matched_drop_h8_d5` | `3_far_to_near_matched` | `dls_nullspace` | **FAIL** | 190.8 | 36.7->36.7 | +2.1 | 2 (22.4 mm) | 15.2 | 30.0 |
| `matched_drop_h8_d5` | `3_far_to_near_matched` | `trajopt_spline` | **FAIL** | 170.7 | 36.7->36.7 | +2.1 | 2 (22.9 mm) | 25.1 | 94.0 |
| `matched_drop_h8_d5` | `3_far_to_near_matched` | `rrt_connect` | **FAIL** | 153.4 | 36.7->36.7 | +2.1 | 2 (27.7 mm) | 37.8 | 94.0 |
| `matched_drop_h8_d5` | `3_far_to_near_matched` | `hybrid_trajopt_rrt` | **FAIL** | 170.4 | 36.7->36.7 | +2.1 | 2 (22.9 mm) | 25.1 | 94.0 |
| `far_ceiling_g1_h3` | `4_far_to_far_ceiling` | `legacy_chess_traj` | **FAIL** | 23.7 | 28.8->36.0 | -29.4 | 1 (16.5 mm) | 9.6 | 18.5 |
| `far_ceiling_g1_h3` | `4_far_to_far_ceiling` | `dls_nullspace` | **FAIL** | 211.4 | 28.4->28.4 | +2.1 | 1 (12.7 mm) | 134.7 | 93.7 |
| `far_ceiling_g1_h3` | `4_far_to_far_ceiling` | `trajopt_spline` | **FAIL** | 173.2 | 28.4->28.4 | +2.1 | 1 (5.8 mm) | 331.9 | 100.4 |
| `far_ceiling_g1_h3` | `4_far_to_far_ceiling` | `rrt_connect` | **FAIL** | 169.2 | 28.4->28.4 | +2.1 | 2 (6.4 mm) | 35.3 | 93.7 |
| `far_ceiling_g1_h3` | `4_far_to_far_ceiling` | `hybrid_trajopt_rrt` | **FAIL** | 175.3 | 28.4->28.4 | +2.1 | 1 (5.8 mm) | 331.9 | 100.4 |
| `far_ceiling_f1_h6` | `4_far_to_far_ceiling` | `legacy_chess_traj` | **FAIL** | 23.6 | 14.7->35.6 | -22.6 | 1 (15.6 mm) | 11.0 | 0.0 |
| `far_ceiling_f1_h6` | `4_far_to_far_ceiling` | `dls_nullspace` | **FAIL** | 194.8 | 28.8->28.8 | +2.1 | 1 (8.2 mm) | 214.8 | 8.2 |
| `far_ceiling_f1_h6` | `4_far_to_far_ceiling` | `trajopt_spline` | **FAIL** | 166.6 | 28.8->28.8 | +2.1 | 1 (8.7 mm) | 215.5 | 0.0 |
| `far_ceiling_f1_h6` | `4_far_to_far_ceiling` | `rrt_connect` | **FAIL** | 160.7 | 28.8->28.8 | +2.1 | 1 (8.2 mm) | 215.5 | 9.2 |
| `far_ceiling_f1_h6` | `4_far_to_far_ceiling` | `hybrid_trajopt_rrt` | **FAIL** | 166.6 | 28.8->28.8 | +2.1 | 1 (8.7 mm) | 215.5 | 0.0 |
| `far_ceiling_g5_g8` | `4_far_to_far_ceiling` | `legacy_chess_traj` | **FAIL** | 21.3 | 16.6->29.9 | -12.7 | 1 (4.5 mm) | 2.1 | 24.9 |
| `far_ceiling_g5_g8` | `4_far_to_far_ceiling` | `dls_nullspace` | **FAIL** | 183.9 | 23.9->23.9 | +2.1 | 0 (0.8 mm) | 2.3 | 38.3 |
| `far_ceiling_g5_g8` | `4_far_to_far_ceiling` | `trajopt_spline` | **FAIL** | 155.4 | 23.9->23.9 | +2.1 | 0 (0.8 mm) | 2.1 | 41.4 |
| `far_ceiling_g5_g8` | `4_far_to_far_ceiling` | `rrt_connect` | **FAIL** | 160.8 | 23.9->23.9 | +2.1 | 0 (0.9 mm) | 2.2 | 42.6 |
| `far_ceiling_g5_g8` | `4_far_to_far_ceiling` | `hybrid_trajopt_rrt` | **FAIL** | 150.5 | 23.9->23.9 | +2.1 | 0 (0.8 mm) | 2.1 | 41.4 |
| `capture_center_e4xd5` | `5_two_leg_capture` | `legacy_chess_traj` | **FAIL** | 102.5 | 10.7->10.9 | -5.8 | 1 (4.5 mm) | 59.9 | 34.7 |
| `capture_center_e4xd5` | `5_two_leg_capture` | `dls_nullspace` | **PASS** | 275.9 | 0.0->0.0 | +2.1 | 0 (0.0 mm) | 5.1 | 0.1 |
| `capture_center_e4xd5` | `5_two_leg_capture` | `trajopt_spline` | **PASS** | 216.4 | 0.0->0.0 | +2.1 | 0 (0.0 mm) | 5.2 | 0.0 |
| `capture_center_e4xd5` | `5_two_leg_capture` | `rrt_connect` | **PASS** | 244.1 | 0.0->0.0 | +2.1 | 0 (0.0 mm) | 5.2 | 0.0 |
| `capture_center_e4xd5` | `5_two_leg_capture` | `hybrid_trajopt_rrt` | **PASS** | 225.4 | 0.0->0.0 | +2.1 | 0 (0.0 mm) | 5.2 | 0.0 |
| `capture_far_c1xh6` | `5_two_leg_capture` | `legacy_chess_traj` | **FAIL** | 75.2 | 9.3->35.6 | -12.7 | 3 (300.9 mm) | 2.8 | 36.8 |
| `capture_far_c1xh6` | `5_two_leg_capture` | `dls_nullspace` | **FAIL** | 357.0 | 28.8->28.8 | +2.1 | 0 (0.0 mm) | 13.5 | 26.2 |
| `capture_far_c1xh6` | `5_two_leg_capture` | `trajopt_spline` | **FAIL** | 298.3 | 28.8->28.8 | +2.1 | 0 (0.0 mm) | 456.1 | 18.6 |
| `capture_far_c1xh6` | `5_two_leg_capture` | `rrt_connect` | **FAIL** | 306.0 | 28.8->28.8 | +2.1 | 0 (0.0 mm) | 28.6 | 36.8 |
| `capture_far_c1xh6` | `5_two_leg_capture` | `hybrid_trajopt_rrt` | **FAIL** | 310.6 | 28.8->28.8 | +2.1 | 0 (0.0 mm) | 456.1 | 18.6 |
