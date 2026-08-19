# Verified Candidate Supplement

The original 3x3 run used 100 move steps per waypoint. The following targeted
replays were run afterward with the same rook configuration and placement
offset:

| Pickup point | Grasp offset | Result | XY error | Tilt |
| --- | --- | --- | ---: | ---: |
| `f1_r1_c2` | `[-0.019, 0.002, -0.003]` | verified success | `0.0001733 m` | `0.0500 deg` |
| `f1_r0_c0` | `[-0.019, 0.002, -0.007]` | verification failed | `0.0153483 m` | `0.0504 deg` |

The `f1_r1_c2` success is treated as an additional valid candidate for the
continuous grid. The outward expansion uses the two closest successful grasp
offsets:

1. `[-0.019, 0.002, -0.005]` (base offset)
2. `[-0.019, 0.002, -0.003]` (verified `+z` offset)
