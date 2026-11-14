# f4 → 3×3 neighbourhood, blind mode: before vs after

Each cell: **≤5 mm · ≤2 mm · missed (M) / fell (F) · median error mm · median landing point (x, y) mm**

| target | before | after (median cal) |
|---|---|---|
| d5 | 9/150 · 0/150 · M25 F6 · 7.91 · (-3.6, +6.2) | 77/150 · 22/150 · M25 F8 · 3.75 · (-0.3, +0.0) |
| e5 | 11/150 · 0/150 · M25 F9 · 7.81 · (-4.2, +6.0) | 79/150 · 21/150 · M25 F6 · 3.92 · (+0.4, -0.2) |
| f5 | 22/150 · 0/150 · M25 F13 · 7.41 · (-2.3, +5.2) | 68/150 · 17/150 · M25 F11 · 4.38 · (+0.1, -0.1) |
| d4 | 17/150 · 0/150 · M25 F6 · 7.17 · (-3.9, +5.5) | 85/150 · 19/150 · M25 F3 · 3.97 · (-0.1, -0.2) |
| e4 | 14/150 · 0/150 · M25 F4 · 7.32 · (-4.0, +5.3) | 84/150 · 19/150 · M25 F5 · 3.86 · (-0.0, -0.0) |
| f4 | 23/150 · 3/150 · M25 F9 · 7.09 · (-2.6, +4.8) | 70/150 · 14/150 · M25 F9 · 4.27 · (+0.1, -0.1) |
| d3 | 25/150 · 0/150 · M25 F5 · 6.99 · (-4.5, +4.7) | 79/150 · 20/150 · M25 F7 · 3.68 · (-0.0, -0.0) |
| e3 | 25/150 · 0/150 · M25 F5 · 7.12 · (-4.5, +4.7) | 80/150 · 22/150 · M25 F6 · 3.60 · (+0.3, -0.1) |
| f3 | 23/150 · 6/150 · M25 F5 · 6.59 · (-1.8, +4.6) | 65/150 · 14/150 · M25 F9 · 4.26 · (-0.4, -0.1) |
| **total (d5, e5, f5, d4, e4, f4, d3, e3, f3)** | **169/1350** · 9/1350 · M225 F62 | **687/1350** · 168/1350 · M225 F64 |

Place offset used (mm, gripper frame):

| target | before | after (median cal) |
|---|---|---|
| d5 | [-8.75, 2.23, -2.67] | [-14.73, 6.16, -3.23] |
| e5 | [-8.75, 2.23, -2.67] | [-14.65, 6.51, -3.36] |
| f5 | [-8.75, 2.23, -2.67] | [-13.88, 4.71, -3.09] |
| d4 | [-8.75, 2.23, -2.67] | [-13.99, 6.41, -3.5] |
| e4 | [-8.75, 2.23, -2.67] | [-13.88, 6.38, -3.52] |
| f4 | [-8.75, 2.23, -2.67] | [-13.3, 5.28, -3.33] |
| d3 | [-8.75, 2.23, -2.67] | [-13.0, 6.99, -3.73] |
| e3 | [-8.75, 2.23, -2.67] | [-13.22, 6.82, -3.72] |
| f3 | [-8.75, 2.23, -2.67] | [-13.02, 4.68, -3.28] |

Run folders:

- d5 / before: `f4_to_d5_bullseye_blind_corr0_random150_blk_20261107_045844`
- d5 / after (median cal): `f4_to_d5_bullseye_blind_corr0_random150_blk_cal_median_20261107_050341`
- e5 / before: `f4_to_e5_bullseye_blind_corr0_random150_blk_20261107_050434`
- e5 / after (median cal): `f4_to_e5_bullseye_blind_corr0_random150_blk_cal_median_20261107_050940`
- f5 / before: `f4_to_f5_bullseye_blind_corr0_random150_blk_20261107_050829`
- f5 / after (median cal): `f4_to_f5_bullseye_blind_corr0_random150_blk_cal_median_20261107_051432`
- d4 / before: `f4_to_d4_bullseye_blind_corr0_random150_blk_20261107_051437`
- d4 / after (median cal): `f4_to_d4_bullseye_blind_corr0_random150_blk_cal_median_20261107_051934`
- e4 / before: `f4_to_e4_bullseye_blind_corr0_random150_blk_20261107_052031`
- e4 / after (median cal): `f4_to_e4_bullseye_blind_corr0_random150_blk_cal_median_20261107_052521`
- f4 / before: `f4_to_f4_bullseye_blind_corr0_random150_blk_20261107_052424`
- f4 / after (median cal): `f4_to_f4_bullseye_blind_corr0_random150_blk_cal_median_20261107_053042`
- d3 / before: `f4_to_d3_bullseye_blind_corr0_random150_blk_20261107_053010`
- d3 / after (median cal): `f4_to_d3_bullseye_blind_corr0_random150_blk_cal_median_20261107_053454`
- e3 / before: `f4_to_e3_bullseye_blind_corr0_random150_blk_20261107_053658`
- e3 / after (median cal): `f4_to_e3_bullseye_blind_corr0_random150_blk_cal_median_20261107_054140`
- f3 / before: `f4_to_f3_bullseye_blind_corr0_random150_blk_20261107_053954`
- f3 / after (median cal): `f4_to_f3_bullseye_blind_corr0_random150_blk_cal_median_20261107_054605`
