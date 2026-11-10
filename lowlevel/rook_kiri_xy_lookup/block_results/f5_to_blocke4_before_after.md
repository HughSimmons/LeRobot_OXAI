# f5 → 3×3 neighbourhood, blind mode: before vs after

Each cell: **≤5 mm · ≤2 mm · missed (M) / fell (F) · median error mm · median landing point (x, y) mm**

| target | before | after (median cal) |
|---|---|---|
| d5 | 17/150 · 1/150 · M25 F7 · 8.43 · (-5.3, +5.1) | 77/150 · 23/150 · M24 F9 · 3.71 · (-0.6, -0.0) |
| e5 | 19/150 · 1/150 · M24 F8 · 8.38 · (-5.9, +5.0) | 74/150 · 25/150 · M24 F8 · 3.86 · (-0.4, -0.1) |
| f5 | 20/150 · 1/150 · M25 F6 · 8.08 · (-3.6, +4.9) | 73/150 · 17/150 · M25 F10 · 4.34 · (-0.4, -0.1) |
| d4 | 23/150 · 1/150 · M24 F9 · 7.69 · (-5.9, +4.7) | 79/150 · 23/150 · M24 F7 · 3.46 · (-0.2, -0.3) |
| e4 | 21/150 · 1/150 · M24 F10 · 7.84 · (-5.9, +4.5) | 78/150 · 24/150 · M24 F8 · 3.57 · (+0.0, -0.3) |
| f4 | 25/150 · 0/150 · M25 F10 · 7.08 · (-2.9, +4.4) | 71/150 · 20/150 · M25 F7 · 4.08 · (-0.2, +0.1) |
| d3 | 27/150 · 2/150 · M24 F7 · 7.36 · (-6.1, +3.5) | 77/150 · 28/150 · M24 F6 · 3.71 · (-0.2, -0.0) |
| e3 | 27/150 · 3/150 · M24 F8 · 7.51 · (-6.3, +3.3) | 74/150 · 31/150 · M24 F7 · 3.84 · (-0.0, -0.1) |
| f3 | 37/150 · 2/150 · M25 F10 · 6.55 · (-3.1, +3.9) | 70/150 · 21/150 · M25 F6 · 4.11 · (+0.3, -0.1) |
| **total (d5, e5, f5, d4, e4, f4, d3, e3, f3)** | **216/1350** · 12/1350 · M220 F75 | **673/1350** · 212/1350 · M219 F68 |

Place offset used (mm, gripper frame):

| target | before | after (median cal) |
|---|---|---|
| d5 | [-8.75, 2.23, -2.67] | [-13.75, 7.55, -3.55] |
| e5 | [-8.75, 2.23, -2.67] | [-13.66, 8.05, -3.7] |
| f5 | [-8.75, 2.23, -2.67] | [-13.47, 5.9, -3.37] |
| d4 | [-8.75, 2.23, -2.67] | [-13.15, 8.2, -3.86] |
| e4 | [-8.75, 2.23, -2.67] | [-13.06, 8.14, -3.87] |
| f4 | [-8.75, 2.23, -2.67] | [-12.82, 5.5, -3.38] |
| d3 | [-8.75, 2.23, -2.67] | [-11.82, 8.42, -3.96] |
| e3 | [-8.75, 2.23, -2.67] | [-11.85, 8.44, -3.99] |
| f3 | [-8.75, 2.23, -2.67] | [-12.09, 5.81, -3.52] |

Run folders:

- d5 / before: `f5_to_d5_bullseye_blind_corr0_random150_blk_20261107_054630`
- d5 / after (median cal): `f5_to_d5_bullseye_blind_corr0_random150_blk_cal_median_20261107_055124`
- e5 / before: `f5_to_e5_bullseye_blind_corr0_random150_blk_20261107_055201`
- e5 / after (median cal): `f5_to_e5_bullseye_blind_corr0_random150_blk_cal_median_20261107_055647`
- f5 / before: `f5_to_f5_bullseye_blind_corr0_random150_blk_20261107_055606`
- f5 / after (median cal): `f5_to_f5_bullseye_blind_corr0_random150_blk_cal_median_20261107_060147`
- d4 / before: `f5_to_d4_bullseye_blind_corr0_random150_blk_20261107_060149`
- d4 / after (median cal): `f5_to_d4_bullseye_blind_corr0_random150_blk_cal_median_20261107_060703`
- e4 / before: `f5_to_e4_bullseye_blind_corr0_random150_blk_20261107_060813`
- e4 / after (median cal): `f5_to_e4_bullseye_blind_corr0_random150_blk_cal_median_20261107_061324`
- f4 / before: `f5_to_f4_bullseye_blind_corr0_random150_blk_20261107_061208`
- f4 / after (median cal): `f5_to_f4_bullseye_blind_corr0_random150_blk_cal_median_20261107_061810`
- d3 / before: `f5_to_d3_bullseye_blind_corr0_random150_blk_20261107_061820`
- d3 / after (median cal): `f5_to_d3_bullseye_blind_corr0_random150_blk_cal_median_20261107_062318`
- e3 / before: `f5_to_e3_bullseye_blind_corr0_random150_blk_20261107_062412`
- e3 / after (median cal): `f5_to_e3_bullseye_blind_corr0_random150_blk_cal_median_20261107_062908`
- f3 / before: `f5_to_f3_bullseye_blind_corr0_random150_blk_20261107_062802`
- f3 / after (median cal): `f5_to_f3_bullseye_blind_corr0_random150_blk_cal_median_20261107_063358`
