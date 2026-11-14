# e4 → 3×3 neighbourhood, blind mode: before vs after

Each cell: **≤5 mm · ≤2 mm · missed (M) / fell (F) · median error mm · median landing point (x, y) mm**

| target | before | after (median cal; f: pivot + median cal) |
|---|---|---|
| d5 | 125/150 · 88/150 · M1 F6 · 1.49 · (-1.0, +0.2) | 128/150 · 93/150 · M1 F9 · 1.24 · (+0.0, +0.0) |
| e5 | 126/150 · 91/150 · M1 F5 · 1.28 · (-0.7, +0.2) | 131/150 · 96/150 · M1 F3 · 1.32 · (-0.1, +0.0) |
| f5 | 0/150 · 0/150 · M1 F148 · 14.89 · (+3.3, -14.5) | 115/150 · 68/150 · M1 F26 · 1.81 · (+0.2, +0.0) |
| d4 | 128/150 · 80/150 · M1 F7 · 1.78 · (-0.9, -0.6) | 128/150 · 92/150 · M1 F7 · 1.24 · (+0.0, +0.0) |
| e4 | 129/150 · 85/150 · M1 F6 · 1.59 · (-0.7, -0.5) | 126/150 · 89/150 · M1 F9 · 1.31 · (-0.1, +0.0) |
| f4 | 0/150 · 0/150 · M1 F149 · – · – | 115/150 · 56/150 · M1 F22 · 2.31 · (-0.7, -0.2) |
| d3 | 125/150 · 30/150 · M1 F3 · 2.65 · (-0.9, -1.7) | 128/150 · 88/150 · M1 F3 · 1.35 · (+0.0, +0.0) |
| e3 | 126/150 · 54/150 · M1 F3 · 2.25 · (-0.6, -1.4) | 129/150 · 87/150 · M1 F4 · 1.42 · (-0.1, +0.1) |
| f3 | 0/150 · 0/150 · M1 F148 · 44.49 · (-8.3, -43.7) | 125/150 · 66/150 · M1 F13 · 2.04 · (+0.2, -0.5) |
| **total (d5, e5, f5, d4, e4, f4, d3, e3, f3)** | **759/1350** · 428/1350 · M9 F475 | **1125/1350** · 735/1350 · M9 F96 |

Place offset used (mm, gripper frame):

| target | before | after (median cal; f: pivot + median cal) |
|---|---|---|
| d5 | [-8.75, 2.23, -2.67] | [-9.19, 1.36, -2.85] |
| e5 | [-8.75, 2.23, -2.67] | [-9.08, 1.56, -2.81] |
| f5 | [-8.75, 2.23, -2.67] | [-5.24, -0.3, -2.21] |
| d4 | [-8.75, 2.23, -2.67] | [-9.82, 1.9, -2.83] |
| e4 | [-8.75, 2.23, -2.67] | [-9.57, 1.98, -2.8] |
| f4 | [-8.75, 2.23, -2.67] | [-4.24, -0.28, -2.01] |
| d3 | [-8.75, 2.23, -2.67] | [-10.62, 2.68, -2.73] |
| e3 | [-8.75, 2.23, -2.67] | [-10.22, 2.68, -2.72] |
| f3 | [-8.75, 2.23, -2.67] | [-4.28, 0.02, -1.98] |

Run folders:

- d5 / before: `e4_to_d5_bullseye_blind_corr0_random150_e4ring_20261106_110758`
- d5 / after (median cal; f: pivot + median cal): `e4_to_d5_bullseye_blind_corr0_random150_e4cal_median_20261106_155336`
- e5 / before: `e4_to_e5_bullseye_blind_corr0_random150_e4ring_20261106_110225`
- e5 / after (median cal; f: pivot + median cal): `e4_to_e5_bullseye_blind_corr0_random150_e4cal_median_20261106_154706`
- f5 / before: `e4_to_f5_bullseye_blind_corr0_random150_e4ring_20261106_111313`
- f5 / after (median cal; f: pivot + median cal): `e4_to_f5_bullseye_blind_corr0_random150_e4fpivot_cal_median_20261106_204417`
- d4 / before: `e4_to_d4_bullseye_blind_corr0_random150_e4ring_20261106_111918`
- d4 / after (median cal; f: pivot + median cal): `e4_to_d4_bullseye_blind_corr0_random150_e4cal_median_20261106_160005`
- e4 / before: `e4_to_e4_bullseye_blind_corr0_random150_e4ring_20261106_130544`
- e4 / after (median cal; f: pivot + median cal): `e4_to_e4_bullseye_blind_corr0_random150_e4cal_median_20261106_161853`
- f4 / before: `e4_to_f4_bullseye_blind_corr0_random150_e4ring_20261106_112439`
- f4 / after (median cal; f: pivot + median cal): `e4_to_f4_bullseye_blind_corr0_random150_e4fpivot_cal_median_20261106_205716`
- d3 / before: `e4_to_d3_bullseye_blind_corr0_random150_e4ring_20261106_113108`
- d3 / after (median cal; f: pivot + median cal): `e4_to_d3_bullseye_blind_corr0_random150_e4cal_median_20261106_160630`
- e3 / before: `e4_to_e3_bullseye_blind_corr0_random150_e4ring_20261106_125336`
- e3 / after (median cal; f: pivot + median cal): `e4_to_e3_bullseye_blind_corr0_random150_e4cal_median_20261106_161254`
- f3 / before: `e4_to_f3_bullseye_blind_corr0_random150_e4ring_20261106_125854`
- f3 / after (median cal; f: pivot + median cal): `e4_to_f3_bullseye_blind_corr0_random150_e4fpivot_cal_median_20261106_211254`
