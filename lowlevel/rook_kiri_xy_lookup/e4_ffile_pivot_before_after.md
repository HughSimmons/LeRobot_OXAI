# e4 → 3×3 neighbourhood, blind mode: before vs after

Each cell: **≤5 mm · ≤2 mm · missed (M) / fell (F) · median error mm · median landing point (x, y) mm**

| target | original far pose | pivot, 16 Oct place | pivot + median cal |
|---|---|---|---|
| d5 | 125/150 · 88/150 · M1 F6 · 1.49 · (-1.0, +0.2) | not run | not run |
| e5 | 126/150 · 91/150 · M1 F5 · 1.28 · (-0.7, +0.2) | not run | not run |
| f5 | 0/150 · 0/150 · M1 F148 · 14.89 · (+3.3, -14.5) | 75/150 · 1/150 · M1 F25 · 4.78 · (+2.0, +3.9) | 115/150 · 68/150 · M1 F26 · 1.81 · (+0.2, +0.0) |
| d4 | 128/150 · 80/150 · M1 F7 · 1.78 · (-0.9, -0.6) | not run | not run |
| e4 | 129/150 · 85/150 · M1 F6 · 1.59 · (-0.7, -0.5) | not run | not run |
| f4 | 0/150 · 0/150 · M1 F149 · – · – | 45/150 · 2/150 · M1 F23 · 5.41 · (+3.5, +3.8) | 115/150 · 56/150 · M1 F22 · 2.31 · (-0.7, -0.2) |
| d3 | 125/150 · 30/150 · M1 F3 · 2.65 · (-0.9, -1.7) | not run | not run |
| e3 | 126/150 · 54/150 · M1 F3 · 2.25 · (-0.6, -1.4) | not run | not run |
| f3 | 0/150 · 0/150 · M1 F148 · 44.49 · (-8.3, -43.7) | 51/150 · 1/150 · M1 F18 · 5.47 · (+4.1, +3.0) | 125/150 · 66/150 · M1 F13 · 2.04 · (+0.2, -0.5) |
| **total (f5, f4, f3)** | **0/450** · 0/450 · M3 F445 | **171/450** · 4/450 · M3 F66 | **355/450** · 190/450 · M3 F61 |

Place offset used (mm, gripper frame):

| target | original far pose | pivot, 16 Oct place | pivot + median cal |
|---|---|---|---|
| d5 | [-8.75, 2.23, -2.67] | – | – |
| e5 | [-8.75, 2.23, -2.67] | – | – |
| f5 | [-8.75, 2.23, -2.67] | [-8.75, 2.23, -2.67] | [-5.24, -0.3, -2.21] |
| d4 | [-8.75, 2.23, -2.67] | – | – |
| e4 | [-8.75, 2.23, -2.67] | – | – |
| f4 | [-8.75, 2.23, -2.67] | [-8.75, 2.23, -2.67] | [-4.24, -0.28, -2.01] |
| d3 | [-8.75, 2.23, -2.67] | – | – |
| e3 | [-8.75, 2.23, -2.67] | – | – |
| f3 | [-8.75, 2.23, -2.67] | [-8.75, 2.23, -2.67] | [-4.28, 0.02, -1.98] |

Run folders:

- d5 / original far pose: `e4_to_d5_bullseye_blind_corr0_random150_e4ring_20261106_110758`
- e5 / original far pose: `e4_to_e5_bullseye_blind_corr0_random150_e4ring_20261106_110225`
- f5 / original far pose: `e4_to_f5_bullseye_blind_corr0_random150_e4ring_20261106_111313`
- f5 / pivot, 16 Oct place: `e4_to_f5_bullseye_blind_corr0_random150_e4fpivot_20261106_200125`
- f5 / pivot + median cal: `e4_to_f5_bullseye_blind_corr0_random150_e4fpivot_cal_median_20261106_204417`
- d4 / original far pose: `e4_to_d4_bullseye_blind_corr0_random150_e4ring_20261106_111918`
- e4 / original far pose: `e4_to_e4_bullseye_blind_corr0_random150_e4ring_20261106_130544`
- f4 / original far pose: `e4_to_f4_bullseye_blind_corr0_random150_e4ring_20261106_112439`
- f4 / pivot, 16 Oct place: `e4_to_f4_bullseye_blind_corr0_random150_e4fpivot_20261106_201404`
- f4 / pivot + median cal: `e4_to_f4_bullseye_blind_corr0_random150_e4fpivot_cal_median_20261106_205716`
- d3 / original far pose: `e4_to_d3_bullseye_blind_corr0_random150_e4ring_20261106_113108`
- e3 / original far pose: `e4_to_e3_bullseye_blind_corr0_random150_e4ring_20261106_125336`
- f3 / original far pose: `e4_to_f3_bullseye_blind_corr0_random150_e4ring_20261106_125854`
- f3 / pivot, 16 Oct place: `e4_to_f3_bullseye_blind_corr0_random150_e4fpivot_20261106_202946`
- f3 / pivot + median cal: `e4_to_f3_bullseye_blind_corr0_random150_e4fpivot_cal_median_20261106_211254`
