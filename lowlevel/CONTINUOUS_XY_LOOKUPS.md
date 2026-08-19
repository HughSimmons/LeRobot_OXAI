# Continuous XY Pick And Place

The continuous flow accepts arbitrary start and target XY coordinates while
leaving the existing square lookup files and commands unchanged.

## Coordinate frames

- `world`: metres in the robot/PyBullet world frame.
- `board`: metres relative to the centre of the board. The board centre is at
  world `(0.25, 0.0)`.

For reference, square-centre `d4` is board `(-0.02, -0.02)` and world
`(0.23, -0.02)`. Square-centre `d6` is board `(-0.02, 0.06)` and world
`(0.23, 0.06)`.

## Rook command

Run a continuous rook search and replay the saved result without video:

```bash
/opt/miniconda3/envs/IKsim_mj/bin/python -B \
  lowlevel/drafts/run_rook_xy_banded_lookup.py \
  --from-xy -0.02 -0.02 \
  --to-xy -0.02 0.06 \
  --frame board \
  --from-name d4_center \
  --to-name d6_center \
  --verify
```

Add `--video` to record the verification. Extra arguments that the wrapper
does not consume are forwarded to `build_general_xy_lookup.py`, for example:

```bash
  --grid-radius 2 --grid-z-radius 1 --placement-corrections 4 --full-grid
```

The default search orders grasp candidates by nearest physical offset and
stops on the first complete success. `--full-grid` evaluates every candidate.

## Output schema

Each search writes one `continuous_xy_lookup_v1` JSON. It stores:

- original input coordinates and frame;
- resolved world XY endpoints;
- selected grasp and placement offsets;
- lift, interpolation, correction, and grid settings;
- piece and collision configuration;
- every attempted grasp and placement-correction result.

The square lookup builder remains separate. Continuous runs do not read square
JSONs as donors, and square lookup files are not overwritten.

## Candidate database

Every evaluated placement pass that meets the normal final placement gate and
does not have an excessive premature drop is saved to:

```text
lowlevel/rook_kiri_xy_lookup/continuous_xy_candidates.sqlite3
```

The database is deduplicated by endpoint, offsets, search settings, piece
configuration, and drop policy. Re-running an identical candidate increments
its `seen_count` instead of adding an indistinguishable duplicate. A detected
drop is allowed only when it is within two trajectory waypoints of release and
less than 2 mm below the drop threshold; this preserves stable near-release
placements while excluding transport drops.

Pass `--candidate-db <path>` to `build_general_xy_lookup.py` (or through the
rook wrapper) to use a separate candidate database.

## Coarse pickup-region sweep

Sweep nine pickup coordinates across `d4`, command placement at `d6` centre,
and retain any stable final placement within the `d6` footprint:

```bash
/opt/miniconda3/envs/IKsim_mj/bin/python -B \
  lowlevel/drafts/run_rook_square_pickup_grid.py \
  --from-square d4 \
  --to-square d6
```

The default `3x3` grid uses offsets of `-10`, `0`, and `+10 mm` from the source
square centre on both axes. It intentionally uses a single grasp offset per
pickup coordinate so that this is a spatial pickup sweep, not a nested grasp
grid. Use `--dry-run` to print the nine commands without simulating them.

## Current policy compatibility

Reach pose and lift policies are selected from physical board position. A
continuous point in the `f`, `g`, or `h` board region receives the same far
reach policy as a square there, and an `h`-region target receives the existing
lower lift height. The exact square-centre `f1` lowering strategy is also
preserved for a continuous point at that coordinate.
