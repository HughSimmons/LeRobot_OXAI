# two_camera

Testbed for a two-camera setup: a stationary camera (e.g. tripod-mounted to
the side) plus a second camera mounted on the arm. Before this is pointed at
real hardware, this folder validates two things against known ground truth
from a PyBullet render: recovering a camera's distance to the board from a
single image (`cv2.solvePnP`), and relating the two cameras' views to each
other by synthesizing a common overhead view from each one independently.

Three scripts, two environments (same split as the rest of `lowlevel/`):

## 1. Generate sim images (`pybullet` env, e.g. `../realsimenv.yml`)

```
python generate_sim_images.py
```

Builds a board-only scene (no robot, no pieces by default) and renders it
from three hardcoded camera poses defined in `camera_rig.py`:

- `topdown`: straight down at the board center -- a ground-truth reference
  view, used only for evaluation, never as an input to the common-view
  synthesis below.
- `tripod`: stationary, off to the side, moderate elevation, further back.
- `wrist`: closer, steeper look-down, simulating a camera mounted on the arm.

Writes `output/{topdown,tripod,wrist}_camera.png` and `output/manifest.json`
(image size, derived camera intrinsics, and each camera's ground-truth
distance to the board plane).

By default the wrist camera is a **second hardcoded pose**, not driven by the
robot's forward kinematics -- there's no robot loaded in this scene.

Pass `--with-robot [--wrist-joints-deg PAN LIFT ELBOW WRIST_FLEX WRIST_ROLL GRIPPER]`
to load the SO101 with the wrist camera mount (`paths.MOUNT_URDF_PATH`) and
place the wrist camera where the arm actually puts it at that pose. Output
goes to `output/with_robot/`, so the baseline above is never overwritten.
The camera's placement on the mount, its field of view and the default arm
pose are in `camera_rig.py` (`WRIST_*`, `DEFAULT_WRIST_JOINTS_DEG`). Notes:

- The arm is in the scene, so it also shows up in the topdown/tripod views.
- The wrist camera gets its own FOV (`WRIST_FOV_DEG`, placeholder 80 deg until
  the real module is calibrated): at the shared 60 deg, no reachable arm pose
  sees all 49 inner corners. Each camera's intrinsics are written to the
  manifest, and scripts 2 and 3 use them when present.
- Image "up" (`WRIST_IMAGE_UP`) is chosen so the jaws sit at the bottom of the
  frame; check against a real photo from the camera.

Pass `--with-piece [--piece-square e4]` to render the same three views again
with a piece placed on the board, written to `output/with_piece/` so the
piece-free baseline above is never overwritten. Only do this after the
piece-free views have been calibrated (step 2) -- the piece is for
downstream piece-detection work, not for calibration.

## 2. Calibrate distance from one image (OpenCV env, e.g. `../coord_finder/live2fen_env.yml`)

```
python calibrate_camera_distance.py \
  --image output/tripod_camera.png \
  --manifest output/manifest.json --camera tripod
```

Detects the board's 7x7 interior corners with `cv2.findChessboardCornersSB`,
builds 3D object points from the known square size (default `0.04125` m,
matching `board_coordinates.DEFAULT_SQUARE_SIZE`), and runs `cv2.solvePnP` to
recover the camera pose. Reports the **perpendicular distance from the
camera to the board plane** (invariant to which detected corner the SB
detector happened to label first) and, in sim mode, compares it against the
manifest's ground truth, asserting the error is under `--tolerance-mm`
(default 1.0 mm).

For a real (non-sim) camera, pass intrinsics directly instead of a manifest:

```
python calibrate_camera_distance.py \
  --image some_real_photo.jpg \
  --fx 950 --fy 950 --cx 640 --cy 360 --square-size 0.04125
```

In this mode there's no ground truth to check against, so the script just
prints the recovered distance.

## 3. Synthesize a common view from each camera (OpenCV env)

```
python synthesize_common_view.py --manifest output/manifest.json --output-dir output
```

This is the "relate the two camera views to each other" piece: each of
`tripod_camera.png` and `wrist_camera.png` is independently self-calibrated
(corner detection + `solvePnP`, same as script 2) using nothing but that
camera's own image, the shared intrinsics, and the known square size. The
recovered pose gives an exact analytic board-plane-to-image homography
(`H = K @ [r1 r2 t]`, since the board is flat); inverting and warping each
image with `cv2.warpPerspective` produces a canonical overhead "common view"
for that camera. Writes `output/common_view_from_tripod.png` and
`output/common_view_from_wrist.png`, and prints the mean pixel difference
between them -- the two independently-derived views should closely agree.

`topdown_camera.png` is then run through the exact same self-calibration +
warp pipeline (`output/common_view_from_topdown.png`) purely so its result
can be diffed against the other two as a ground-truth check -- its pose is
never used to build the tripod/wrist common views, only its image, and only
after the fact.

### The checkerboard orientation ambiguity (and why it's handled here)

A plain, markerless NxN checkerboard is a real source of pose ambiguity:
`cv2.findChessboardCornersSB` can return any of 8 row/column orderings of
the same detected points, and for an even cell count (this board is 6x6
interior cells) exactly half of those orderings reproduce the board's true
light/dark coloring while the other half invert every square. `solvePnP`
fits whichever ordering it's given just as well either way (it's a genuine
symmetry of the correspondence, not a detector error), so two cameras
viewed from different enough angles can land in different halves and
produce synthesized views that are each individually "correct" but
color-inverted relative to each other.

`synthesize_common_view.py` resolves this with `resolve_corner_orientation`:
it samples real pixel brightness at the 36 interior cell centers under both
the raw corner order and a single row-flip, and keeps whichever matches the
board's known coloring. This uses only the board's own pixels, never a
camera's ground-truth pose, and is enough to fully disambiguate this board
(it has no other asymmetric features) -- a real board with a frame, marked
corner, or ChArUco-style markers would resolve the same ambiguity more
robustly.

## Caveats

- The real-photo detector in `coord_finder/test_chessboard_sb_detection.py`
  works around a spurious extra row/column caused by the physical wooden
  frame around a real board. The simulated board here has no such frame, so
  that confound is not expected -- but verify by overlaying detected corners
  on the rendered images (`cv2.drawChessboardCorners`) before trusting this
  on a real camera.
- `calibrate_camera_distance.py` never recomputes intrinsics from FOV itself
  -- it only reads them from the manifest or from CLI flags. `camera_rig.py`
  is the one place the FOV-to-intrinsics formula lives, so the two scripts
  can't silently drift apart.
