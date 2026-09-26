# coord_finder/

Vision pipeline: detect the chessboard in a camera photo, warp it to a
canonical top-down view, split it into 64 squares, and localise/extract any
piece sitting on each square. Pure image processing -- no PyBullet, no robot
control (that moved to `../kinematicscalibration/`).

## Environment

`pip install -r requirements.txt` (numpy, opencv-python-headless,
scikit-learn). The vendored `../../vision/LiveChess2FEN/` project supplies the
board-corner detector (`detect_board`); 8 files here `sys.path`-hack into it
at import time (`board_detect_lc2fen.py`, `robot_to_board_calibration.py`,
`run_pipeline_test01.py`, `run_pipeline_test02.py`, `localise_all_squares.py`,
`split_squares_trivial.py`, `test_chessboard_sb_detection.py`,
`test_03_svm_servo.py`).

The conda env used throughout this README is `Live2FEN`;
[`live2fen_env.yml`](live2fen_env.yml) is its full `conda env export` --
create it with:
```
conda env create -n Live2FEN -f lowlevel/coord_finder/live2fen_env.yml
```
Keep it in sync if you add a dependency:
`conda env export -n Live2FEN > lowlevel/coord_finder/live2fen_env.yml`.

## Example: run the pipeline on a bundled sample image

No camera needed -- `input/im2.jpeg` ships with the repo. From the repo root:
```bash
/opt/miniconda3/envs/Live2FEN/bin/python lowlevel/coord_finder/run_pipeline_test02.py \
  --image lowlevel/coord_finder/input/im2.jpeg \
  --out-dir /tmp/vision_smoketest \
  --a1-pos TR
```
This warps the board, splits it into 64 squares, isolates any piece on each
one, and writes an annotated crop per square plus a `summary.json` (occupancy
counts, per-square piece polarity) to `/tmp/vision_smoketest`. `--a1-pos`
controls which corner of the warped image the a1 square lands in -- `TR`
matches this sample photo's camera angle.

To try the standalone chessboard-corner diagnostic on the same image:
```bash
/opt/miniconda3/envs/Live2FEN/bin/python lowlevel/coord_finder/test_chessboard_sb_detection.py
```
writes its debug crop and undistorted-photo check under
`lowlevel/coord_finder/output/robot_to_board_calibration/`.

## Layout

- **Shared modules**:
  - `board_geometry.py` -- board-transform dataclass, classical board-corner
    detection, crop/mask/debug-save helpers
  - `board_detect_lc2fen.py` -- wraps LiveChess2FEN's corner detector;
    `detect_and_warp_board()` is imported by most scripts here
  - `localise_all_squares.py`, `split_squares_trivial.py` -- per-square
    segmentation / naive 8x8 grid split
  - `piece_from_square_crop.py` -- piece-centroid extraction from a square
    crop; imported directly by `run_pipeline_test02.py` (not superseded --
    despite the existence of a "signed" variant, see below)

- **Canonical end-to-end pipeline**: `run_pipeline_test02.py` (board warp →
  crops → piece isolation → SVM segmentation via `test_03_svm_servo.py`). It
  also imports `annotate_piece_crop` from `run_pipeline_test01.py` directly,
  so **both scripts are live dependencies** -- don't archive either one just
  because a newer-numbered file exists.
  ```
  python run_pipeline_test02.py --image input/im2.jpeg --out-dir output/pipeline_test02 --a1-pos TR
  ```

- **`robot_to_board_calibration.py`** -- interactive click-based measurement
  combining the LC2FEN homography with a tape-measure reading to compute the
  robot-to-board offset; writes `output/robot_to_board_calibration/board_meta.json`.

- **`test_chessboard_sb_detection.py`** -- standalone diagnostic, *not* wired
  into the pipeline above (see its own docstring): cross-checks
  `cv2.findChessboardCornersSB`'s interior-corner detection against LC2FEN's
  corner homography, and separately fits a single-photo camera calibration.
  Currently the most actively developed file here (see recent git log).
  ```
  python test_chessboard_sb_detection.py
  Hugh is looking into this to improve square cropping
  ```

- **`legacy/`** -- `signed_piece_from_square_crop.py`: despite its name
  suggesting it's the *current* piece-isolation approach, nothing in the repo
  imports it (verified) -- it's an unused standalone variant of
  `piece_from_square_crop.py`, kept for reference only.

- **`input/`** -- two bundled sample photos (`im2.jpeg`,
  `robot_to_board_calibration.png`) for running the pipeline without a camera.

- **`output/`** -- generated/scratch data (pipeline runs, calibration batches,
  debug crops). Not hand-maintained; safe to delete and regenerate.
