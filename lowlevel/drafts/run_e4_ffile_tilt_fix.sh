#!/bin/bash
# f-file fix for e4 -> f3/f4/f5. The far-file place pose released the rook at a 33-35 deg
# gripper tilt, past the rook's tipping angle, so it toppled (0-1/150 upright). With
# --far-place-max-tilt 11 the arm bends (shoulder/elbow/wrist) so the gripper is at most
# 11 deg from vertical when it lowers and releases, like the d/e files.
#   1. known 9x9 (+-10 mm), 16 Oct place offset           tag e4ftilt11known
#   2. blind 150 (sigma 3.3 mm, seed 0), 16 Oct place     tag e4ftilt11          ("before calibration")
#   3. median place calibration from step 2, blind 150    tag e4ftilt11_cal_median ("after")
# Log: lowlevel/rook_kiri_xy_lookup/2026-11-06_e4_neighbourhood/logs/e4_ffile_tilt_fix_log.txt
cd /home/vince/LeRobot_OXAI
PY=/home/vince/miniforge3/envs/IKsim_mj/bin/python
exec > >(tee -a lowlevel/rook_kiri_xy_lookup/2026-11-06_e4_neighbourhood/logs/e4_ffile_tilt_fix_log.txt) 2>&1
echo "### started $(date "+%F %T") on $(hostname)"
LOOK=lowlevel/rook_kiri_xy_lookup
CALPY=lowlevel/drafts/calibrate_place_offset_from_bullseye.py
BULL=lowlevel/drafts/run_rook_offset_bullseye.py
GRASP="-0.02 -0.001 -0.003"
PLACE0="-0.00875 0.00223 -0.00267"
TILT=11
TARGETS="f5 f4 f3"
newest() { ls -d $LOOK/$1 2>/dev/null | tail -1; }
SHOW='^[0-9]+ samples,|Results|Error|Traceback'

for T in $TARGETS; do
  echo "=== $(date +%T) known e4 -> $T (tilt cap $TILT) ==="
  $PY -B $BULL --mode known --from-square e4 --to-square $T --grid-size 9 --inset 0.010 --far-place-max-tilt $TILT \
    --grasp-offset $GRASP --place-offset $PLACE0 --workers 6 --tag e4ftilt11known 2>&1 | grep -E "$SHOW"
done

BEFORE=()
for T in $TARGETS; do
  echo "=== $(date +%T) blind e4 -> $T (tilt cap $TILT, 16 Oct place) ==="
  $PY -B $BULL --mode blind --from-square e4 --to-square $T --random-samples 150 --noise-sigma 0.0033 --seed 0 \
    --far-place-max-tilt $TILT --grasp-offset $GRASP --place-offset $PLACE0 --workers 6 --tag e4ftilt11 2>&1 | grep -E "$SHOW"
  BEFORE+=("$(newest "e4_to_${T}_bullseye_blind_corr0_random150_e4ftilt11_2026*")")
done

CAL=$LOOK/e4_ffile_tilt11_place_offsets_median.json
$PY -B $CALPY "${BEFORE[@]}" --estimator median --output $CAL 2>&1 | grep -vi collid
for T in $TARGETS; do
  P=$($PY -c "import json;d=json.load(open('$CAL'));print(*d['$T']['new_place_offset']) if '$T' in d else None")
  if [ -z "$P" ] || [ "$P" = "None" ]; then echo "=== median $T skipped ==="; continue; fi
  echo "=== $(date +%T) median e4 -> $T  place=[$P] ==="
  $PY -B $BULL --mode blind --from-square e4 --to-square $T --random-samples 150 --noise-sigma 0.0033 --seed 0 \
    --far-place-max-tilt $TILT --grasp-offset $GRASP --place-offset $P --workers 6 --tag e4ftilt11_cal_median 2>&1 | grep -E "$SHOW"
done
echo "=== $(date +%T) ALL DONE ==="
