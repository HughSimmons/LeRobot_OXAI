#!/bin/bash
# Place-offset calibration comparison for e4 -> 3x3 block. Three ways to calibrate:
#   clean  : mean landing error of a KNOWN-mode 9x9 (+-10 mm) run (no noise)
#   mean   : mean landing error of the 150-sample BLIND "before" run
#   median : median landing error of the same BLIND run
# Each calibrated place offset is then re-tested on the same 150 blind starts (seed 0).
# Targets with < 30 upright blind landings (e.g. the f file, where the rook topples)
# are skipped: there is no landing cloud to centre.
# Log of every step and run folder: lowlevel/rook_kiri_xy_lookup/e4_neighbourhood_calibration_log.txt
cd /home/vince/LeRobot_OXAI
PY=/home/vince/miniforge3/envs/IKsim_mj/bin/python
exec > >(tee -a lowlevel/rook_kiri_xy_lookup/e4_neighbourhood_calibration_log.txt) 2>&1
echo "### started $(date "+%F %T") on $(hostname)"
LOOK=lowlevel/rook_kiri_xy_lookup
CALPY=lowlevel/drafts/calibrate_place_offset_from_bullseye.py
BULL=lowlevel/drafts/run_rook_offset_bullseye.py
GRASP="-0.02 -0.001 -0.003"
PLACE0="-0.00875 0.00223 -0.00267"
ALL="e5 d5 f5 d4 f4 d3 e3 f3 e4"
newest() { ls -d $LOOK/$1 2>/dev/null | tail -1; }

# --- which targets can be calibrated (enough upright blind landings) ---
BEFORE=(); for T in $ALL; do BEFORE+=("$(newest "e4_to_${T}_bullseye_blind_corr0_random150_e4ring_2026*")"); done
$PY -B $CALPY "${BEFORE[@]}" --estimator median --output $LOOK/e4_neighbourhood_place_offsets_median.json 2>&1 | grep -vi collid
$PY -B $CALPY "${BEFORE[@]}" --estimator trimmed-mean --trim-mm 1000 --output $LOOK/e4_neighbourhood_place_offsets_mean.json 2>&1 | grep -vi collid
TARGETS=$($PY -c "import json;print(*json.load(open('$LOOK/e4_neighbourhood_place_offsets_median.json')))")
echo "calibratable targets: $TARGETS"

# --- phase A: clean (known-mode) runs for those targets ---
KNOWN=()
for T in $TARGETS; do
  echo "=== $(date +%T) known e4 -> $T ==="
  $PY -B $BULL --mode known --from-square e4 --to-square $T --grid-size 9 --inset 0.010 \
    --grasp-offset $GRASP --place-offset $PLACE0 --workers 6 --tag e4ringknown 2>&1 | grep -E "^[0-9]+ samples,|Results|Error|Traceback"
  KNOWN+=("$(newest "e4_to_${T}_bullseye_known_corr0_grid9_e4ringknown_2026*")")
done
$PY -B $CALPY "${KNOWN[@]}" --estimator trimmed-mean --trim-mm 1000 --output $LOOK/e4_neighbourhood_place_offsets_clean.json 2>&1 | grep -vi collid

# --- phase B: blind re-test with each calibrated place offset ---
for METHOD in clean mean median; do
  CAL=$LOOK/e4_neighbourhood_place_offsets_$METHOD.json
  for T in $TARGETS; do
    P=$($PY -c "import json;d=json.load(open('$CAL'));print(*d['$T']['new_place_offset']) if '$T' in d else None")
    if [ -z "$P" ] || [ "$P" = "None" ]; then echo "=== $METHOD $T skipped ==="; continue; fi
    echo "=== $(date +%T) $METHOD e4 -> $T  place=[$P] ==="
    $PY -B $BULL --mode blind --from-square e4 --to-square $T --random-samples 150 --noise-sigma 0.0033 --seed 0 \
      --grasp-offset $GRASP --place-offset $P --workers 6 --tag e4cal_$METHOD 2>&1 | grep -E "^[0-9]+ samples,|Results|Error|Traceback"
  done
done
echo "=== $(date +%T) ALL DONE ==="
