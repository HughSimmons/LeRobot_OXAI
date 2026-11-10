#!/bin/bash
# f sources -> every square of the d3-f5 block, with the pivot-first PICKUP (--far-pickup-max-tilt 11,
# lift 'direct') and the pivot-first place on f targets. Place offset calibrated in KNOWN mode, then
# tested blind. Per (source, target) pair:
#   1. known 9x9 (+-10 mm), place offset P0 = f3->f3 calibrated offset      tag fpp
#   2. median place calibration from 1                                       fpickpiv_offsets/<src>_to_<dst>_median.json
#   3. known 9x9 with the calibrated offset (check that known mode is good)  tag fpp_cal
#   4. blind 150, sigma 3.3 mm, seed 0, calibrated offset                    tag fpp_blind
# Steps 1-3 for f->f pairs are shared with run_f_known_pickpivot.sh (same tags/files), so they are reused.
# Resumable: steps whose results exist are skipped.
# Usage: run_f_pickpivot_block.sh [sources...]   (default f3)   CHAINS=2 WORKERS=8
# Log: lowlevel/rook_kiri_xy_lookup/f_pickpivot_block_log.txt
cd /home/vince/LeRobot_OXAI
export PY=/home/vince/miniforge3/envs/IKsim_mj/bin/python LOOK=lowlevel/rook_kiri_xy_lookup
export BULL=lowlevel/drafts/run_rook_offset_bullseye.py CALPY=lowlevel/drafts/calibrate_place_offset_from_bullseye.py
export P0="-0.02151 -0.002751 -0.001731"
export FAR="--far-place-max-tilt 11 --far-place-style pivot --far-place-pivot-height 0.02 --far-pickup-max-tilt 11 --far-pickup-lift direct"
export WORKERS=${WORKERS:-8}
export GRASP="--grasp-offset -0.02 -0.001 -0.003"
export KNOWN="--mode known --grid-size 9 --inset 0.010"
export BLIND="--mode blind --random-samples 150 --noise-sigma 0.0033 --seed 0"
CHAINS=${CHAINS:-2}
SOURCES=${*:-f3}
TARGETS="d5 e5 f5 d4 e4 f4 d3 e3 f3"
mkdir -p $LOOK/fpickpiv_offsets
exec > >(tee -a $LOOK/f_pickpivot_block_log.txt) 2>&1
echo "### started $(date "+%F %T"): sources [$SOURCES], $CHAINS chains x $WORKERS workers"
score() { $PY -c "
import json; s=[x for x in json.load(open('$1'))['samples'] if 'error' not in x]; up=[x for x in s if x.get('landed_upright')]; m=sum(not x.get('pickup_success') for x in s)
print(f'{sum(x[\"xy_error_mm\"]<5 for x in up)}/{len(s)} within 5 mm, {sum(x[\"xy_error_mm\"]<2 for x in up)} within 2 mm, missed {m}, fell {len(s)-len(up)-m}')"; }
export -f score
chain() {
  local S=$1 T=$2 r cal P
  newest() { ls -d $LOOK/${S}_to_${T}_bullseye_$1_2026*/results.json 2>/dev/null | tail -1; }
  run() { $PY -B $BULL $1 --from-square $S --to-square $T $GRASP --place-offset $2 $FAR --workers $WORKERS --tag $3 > /dev/null 2>&1; }
  r=$(newest known_corr0_grid9_fpp)
  [ -z "$r" ] && { run "$KNOWN" "$P0" fpp; r=$(newest known_corr0_grid9_fpp); }
  [ -z "$r" ] && { echo "$(date +%T) [$S->$T] known start: FAILED"; return; }
  echo "$(date +%T) [$S->$T] known, start offset: $(score $r)"
  cal=$LOOK/fpickpiv_offsets/${S}_to_${T}_median.json
  [ -f $cal ] || $PY -B $CALPY "$(dirname $r)" --estimator median --output $cal > /dev/null 2>&1
  P=$($PY -c "import json;d=json.load(open('$cal'));print(*d['$T']['new_place_offset']) if '$T' in d else None" 2>/dev/null)
  if [ -z "$P" ] || [ "$P" = "None" ]; then echo "$(date +%T) [$S->$T] calibration skipped (too few upright)"; return; fi
  r=$(newest known_corr0_grid9_fpp_cal)
  [ -z "$r" ] && { run "$KNOWN" "$P" fpp_cal; r=$(newest known_corr0_grid9_fpp_cal); }
  echo "$(date +%T) [$S->$T] known, calibrated [$P]: $(score $r)"
  r=$(newest blind_corr0_random150_fpp_blind)
  [ -z "$r" ] && { run "$BLIND" "$P" fpp_blind; r=$(newest blind_corr0_random150_fpp_blind); }
  [ -z "$r" ] && { echo "$(date +%T) [$S->$T] blind: FAILED"; return; }
  echo "$(date +%T) [$S->$T] BLIND, calibrated [$P]: $(score $r)"
}
export -f chain
for S in $SOURCES; do for T in $TARGETS; do echo "$S $T"; done; done | xargs -P $CHAINS -L 1 bash -c 'chain $0 $1'
echo "=== $(date +%T) ALL DONE ==="
