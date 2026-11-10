#!/bin/bash
# f3/f4/f5 -> f3/f4/f5, KNOWN mode (9x9 grid, +-10 mm), with the pivot-first PICKUP
# (--far-pickup-max-tilt 11, lift 'direct') and the pivot-first place. Per pair:
#   1. place offset = f3->f3 calibrated offset (start point)      tag fpp
#   2. median place calibration from 1                             fpickpiv_offsets/<src>_to_<dst>_median.json
#   3. rerun with the calibrated offset                            tag fpp_cal
# Resumable. Log: lowlevel/rook_kiri_xy_lookup/f_known_pickpivot_log.txt
cd /home/vince/LeRobot_OXAI
export PY=/home/vince/miniforge3/envs/IKsim_mj/bin/python LOOK=lowlevel/rook_kiri_xy_lookup
export BULL=lowlevel/drafts/run_rook_offset_bullseye.py CALPY=lowlevel/drafts/calibrate_place_offset_from_bullseye.py
export P0="-0.02151 -0.002751 -0.001731"
export FAR="--far-place-max-tilt 11 --far-place-style pivot --far-place-pivot-height 0.02 --far-pickup-max-tilt 11 --far-pickup-lift direct"
export COMMON="--mode known --grid-size 9 --inset 0.010 --grasp-offset -0.02 -0.001 -0.003 --workers 8"
exec > >(tee -a $LOOK/f_known_pickpivot_log.txt) 2>&1
echo "### started $(date "+%F %T")"
score() { $PY -c "
import json; s=json.load(open('$1'))['samples']; up=[x for x in s if x.get('landed_upright')]; m=sum(not x.get('pickup_success') for x in s)
print(f'{sum(x[\"xy_error_mm\"]<5 for x in up)}/{len(s)} within 5 mm, {sum(x[\"xy_error_mm\"]<2 for x in up)} within 2 mm, missed {m}, fell {len(s)-len(up)-m}')"; }
export -f score
chain() {
  local S=$1 T=$2 r cal P
  newest() { ls -d $LOOK/${S}_to_${T}_bullseye_known_corr0_grid9_$1_2026*/results.json 2>/dev/null | tail -1; }
  r=$(newest fpp)
  [ -z "$r" ] && { $PY -B $BULL $COMMON --from-square $S --to-square $T --place-offset $P0 $FAR --tag fpp > /dev/null 2>&1; r=$(newest fpp); }
  [ -z "$r" ] && { echo "$(date +%T) [$S->$T] FAILED"; return; }
  echo "$(date +%T) [$S->$T] start offset: $(score $r)"
  cal=$LOOK/fpickpiv_offsets/${S}_to_${T}_median.json
  [ -f $cal ] || $PY -B $CALPY "$(dirname $r)" --estimator median --output $cal > /dev/null 2>&1
  P=$($PY -c "import json;print(*json.load(open('$cal'))['$T']['new_place_offset'])")
  r=$(newest fpp_cal)
  [ -z "$r" ] && { $PY -B $BULL $COMMON --from-square $S --to-square $T --place-offset $P $FAR --tag fpp_cal > /dev/null 2>&1; r=$(newest fpp_cal); }
  echo "$(date +%T) [$S->$T] calibrated [$P]: $(score $r)"
}
export -f chain
for S in f3 f4 f5; do for T in f3 f4 f5; do echo "$S $T"; done; done | xargs -P 2 -L 1 bash -c 'chain $0 $1'
echo "=== $(date +%T) ALL DONE ==="
