#!/bin/bash
# f3/f4/f5 -> f3/f4/f5, KNOWN mode (robot told the true start offset), 9x9 grid within +-10 mm,
# grasp (-20,-1,-3) mm, pivot-first place on f, place offset = that pair's blind median calibration
# (block_place_offsets/<src>_to_<dst>_median.json). Tag fknown_cal. Resumable.
# Second pass: OFFDIR=block_place_offsets_known TAG=fknown_cal2 (offsets recalibrated from the known runs).
# Log: lowlevel/rook_kiri_xy_lookup/f_known_log.txt
cd /home/vince/LeRobot_OXAI
export PY=/home/vince/miniforge3/envs/IKsim_mj/bin/python LOOK=lowlevel/rook_kiri_xy_lookup
export BULL=lowlevel/drafts/run_rook_offset_bullseye.py
export OFFDIR=${OFFDIR:-block_place_offsets} TAG=${TAG:-fknown_cal}
export FAR="--far-place-max-tilt 11 --far-place-style pivot --far-place-pivot-height 0.02"
exec > >(tee -a $LOOK/f_known_log.txt) 2>&1
echo "### started $(date "+%F %T")"
one() {
  local S=$1 T=$2
  local P=$($PY -c "import json;print(*json.load(open('$LOOK/$OFFDIR/${S}_to_${T}_median.json'))['$T']['new_place_offset'])")
  local r=$(ls -d $LOOK/${S}_to_${T}_bullseye_known_corr0_grid9_${TAG}_2026*/results.json 2>/dev/null | tail -1)
  if [ -z "$r" ]; then
    $PY -B $BULL --mode known --from-square $S --to-square $T --grid-size 9 --inset 0.010 \
      --grasp-offset -0.02 -0.001 -0.003 --place-offset $P $FAR --workers 8 --tag $TAG > /dev/null 2>&1
    r=$(ls -d $LOOK/${S}_to_${T}_bullseye_known_corr0_grid9_${TAG}_2026*/results.json 2>/dev/null | tail -1)
  fi
  echo "$(date +%T) $TAG [$S->$T] place [$P]: $($PY -c "
import json,collections; s=json.load(open('$r'))['samples']; up=[x for x in s if x.get('landed_upright')]
print(f'{sum(x[\"xy_error_mm\"]<5 for x in up)}/{len(s)} within 5 mm, {sum(x[\"xy_error_mm\"]<2 for x in up)} within 2 mm, {len(up)} upright, missed {sum(not x.get(\"pickup_success\") for x in s)}')")"
}
export -f one
for S in f3 f4 f5; do for T in f3 f4 f5; do echo "$S $T"; done; done | xargs -P 2 -L 1 bash -c 'one $0 $1'
echo "=== $(date +%T) ALL DONE ==="
