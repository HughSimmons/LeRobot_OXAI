#!/bin/bash
# Queue: rook from each source square to every square of the d3-f5 block (3x3 around e4),
# same protocol as the e4 sweep, run as one resumable queue.
#
#   per (source, target) pair, a "chain":
#     1. blind 150 (sigma 3.3 mm, seed 0), 16 Oct place offset             tag blk
#     2. median place calibration from 1 (needs >= 30 upright landings)   place_offsets/block_blind_median/<src>_to_<dst>_median.json
#     3. blind 150 with the calibrated place offset                        tag blk_cal_median
#   f targets use the pivot-first place (--far-place-max-tilt 11 --far-place-style pivot);
#   the flags have no effect on d/e targets.
#
# Speed: workers load no robot driver/torch and no IK meshes (sim_no_robot_driver, ~340 MB
# instead of ~1 GB per worker, results bit-identical), so 16 workers fit in RAM. CHAINS chains
# run at once with WORKERS workers each, so one run's slow tail overlaps the next run's start.
# Resumable: a step whose results already exist is skipped, so the script can simply be re-run.
#
# Usage: run_block_queue.sh [sources...]        (default: e3 e5 d3 d4 d5)
# Log:   lowlevel/rook_kiri_xy_lookup/2026-11-07_block_d3-f5/logs/block_queue_log.txt
cd /home/vince/LeRobot_OXAI
export PY=/home/vince/miniforge3/envs/IKsim_mj/bin/python
export LOOK=lowlevel/rook_kiri_xy_lookup
export CALPY=lowlevel/drafts/calibrate_place_offset_from_bullseye.py
export BULL=lowlevel/drafts/run_rook_offset_bullseye.py
export GRASP="-0.02 -0.001 -0.003"
export PLACE0="-0.00875 0.00223 -0.00267"
export FAR="--far-place-max-tilt 11 --far-place-style pivot --far-place-pivot-height 0.02"
export NOISE="--mode blind --random-samples 150 --noise-sigma 0.0033 --seed 0"
CHAINS=${CHAINS:-2}
export WORKERS=${WORKERS:-8}
SOURCES=${*:-e3 e5 d3 d4 d5}
TARGETS="d5 e5 f5 d4 e4 f4 d3 e3 f3"
LOG=$LOOK/2026-11-07_block_d3-f5/logs/block_queue_log.txt
mkdir -p $LOOK/place_offsets/block_blind_median
exec > >(tee -a $LOG) 2>&1
echo "### started $(date "+%F %T") on $(hostname): sources [$SOURCES], $CHAINS chains x $WORKERS workers"

chain() {
  local S=$1 T=$2
  local tag="[$S->$T]"
  newest() { ls -d $LOOK/$1 2>/dev/null | tail -1; }
  local before after cal P
  before=$(newest "${S}_to_${T}_bullseye_blind_corr0_random150_blk_2026*/results.json")
  if [ -z "$before" ]; then
    echo "$(date +%T) $tag before: start"
    $PY -B $BULL $NOISE --from-square $S --to-square $T --grasp-offset $GRASP --place-offset $PLACE0 \
      $FAR --workers $WORKERS --tag blk > /dev/null 2>&1
    before=$(newest "${S}_to_${T}_bullseye_blind_corr0_random150_blk_2026*/results.json")
    [ -z "$before" ] && { echo "$(date +%T) $tag before: FAILED (no results.json)"; return; }
  fi
  echo "$(date +%T) $tag before: $($PY -c "
import json; s=[x for x in json.load(open('$before'))['samples'] if 'error' not in x]
up=[x for x in s if x.get('landed_upright')]; print(f'{sum(x[\"xy_error_mm\"]<5 for x in up)}/{len(s)} within 5 mm, {len(up)} upright')")"
  cal=$LOOK/place_offsets/block_blind_median/${S}_to_${T}_median.json
  [ -f "$cal" ] || $PY -B $CALPY "$(dirname $before)" --estimator median --output $cal > /dev/null 2>&1
  P=$($PY -c "import json;d=json.load(open('$cal'));print(*d['$T']['new_place_offset']) if '$T' in d else None" 2>/dev/null)
  if [ -z "$P" ] || [ "$P" = "None" ]; then echo "$(date +%T) $tag after: skipped (fewer than 30 upright landings)"; return; fi
  after=$(newest "${S}_to_${T}_bullseye_blind_corr0_random150_blk_cal_median_2026*/results.json")
  if [ -z "$after" ]; then
    $PY -B $BULL $NOISE --from-square $S --to-square $T --grasp-offset $GRASP --place-offset $P \
      $FAR --workers $WORKERS --tag blk_cal_median > /dev/null 2>&1
    after=$(newest "${S}_to_${T}_bullseye_blind_corr0_random150_blk_cal_median_2026*/results.json")
    [ -z "$after" ] && { echo "$(date +%T) $tag after: FAILED (no results.json)"; return; }
  fi
  echo "$(date +%T) $tag after (place [$P]): $($PY -c "
import json; s=[x for x in json.load(open('$after'))['samples'] if 'error' not in x]
up=[x for x in s if x.get('landed_upright')]; print(f'{sum(x[\"xy_error_mm\"]<5 for x in up)}/{len(s)} within 5 mm, {len(up)} upright')")"
}
export -f chain

for S in $SOURCES; do for T in $TARGETS; do echo "$S $T"; done; done | xargs -P $CHAINS -L 1 bash -c 'chain $0 $1'
echo "=== $(date +%T) ALL DONE ==="
