#!/bin/bash
# Figures + tables for one source square of the block queue (run_block_queue.sh):
#   <src>_to_blocke4_bullseye_blind_before.png   bullseye 3x3 (d3-f5), tag blk
#   <src>_to_blocke4_bullseye_blind_after.png    bullseye 3x3, tag blk_cal_median (+ before total)
#   <src>_to_blocke4_before_after.{md,csv,png}   numbers + dot plot
# Usage: report_block_source.sh e3 [e5 ...]      Output: lowlevel/rook_kiri_xy_lookup/block_results/
cd /home/vince/LeRobot_OXAI
PY=/home/vince/miniforge3/envs/IKsim_mj/bin/python
D=lowlevel/drafts
OUT=lowlevel/rook_kiri_xy_lookup/block_results
mkdir -p $OUT
for S in "$@"; do
  stem=$OUT/${S}_to_blocke4
  $PY -B $D/plot_bullseye_neighbourhood.py --from-square $S --centre e4 --mode blind --tag blk \
    --label Before --note "16 Oct place offset; f targets: pivot-first place" \
    --missing-text "not run yet" --output ${stem}_bullseye_blind_before.png
  $PY -B $D/plot_bullseye_neighbourhood.py --from-square $S --centre e4 --mode blind --tag blk_cal_median \
    --label After --note "median place calibration per target; f targets: pivot-first place" \
    --compare-tag blk --missing-text "not run yet" --output ${stem}_bullseye_blind_after.png
  $PY -B $D/summarise_bullseye_conditions.py --from-square $S --centre e4 --mode blind \
    --condition "before=blk" --condition "after (median cal)=blk_cal_median" --output-stem ${stem}_before_after
  $PY -B $D/plot_before_after_summary.py ${stem}_before_after.csv \
    --source $S --title "$S → d3–f5 block: before vs after median place calibration" --output ${stem}_before_after.png
done
