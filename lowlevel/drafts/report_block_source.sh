#!/bin/bash
# Figures + tables for one source square of the block queue (run_block_queue.sh):
#   figures/from_<src>__landings_on_9_targets__before_calibration.png   bullseye 3x3 (d3-f5), tag blk
#   figures/from_<src>__landings_on_9_targets__after_calibration.png    bullseye 3x3, tag blk_cal_median (+ before total)
#   tables/from_<src>__summary_before_vs_after.{md,csv}, figures/…png   numbers + dot plot
# Needs the run folders: extract ../rook_kiri_xy_lookup/_archive/raw_run_results_*.tar.xz first (see README).
# Usage: report_block_source.sh e3 [e5 ...]      Output: lowlevel/rook_kiri_xy_lookup/2026-11-07_block_d3-f5/figures/
cd /home/vince/LeRobot_OXAI
PY=/home/vince/miniforge3/envs/IKsim_mj/bin/python
D=lowlevel/drafts
OUT=lowlevel/rook_kiri_xy_lookup/2026-11-07_block_d3-f5/figures
TAB=lowlevel/rook_kiri_xy_lookup/2026-11-07_block_d3-f5/tables
mkdir -p $OUT $TAB
for S in "$@"; do
  fig=$OUT/from_${S}__landings_on_9_targets
  stem=$TAB/from_${S}__summary_before_vs_after
  $PY -B $D/plot_bullseye_neighbourhood.py --from-square $S --centre e4 --mode blind --tag blk \
    --label Before --note "16 Oct place offset; f targets: pivot-first place" \
    --missing-text "not run yet" --output ${fig}__before_calibration.png
  $PY -B $D/plot_bullseye_neighbourhood.py --from-square $S --centre e4 --mode blind --tag blk_cal_median \
    --label After --note "median place calibration per target; f targets: pivot-first place" \
    --compare-tag blk --missing-text "not run yet" --output ${fig}__after_calibration.png
  $PY -B $D/summarise_bullseye_conditions.py --from-square $S --centre e4 --mode blind \
    --condition "before=blk" --condition "after (median cal)=blk_cal_median" --output-stem ${stem}
  $PY -B $D/plot_before_after_summary.py ${stem}.csv \
    --source $S --title "$S → d3–f5 block: before vs after median place calibration" --output $OUT/from_${S}__summary_before_vs_after.png
done
