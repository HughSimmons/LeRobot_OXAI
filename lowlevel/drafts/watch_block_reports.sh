#!/bin/bash
# Make report_block_source.sh figures/tables for each source as soon as all 9 of its chains
# have finished (after-run done, skipped or failed) in 2026-11-07_block_d3-f5/logs/block_queue_log.txt.
cd /home/vince/LeRobot_OXAI
LOG=lowlevel/rook_kiri_xy_lookup/2026-11-07_block_d3-f5/logs/block_queue_log.txt
for S in ${*:-e5 d3 d4 d5 f3 f4 f5}; do
  until [ "$(grep -cE "\[$S->[a-h][1-8]\] (after|before: FAILED)" $LOG)" -ge 9 ]; do sleep 60; done
  lowlevel/drafts/report_block_source.sh $S > /dev/null 2>&1
  echo "$(date +%T) report for $S: $(grep -o '\*\*total.*' lowlevel/rook_kiri_xy_lookup/2026-11-07_block_d3-f5/tables/from_${S}__summary_before_vs_after.md)"
done
