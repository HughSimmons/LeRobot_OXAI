#!/usr/bin/env bash
# Pickup tolerance to joint execution error, c/d/e files. Resumable: re-run to continue.
#   setsid nohup bash lowlevel/drafts/run_pickup_tolerance_cde.sh > lowlevel/rook_kiri_xy_lookup/stability/pickup_tolerance/run_log.txt 2>&1 &
# Stages: FK bound -> e4 grasp search (k=2) -> best grasp -> e4 error-scale curve -> 18 squares
# (baseline vs best, k=1,2) -> report -> d4 grasp search (transfer check) -> report again.
set -u
cd /home/vince/LeRobot_OXAI
PY=/home/vince/miniforge3/envs/IKsim_mj/bin/python
D=lowlevel/drafts
OUT=lowlevel/rook_kiri_xy_lookup/stability/pickup_tolerance
mkdir -p "$OUT"
WORKERS=${WORKERS:-16}
SQUARES="d4 e4 d5 e5 c4 c5 d3 e3 d6 e6 c3 c6 c2 d2 e2 c7 d7 e7"
GX="-26 -23 -20 -17 -14"; GY="-7 -4 -1 2 5"; GZ="-9 -6 -3 0 3"
stamp() { date '+%F %T'; }
carry() { local f=${1:0:1} r=${1:1:1}; if [ "$r" -lt 7 ]; then echo "$f$((r + 1))"; else echo "$f$((r - 1))"; fi; }

# Keep Windows awake (idle sleep) while this script runs; the lid action is set separately (powercfg).
KEEP=""
if command -v powershell.exe >/dev/null 2>&1; then
  ENC=$(iconv -t UTF-16LE $D/keep_awake.ps1 | base64 -w0)
  (cd /mnt/c && exec powershell.exe -NoProfile -NonInteractive -EncodedCommand $ENC) >/dev/null 2>&1 &
  KEEP=$!
  echo "$(stamp) keep-awake pid $KEEP"
fi
trap '[ -n "$KEEP" ] && kill $KEEP 2>/dev/null' EXIT

echo "$(stamp) stage 0: FK bound"
[ -f "$OUT/fk_bound/fk_bound.csv" ] || $PY -B $D/joint_error_fk_bound.py --squares $SQUARES --out "$OUT/fk_bound" 2>&1 | grep -E "^[cde][0-9] |Error|Traceback"

echo "$(stamp) stage 1: e4 grasp search, k=2, 40 draws"
$PY -B $D/run_pickup_tolerance.py --square e4 --to-square e5 --k 2 --draws 40 --seed 0 \
  --grasp-x $GX --grasp-y $GY --grasp-z $GZ --workers $WORKERS --output-dir "$OUT/e4_search" | grep -vE "^\[[0-9]+/"
$PY -B $D/pickup_tolerance_select.py "$OUT/e4_search" --k 2 > /dev/null
BEST=$($PY -c "import json; print(' '.join(str(v) for v in json.load(open('$OUT/e4_search/best.json'))['best']['grasp_mm']))")
echo "$(stamp) best grasp (mm): $BEST"
GRASPS=(--grasp -20 -1 -3)
[ "$BEST" != "-20.0 -1.0 -3.0" ] && GRASPS+=(--grasp $BEST)

echo "$(stamp) stage 2: e4 error-scale curve (new draws, seed 1)"
$PY -B $D/run_pickup_tolerance.py --square e4 --to-square e5 --k 1 2 3 4 --draws 64 --seed 1 \
  "${GRASPS[@]}" --workers $WORKERS --output-dir "$OUT/e4_scale" | grep -vE "^\[[0-9]+/"

echo "$(stamp) stage 3: squares"
for sq in $SQUARES; do
  echo "$(stamp)   $sq -> $(carry $sq)"
  $PY -B $D/run_pickup_tolerance.py --square $sq --to-square $(carry $sq) --k 1 2 --draws 64 --seed 1 \
    "${GRASPS[@]}" --workers $WORKERS --output-dir "$OUT/squares/$sq" | grep -vE "^\[[0-9]+/"
done

echo "$(stamp) stage 4: report"
$PY -B $D/pickup_tolerance_report.py "$OUT" || echo "report failed"

echo "$(stamp) stage 5: d4 grasp search (does the e4 optimum transfer?)"
$PY -B $D/run_pickup_tolerance.py --square d4 --to-square d5 --k 2 --draws 40 --seed 0 \
  --grasp-x $GX --grasp-y $GY --grasp-z $GZ --workers $WORKERS --output-dir "$OUT/d4_search" | grep -vE "^\[[0-9]+/"
$PY -B $D/pickup_tolerance_select.py "$OUT/d4_search" --k 2 > /dev/null
$PY -B $D/pickup_tolerance_report.py "$OUT" || echo "report failed"
echo "$(stamp) ALL DONE"
