#!/bin/bash
# W41 반복 측정 — 자산 1종당 Baseline 3회 + Ours 3회.
# 로그 각 줄에 경과초를 찍는다 (run_voxhammer_checked.sh 와 같은 방식). 단계별 시간을 사후에 쪼개려면 필요하다.
set -uo pipefail
VOXHAMMER_DIR="${VOXHAMMER_DIR:-<HOME>/ai-ar-v2/VoxHammer}"
TRACK_ROOT="${TRACK_ROOT:-<HOME>/ai-ar-data/vh-track}"
VH_TOOLS="${VH_TOOLS:-<HOME>/ai-ar-v3/tools}"
PY="$VOXHAMMER_DIR/.venv/bin/python"
REPS="${REPS:-3}"
STAMP='import sys,time
t0=time.time()
for line in sys.stdin:
    sys.stdout.write("[%8.2f] %s" % (time.time()-t0, line)); sys.stdout.flush()'

# $1 = 자산 태그 (t1s|t2s)  $2 = 자산 디렉터리명
TAG="$1"; ASSET="$2"
for REP in $(seq 1 "$REPS"); do
  for ARM in baseline ours; do
    OUT="$TRACK_ROOT/w41-$TAG-$ARM-r$REP"
    [ -f "$OUT/timing.json" ] && { echo "SKIP $OUT (이미 있음)"; continue; }
    rm -rf "$OUT"; mkdir -p "$OUT"
    echo "=== $TAG $ARM rep$REP  $(date '+%H:%M:%S') ==="
    if [ "$ARM" = baseline ]; then
      ARGS=(--arm baseline --model "$TRACK_ROOT/$ASSET/model.glb" --mask_glb "$TRACK_ROOT/$ASSET/mask.glb")
    else
      ARGS=(--arm ours --model "$TRACK_ROOT/$ASSET/model.glb" \
            --slat "$TRACK_ROOT/$ASSET/slat.safetensors" --mask_ply "$TRACK_ROOT/$TAG-mask/voxels_delete.ply")
    fi
    /usr/bin/time -v "$PY" -u "$VH_TOOLS/repeat_bench.py" "${ARGS[@]}" \
        --image_dir "$TRACK_ROOT/$TAG-inpaint" --out_dir "$OUT" --seed 0 2>&1 \
      | "$PY" -c "$STAMP" > "$OUT/run.log"
    grep -qF "Error happened during processing." "$OUT/run.log" \
      && echo "  🔴 FAIL 삼킨 예외 마커 검출" || echo "  ok 삼킨 예외 없음"
    grep -E "^\[.*\] DONE|Maximum resident set size" "$OUT/run.log" | tail -2
  done
done
echo "=== $TAG 완료 $(date '+%H:%M:%S') ==="
