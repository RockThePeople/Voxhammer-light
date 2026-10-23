#!/bin/bash
# W43 — **조건까지 갱신**한 재실행.  마스크와 2d_mask/2d_edit 이 이제 같은 3D 마스크에서 나온다.
# 이전 웨이브와 다른 것은 조건 하나뿐이다 (마스크·시드·프롬프트·자산 동일).
set -uo pipefail
VOXHAMMER_DIR=<HOME>/ai-ar-v2/VoxHammer; TRACK_ROOT=<HOME>/ai-ar-data/vh-track
VH_TOOLS=<HOME>/ai-ar-v3/tools; PY="$VOXHAMMER_DIR/.venv/bin/python"
STAMP='import sys,time
t0=time.time()
for line in sys.stdin:
    sys.stdout.write("[%8.2f] %s" % (time.time()-t0, line)); sys.stdout.flush()'
declare -A A=( [t1s]=t1s-snowman [t2s]=t2s-motorcycle )
for TAG in t1s t2s; do
  [ -f "$TRACK_ROOT/$TAG-inpaint-w42/2d_edit.png" ] || { echo "🔴 $TAG 새 조건 없음 — 건너뜀"; continue; }
  for ARM in ours baseline; do
    OUT="$TRACK_ROOT/w43-$TAG-$ARM-r1"
    [ -f "$OUT/timing.json" ] && { echo "SKIP $OUT"; continue; }
    rm -rf "$OUT"; mkdir -p "$OUT"
    echo "=== $TAG $ARM $(date +%H:%M:%S) ==="
    ARGS=(--arm $ARM --model "$TRACK_ROOT/${A[$TAG]}/model.glb" --mask_glb "$TRACK_ROOT/$TAG-mask/w42_mask.glb"
          --mask_json "$TRACK_ROOT/$TAG-mask/w42_cells.json")
    [ "$ARM" = ours ] && ARGS+=(--slat "$TRACK_ROOT/${A[$TAG]}/slat.safetensors")
    /usr/bin/time -v "$PY" -u "$VH_TOOLS/repeat_bench.py" "${ARGS[@]}" \
        --image_dir "$TRACK_ROOT/$TAG-inpaint-w42" --out_dir "$OUT" --seed 0 2>&1 \
      | "$PY" -c "$STAMP" > "$OUT/run.log"
    grep -qF "Error happened during processing." "$OUT/run.log" && echo "  🔴 삼킨 예외" || true
    grep -E "DONE|마스크 셀" "$OUT/run.log" | tail -2
  done
done
echo "=== W43 편집 완료 $(date +%H:%M:%S) ==="
