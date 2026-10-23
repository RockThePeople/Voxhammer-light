#!/bin/bash
# W42 — 새 마스크(점유 셀 기반)로 재측정. 조건·프롬프트·시드·자산은 그대로. 마스크만 변수.
# coords 지표는 W41 에서 12런 표준편차가 정확히 0 이었으므로 1회면 확정. 시간·feats 용으로 2회 돌린다.
set -uo pipefail
VOXHAMMER_DIR=<HOME>/ai-ar-v2/VoxHammer
TRACK_ROOT=<HOME>/ai-ar-data/vh-track
VH_TOOLS=<HOME>/ai-ar-v3/tools
PY="$VOXHAMMER_DIR/.venv/bin/python"
STAMP='import sys,time
t0=time.time()
for line in sys.stdin:
    sys.stdout.write("[%8.2f] %s" % (time.time()-t0, line)); sys.stdout.flush()'
declare -A ASSET=( [t1s]=t1s-snowman [t2s]=t2s-motorcycle )
for TAG in t1s t2s; do
  A=${ASSET[$TAG]}
  for REP in 1 2; do
    for ARM in ours baseline; do       # ours 를 먼저 — 빠르고 코가 사라졌는지 먼저 본다
      OUT="$TRACK_ROOT/w42-$TAG-$ARM-r$REP"
      [ -f "$OUT/timing.json" ] && { echo "SKIP $OUT"; continue; }
      rm -rf "$OUT"; mkdir -p "$OUT"
      echo "=== $TAG $ARM rep$REP $(date +%H:%M:%S) ==="
      ARGS=(--arm $ARM --model "$TRACK_ROOT/$A/model.glb" --mask_glb "$TRACK_ROOT/$A/mask.glb"
            --mask_json "$TRACK_ROOT/$TAG-mask/w42_cells.json")
      [ "$ARM" = ours ] && ARGS+=(--slat "$TRACK_ROOT/$A/slat.safetensors")
      /usr/bin/time -v "$PY" -u "$VH_TOOLS/repeat_bench.py" "${ARGS[@]}" \
          --image_dir "$TRACK_ROOT/$TAG-inpaint" --out_dir "$OUT" --seed 0 2>&1 \
        | "$PY" -c "$STAMP" > "$OUT/run.log"
      grep -qF "Error happened during processing." "$OUT/run.log" && echo "  🔴 삼킨 예외" || true
      grep -E "DONE|마스크 셀" "$OUT/run.log" | tail -2
    done
  done
done
echo "=== W42 완료 $(date +%H:%M:%S) ==="
