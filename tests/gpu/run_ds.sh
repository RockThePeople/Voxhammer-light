#!/usr/bin/env bash
# 보조 DS 실행: A = 시나리오마다 run_ds.py(포크 2·이식 2 비트 비교), B = 토글 켬 이식 CLI 3회(구간 6 시간, 분포 비교용 slat).
# B 는 다른 GPU·CPU 작업이 없을 때 돌린다(시간 측정). 사용: run_ds.sh A|B   (결과 $RUNS_DIR/01a-ds/)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
: "${RUNS_DIR:?}" "${ASSETS_DIR:?}"
PY="${VH_PY:-$HOME/ai-ar-v2/VoxHammer/.venv/bin/python}"
OUT="$RUNS_DIR/01a-ds"; cd "$ROOT"
A=$ASSETS_DIR/track_a; B=$ASSETS_DIR/track_b; V=third_party/ai-ar-v3/handoff/lasso
list() {
  for s in A-M1:moto A-M2:moto A-S1:snowman A-S2:snowman A-C1:chair A-C2:chair; do id=${s%%:*}; a=${s##*:}
    echo "track_a-$id $A/$a/gen/slat.safetensors $A/$a/masks/$id.mask.json $A/$a/cond/$id"; done
  for i in 1 2 3 4 5 6 7 8; do echo "track_b-B$i $B/B$i/slat.safetensors $B/B$i/mask.json $B/B$i/cond"; done
  if [ "$1" = A ]; then for m in rider-head rear-wheel; do
    echo "moto-b-$m $HOME/ai-ar-data/assets/v3-moto-b/v1/slat.safetensors $V/moto-b.$m.mask.json $RUNS_DIR/2026-10-22/moto-b/inpaint/$m-run1"; done; fi
}
list "$1" | while read -r name slat mask img; do
  if [ "$1" = A ]; then
    [ -f "$OUT/e2e/$name.json" ] && continue
    "$PY" -u tests/gpu/run_ds.py --slat "$slat" --mask_json "$mask" --image_dir "$img" --out "$OUT/e2e/$name.json" > "$OUT/e2e-$name.log" 2>&1
    echo "A $name rc=$? $(grep -o '"all_pairs_bit_equal": [a-z]*' "$OUT/e2e/$name.json" 2>/dev/null)"
  else
    for r in r1 r2 r3; do
      d="$OUT/on/$name"; mkdir -p "$d"; [ -f "$d/$r.json" ] && continue
      VHL_DS_DETERMINISTIC=1 "$PY" -u -m editcore.cli --slat "$slat" --mask_json "$mask" --image_dir "$img" --seed 0 \
        --out "$d/$r/output.glb" --slat_out "$d/$r/slat_tgt.safetensors" --timing "$d/$r.json" > "$d/$r.log" 2>&1
      echo "B $name $r rc=$? $(grep -a '^\[ds\] deterministic' "$d/$r.log")"
    done
  fi
done
echo "DS_$1_DONE"
