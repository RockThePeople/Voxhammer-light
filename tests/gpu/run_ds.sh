#!/usr/bin/env bash
# 보조 DS 실행. A = 시나리오마다 run_ds.py(포크 2·이식 2 비트 비교, 이식 1회차 slat_tgt 저장),
# B = 토글 켬 이식 CLI(구간 6 시간·분포 비교용 slat). 포크 대조라 25 스텝을 명시한다(E20 뒤 editcore 기본 15).
#   DS_ONLY="이름 …"   A·B 대상 제한(chief 2026-10-25: A 는 크기 범위를 덮는 건으로 축소)
#   B: DS_TIME="이름 …" 은 3회(장비가 빈 상태, 시간 측정), DS_DIST="이름 …" 은 1회(분포 비교용 slat 보충)
# 사용: run_ds.sh A|B   (결과 $RUNS_DIR/01a-ds/)
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
  for m in rider-head rear-wheel; do
    echo "moto-b-$m $HOME/ai-ar-data/assets/v3-moto-b/v1/slat.safetensors $V/moto-b.$m.mask.json $RUNS_DIR/2026-10-22/moto-b/inpaint/$m-run1"; done
}
want() { local n=$1 l=$2; [ -z "$l" ] && return 0; [[ " $l " == *" $n "* ]]; }
list | while read -r name slat mask img; do
  if [ "$1" = A ]; then
    want "$name" "${DS_ONLY:-}" || continue
    [ -f "$OUT/e2e/$name.json" ] && continue
    mkdir -p "$OUT/on/$name/a"
    "$PY" -u tests/gpu/run_ds.py --slat "$slat" --mask_json "$mask" --image_dir "$img" --out "$OUT/e2e/$name.json" \
      --slat_out "$OUT/on/$name/a/slat_tgt.safetensors" > "$OUT/e2e-$name.log" 2>&1
    echo "A $name rc=$? $(grep -o '"all_pairs_bit_equal": [a-z]*' "$OUT/e2e/$name.json" 2>/dev/null)"
  else
    if want "$name" "${DS_TIME:-__none__}"; then runs="r1 r2 r3"; elif want "$name" "${DS_DIST:-__none__}"; then runs="r1"; else continue; fi
    for r in $runs; do
      d="$OUT/on/$name"; mkdir -p "$d"; [ -f "$d/$r.json" ] && continue
      VHL_E2A_STEPS_S1=25 VHL_E2A_STEPS_S2=25 VHL_DS_DETERMINISTIC=1 "$PY" -u -m editcore.cli --slat "$slat" --mask_json "$mask" --image_dir "$img" --seed 0 \
        --out "$d/$r/output.glb" --slat_out "$d/$r/slat_tgt.safetensors" --timing "$d/$r.json" > "$d/$r.log" 2>&1
      echo "B $name $r rc=$? $(grep -a '^\[ds\] deterministic' "$d/$r.log")"
    done
  fi
done
echo "DS_$1_DONE"
