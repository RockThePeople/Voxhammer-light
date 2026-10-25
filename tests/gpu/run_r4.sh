#!/usr/bin/env bash
# 2번 R4 실행. 비교 기준은 E2a 채택 15/15(정오표 E20) — 기본값 반영 전 main 에서도 같게 15 를 명시한다.
#   run_r4.sh parents | decode | on | render | r11
#   r11: R11 정정 규칙(부기 ∪ 점유 변경, chief 2026-10-26) 코드로 R4_ONLY(기본 A-M1·B2·B6) × 3회 다시 → $RUNS_DIR/02-r4/<이름>/r4-r11/
#   on: 14건 × 3회, VHL_R4=1, --timing(구간 7 세부) --slat_out --parent_cbin. 결과 $RUNS_DIR/02-r4/<이름>/r4-on/r{1,2,3}{.json,/}
#   끔(기존 GLB) 대조는 E2a c15 r1–r3(같은 15 스텝, 같은 장비, 깨끗한 측정)을 재사용한다.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
: "${ASSETS_DIR:?}" "${RUNS_DIR:?}"
PY="${VH_PY:-$HOME/ai-ar-v2/VoxHammer/.venv/bin/python}"
export VHL_E2A_STEPS_S1=15 VHL_E2A_STEPS_S2=15
cd "$ROOT"
case "$1" in
  parents|decode|render|decode2|render2) "$PY" -u tests/gpu/r4_checks.py "$1"; exit $? ;;
esac
A=$ASSETS_DIR/track_a; B=$ASSETS_DIR/track_b; P=$RUNS_DIR/02-r4/parents
jobs=()
for s in A-M1:moto A-M2:moto A-S1:snowman A-S2:snowman A-C1:chair A-C2:chair; do id=${s%%:*}; a=${s##*:}
  jobs+=("track_a-$id|$A/$a/gen/slat.safetensors|$A/$a/masks/$id.mask.json|$A/$a/cond/$id|$A/$a/gen/chunks"); done
for i in 1 2 3 4 5 6 7 8; do jobs+=("track_b-B$i|$B/B$i/slat.safetensors|$B/B$i/mask.json|$B/B$i/cond|$P/B$i/chunks"); done
tag=r4-on; only=""
if [ "$1" = r11 ]; then tag=r4-r11; only="${R4_ONLY:-track_a-A-M1 track_b-B2 track_b-B6}"; fi
for r in r1 r2 r3; do
  for j in "${jobs[@]}"; do IFS='|' read -r name slat mask img parent <<< "$j"
    [ -n "$only" ] && [[ " $only " != *" $name "* ]] && continue
    d="$RUNS_DIR/02-r4/$name/$tag"; mkdir -p "$d"; [ -f "$d/$r.json" ] && continue
    VHL_R4=1 "$PY" -u -m editcore.cli --slat "$slat" --mask_json "$mask" --image_dir "$img" --seed 0 \
      --out "$d/$r/output.glb" --slat_out "$d/$r/slat_tgt.safetensors" --cbin_out "$d/$r/cbin" --parent_cbin "$parent" \
      --timing "$d/$r.json" > "$d/$r.log" 2>&1
    echo "$name $r rc=$? $(grep -a '^\[r4\]' "$d/$r.log" | tr '\n' ' ')"
  done
done
echo "R4_${1^^}_DONE"
