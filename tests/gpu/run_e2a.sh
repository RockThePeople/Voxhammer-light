#!/usr/bin/env bash
# 1번 E2a 실행: 토글 꺼짐 확인 1회 → 설정별 14건 × 3회. 결과 $RUNS_DIR/<날짜>/<자산>/e2a-<설정>/r{1,2,3}{.json,/}
# 사용: run_e2a.sh offcheck | <설정 이름> <S1> <S2>   (이미 있는 회차는 건너뛴다)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
: "${ASSETS_DIR:?}" "${RUNS_DIR:?}"
PY="${VH_PY:-$HOME/ai-ar-v2/VoxHammer/.venv/bin/python}"
DAY="${E2A_DAY:-$(date +%F)}"
A=$ASSETS_DIR/track_a; B=$ASSETS_DIR/track_b
cd "$ROOT"
jobs=()
for s in A-M1:moto A-M2:moto A-S1:snowman A-S2:snowman A-C1:chair A-C2:chair; do id=${s%%:*}; a=${s##*:}
  jobs+=("track_a-$id|$A/$a/gen/slat.safetensors|$A/$a/masks/$id.mask.json|$A/$a/cond/$id"); done
for i in 1 2 3 4 5 6 7 8; do jobs+=("track_b-B$i|$B/B$i/slat.safetensors|$B/B$i/mask.json|$B/B$i/cond"); done
run1(){ cfg=$1; r=$2; j=$3; shift 3
  IFS='|' read -r name slat mask img <<<"$j"
  d="$RUNS_DIR/$DAY/$name/e2a-$cfg"; mkdir -p "$d"
  [ -f "$d/$r.json" ] && { echo "skip $name $cfg $r"; return; }
  "$PY" -u -m editcore.cli --slat "$slat" --mask_json "$mask" --image_dir "$img" --seed 0 "$@" \
     --out "$d/$r/output.glb" --slat_out "$d/$r/slat_tgt.safetensors" --timing "$d/$r.json" > "$d/$r.log" 2>&1
  echo "$name $cfg $r rc=$? $(grep -o '\[e2a\] steps.*' "$d/$r.log" | head -1)"
}
if [ "$1" = offcheck ]; then
  ( unset VHL_E2A_STEPS_S1 VHL_E2A_STEPS_S2; run1 off r1 "${jobs[0]}" )
  exit 0
fi
cfg=$1; s1=$2; s2=$3
for r in r1 r2 r3; do for j in "${jobs[@]}"; do run1 "$cfg" "$r" "$j" --steps_s1 "$s1" --steps_s2 "$s2"; done; done
