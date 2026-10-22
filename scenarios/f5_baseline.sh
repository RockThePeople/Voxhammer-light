#!/usr/bin/env bash
# 00s2 F-5 14건 × 3회 기준선 (W34 경로, 기준 구현, run_edit 직전 torch.manual_seed(0)).
# 입력 = 동결 SLat(트랙 A gen/, 트랙 B 판본 0), 조건 = 동결 이미지(cond/). 결과는 RUNS_DIR/<날짜>/<자산>/w34-<지문>/r{1,2,3}.json
# 이미 있는 회차 JSON 은 건너뛴다(재시작 가능).
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MAIN="$(cd "$HERE/../Voxhammer-light" && pwd)"
: "${ASSETS_DIR:?}" "${RUNS_DIR:?}"
PY="${VH_PY:-$HOME/ai-ar-v2/VoxHammer/.venv/bin/python}"
DAY="${F5_DAY:-$(date +%F)}"
A="$ASSETS_DIR/track_a"; B="$ASSETS_DIR/track_b"
jobs=()
for s in A-M1:moto A-M2:moto A-S1:snowman A-S2:snowman A-C1:chair A-C2:chair; do
  id=${s%%:*}; a=${s##*:}
  jobs+=("track_a-$id|$A/$a/gen/slat.safetensors|$A/$a/masks/$id.mask.json|$A/$a/cond/$id")
done
for i in 1 2 3 4 5 6 7 8; do jobs+=("track_b-B$i|$B/B$i/slat.safetensors|$B/B$i/mask.json|$B/B$i/cond"); done
for r in r1 r2 r3; do
  for j in "${jobs[@]}"; do
    IFS='|' read -r name slat mask img <<<"$j"
    fp=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['meta']['mask_fingerprint'][:12])" "$mask")
    out="$RUNS_DIR/$DAY/$name/w34-$fp/$r.json"
    [ -f "$out" ] && { echo "skip $name $r"; continue; }
    mkdir -p "$RUNS_DIR/$DAY/logs"
    ( cd "$MAIN" && timeout 2400 "$PY" -u reference/run_reference.py --mode w34 --asset "$name" --run "$r" --seed 0 \
        --slat "$slat" --mask_json "$mask" --image_dir "$img" --runs_dir "$RUNS_DIR" \
        > "$RUNS_DIR/$DAY/logs/f5-$name-$r.log" 2>&1 )
    echo "$name $r rc=$? $(grep -o 'edit_total.*' "$RUNS_DIR/$DAY/logs/f5-$name-$r.log" | head -1)"
  done
done
