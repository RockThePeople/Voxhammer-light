#!/usr/bin/env bash
# 0-1 P-4 종단 동등성: 시나리오마다 기준(포크 W34, seed 0) 3회 + 이식본(editcore.cli, seed 0) 3회.
# 사용: run_e2e_port.sh <이름> <slat> <mask_json> <image_dir>   (결과 $RUNS_DIR/01-port/<이름>/)
# 기준 회차가 이미 있으면(F-5 결과 등) REF_DIR=<w34-지문 디렉터리> 로 넘겨 다시 돌리지 않는다.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
: "${RUNS_DIR:?}"
PY="${VH_PY:-$HOME/ai-ar-v2/VoxHammer/.venv/bin/python}"
name=$1; slat=$2; mask=$3; img=$4
OUT="$RUNS_DIR/01-port/$name"; mkdir -p "$OUT"
cd "$ROOT"
REF_DIR="${REF_DIR:-}"
if [ -z "$REF_DIR" ]; then
  for r in r1 r2 r3; do
    "$PY" -u reference/run_reference.py --mode w34 --asset "01ref-$name" --run $r --seed 0 --slat "$slat" \
        --mask_json "$mask" --image_dir "$img" --runs_dir "$RUNS_DIR" > "$OUT/ref-$r.log" 2>&1
    echo "ref $r rc=$?"
  done
  fp=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['meta']['mask_fingerprint'][:12])" "$mask")
  REF_DIR=$(ls -d "$RUNS_DIR"/*/01ref-$name/w34-$fp | tail -1)
fi
for r in r1 r2 r3; do
  VHL_E2A_STEPS_S1=25 VHL_E2A_STEPS_S2=25 "$PY" -u -m editcore.cli --slat "$slat" --mask_json "$mask" --image_dir "$img" --seed 0 \
      --out "$OUT/port-$r/output.glb" --slat_out "$OUT/port-$r/slat_tgt.safetensors" \
      --timing "$OUT/port-$r.json" > "$OUT/port-$r.log" 2>&1
  echo "port $r rc=$?"
done
echo "REF_DIR=$REF_DIR" > "$OUT/ref_dir.txt"
