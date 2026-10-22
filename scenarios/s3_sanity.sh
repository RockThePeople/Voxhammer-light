#!/usr/bin/env bash
# 00s S-3 평가 환경 정상 동작 확인 (Edit3D-Bench, 객체 1개: GSO/3D_Dollhouse_Lamp prompt_1).
#  (1) GT 복사: GT source_model/render/render_*.png 를 예측 자리에 그대로, edit.glb = 원본 model.glb
#      → 마스크 밖 PSNR 매우 큼, LPIPS≈0, Chamfer≈0 이어야 한다.
#  (2) 재렌더: 원본 model.glb 를 포크 Edit3D-Bench/render.py 의 render_images 로 렌더(파일 수정 없음, 영상 생략)
#      → GT visual 과 비교. 같은 렌더러·같은 카메라라 (1) 에 가까워야 한다.
# 평가 코드는 포크 파일을 호출만 한다. env: $EVAL_VENV (requirements.txt + bpy 4.0.0, torch cu126).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_SRC="$ROOT/third_party/VoxHammer/Edit3D-Bench"
GT="${ASSETS_DIR:?}/Edit3D-Bench/data"
PY="${EVAL_VENV:-$HOME/vhl-venvs/edit3dbench}/bin/python"
OUT="${RUNS_DIR:?}/s3_sanity"
OBJ="GSO/3D_Dollhouse_Lamp"; P=prompt_1
METRICS="psnr ssim lpips dino_i chamfer clip_t"

mkdir -p "$OUT/gtcopy/$OBJ/$P/images" "$OUT/rerender/$OBJ/$P/images"
cp "$GT/$OBJ/source_model/render/"render_*.png "$OUT/gtcopy/$OBJ/$P/images/"
cp "$GT/$OBJ/source_model/model.glb" "$OUT/gtcopy/$OBJ/$P/edit.glb"
cp "$GT/$OBJ/source_model/model.glb" "$OUT/rerender/$OBJ/$P/edit.glb"

cd "$BENCH_SRC"
"$PY" -c "
import sys, time; sys.path.insert(0, '.')
import render
t = time.time(); render.render_images(sys.argv[1], sys.argv[2]); print(f'render_images {time.time()-t:.1f}s')
" "$OUT/rerender/$OBJ/$P/edit.glb" "$OUT/rerender/$OBJ/$P/images"

for arm in gtcopy rerender; do
  "$PY" eval_main.py --gt_root "$GT" --pred_root "$OUT/$arm" --metrics $METRICS \
      --device cuda:0 --output_dir "$OUT/result-$arm" 2>&1 | tail -25
done
