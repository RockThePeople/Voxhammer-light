#!/bin/bash
# :8082 편집 서비스 복구 — vh-track 이 GPU 를 통째로 써야 할 때 내렸다가 되살린다.
# 실측 근거: pA 이전 런이 decode_slat 에서 OOM 났다 (이 서비스가 같은 카드에서 5.8 GiB 점유).
# 키·포트·호스트는 여기 박지 않는다 — 환경변수로만 받는다.
set -uo pipefail
cd <HOME>/ai-ar-prototype || exit 1
if pgrep -f '[p]ython.*server/api/main\.py' >/dev/null; then echo "이미 떠 있다: PID $(pgrep -f '[p]ython.*server/api/main\.py'|head -1)"; exit 0; fi
export CUDA_HOME="${CUDA_HOME:-<HOME>/cuda-11.8}"
: "${TRELLIS_API_KEY:?TRELLIS_API_KEY 를 환경에서 넘겨라 (문서·스크립트에 값을 박지 않는다)}"
setsid <HOME>/ai-ar-v2/VoxHammer/.venv/bin/python server/api/main.py \
  > <HOME>/ai-ar-data/vh-track/edit_service.log 2>&1 < /dev/null &
echo "기동 PID $! — 로그 <HOME>/ai-ar-data/vh-track/edit_service.log"
