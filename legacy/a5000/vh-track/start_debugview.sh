#!/bin/bash
# DebugView 단독 기동. GPU 를 안 쓰므로 편집 서비스와 **독립적으로** 살아 있어야 한다.
# 포트는 환경변수로만 받는다 (문서·코드에 조합을 박지 않는다).
set -uo pipefail
PORT="${DEBUGVIEW_PORT:-8081}"
SCRIPT=<HOME>/ai-ar-v3/tools/debugview_standalone.py
LOG="${DEBUGVIEW_LOG:-<HOME>/ai-ar-data/vh-track/debugview.log}"
if pgrep -f "[p]ython.*debugview_standalone\.py" >/dev/null; then
  echo "이미 떠 있다: PID $(pgrep -f '[p]ython.*debugview_standalone\.py' | head -1)"; exit 0
fi
setsid nohup <HOME>/ai-ar-v2/VoxHammer/.venv/bin/python "$SCRIPT" "$PORT" \
  > "$LOG" 2>&1 < /dev/null &
echo "기동 PID $! · 포트 $PORT · 로그 $LOG"
