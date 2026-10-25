#!/usr/bin/env bash
# R9 인페인팅 워커 기동·정지·재기동·상태 (rtx3090). 사용법은 README.md.
#
#   bash services/rtx3090_inpaint/run.sh start|stop|restart|status
#
# 저장소 루트의 .env 를 읽는다. 워커는 setsid 로 띄워 이 셸·세션이 끝나도 살아 있다.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }

PY="${RTX3090_PYTHON:-}"
PORT="${INPAINT_PORT:-8082}"
RUNS="${RUNS_DIR:-$ROOT/runs}"
LOG="$RUNS/r9-inpaint/worker.log"

pid_of() { ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1 || true; }

start() {
  [ "${VHL_R9:-0}" = "1" ] || { echo "VHL_R9=1 이 아니다 (.env 또는 환경변수로 켠다). 기본 꺼짐"; exit 1; }
  [ -n "$PY" ] && [ -x "$PY" ] || { echo "RTX3090_PYTHON 이 없다 (flux-fill env 의 python)"; exit 1; }
  if [ -n "$(pid_of)" ]; then echo "이미 :$PORT 에 떠 있다 (pid $(pid_of))"; exit 0; fi
  mkdir -p "$(dirname "$LOG")"
  echo "── start $(date -Is)" >> "$LOG"
  (cd "$ROOT" && VHL_R9=1 setsid --fork "$PY" -u services/rtx3090_inpaint/worker.py >> "$LOG" 2>&1)
  for _ in $(seq 1 600); do        # 최대 60 s (실측 기동~ready 7.4 s)
    if [ -n "$(pid_of)" ] && tail -n 5 "$LOG" | grep -q '\[ready\]'; then
      echo "ready :$PORT pid $(pid_of)  log $LOG"; return 0
    fi
    sleep 0.1
  done
  echo "FAIL — 60 s 안에 ready 가 안 났다. 로그:"; tail -20 "$LOG"; exit 1
}

stop() {
  local p; p="$(pid_of)"
  [ -n "$p" ] || { echo "떠 있지 않다"; return 0; }
  kill "$p"
  for _ in $(seq 1 100); do [ -z "$(pid_of)" ] && { echo "stopped (pid $p)"; return 0; }; sleep 0.1; done
  echo "10 s 안에 안 내려가 SIGKILL"; kill -9 "$p" || true
}

status() {
  curl -s -m 5 "http://127.0.0.1:$PORT/health" || { echo "응답 없음 (:$PORT)"; exit 1; }
  echo
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  *) echo "usage: $0 start|stop|restart|status"; exit 2 ;;
esac
