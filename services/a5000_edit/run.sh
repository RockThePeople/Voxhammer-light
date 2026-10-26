#!/usr/bin/env bash
# 04L A5000 상주 워커 기동. .env(RUNS_DIR·ASSETS_DIR·TRELLIS_API_KEY 선택)를 읽고, VHL_R4=1, editcore 기본 15/15.
#   bash services/a5000_edit/run.sh            # 0.0.0.0:8082, 시드 판본(트랙 A v1) 확인 후 기동
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"; [ -f "$ENV_FILE" ] || ENV_FILE="$ROOT/../Voxhammer-light/.env"
set -a; . "$ENV_FILE"; set +a
export VHL_R4=1 HF_HUB_OFFLINE=1
PY="${VH_PY:-$HOME/ai-ar-v2/VoxHammer/.venv/bin/python}"
cd "$ROOT"
"$PY" services/a5000_edit/seed.py
exec "$PY" -u -m services.a5000_edit.app --host "${EDIT_HOST_BIND:-0.0.0.0}" --port "${EDIT_PORT:-8082}"
