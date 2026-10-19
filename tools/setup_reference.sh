#!/usr/bin/env bash
# 기준 구현(포크) 빌드 디렉터리를 만든다.
#   third_party/VoxHammer (f33cd96, 수정 안 함) 를 build/reference/ 로 복사하고
#   patches/edit_pipeline.f33cd96.patch 를 그 사본에 적용한다.
# submodule 자체에는 패치를 적용하지 않는다 (tools/check_pins.py 가 작업 트리 수정을 검출한다).
#
# 기준 실행 조건: W39_ATTN_MASK 를 설정하지 않는다 (요구사항서 5장, W39 제외).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${REFERENCE_DIR:-$ROOT/build/reference}"

python3 "$ROOT/tools/check_pins.py"

rm -rf "$OUT"
mkdir -p "$(dirname "$OUT")"
git -C "$ROOT/third_party/VoxHammer" archive --format=tar HEAD | (mkdir -p "$OUT" && tar -x -C "$OUT")
git -C "$OUT" init -q 2>/dev/null || true
(cd "$OUT" && git apply "$ROOT/patches/edit_pipeline.f33cd96.patch")

if [ -n "${W39_ATTN_MASK:-}" ]; then
  echo "경고: W39_ATTN_MASK=${W39_ATTN_MASK} 가 설정돼 있다. 기준 실행에서는 해제할 것." >&2
fi
echo "reference ready: $OUT"
