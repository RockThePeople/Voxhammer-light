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

# 삭제(rm -rf)를 하지 않는다 — 세션 권한 정책이 막는 경우가 있다(00a 3090 보고).
# 이미 있으면: 패치가 적용된 정상 사본인지 확인하고 그대로 쓴다. 아니면 직접 지우라고 안내하고 멈춘다.
STAMP_WANT="$(git -C "$ROOT/third_party/VoxHammer" rev-parse HEAD) $(sha256sum "$ROOT/patches/edit_pipeline.f33cd96.patch" | cut -d' ' -f1)"
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  if [ "$(cat "$OUT/.reference-stamp" 2>/dev/null)" = "$STAMP_WANT" ] \
     && [ -z "$(git -C "$OUT" status --porcelain 2>/dev/null)" ] \
     && git -C "$OUT" rev-parse -q --verify HEAD >/dev/null 2>&1; then
    echo "reference already built (stamp ok, unmodified): $OUT"
    exit 0
  fi
  echo "오류: $OUT 가 이미 있고 이 고정값의 기준 사본이 아니다. 직접 지운 뒤 다시 실행하라." >&2
  exit 1
fi
mkdir -p "$(dirname "$OUT")"
git -C "$ROOT/third_party/VoxHammer" archive --format=tar HEAD | (mkdir -p "$OUT" && tar -x -C "$OUT")
git -C "$OUT" init -q 2>/dev/null || true
(cd "$OUT" && git apply "$ROOT/patches/edit_pipeline.f33cd96.patch")
echo "$STAMP_WANT" > "$OUT/.reference-stamp"
# 사본을 커밋해 두어 이후 수정 여부를 git status 로 검출한다
git -C "$OUT" add -A && git -C "$OUT" -c user.name=reference -c user.email=reference@local commit -qm "f33cd96 + edit_pipeline patch"

if [ -n "${W39_ATTN_MASK:-}" ]; then
  echo "경고: W39_ATTN_MASK=${W39_ATTN_MASK} 가 설정돼 있다. 기준 실행에서는 해제할 것." >&2
fi
echo "reference ready: $OUT"
