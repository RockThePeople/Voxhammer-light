"""기준 구현 고정 검사 (요구사항서 2장 '기준 코드', 4.2-1).

검사 항목
  1. third_party/VoxHammer submodule 이 f33cd96 에 고정되어 있다.
  2. patches/edit_pipeline.f33cd96.patch 의 sha256 이 SHA256SUMS 와 같다.
  3. 패치가 고정 커밋에 충돌 없이 적용된다 (git apply --check).
  4. submodule 작업 트리에 수정이 없다 (포크 읽기 전용).

실패하면 종료 코드 1. CI 와 각 워크스테이션 세션이 작업 시작 전에 실행한다.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VH = ROOT / "third_party" / "VoxHammer"
PATCH = ROOT / "patches" / "edit_pipeline.f33cd96.patch"
SUMS = ROOT / "patches" / "SHA256SUMS"
VH_PIN = "f33cd965388faf4bd89252068cac5a0b8030e7de"


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def main() -> int:
    errors: list[str] = []

    head = _git("rev-parse", "HEAD", cwd=VH)
    if head != VH_PIN:
        errors.append(f"VoxHammer HEAD {head} != pin {VH_PIN}")

    want = SUMS.read_text().split()[0]
    got = hashlib.sha256(PATCH.read_bytes()).hexdigest()
    if got != want:
        errors.append(f"patch sha256 {got} != SHA256SUMS {want}")

    dirty = _git("status", "--porcelain", cwd=VH)
    if dirty:
        errors.append("third_party/VoxHammer 작업 트리가 수정됨:\n" + dirty)

    chk = subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=VH, capture_output=True, text=True)
    if chk.returncode != 0:
        errors.append("patch apply --check 실패:\n" + chk.stderr)

    if errors:
        print("PIN CHECK FAILED")
        for e in errors:
            print(" -", e)
        return 1
    print(f"PIN CHECK OK  voxhammer={VH_PIN[:7]}  patch={got[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
