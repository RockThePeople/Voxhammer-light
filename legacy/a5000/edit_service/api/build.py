"""돌고 있는 코드가 무엇인가 (계약 3.16.0).

⚠️ **값은 import 시점에 한 번 굳힌다. 요청마다 다시 재지 않는다.**

요청마다 `git rev-parse HEAD` 를 부르면 이 필드가 존재하는 이유가 사라진다.
고친 뒤 커밋만 하고 프로세스를 안 내리면 /health 는 새 해시를 보고하는데
와이어는 옛 동작이다 — 즉 "커밋했는데 안 떴다" 가 정확히 안 보이게 된다.
그 상황을 잡으려고 넣은 필드다.

여기서 읽는 건 **이 프로세스가 기동할 때의 작업 트리 상태**다. 기동 후에 무슨
커밋을 하든 값은 안 변한다. 그래서 커밋 시각 > started_at 이면 "이 수정은
안 떠 있다" 가 한 줄로 판정된다.

git 이 없거나 저장소가 아니어도 죽지 않는다 — build=None 으로 나간다.
"""
import os
import subprocess
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _git(*args):
    """실패는 전부 None. health 가 git 때문에 500 이 되면 안 된다."""
    try:
        r = subprocess.run(("git", "-C", REPO) + args, capture_output=True,
                           text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _read():
    rev = _git("rev-parse", "--short", "HEAD")
    if rev is None:
        return None, False, None
    # 🔴 **두 가지를 갈라서 낸다 (3.19.x).**
    #
    # 예전엔 `git status --porcelain` 하나로 판정했는데, 그건 untracked 파일까지
    # 센다. 실측(2026-08-03): 커밋을 다 하고 tracked 변경이 0건인데도 문서 파일
    # 하나(`?? HANDOFF.md`)가 남아 있어 `build_dirty=true` 가 내려가지 않았다.
    # RUNBOOK 은 그 값을 "해시로 코드를 특정할 수 없다"로 읽으므로, 스크래치 파일
    # 하나가 그 판정을 영구히 켜둔다.
    #
    # ⚠️ 그렇다고 untracked 를 **버리면 안 된다.** untracked `.py` 가 패키지 안에
    # 있으면 임포트될 수 있고, 그러면 "코드를 특정했다"고 말하는데 실제로는 못 한
    # 상태가 된다 — 지금의 거짓 dirty 보다 그쪽이 나쁘다. 그래서 버리지 않고 **센다.**
    #
    #   build_dirty      tracked 변경만 있는가 (bool)
    #   build_untracked  untracked 파일 수   (int)
    #
    # 실패(None)면 "깨끗하다"고 단정하지 않는다 — 모르면 dirty 로 본다.
    tracked = _git("status", "--porcelain", "--untracked-files=no")
    others = _git("ls-files", "--others", "--exclude-standard")
    n_other = None if others is None else len([l for l in others.splitlines() if l])
    return rev, (tracked is None or tracked != ""), n_other


BUILD, BUILD_DIRTY, BUILD_UNTRACKED = _read()
STARTED_AT = datetime.now(timezone.utc).isoformat(timespec="seconds")


def fields():
    """BHealth/ServerHealth 에 실을 build 필드.

    ⚠️ `build_untracked` 는 **아직 스키마에 없다.** BHealth/ServerHealth 는 모르는
    키를 조용히 **드롭한다**(pydantic 기본 extra='ignore', 실측 확인). 그래서 지금
    이 값은 계산되지만 와이어에 안 실린다. 스키마가 내려오면 그 순간부터 흐른다.
    여기서 미리 넣어두는 이유는 스키마 드롭과 서버 배포를 같은 순간에 맞출 필요를
    없애기 위해서다 — 다만 **그때까지는 untracked 가 안 보인다**는 것을 알고 있어야
    한다 (build_dirty 가 tracked 만 보게 됐으므로).
    """
    return {"build": BUILD, "build_dirty": BUILD_DIRTY,
            "build_untracked": BUILD_UNTRACKED, "started_at": STARTED_AT}
