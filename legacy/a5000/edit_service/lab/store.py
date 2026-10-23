"""LAB_ROOT 아래 run 디렉터리 관리 + 진행 상태 기록.

meta.json 은 **시작 시점에 먼저 쓰고 단계마다 갱신한다.** 끝나고 한 번에 쓰면
3090 디버깅 페이지에서 진행이 안 보이는데, 그게 그 페이지의 핵심 가치다.
"""
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

LAB_ROOT = os.environ.get("LAB_ROOT", "<HOME>/ai-ar-lab")
RUNS = os.path.join(LAB_ROOT, "runs")
RUN_ID_RE = re.compile(r"^(gen|edit)_(\d{4})$")

TERMINAL = {"succeeded", "failed"}

# LAB_API.md §RunSummary 의 stage 어휘: prompt → t2i → structure → slat →
# repaint → decode → chunk.
#
# ⚠️ 이 목록이 우리 실제 단계를 다 덮지 못한다. 모델 로드(34s, 최장)와 렌더가
# 빠져 있다. 없는 값을 지어내면 3090이 모르는 값으로 떨어뜨리므로, stage 는
# 어휘 안에서만 고르고 정확한 이름은 stage_detail(계약 밖 추가 필드, 3090은
# extra="allow")로 따로 보낸다. 보고서에 문서 갭으로 적었다.
STAGE_FIRST = {"gen": "structure", "edit": "repaint"}

# (stage_detail, 계약 stage, 진행률) — 진행률은 실측 소요시간 비중에 맞췄다.
STAGE_MAP = {
    "start":            ("structure", 0.00),
    "image_load":       ("structure", 0.02),
    "model_load":       ("structure", 0.05),
    "mask":             ("repaint",   0.05),
    "sparse_structure": ("structure", 0.45),
    "slat_sample":      ("slat",      0.75),
    "slat_save":        ("slat",      0.80),
    "repaint":          ("repaint",   0.70),
    "decode":           ("decode",    0.88),
    "chunk":            ("chunk",     0.95),
    "render":           ("chunk",     0.98),
}


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def list_run_ids():
    """번호 오름차순. 알파벳순으로 정렬하면 edit_0002 가 gen_0001 보다 앞서서
    역순을 '최신순'으로 쓸 수 없다."""
    if not os.path.isdir(RUNS):
        return []
    ids = [d for d in os.listdir(RUNS) if RUN_ID_RE.match(d)]
    return sorted(ids, key=lambda d: int(RUN_ID_RE.match(d).group(2)))


def next_run_id(kind):
    """gen_0001 / edit_0002 — 종류+순번. 번호는 종류를 가로질러 하나의 수열이다.

    사람이 읽고 타이핑할 수 있어야 하므로 해시나 UUID 를 쓰지 않는다.
    """
    n = 0
    for r in list_run_ids():
        n = max(n, int(RUN_ID_RE.match(r).group(2)))
    return f"{kind}_{n + 1:04d}"


def run_dir(run_id):
    if not RUN_ID_RE.match(run_id):
        raise ValueError(f"run_id 형식이 아니다: {run_id!r}")
    return os.path.join(RUNS, run_id)


def read_meta(run_id):
    p = os.path.join(run_dir(run_id), "meta.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


class Run:
    """한 실행. 로그를 터미널과 log.txt 에 동시에 쓰고 meta 를 단계마다 갱신한다."""

    def __init__(self, kind, run_id=None):
        self.kind = kind
        self.run_id = run_id or next_run_id(kind)
        self.dir = run_dir(self.run_id)
        os.makedirs(os.path.join(self.dir, "render"), exist_ok=True)
        os.makedirs(os.path.join(self.dir, "chunks"), exist_ok=True)
        self._log = open(os.path.join(self.dir, "log.txt"), "a", buffering=1)
        self.t0 = time.time()
        # state/stage 어휘는 계약을 재사용한다 (contract/LAB_API.md §1, JobState).
        # done/error 같은 동의어를 쓰면 3090이 회색 queued 배지로 떨어뜨린다 —
        # 예외가 안 나고 조용히 틀리는 종류라 여기서 못박는다.
        self.meta = {
            "run_id": self.run_id,
            "kind": kind,
            "asset_id": self.run_id,
            "state": "running",
            "stage": STAGE_FIRST[kind],
            "stage_detail": "start",
            "progress": 0.0,
            "started_at": now_iso(),
            "finished_at": None,
            "duration_s": 0.0,
            "error": None,
            "error_code": None,
            "log_path": "log.txt",
        }
        self.save()

    # ── 로그 ───────────────────────────────────────────────────────────
    def log(self, msg, indent=False):
        prefix = " " * (len(self.run_id) + 3) if indent else f"[{self.run_id}] "
        line = prefix + msg
        print(line, flush=True)
        self._log.write(line + "\n")

    def warn(self, msg):
        self.log(f"\033[33m⚠  {msg}\033[0m")

    def fail_loud(self, msg):
        bar = "━" * 68
        for line in (bar, f"실패: {msg}", bar):
            print(f"\033[1;31m{line}\033[0m", flush=True)
            self._log.write(line + "\n")

    def _set_stage(self, key):
        stage, prog = STAGE_MAP.get(key, (self.meta.get("stage"), self.meta.get("progress", 0.0)))
        self.meta["stage"] = stage
        self.meta["stage_detail"] = key
        self.meta["progress"] = prog

    def begin(self, key):
        """긴 작업 **시작 전에** stage 를 올린다.

        끝난 뒤에만 기록하면 가장 오래 걸리는 단계(모델 로드 ~34s)가 3090 페이지에서
        내내 이전 단계로 보인다. 진행이 보이는 게 그 페이지의 핵심 가치라 먼저 쓴다.
        """
        self._set_stage(key)
        self.touch()

    def stage(self, key, label, detail="", secs=None):
        self._set_stage(key)
        self.touch()
        t = f"{secs:6.1f}s  " if secs is not None else " " * 9
        self.log(f"{label:<16} {t}{detail}")

    # ── meta ───────────────────────────────────────────────────────────
    def set(self, **kw):
        self.meta.update(kw)
        self.save()

    def touch(self):
        self.meta["duration_s"] = round(time.time() - self.t0, 2)
        self.save()

    def save(self):
        p = os.path.join(self.dir, "meta.json")
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, p)   # 원자적 — 3090 이 읽는 중에 반쪽 파일을 보지 않게

    def done(self):
        self.meta["state"] = "succeeded"      # 계약 JobState. "done" 이 아니다.
        self.meta["stage"] = "chunk"
        self.meta["stage_detail"] = "done"
        self.meta["progress"] = 1.0
        self.meta["finished_at"] = now_iso()
        self.touch()
        self.log(f"→ {self.dir}")
        self._log.close()

    def failed(self, err):
        from deltacontract.errors import DeltaContractError

        self.meta["state"] = "failed"
        self.meta["error"] = str(err)
        # LAB_API.md §6 — 계약 코드에 대응되면 채우고, 없으면 null.
        self.meta["error_code"] = (
            err.error_code if isinstance(err, DeltaContractError) else None
        )
        self.meta["finished_at"] = now_iso()
        self.touch()
        if not self._log.closed:
            self._log.close()

    def path(self, *parts):
        return os.path.join(self.dir, *parts)
