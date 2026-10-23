"""단일 워커 큐. GPU 가 직렬 자원이라 큐도 하나여야 한다.

lab 요청과 제품 요청이 **같은 큐**에 줄 선다. 별도 프로세스로 나누면 큐가 둘이
되어 서로 GPU 를 뺏는다.
"""
import queue
import threading
import time
import traceback
import uuid

from deltacontract.errors import DeltaContractError

JOBS = {}
Q = queue.Queue()
_STARTED = False

# 계약 stage 어휘. ops 가 진행 콜백으로 이 값을 넘긴다.
STAGES = ("load", "structure", "slat", "repaint", "decode", "chunk")


def new_job(kind, asset_id, job_id=None, **extra):
    # job_id 를 넘기면 그걸 쓴다 — 재생이 원본 id 를 되살릴 때 필요하다.
    jid = job_id or f"j-{uuid.uuid4().hex[:12]}"
    JOBS[jid] = {
        "job_id": jid, "kind": kind, "asset_id": asset_id,
        "state": "queued", "stage": None, "progress": 0.0,
        "error": None, "error_code": None, "result": None,
        "submitted": time.time(), **extra,
    }
    return jid


def submit(jid, fn):
    Q.put((jid, fn))
    return jid


def _run_one(jid, fn):
    job = JOBS[jid]
    job.update(state="running", started=time.time())

    def progress(stage):
        job["stage"] = stage
        if stage in STAGES:
            job["progress"] = round((STAGES.index(stage) + 1) / len(STAGES), 3)

    try:
        job["result"] = fn(progress)
        job.update(state="succeeded", stage="chunk", progress=1.0,
                   finished=time.time())
    except DeltaContractError as e:
        job.update(state="failed", error=str(e), error_code=e.error_code,
                   finished=time.time())
    except Exception as e:                                  # noqa: BLE001
        job.update(state="failed", error=f"{type(e).__name__}: {e}",
                   error_code="INTERNAL", finished=time.time())
        traceback.print_exc()


def _worker():
    # ⚠️ 루프 전체를 감싼다. 워커가 예외로 죽으면 큐가 영구 정지하고, 그러면
    # 이후 모든 요청이 queued 에서 멈춘 채 아무 에러도 안 난다.
    while True:
        try:
            jid, fn = Q.get()
        except Exception:                                   # noqa: BLE001
            time.sleep(0.5)
            continue
        try:
            _run_one(jid, fn)
        except Exception:                                   # noqa: BLE001
            traceback.print_exc()
            JOBS.get(jid, {}).update(state="failed", error_code="INTERNAL",
                                     error="worker crashed")
        finally:
            Q.task_done()


def start(preload=None):
    """워커 1개 기동. preload 는 모델 상주 로드(34초)를 큐 맨 앞에 태운다."""
    global _STARTED
    if _STARTED:
        return
    _STARTED = True
    threading.Thread(target=_worker, daemon=True, name="gpu-worker").start()
    if preload:
        jid = new_job("preload", None)
        submit(jid, lambda _p: preload())
