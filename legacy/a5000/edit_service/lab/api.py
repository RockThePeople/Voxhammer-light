"""lab 읽기 전용 API. 응답 모양은 contract/LAB_API.md 를 따른다.

쓰기·삭제·실행 엔드포인트는 **의도적으로 없다.** 이 서버는 관찰만 한다.
실행은 `lab shell`(로컬 대화형)이 맡는다 — 네트워크 표면을 늘리지 않는다.

3090이 이미 이 문서 기준으로 소비 측을 만들었으므로, 어긋나면 예외가 아니라
조용한 오표시로 나타난다. 아래 필드명·어휘는 문서와 1:1로 맞춰져 있다.
"""
import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from .store import LAB_ROOT, RUN_ID_RE, list_run_ids, read_meta, run_dir
from .view import chunk_map, load_chunks

app = FastAPI(title="lab (read-only)", version="2.0")


def _summary(run_id):
    """LAB_API.md §RunSummary. 없는 값은 None 으로 둔다 — 0 으로 채우지 않는다."""
    m = read_meta(run_id) or {}
    return {
        "run_id": run_id,
        "kind": m.get("kind"),
        "parent_run_id": m.get("parent_run_id"),
        "asset_id": m.get("asset_id"),

        "state": m.get("state"),
        "stage": m.get("stage"),
        "stage_detail": m.get("stage_detail"),   # 계약 밖. 어휘가 못 덮는 단계용
        "progress": m.get("progress"),
        "error": m.get("error"),
        "error_code": m.get("error_code"),

        "started_at": m.get("started_at"),
        "finished_at": m.get("finished_at"),
        "duration_s": m.get("duration_s"),

        "chunk_count": m.get("chunk_count"),
        "payload": m.get("payload"),
        "delta": m.get("delta"),
        "seam": m.get("seam"),
        "latent_preserved": m.get("latent_preserved"),
        "preserved_rows": m.get("preserved_rows"),

        "thumbs": m.get("thumbs"),
    }


@app.get("/lab/runs")
def runs(limit: int = Query(5, ge=1, le=200)):
    # 문서 §엔드포인트: 배열이 아니라 {"runs": [...]} 로 감싼다.
    return {"runs": [_summary(r) for r in list_run_ids()[::-1][:limit]]}


@app.get("/lab/runs/{run_id}")
def run_detail(run_id: str):
    if not RUN_ID_RE.match(run_id):
        raise HTTPException(400, "run_id 형식이 아니다")
    m = read_meta(run_id)
    if m is None:
        raise HTTPException(404, "없는 run")
    ch = load_chunks(run_id)
    masked = set((m.get("mask") or {}).get("chunk_keys", []))
    out = dict(_summary(run_id))
    out.update({
        "input_image": m.get("input_image"),
        "mask": m.get("mask"),
        "renders": m.get("renders"),
        "chunk_map_ascii": chunk_map(set(ch), masked=masked),
        "comparisons": m.get("comparisons"),
        "log_path": m.get("log_path", "log.txt"),   # log_text 는 싣지 않는다 (§5)
        "meta": m,
    })
    return out


@app.get("/lab/runs/{run_id}/files/{path:path}")
def run_file(run_id: str, path: str):
    if not RUN_ID_RE.match(run_id):
        raise HTTPException(400, "run_id 형식이 아니다")
    root = os.path.realpath(run_dir(run_id))
    if not os.path.isdir(root):
        raise HTTPException(404, "없는 run")
    # 경로 탈출 방지: realpath 로 심볼릭 링크까지 풀어 root 안인지 확인한다.
    # 3090도 같은 검증을 한다 — 양쪽 모두 검증하는 게 문서 규칙이다.
    target = os.path.realpath(os.path.join(root, path))
    if target != root and not target.startswith(root + os.sep):
        raise HTTPException(403, "run 디렉터리 밖은 접근할 수 없다")
    if not os.path.isfile(target):
        raise HTTPException(404, "없는 파일")
    return FileResponse(target)


def serve(host="0.0.0.0", port=8082):
    import uvicorn

    print(f"lab API (읽기 전용)  LAB_ROOT={LAB_ROOT}  http://{host}:{port}/lab/runs")
    uvicorn.run(app, host=host, port=port, log_level="info")
