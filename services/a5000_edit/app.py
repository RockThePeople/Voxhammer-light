"""04L A5000 상주 편집·렌더 워커 (R8a A절 규칙 + 04L 변경). 포트 8082, 3090 → A5000 한 방향(정오표 E4).

    POST /v2/trellis/render                       조건 렌더(구간 1): 요청 시점 1장 + 2D 마스크, render_id·두 해시 (계약 3.28.0)
    GET  /v2/trellis/renders/{id}/2d_render.png   ·  .../2d_mask.png
    POST /v2/trellis/edit                         multipart meta + 2d_render · 2d_mask · 2d_edit → 202 {job_id}
    GET  /v2/trellis/jobs/{id}                    기존 응답(BChunkResponse: chunks · to_version …)
    GET  /v2/assets/{a}/staging/{job}/chunks/{k}.cbin · GET /v2/trellis/assets/{a}/chunks/{k}.v{n}.cbin
    POST /v2/trellis/assets/{a}/commit            (기존)
    GET  /v2/trellis/health

경로·응답 모양·오류 본문은 현행 서비스(`legacy/a5000/edit_service/api/routes.py`)를 따른다. 편집 연산만 editcore 로 바꿨다.
인증: X-Blockedit-Key = TRELLIS_API_KEY(.env). 기동: `bash services/a5000_edit/run.sh` (README).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "third_party" / "ai-ar-v3" / "contract" / "python"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fastapi import APIRouter, Depends, FastAPI, Header, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402

from deltacontract import COORD_ORDER, mask_fingerprint, validate_job_id  # noqa: E402
from deltacontract.errors import (DeltaContractError, InvalidRequest, NotFound, AssetNotFound,  # noqa: E402
                                  StagingExpired, Unauthorized, VersionConflict)

from services.a5000_edit import jobs, store  # noqa: E402

API_KEY = os.environ.get("TRELLIS_API_KEY", "dev-key")
RENDER_TTL_S = 24 * 3600
RENDER_ROOT = Path(os.environ.get("VHL_RENDER_ROOT") or Path(os.environ.get("RUNS_DIR", "runs")) / "04l" / "renders")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ── 계약 3.28.0 거절 사유 (고정 계약 19ed91f 에는 아직 없어 같은 기반 클래스로 정의, ai-ar-v3 PR #27 과 같은 코드·상태) ──
class ConditionImagesRequired(DeltaContractError):
    error_code, http_status = "CONDITION_IMAGES_REQUIRED", 422


class RenderMismatch(DeltaContractError):
    error_code, http_status = "RENDER_MISMATCH", 409


class RenderUnknown(DeltaContractError):
    error_code, http_status = "RENDER_UNKNOWN", 404


class BadConditionImage(DeltaContractError):
    error_code, http_status = "BAD_CONDITION_IMAGE", 400


def require_key(x_blockedit_key: str = Header(None)):
    if x_blockedit_key is None:
        raise Unauthorized("X-Blockedit-Key 없음")
    if x_blockedit_key != API_KEY:
        raise Unauthorized("X-Blockedit-Key 불일치")


router = APIRouter(prefix="/v2/trellis", dependencies=[Depends(require_key)])
v2 = APIRouter(prefix="/v2", dependencies=[Depends(require_key)])


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ── health ────────────────────────────────────────────────────────────
@router.get("/health")
def health():
    import torch
    from services.a5000_edit import engine
    return {"ok": engine.STATE["pipe"] is not None, "service": "vhl-04l a5000 worker", "load": engine.STATE["load"],
            "queue": jobs.Q.qsize(), "gpu_mem_gb": round(torch.cuda.memory_allocated() / 1024 ** 3, 3),
            "trellis_variant": "TRELLIS-image-large", "coord_order": COORD_ORDER, "vhl_r4": os.environ.get("VHL_R4"),
            "data_root": store.DATA_ROOT, "render_root": str(RENDER_ROOT)}


# ── render (계약 3.28.0 BRenderRequest/Response) ───────────────────────
def _render_dir(rid: str) -> Path:
    if not rid or not all(c.isalnum() or c in "-_" for c in rid) or len(rid) > 64:
        raise InvalidRequest(f"render_id 형식 위반: {rid!r}")
    return RENDER_ROOT / rid


def _render_record(rid: str):
    d = _render_dir(rid)
    p = d / "record.json"
    if not p.is_file():
        return None, d
    rec = json.loads(p.read_text())
    if time.time() - rec["created"] > RENDER_TTL_S:
        return None, d
    return rec, d


@router.post("/render")
def render(body: dict):
    try:
        asset_id, base_version, mask = body["asset_id"], int(body["base_version"]), body["mask"]
    except (KeyError, TypeError, ValueError) as e:
        raise InvalidRequest(f"render 요청 필드 오류: {e}")
    view = body.get("view")
    if view is not None and (not isinstance(view, int) or not 0 <= view <= 4):
        raise InvalidRequest(f"view 는 0–4 또는 null: {view!r}")
    if not store.has_committed_version(asset_id, base_version):
        raise VersionConflict(f"base_version={base_version} 은 커밋되지 않았다 (committed latest={store.committed(asset_id)})",
                              detail={"latest_version": str(store.committed(asset_id))})
    from services.a5000_edit import engine
    cells = engine.mask_cells(mask)
    if not len(cells):
        raise InvalidRequest("마스크 셀이 없다")
    rid = f"r-{uuid.uuid4().hex[:16]}"
    d = _render_dir(rid)
    d.mkdir(parents=True, exist_ok=True)
    jid = jobs.new_job("render", asset_id)
    jobs.submit(jid, lambda _p: engine.render(Path(store.version_dir(asset_id, base_version)), cells, view, d))
    while jobs.JOBS[jid]["state"] in ("queued", "running"):   # 단일 GPU 큐를 지나 동기로 돌려준다
        time.sleep(0.05)
    j = jobs.JOBS.pop(jid)
    if j["state"] != "succeeded":
        raise DeltaContractError(f"조건 렌더 실패: {j.get('error')}")
    r = j["result"]
    rb, mb = (d / "2d_render.png").read_bytes(), (d / "2d_mask.png").read_bytes()
    rec = {"render_id": rid, "asset_id": asset_id, "base_version": base_version, "view": r["view"],
           "render_sha256": _sha(rb), "mask_sha256": _sha(mb), "renderer": r["renderer"], "created": time.time(),
           "mask_fingerprint": mask_fingerprint(cells), "timing": {**r["timing"], "queue_wait_s": j.get("queue_wait_s")},
           "view_auto": r["view_auto"], "view_areas": r["view_areas"]}
    (d / "record.json").write_text(json.dumps(rec, indent=1))
    return {k: rec[k] for k in ("render_id", "asset_id", "base_version", "view", "render_sha256", "mask_sha256")} | {
        "renderer": {**rec["renderer"], "timing": rec["timing"], "view_auto": rec["view_auto"], "view_areas": rec["view_areas"]}}


@router.get("/renders/{rid}/{name}")
def render_file(rid: str, name: str):
    if name not in ("2d_render.png", "2d_mask.png"):
        raise NotFound(f"없는 렌더 파일: {name}")
    rec, d = _render_record(rid)
    if rec is None:
        raise RenderUnknown(f"render_id 기록이 없다(만료 포함): {rid}")
    return FileResponse(d / name, media_type="image/png")


# ── edit (계약 3.28.0 multipart) ─────────────────────────────────────
def _check_png(name: str, b: bytes):
    from PIL import Image
    if not b.startswith(PNG_MAGIC):
        raise BadConditionImage(f"{name} 가 PNG 가 아니다", detail={"part": name})
    try:
        im = Image.open(io.BytesIO(b))
        size = im.size
    except Exception as e:  # noqa: BLE001
        raise BadConditionImage(f"{name} 를 읽을 수 없다: {e}", detail={"part": name})
    if size != (1024, 1024):
        raise BadConditionImage(f"{name} 가 1024² 가 아니다: {size}", detail={"part": name, "size": f"{size[0]}x{size[1]}"})


@router.post("/edit", status_code=202)
async def edit(request: Request):
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if ctype != "multipart/form-data":
        raise ConditionImagesRequired("편집은 multipart(meta + 2d_render · 2d_mask · 2d_edit)만 받는다 — 조건 없이 진행하지 않는다",
                                      detail={"missing": "2d_render,2d_mask,2d_edit"})
    form = await request.form()
    if form.get("image") is not None:
        raise InvalidRequest("기존 image(t2i 조건) 필드는 받지 않는다(규약 §4 결정 2)")
    imgs = {}
    for n in ("2d_render", "2d_mask", "2d_edit"):
        up = form.get(n)
        b = await up.read() if up is not None and hasattr(up, "read") else None
        if b:
            imgs[n] = b
    missing = [n for n in ("2d_render", "2d_mask", "2d_edit") if n not in imgs]
    if missing:
        raise ConditionImagesRequired(f"조건 이미지가 없다: {missing}", detail={"missing": ",".join(missing)})
    if form.get("meta") is None:
        raise InvalidRequest("multipart 인데 meta 파트가 없다")
    try:
        meta = json.loads(form.get("meta"))
        asset_id, base_version, mask = meta["asset_id"], int(meta["base_version"]), meta["mask"]
        seed = int(meta.get("seed", 42))
    except (KeyError, TypeError, ValueError) as e:
        raise InvalidRequest(f"meta 검증 실패: {e}")
    for n, b in imgs.items():
        _check_png(n, b)
    key = meta.get("idempotency_key")
    if not key:
        raise InvalidRequest("idempotency_key 가 필요하다(staging 슬롯·재생 색인을 이 키로 잡는다)")
    rid = meta.get("render_id")
    if not rid:
        raise InvalidRequest("meta.render_id 가 필요하다")
    rec, _ = _render_record(rid)
    if rec is None:
        raise RenderUnknown(f"render_id 기록이 없다(만료 포함): {rid}")
    got = {"render_sha256": _sha(imgs["2d_render"]), "mask_sha256": _sha(imgs["2d_mask"])}
    bad = {k: f"기록 {rec[k][:12]}… / 받음 {got[k][:12]}… / meta {str(meta.get(k))[:12]}…"
           for k in got if got[k] != rec[k] or (meta.get(k) is not None and meta.get(k) != rec[k])}
    if bad:
        raise RenderMismatch("받은 조건 렌더·마스크가 render_id 기록과 다르다", detail=bad)
    if int(rec["base_version"]) != base_version or rec["asset_id"] != asset_id:
        raise VersionConflict(f"base_version/asset 이 렌더 때와 다르다 (렌더 {rec['asset_id']} v{rec['base_version']}, 요청 {asset_id} v{base_version})",
                              detail={"latest_version": str(store.committed(asset_id))})
    from services.a5000_edit import engine
    cells = engine.mask_cells(mask)
    if mask_fingerprint(cells) != rec["mask_fingerprint"]:
        raise RenderMismatch("마스크 셀이 렌더 때와 다르다", detail={"render": rec["mask_fingerprint"][:12], "edit": mask_fingerprint(cells)[:12]})
    store.maybe_gc()
    replay = store.replayable(asset_id, key)
    if replay is not None:
        jid = store.job_id_for_key(asset_id, key) or jobs.new_job("edit", asset_id, to_version=replay.get("to_version"))
        if jid not in jobs.JOBS:
            jobs.new_job("edit", asset_id, job_id=jid, to_version=replay.get("to_version"))
        jobs.JOBS[jid].update(state="succeeded", stage="chunk", progress=1.0, result={**replay, "replayed": True}, replayed=True)
        return {"job_id": jid}
    if not store.has_committed_version(asset_id, base_version):
        raise VersionConflict(f"base_version={base_version} 은 커밋되지 않았다 (committed latest={store.committed(asset_id)})",
                              detail={"latest_version": str(store.committed(asset_id))})
    to_version = store.committed(asset_id) + 1
    halo = int(mask.get("halo_margin_voxels", 2))
    jid = jobs.new_job("edit", asset_id, to_version=to_version, idempotency_key=key)

    def work(progress):
        parent = Path(store.version_dir(asset_id, base_version))
        d = Path(store.staging_dir(asset_id, key))
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        for n, b in imgs.items():
            (d / f"{n}.png").write_bytes(b)
        progress("repaint")
        out = engine.edit(parent, d, cells=cells, halo=halo, seed=seed, asset_id=asset_id, to_version=to_version)
        progress("chunk")
        store.record_job(asset_id, jid, key, to_version, mask={"raw": mask_fingerprint(cells)})
        resp = _response(asset_id, to_version, d, out, cells, halo, jid)
        store.write_staging(asset_id, key, {**resp, "replayed_base_version": base_version})
        (d / "request.json").write_text(json.dumps({"meta": meta, "render": rec, "queue_wait_s": jobs.JOBS[jid].get("queue_wait_s")},
                                                   indent=1))
        return resp

    jobs.submit(jid, work)
    return {"job_id": jid}


def _response(asset_id, to_version, d: Path, out: dict, cells, halo, job_id):
    """BChunkResponse — 현행 서비스 `routes._response` 와 같은 모양."""
    from deltacontract.coords import dilate_cells
    from deltacontract.schemas import BChunkResponse
    r4, t = out["r4"], out["timing"]
    man = json.loads((d / "manifest.json").read_text())["chunks"]
    exist = set(r4["sent"]) | set(r4["reused"]) | set(r4["removed"])
    book = sorted(k for k in r4["book"] if k in exist)
    chunks = [{"chunk_id": k, "hash": man[k]["hash"], "byte_length": man[k]["byte_length"], "vertex_count": man[k]["vertex_count"],
               "index_count": man[k]["index_count"], "voxel_count": man[k]["voxel_count"], "version": to_version}
              for k in sorted(r4["sent"]) if k in man]
    seg = t["segments"]
    seam = (r4.get("seam") or {}).get("after") or {}
    resp = BChunkResponse(
        asset_id=asset_id, to_version=to_version, is_initial=False,
        mask_fingerprint=mask_fingerprint(cells), mask_fingerprint_dilated=mask_fingerprint(dilate_cells(cells, halo)),
        chunks=chunks, removed_chunk_ids=list(r4["removed"]), bookkeeping_affected_chunk_ids=book,
        stats={k: v for k, v in {
            "bookkeeping_chunks": len(book), "regenerated_chunks": len(chunks), "total_chunks": len(man),
            "repaint_ms": round(1000 * sum(seg.get(s, {}).get("seconds", 0.0) for s in ("4", "5", "6")), 1),
            "decode_ms": round(1000 * seg.get("7", {}).get("parts", {}).get("decode_mesh", 0.0), 1),
            "chunk_ms": round(1000 * (seg.get("7", {}).get("seconds", 0.0) - seg.get("7", {}).get("parts", {}).get("decode_mesh", 0.0)), 1),
            "snapped_vertices": (r4.get("seam") or {}).get("snapped"),
            "seam_median_cells": seam.get("cells_median"), "seam_max_cells": seam.get("cells_max"),
            "voxels": t["counts"].get("coords_tgt") if isinstance(t.get("counts"), dict) else None,
            "occupancy_only_chunks": r4.get("occupancy_only_added"), "new_outside_chunks": len(r4.get("new_outside", [])),
        }.items() if isinstance(v, (int, float))})
    return json.loads(resp.model_dump_json())


# ── jobs · 청크 · commit (현행 서비스와 같은 모양) ─────────────────────
@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    j = jobs.JOBS.get(job_id)
    if j is None:
        asset, slot = store.find_job(job_id)
        if slot:
            stored = store.stored_response(asset, slot)
            if stored is not None:
                return stored
        raise NotFound(f"없는 job: {job_id}")
    if j["state"] == "succeeded":
        return j["result"]
    if j["state"] == "failed":
        from deltacontract.errors import ERROR_CODE_TO_EXCEPTION, InternalError
        raise ERROR_CODE_TO_EXCEPTION.get(j.get("error_code"), InternalError)(j.get("error") or "실패")
    return {"job_id": job_id, "state": j["state"], "stage": j["stage"], "progress": j["progress"], "asset_id": j["asset_id"]}


@router.get("/assets/{asset_id}/chunks/{name}")
def chunk(asset_id: str, name: str):
    if not name.endswith(".cbin") or ".v" not in name[:-5]:
        raise InvalidRequest("{key}.v{n}.cbin 형식이어야 한다")
    ckey, ver = name[:-5].rsplit(".v", 1)
    try:
        p = store.chunk_path(asset_id, ckey, int(ver))
    except ValueError as e:
        raise InvalidRequest(str(e))
    if not p:
        raise NotFound("없는 청크")
    return FileResponse(p, media_type="application/octet-stream")


@v2.get("/assets/{asset_id}/staging/{job_id}/chunks/{name}")
def staging_chunk(asset_id: str, job_id: str, name: str):
    if not name.endswith(".cbin"):
        raise InvalidRequest("cbin 아님")
    validate_job_id(job_id)
    try:
        p, status = store.staging_chunk(asset_id, job_id, name[:-5])
    except ValueError as e:
        raise InvalidRequest(str(e))
    if status == "expired":
        raise StagingExpired(f"staging 만료: {name}")
    if status == "no_job":
        raise NotFound(f"없는 job_id: {job_id}")
    if status == "escape":
        raise InvalidRequest(f"경로 규칙 위반: {name}")
    if status == "no_chunk":
        raise AssetNotFound(f"슬롯에 없는 청크: {name} (job={job_id})")
    return FileResponse(p, media_type="application/octet-stream")


@router.post("/assets/{asset_id}/commit")
def commit(asset_id: str, body: dict):
    try:
        n = store.commit(asset_id, body["to_version"], body["idempotency_key"])
    except KeyError as e:
        raise InvalidRequest(f"필수 필드 없음: {e}")
    except FileNotFoundError as e:
        raise NotFound(str(e))
    except ValueError as e:
        raise InvalidRequest(str(e))
    return {"committed": n}


@router.get("/assets/{asset_id}/v{n}/manifest.json")
def version_manifest(asset_id: str, n: int):
    """04L 보조: 판본 매니페스트(3090 이 시드 판본을 맞출 때). 바이트를 재가공하지 않는다."""
    p = Path(store.version_dir(asset_id, n)) / "manifest.json"
    if not store.has_committed_version(asset_id, n) or not p.is_file():
        raise NotFound(f"없는 판본: {asset_id} v{n}")
    return FileResponse(p, media_type="application/json")


app = FastAPI(title="vhl-04l A5000 worker", version="1.0")
app.include_router(router)
app.include_router(v2)


@app.exception_handler(DeltaContractError)
async def _contract_error(request, exc: DeltaContractError):
    return JSONResponse(exc.to_dict(), status_code=exc.http_status)


@app.exception_handler(RequestValidationError)
async def _validation_error(request, exc: RequestValidationError):
    e = InvalidRequest(str(exc.errors()[:3]))
    return JSONResponse(e.to_dict(), status_code=e.http_status)


@app.exception_handler(StarletteHTTPException)
async def _http_error(request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return JSONResponse({"error_code": "NOT_FOUND", "message": f"모르는 경로: {request.url.path}"}, status_code=404)
    return JSONResponse({"error_code": "UNAUTHORIZED" if exc.status_code == 401 else "INTERNAL", "message": str(exc.detail)},
                        status_code=exc.status_code)


def startup(preload=True):
    from services.a5000_edit import engine
    jobs.start(preload=engine.load if preload else None)


def serve(host="0.0.0.0", port=8082, preload=True):
    import uvicorn
    app.add_event_handler("startup", lambda: startup(preload))
    print(f"[04l] /v2/trellis/*  http://{host}:{port}  data={store.DATA_ROOT} renders={RENDER_ROOT}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("EDIT_PORT") or 8082))
    ap.add_argument("--no-preload", action="store_true")
    a = ap.parse_args()
    serve(a.host, a.port, preload=not a.no_preload)
