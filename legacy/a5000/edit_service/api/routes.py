"""/v2/trellis/* — 제품 경로 (HTTP_CONTRACT.md §2).

A5000 은 glTF·매니페스트·버전 발급을 모른다. asset_id 는 3090이 발급해 보낸다.
"""
import json
import os
import shutil
import tempfile

from fastapi import APIRouter, Depends, File, Form, Header, Request, UploadFile
from pydantic import ValidationError as PydanticValidationError
from deltacontract import coords
from fastapi.responses import FileResponse

from deltacontract import COORD_ORDER, validate_job_id
from deltacontract.errors import (
    AssetNotFound,
    InvalidRequest,
    NotFound,
    StagingExpired,
    Unauthorized,
    VersionConflict,
)
from deltacontract.schemas import (BAssembleRequest, BChunkResponse, BEditRequest,
                                   BGenerateRequest, BHealth)

from . import build, jobs, store

API_KEY = os.environ.get("TRELLIS_API_KEY", "dev-key")
SNAP_CELLS = float(os.environ.get("TRELLIS_SNAP_CELLS", "2"))


def require_key(x_blockedit_key: str = Header(None)):
    """라우터 단위 인증. 프리픽스 면제를 만들지 않는다 — 그게 구멍이 된다.

    ⚠️ HTTPException 을 쓰지 않는다. FastAPI 가 본문을 {"detail": ...} 로 한 겹
    싸는데, 그러면 4xx/5xx 가 전부 같은 모양이라는 계약이 무너지고 소비 측만
    예외 처리를 들게 된다. DeltaContractError 를 던지면 main.py 의 핸들러가
    exc.to_dict() 를 그대로 본문으로 낸다.

    키 값은 메시지에 넣지 않는다 — 로그·응답에 유출된다.
    """
    if x_blockedit_key is None:
        raise Unauthorized("X-Blockedit-Key 없음")
    if x_blockedit_key != API_KEY:
        raise Unauthorized("X-Blockedit-Key 불일치")


router = APIRouter(prefix="/v2/trellis", dependencies=[Depends(require_key)])


# ── health ────────────────────────────────────────────────────────────
@router.get("/health")
def health():
    import torch

    from lab.pipeline import _PIPE

    # deterministic_algorithms 는 **실제 상태를 정직하게** 보고한다.
    # 계약 3.4.0 이후 이 값은 참고 정보이고 3090은 이걸로 거부하지 않는다.
    det = bool(torch.are_deterministic_algorithms_enabled())
    gb = (torch.cuda.memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0
    return BHealth(
        ok=_PIPE is not None,
        # 계약 3.16.0 — 돌고 있는 코드. 기동 시 굳힌 값이다 (build.py 주석 참조).
        **build.fields(),
        low_vram=False,
        gpu_mem_gb=round(gb, 3),
        trellis_variant="TRELLIS-image-large",
        stages=["sparse_structure", "slat"],
        slat_resolution=64,
        deterministic_algorithms=det,
        coord_order=COORD_ORDER,
    )


# ── generate ──────────────────────────────────────────────────────────
@router.post("/generate", status_code=202)
async def generate(meta: str = Form(...), image: UploadFile = File(...)):
    try:
        req = BGenerateRequest(**json.loads(meta))
    except Exception as e:                                   # noqa: BLE001
        raise InvalidRequest(f"meta 파싱 실패: {e}")

    if store.committed(req.asset_id):
        raise VersionConflict("이미 생성된 asset_id 다")

    tmp = tempfile.mkdtemp(prefix="gen_")
    ipath = os.path.join(tmp, "input.png")
    with open(ipath, "wb") as f:
        f.write(await image.read())

    key = f"gen-{req.asset_id}"
    jid = jobs.new_job("generate", req.asset_id, to_version=1, idempotency_key=key)

    def work(progress):
        from . import ops as ops_mod
        from .ops import op_generate

        d = store.staging_dir(req.asset_id, key)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(os.path.join(d, "chunks"), exist_ok=True)
        r = op_generate(d, ipath, req.seed, progress)
        # generate 는 승계가 없어 엔트리가 곧 합성물이지만, asset_id/version 을
        # 실제 값으로 다시 쓴다 — decode_and_chunk 는 staging 키를 asset_id 로 넣는다.
        ops_mod.write_manifest(d, req.asset_id, 1, r["entries"], r.get("sent") or ())
        store.record_job(req.asset_id, jid, key, 1)
        resp = _response(req.asset_id, 1, r, is_initial=True, job_id=jid)
        store.write_staging(req.asset_id, key, resp)
        shutil.rmtree(tmp, ignore_errors=True)
        return resp

    jobs.submit(jid, work)
    return {"job_id": jid}


# ── ingest (외부 판본 수납) ────────────────────────────────────────────
#
# 🔴 왜 필요한가: 커밋 API 는 `commit(job_id + staging)` 뿐이라 **이 서버가 만든 것만**
#    커밋할 수 있다. 3090 이 로컬에서 만든 판본에는 staging 이 없어 밀어 넣을 길이 없었고,
#    그래서 v1 을 손으로 세웠다. 그것이 없으면 **v2 기반 편집이 영구히 막힌다**.
#
# 검사는 V1-REQUIRED-FILES.md 그대로 — 진입점에서 전부 본다. "202 뒤 워커에서 죽는" 셋
# (slat / input.png / slat 메타) 을 여기서 잡는다.
#
#   POST /v2/trellis/assets/{asset_id}/ingest      multipart/form-data
#     meta      = {"version": N}                (필수)
#     manifest  = manifest.json                 (필수)
#     slat      = slat.safetensors              (필수, coords+feats+norm 메타)
#     input     = input.png                     (필수)
#     chunks    = chunks.tar.gz                 (필수, 내부에 {key}.cbin)
@router.post("/assets/{asset_id}/ingest", status_code=201)
async def ingest(asset_id: str, request: Request):
    import hashlib, io, tarfile
    form = await request.form()

    def part(name, required=True):
        v = form.get(name)
        if v is None and required:
            raise InvalidRequest(f"{name} 파트가 없다 (V1-REQUIRED-FILES)")
        return v

    try:
        version = int(json.loads(await part("meta").read() if hasattr(part("meta"), "read")
                                 else form.get("meta"))["version"])
    except Exception as e:
        raise InvalidRequest(f"meta 파싱 실패(version 필요): {e}")
    if version < 1:
        raise InvalidRequest("version 은 1 이상이어야 한다")

    man_b = await part("manifest").read()
    slat_b = await part("slat").read()
    img_b = await part("input").read()
    tar_b = await part("chunks").read()
    try:
        man = json.loads(man_b)
    except Exception as e:
        raise InvalidRequest(f"manifest 파싱 실패: {e}")

    # --- 계약 판본 가드 ---
    c = man.get("contract") or {}
    if int(c.get("contract_version", -1)) != coords.CONTRACT_VERSION:
        raise InvalidRequest(
            f"contract_version 불일치: 수신 {c.get('contract_version')} vs 로컬 "
            f"{coords.CONTRACT_VERSION} — 옛 판본을 새 코드에 물리지 않는다")
    if int(c.get("chunk_size", -1)) != coords.CHUNK_SIZE:
        raise InvalidRequest(f"chunk_size 불일치: {c.get('chunk_size')} vs {coords.CHUNK_SIZE}")

    # --- slat 메타 가드 (없으면 다음 편집이 죽는 자리) ---
    try:
        from safetensors import safe_open as _so
        import tempfile as _tf
        with _tf.NamedTemporaryFile(suffix=".safetensors", delete=False) as _t:
            _t.write(slat_b); _tmp = _t.name
        with _so(_tmp, "pt") as _f:
            md = _f.metadata() or {}
            keys = set(_f.keys())
        os.unlink(_tmp)
    except InvalidRequest:
        raise
    except Exception as e:
        raise InvalidRequest(f"slat.safetensors 를 읽을 수 없다: {e}")
    if not {"coords", "feats"} <= keys:
        raise InvalidRequest(f"slat 텐서 누락: {sorted({'coords','feats'} - keys)}")
    miss = [k for k in ("norm_mean", "norm_std", "slat_space") if k not in md]
    if miss:
        raise InvalidRequest(
            f"slat 메타 누락 {miss} — 이대로 커밋하면 다음 op_edit 이 KeyError 로 죽는 "
            f"**막다른 판본**이 된다 (ops.py:265)")

    entries = man.get("chunks") or {}
    if not isinstance(entries, dict):
        entries = {e["chunk_id"]: e for e in entries}

    # --- 청크: 전건 sha256 + byte_length. 부분 수신은 성공이 아니다 ---
    got = {}
    with tarfile.open(fileobj=io.BytesIO(tar_b), mode="r:*") as tf:
        for m in tf.getmembers():
            if not m.isfile() or not m.name.endswith(".cbin"):
                continue
            got[os.path.basename(m.name)[:-5]] = tf.extractfile(m).read()
    missing = sorted(set(entries) - set(got))
    extra = sorted(set(got) - set(entries))
    if missing or extra:
        raise InvalidRequest(f"청크 불일치 — 매니페스트에 있고 없는 것 {missing[:5]} "
                             f"(총 {len(missing)}), 매니페스트에 없는 것 {extra[:5]} (총 {len(extra)})")
    bad = [k for k, e in entries.items()
           if hashlib.sha256(got[k]).hexdigest() != e["hash"] or len(got[k]) != e["byte_length"]]
    if bad:
        raise InvalidRequest(f"sha256/byte_length 불일치 {len(bad)}건: {bad[:5]}")

    # --- 전부 통과한 뒤에만 쓴다 ---
    vd = store.version_dir(asset_id, version)
    tmp = vd + ".ingest.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "chunks"), exist_ok=True)
    for k, b in got.items():
        with open(os.path.join(tmp, "chunks", f"{k}.cbin"), "wb") as f:
            f.write(b)
    for name, blob in (("manifest.json", man_b), ("slat.safetensors", slat_b), ("input.png", img_b)):
        with open(os.path.join(tmp, name), "wb") as f:
            f.write(blob)
    shutil.rmtree(vd, ignore_errors=True)
    os.replace(tmp, vd)
    prev = store.committed(asset_id)
    if version > prev:
        with open(os.path.join(store.asset_dir(asset_id), "committed.json"), "w") as f:
            json.dump({"latest": version}, f)
    return {"asset_id": asset_id, "version": version, "chunks": len(got),
            "committed_latest": store.committed(asset_id), "verified": "sha256+byte_length"}


# ── edit ──────────────────────────────────────────────────────────────
#
# 🔴 조건 이미지 슬롯 (D40).  A5000 실측: 조건 이미지가 프롬프트를 이긴다 —
#    조건을 3-머리 이미지로 바꾸자 제거 730 → 471 로 **방향이 뒤집혔다**(W11).
#    `prompt` 는 no-op 이므로(3.17.0) 형태 편집의 유일한 지시는 **이미지**다.
#
# 요청 형식 둘을 **Content-Type 으로 가른다.** 기존 JSON 경로를 깨지 않는다 —
# recolor 왕복(W27②)이 그 경로로 성립해 있고, 그것을 깨면 무엇이 실패했는지 못 가른다.
#   application/json  : 기존 그대로.  BEditRequest 본문
#   multipart/form-data: meta=<BEditRequest JSON 문자열> + image=<PNG/JPEG 파일>
def _as_edit_request(payload):
    """dict -> BEditRequest. 검증 실패를 **422 로** 돌려준다.

    ⚠️ 시그니처를 `req: BEditRequest` 에서 `Request` 로 바꾸면 FastAPI 의 자동
    422 가 사라진다. 그대로 두면 스키마 위반이 **평문 500** 으로 나가고
    클라이언트가 원인을 못 읽는다 (A5000 이 이 회귀를 자기 테스트로 잡았다).
    """
    try:
        return BEditRequest(**payload)
    except PydanticValidationError as e:
        raise InvalidRequest(f"EditRequest 검증 실패: {e.errors()[0].get('msg', e)}")


async def _parse_edit(request):
    """(BEditRequest, 조건이미지 바이트 | None) 를 돌려준다."""
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if ctype == "multipart/form-data":
        form = await request.form()
        meta = form.get("meta")
        if meta is None:
            raise InvalidRequest("multipart 인데 meta 파트가 없다")
        up = form.get("image")
        img = await up.read() if up is not None and hasattr(up, "read") else None
        if img is not None and len(img) == 0:
            raise InvalidRequest("image 파트가 비었다")
        return _as_edit_request(json.loads(meta)), img
    return _as_edit_request(await request.json()), None


@router.post("/edit", status_code=202)
async def edit(request: Request):
    req, cond_image = await _parse_edit(request)
    # ⚠️ 계약은 idempotency_key 를 Optional 로 두지만 **이 구현은 그것 없이 못 돈다** —
    #    staging 슬롯·재생 색인·job 기록이 전부 이 키로 잡힌다(store.staging_dir /
    #    record_idempotency). 없으면 라우트는 202 를 주고 워커가 죽어 job 이 INTERNAL 이 된다.
    #    조용히 키를 지어내면 두 번의 같은 요청이 다른 슬롯을 갖게 되어 멱등이 깨진다.
    #    ⇒ 지어내지 말고 **명시적으로 거절**한다. 계약/구현 불일치는 별도로 보고했다.
    if req.idempotency_key is None:
        raise InvalidRequest(
            "idempotency_key 가 필요하다. 계약상 Optional 이지만 편집 구현은 이 키로 "
            "staging 슬롯과 재생 색인을 잡는다 — 없으면 재시도 안전성이 성립하지 않는다")
    # ⚠️ 지금은 받아서 **보관만** 한다. 형태 편집 op 가 켜질 때 소비한다 (W26f 준비서).
    #    받자마자 쓰는 척하면 D40 을 고친 것처럼 보이고 실제로는 안 고쳐진다.
    store.maybe_gc()          # 지연 GC. 스케줄러 대신 요청 시 최대 5분에 한 번.
    # ★ 계약 3.7.0 — **멱등 확인이 base_version 검사보다 먼저다.**
    #
    # 순서를 바꾸면 조용히 틀린다. "서버는 v2 로 커밋했는데 응답만 유실" 된
    # 재시도가 들어오면, base_version 을 먼저 보는 순간 409 가 나가고
    # 클라이언트는 **성공한 편집을 실패로 배운다.**
    #
    # 키는 요청 상태가 아니라 **연산을 식별한다.** 그래서 커밋 여부와도 무관하게
    # 조회한다 (store.replayable).
    replay = store.replayable(req.asset_id, req.idempotency_key)
    if replay is not None:
        # ★ 계약 3.15.x 불변식: 응답이 실은 job_id 로 staging GET 이 200 이어야 한다.
        #
        # 재생마다 새 job_id 를 발급하면 슬롯은 원본 job_id 아래 있어서 포인터가
        # 어긋난다 — 바이트는 멀쩡한데 조회만 404 다. 3.12.0 의 chunk_path()
        # sorted() 첫 매치와 같은 형태를 두 번째 밟는 것이다.
        #
        # 원본 id 를 그대로 돌려준다(3090 제안 (a)). 멱등키가 연산을 식별하므로
        # 그 연산을 대표하는 job_id 도 키의 함수여야 한다 — 재생마다 새 id 를
        # 만들 이유가 없다. 3090 의 staging-map 도 1:1 로 유지된다.
        jid = store.job_id_for_key(req.asset_id, req.idempotency_key)
        if jid is None:
            # 3.14.x 이전 슬롯엔 job.json 이 없다. 새 id 를 발급하고 슬롯에 이어준다.
            jid = jobs.new_job("edit", req.asset_id,
                               to_version=replay.get("to_version"),
                               idempotency_key=req.idempotency_key)
            store.record_job(req.asset_id, jid, req.idempotency_key,
                             replay.get("to_version") or 0)
        elif jid not in jobs.JOBS:
            jobs.new_job("edit", req.asset_id, job_id=jid,
                         to_version=replay.get("to_version"),
                         idempotency_key=req.idempotency_key)
        # base_version 이 저장된 것과 달라도 저장된 결과를 준다. 다만 조용히
        # 넘기지 않는다 — 클라이언트가 혼동한 상태라는 신호다.
        stored_base = replay.get("replayed_base_version")
        if stored_base is not None and stored_base != req.base_version:
            print(f"[warn] 멱등 재생 base_version 불일치: 요청={req.base_version} "
                  f"저장={stored_base} key={req.idempotency_key}", flush=True)
            jobs.JOBS[jid]["base_version_mismatch"] = {
                "requested": req.base_version, "stored": stored_base}
        # 계약 3.11.x 의 replayed / replayed_base_version 을 쓴다.
        out = {**replay, "replayed": True}
        if stored_base is not None:
            out["replayed_base_version"] = stored_base
        jobs.JOBS[jid].update(state="succeeded", stage="chunk", progress=1.0,
                              result=out, replayed=True)
        return {"job_id": jid}

    # 멱등 재생이 없을 때만 base_version 을 검증한다 (§10-1).
    if not store.has_committed_version(req.asset_id, req.base_version):
        raise VersionConflict(
            f"base_version={req.base_version} 은 커밋되지 않았다 "
            f"(committed latest={store.committed(req.asset_id)})")

    to_version = store.committed(req.asset_id) + 1
    jid = jobs.new_job("edit", req.asset_id, to_version=to_version,
                       idempotency_key=req.idempotency_key)

    def work(progress):
        from . import ops as ops_mod
        from .ops import op_edit

        parent = store.version_dir(req.asset_id, req.base_version)
        d = store.staging_dir(req.asset_id, req.idempotency_key)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(os.path.join(d, "chunks"), exist_ok=True)
        if cond_image is not None:
            with open(os.path.join(d, "cond.png"), "wb") as _f:
                _f.write(cond_image)
        r = op_edit(parent, d, req.mask, req.seed, snap_cells=SNAP_CELLS,
                    progress=progress)
        ops_mod.write_manifest(d, req.asset_id, to_version, r["entries"],
                               r.get("sent") or ())
        store.record_job(req.asset_id, jid, req.idempotency_key, to_version,
                         mask=r.get("mask_fp"))
        resp = _response(req.asset_id, to_version, r, is_initial=False, job_id=jid)
        # 재생 시 "요청 base 와 저장 base 가 다른가"를 판정하려면 원래 base 가
        # 필요하다. 계약의 replayed_base_version 필드를 쓴다.
        store.write_staging(req.asset_id, req.idempotency_key,
                            {**resp, "replayed_base_version": req.base_version})
        return resp

    jobs.submit(jid, work)
    return {"job_id": jid}


# ── assemble ──────────────────────────────────────────────────────────
@router.post("/assemble", status_code=202)
def assemble(req: BAssembleRequest):
    """부위 교체. **새 경로다** — 이유는 BAssembleRequest docstring 에 있다.

    멱등/스테이징/잡 기계는 edit 과 **똑같이** 쓴다. 다른 건 계산 코어뿐이다.
    """
    store.maybe_gc()
    # 멱등 확인이 base_version 검사보다 먼저다 (3.7.0). edit 과 같은 이유다:
    # 성공한 연산의 재시도가 409 로 나가면 클라이언트가 성공을 실패로 배운다.
    replay = store.replayable(req.asset_id, req.idempotency_key)
    if replay is not None:
        jid = store.job_id_for_key(req.asset_id, req.idempotency_key)
        if jid is None:
            jid = jobs.new_job("assemble", req.asset_id,
                               to_version=replay.get("to_version"),
                               idempotency_key=req.idempotency_key)
            store.record_job(req.asset_id, jid, req.idempotency_key,
                             replay.get("to_version") or 0)
        elif jid not in jobs.JOBS:
            jobs.new_job("assemble", req.asset_id, job_id=jid,
                         to_version=replay.get("to_version"),
                         idempotency_key=req.idempotency_key)
        out = {**replay, "replayed": True}
        jobs.JOBS[jid].update(state="succeeded", stage="chunk", progress=1.0,
                              result=out, replayed=True)
        return {"job_id": jid}

    if not store.has_committed_version(req.asset_id, req.base_version):
        raise VersionConflict(
            f"base_version={req.base_version} 은 커밋되지 않았다 "
            f"(committed latest={store.committed(req.asset_id)})")
    # 기증자는 **읽기만** 한다. 없으면 여기서 끊는다 — 잡 안에서 터지면
    # 클라이언트가 폴링으로만 알게 되고 원인이 안 보인다.
    if not store.has_committed_version(req.donor_asset_id, 1):
        raise NotFound(f"기증자 자산이 없다: {req.donor_asset_id} v1")

    to_version = store.committed(req.asset_id) + 1
    jid = jobs.new_job("assemble", req.asset_id, to_version=to_version,
                       idempotency_key=req.idempotency_key)

    def work(progress):
        from . import ops as ops_mod
        from .ops import op_assemble

        parent = store.version_dir(req.asset_id, req.base_version)
        donor = store.version_dir(req.donor_asset_id, 1)
        d = store.staging_dir(req.asset_id, req.idempotency_key)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(os.path.join(d, "chunks"), exist_ok=True)
        r = op_assemble(parent, donor, d, req.mask,
                        {"fraction": req.donor_crop_fraction,
                         "axis": req.donor_crop_axis,
                         "keep": req.donor_crop_keep},
                        list(req.offset) if req.offset is not None else None,
                        snap_cells=SNAP_CELLS, progress=progress)
        ops_mod.write_manifest(d, req.asset_id, to_version, r["entries"],
                               r.get("sent") or ())
        store.record_job(req.asset_id, jid, req.idempotency_key, to_version,
                         mask=r.get("mask_fp"))
        resp = _response(req.asset_id, to_version, r, is_initial=False, job_id=jid)
        store.write_staging(req.asset_id, req.idempotency_key,
                            {**resp, "replayed_base_version": req.base_version})
        return resp

    jobs.submit(jid, work)
    return {"job_id": jid}


# ── jobs ──────────────────────────────────────────────────────────────
@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    j = jobs.JOBS.get(job_id)
    if j is None:
        # JOBS 는 메모리라 재시작에 날아간다. 슬롯은 디스크에 남으므로 거기서 찾는다.
        # 재생이 원본 job_id 를 돌려주는 이상, 그 id 로 폴링도 돼야 한다.
        asset, slot = store.find_job(job_id)
        if slot:
            stored = store.stored_response(asset, slot)
            if stored is not None:
                return stored
        raise NotFound(f"없는 job: {job_id}")
    if j["state"] == "succeeded":
        return j["result"]
    if j["state"] == "failed":
        # 워커가 붙여둔 계약 코드를 그대로 되살린다.
        # 계약 3.11.x 에서 BY_CODE → ERROR_CODE_TO_EXCEPTION 으로 개명됐다.
        # 옛 이름을 쓰면 실패한 잡을 조회할 때 ImportError 로 500 이 난다.
        from deltacontract.errors import ERROR_CODE_TO_EXCEPTION, InternalError

        cls = ERROR_CODE_TO_EXCEPTION.get(j.get("error_code"), InternalError)
        raise cls(j.get("error") or "실패")
    return {"job_id": job_id, "state": j["state"], "stage": j["stage"],
            "progress": j["progress"], "asset_id": j["asset_id"]}


# ── chunk bytes ───────────────────────────────────────────────────────
@router.get("/assets/{asset_id}/chunks/{name}")
def chunk(asset_id: str, name: str):
    """{chunk_key}.v{version}.cbin. 바이트를 재가공하지 않는다."""
    if not name.endswith(".cbin"):
        raise InvalidRequest("cbin 아님")
    body = name[:-5]
    if ".v" not in body:
        raise InvalidRequest("{key}.v{n}.cbin 형식이어야 한다")
    ckey, ver = body.rsplit(".v", 1)
    try:
        p = store.chunk_path(asset_id, ckey, int(ver))
    except ValueError as e:
        raise InvalidRequest(str(e))
    if not p:
        raise NotFound("없는 청크")
    return FileResponse(p, media_type="application/octet-stream")


# ── 잡 범위 staging 청크 (계약 §13-8) ─────────────────────────────────
#
# 계약 경로가 /v2/assets/... 로 /v2/trellis 프리픽스 **밖**이라 별도 라우터를 둔다.
# 인증 의존성은 동일하게 라우터 단위로 건다 — 프리픽스 면제를 만들지 않는다.
v2 = APIRouter(prefix="/v2", dependencies=[Depends(require_key)])


@v2.get("/assets/{asset_id}/staging/{job_id}/chunks/{name}")
def staging_chunk(asset_id: str, job_id: str, name: str):
    """ephemeral 편집의 바이트. 결과는 버전이 아니라 **잡**에 속한다.

    기존 .v{n}.cbin 라우트는 staging 을 sorted() 로 훑어 첫 매치를 주기 때문에
    슬롯이 2개 이상이면 **틀린 잡의 바이트**를 준다 (실측: 5개 중 4개 도달 불가).
    이 경로는 job_id 로 슬롯을 특정한다.
    """
    if not name.endswith(".cbin"):
        raise InvalidRequest("cbin 아님")
    # 계약 함수로 파일시스템에 닿기 전에 막는다 (3.14.0).
    validate_job_id(job_id)
    try:
        p, status = store.staging_chunk(asset_id, job_id, name[:-5])
    except ValueError as e:
        raise InvalidRequest(str(e))
    if status == "expired":
        raise StagingExpired(f"staging 만료 (TTL {store.STAGING_TTL_S}s): {name}")
    if status == "no_job":
        raise NotFound(f"없는 job_id: {job_id}")
    if status == "escape":
        # 없는 것(404)과 잘못된 것(422)을 뭉치지 않는다. validate_job_id 를 거쳤는데도
        # 여기 왔다면 계약 위반이므로 크게 남긴다.
        print(f"[ERROR] 경로 탈출 시도: asset={asset_id} job={job_id} name={name}",
              flush=True)
        raise InvalidRequest(f"경로 규칙 위반: {name}")
    if status == "no_chunk":
        # 슬롯은 살아있는데 그 청크가 없다. 부기가 지목했는데 청크가 없다는 뜻이라
        # §4.2 전칭 검증이 깨졌다는 신호다 — NOT_FOUND 와 뭉치면 신호가 사라진다.
        raise AssetNotFound(f"슬롯에 없는 청크: {name} (job={job_id})")
    return FileResponse(p, media_type="application/octet-stream")


# ── commit ────────────────────────────────────────────────────────────
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


# ── 응답 조립 ─────────────────────────────────────────────────────────
def _response(asset_id, to_version, r, is_initial, job_id=None):
    ent = r["entries"]
    t = r["timings"]
    seam = r.get("seam") or {}
    # ★ 계약 3.4.3 전칭 규칙:
    #   ∀ c ∈ bookkeeping : c ∈ chunks ∨ c ∈ removed_chunk_ids
    # 기하가 없는 부기 청크는 반드시 removed 다. 아무 데도 안 넣으면 3090이
    # "빠뜨렸다"와 "비었다고 알려줬다"를 구분할 수 없어 BOOKKEEPING_MISMATCH 가 난다.
    book = list(r["book"])
    sent = r.get("sent") or list(ent)
    removed = list(r["removed"])
    # ⚠️ ChunkMeshPayload.version 은 **필수**다. 문서 §2 의 예시에는 빠져 있는데
    # 스키마가 정본이므로 채운다. 이 버전에서 새로 만든 청크이므로 to_version 이다.
    # ⚠️ uri 를 싣지 않는다. ChunkMeshPayload(B→3090)에는 uri 필드가 **없다** —
    # 계약의 changed_chunks[].uri 는 3090→Unity 의 ChunkEntry 쪽이고 3090이 만든다.
    # 3090은 폴링하던 job_id 를 이미 알고 있으므로
    #   /v2/assets/{asset_id}/staging/{job_id}/chunks/{chunk_id}.cbin
    # 을 그대로 조립하면 된다.
    chunks = [
        {"chunk_id": k, "hash": ent[k].hash, "byte_length": ent[k].byte_length,
         "vertex_count": ent[k].vertex_count, "index_count": ent[k].index_count,
         "voxel_count": ent[k].voxel_count, "version": to_version}
        for k in sorted(sent) if k in ent
    ]
    # 계약 3.13.0 — 지문은 stats(Dict[str,float])에 못 실어 별도 필드다.
    fp = r.get("mask_fp") or {}
    resp = BChunkResponse(
        asset_id=asset_id, to_version=to_version, is_initial=is_initial,
        mask_fingerprint=fp.get("raw"),
        mask_fingerprint_dilated=fp.get("dilated"),
        chunks=chunks, removed_chunk_ids=removed,
        bookkeeping_affected_chunk_ids=book,
        # stats 는 Dict[str, float] 다 — None 을 넣으면 검증에서 터진다.
        # 미측정 항목은 키 자체를 뺀다.
        stats={k: v for k, v in {
            "bookkeeping_chunks": len(book),
            "regenerated_chunks": len(chunks),
            "total_chunks": len(ent),
            "repaint_ms": round(t.get("repaint", 0.0), 1),
            "decode_ms": round(t.get("decode", 0.0), 1),
            "chunk_ms": round(t.get("chunk", 0.0), 1),
            "snapped_vertices": r.get("snapped", 0),
            # 편집이 실제로 건드린 latent 행 수. 이게 없어서 "마스크가 정말 달랐나"를
            # 사후에 못 답했고, 출력 latent 를 base 와 비교해 역산해야 했다.
            "masked_rows": r.get("masked_rows"),
            # 승계 drift (3.19.x) — 승계 **전** 신선 바이트로 잰 값이다.
            # 임계는 두지 않는다. 어느 선부터 승계가 위험한지 아직 모른다.
            **(r.get("drift") or {}),
            "seam_median_cells": seam.get("cells_median"),
            "seam_max_cells": seam.get("cells_max"),
            "voxels": r.get("voxels"),
        }.items() if v is not None},
    )
    return json.loads(resp.model_dump_json())
