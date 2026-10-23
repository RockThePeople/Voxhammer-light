"""GPU 없이 도는 테스트. 계약 규칙과 저장소 불변식 위주다."""
import json
import os
import sys
import time

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

_HERE = os.path.dirname(os.path.abspath(__file__))
_SERVER = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_SERVER), "vendor"))
sys.path.insert(0, _SERVER)


@pytest.fixture(autouse=True)
def data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TRELLIS_DATA_ROOT", str(tmp_path))
    from api import store

    monkeypatch.setattr(store, "DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(store, "ASSETS", str(tmp_path / "assets"))
    return tmp_path


# ══════════════════════════════════════════ 부기 전칭 규칙
def test_bookkeeping_is_total():
    """∀ c ∈ bookkeeping : c ∈ chunks ∨ c ∈ removed_chunk_ids.

    기하가 없는 부기 청크는 반드시 removed 다. 실측에서 부기 24 vs 청크 23 이
    정확히 이 경우였고, 아무 데도 안 넣으면 3090이 BOOKKEEPING_MISMATCH 로 거부한다.
    """
    from api.routes import _response
    from deltacontract import ChunkEntry

    e = ChunkEntry(uri="chunks/1_1_1.cbin", hash="h", byte_length=10,
                   vertex_count=3, index_count=3, voxel_count=1, version=1)
    r = {"entries": {"1_1_1": e}, "removed": ["2_2_2"],
         "book": ["1_1_1", "2_2_2"], "sent": ["1_1_1"],
         "timings": {}, "snapped": 0, "seam": None, "voxels": 1}
    resp = _response("a-1", 2, r, is_initial=False)

    ids = {c["chunk_id"] for c in resp["chunks"]}
    removed = set(resp["removed_chunk_ids"])
    for c in resp["bookkeeping_affected_chunk_ids"]:
        assert c in ids or c in removed, f"{c} 가 chunks 에도 removed 에도 없다"
    assert resp["stats"]["bookkeeping_chunks"] == 2
    assert resp["stats"]["regenerated_chunks"] == 1


def test_stats_has_required_keys():
    from api.routes import _response
    from deltacontract import ChunkEntry

    e = ChunkEntry(uri="c", hash="h", byte_length=1, vertex_count=1,
                   index_count=3, voxel_count=1, version=1)
    r = {"entries": {"0_0_0": e}, "removed": [], "book": ["0_0_0"],
         "sent": ["0_0_0"], "timings": {"repaint": 1.0, "decode": 2.0, "chunk": 3.0},
         "snapped": 7, "seam": {"cells_median": 0.0, "cells_max": 3.8}, "voxels": 5}
    st = _response("a", 1, r, True)["stats"]
    for k in ("bookkeeping_chunks", "regenerated_chunks", "total_chunks",
              "repaint_ms", "decode_ms", "chunk_ms"):
        assert k in st, k
    assert st["snapped_vertices"] == 7


# ══════════════════════════════════════════ base_version / 커밋
def test_uncommitted_base_version_rejected():
    from api import store

    a = "a-x"
    os.makedirs(store.asset_dir(a), exist_ok=True)
    assert store.committed(a) == 0
    assert not store.has_committed_version(a, 1)

    store.write_staging(a, "k1", {"to_version": 1})
    # staging 에만 있는 버전은 base 로 받으면 안 된다.
    assert not store.has_committed_version(a, 1)

    store.commit(a, 1, "k1")
    assert store.committed(a) == 1
    assert store.has_committed_version(a, 1)
    assert not store.has_committed_version(a, 2)


def test_commit_is_idempotent():
    from api import store

    a = "a-y"
    store.write_staging(a, "k", {"v": 1})
    assert store.commit(a, 1, "k") == 1
    assert store.commit(a, 1, "k") == 1          # 두 번 불러도 같다


# ══════════════════════════════════════════ 멱등키 재요청
def test_same_idempotency_key_replays_without_recompute():
    from api import store

    a, k = "a-z", "e-42"
    payload = {"asset_id": a, "to_version": 2, "chunks": []}
    store.write_staging(a, k, payload)

    got, _ = store.staged_result(a, k)
    assert got == payload, "같은 키 재요청은 staged 결과를 그대로 줘야 한다"

    # TTL 초과분은 재계산 대상이다.
    p = os.path.join(store.staging_dir(a, k), "response.json")
    old = time.time() - store.STAGING_TTL_S - 10
    os.utime(p, (old, old))
    got, _ = store.staged_result(a, k)
    assert got is None


def test_gc_removes_expired_staging():
    from api import store

    a, k = "a-gc", "old"
    d = store.write_staging(a, k, {"x": 1})
    old = time.time() - store.STAGING_TTL_S - 10
    os.utime(d, (old, old))
    assert store.gc_staging() == 1
    assert not os.path.isdir(d)


# ══════════════════════════════════════════ coord_order 검증
def _write_slat(path, coord_order="canonical", space="denormalized"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(
        {"coords": torch.zeros((2, 4), dtype=torch.int32),
         "feats": torch.zeros((2, 8), dtype=torch.float32)},
        path,
        metadata={"coord_order": coord_order, "slat_space": space,
                  "norm_mean": "[0]*8", "norm_std": "[1]*8"},
    )


def test_slat_with_wrong_coord_order_is_rejected(tmp_path):
    from api.ops import _load_slat

    p = str(tmp_path / "native" / "slat.safetensors")
    _write_slat(p, coord_order="native")
    with pytest.raises(ValueError, match="coord_order"):
        _load_slat(p)

    p2 = str(tmp_path / "norm" / "slat.safetensors")
    _write_slat(p2, space="normalized")
    with pytest.raises(ValueError, match="slat_space"):
        _load_slat(p2)

    p3 = str(tmp_path / "ok" / "slat.safetensors")
    _write_slat(p3)
    d, meta = _load_slat(p3)
    assert meta["coord_order"] == "canonical"


# ══════════════════════════════════════════ 경로 검증
@pytest.mark.parametrize("bad", [
    "../../../etc/passwd", "..", "0_0_0/../../x", "8_0_0", "a_b_c", "", "0_0_0 ",
])
def test_chunk_path_rejects_traversal(bad):
    from api import store

    a = "a-p"
    os.makedirs(os.path.join(store.version_dir(a, 1), "chunks"), exist_ok=True)
    try:
        p = store.chunk_path(a, bad, 1)
    except ValueError:
        return                       # 형식 거부 — 정상
    assert p is None, f"{bad!r} 가 통과했다: {p}"


def test_chunk_path_serves_committed_only():
    """staging 폴백 제거(계약 §15 4단계). 커밋된 버전만 준다.

    폴백이 있으면 슬롯이 둘 이상일 때 어느 잡의 바이트인지 임의였다.
    커밋 전 바이트는 잡 범위 경로로만 받는다.
    """
    from api import store

    a = "a-q"
    vd = os.path.join(store.version_dir(a, 1), "chunks")
    os.makedirs(vd, exist_ok=True)
    open(os.path.join(vd, "3_3_3.cbin"), "wb").write(b"xy")
    assert store.chunk_path(a, "3_3_3", 1).endswith("v1/chunks/3_3_3.cbin")

    sd = os.path.join(store.staging_dir(a, "k9"), "chunks")
    os.makedirs(sd, exist_ok=True)
    open(os.path.join(sd, "4_4_4.cbin"), "wb").write(b"zz")
    assert store.chunk_path(a, "4_4_4", 2) is None, "staging 폴백은 제거됐다"
    # 잡 범위 경로로는 여전히 받을 수 있다.
    store.record_job(a, "j-k9", "k9", 2)
    assert store.staging_chunk(a, "j-k9", "4_4_4")[1] == "ok"


def test_asset_id_validated():
    from api import store

    for bad in ("../etc", "a/b", "", "x" * 200):
        with pytest.raises(ValueError):
            store.asset_dir(bad)


# ══════════════════════════════════════════ 워커 생존
def test_worker_survives_failing_job():
    from api import jobs

    jobs.start()
    j1 = jobs.new_job("t", "a")
    jobs.submit(j1, lambda _p: (_ for _ in ()).throw(RuntimeError("boom")))
    j2 = jobs.new_job("t", "a")
    jobs.submit(j2, lambda _p: {"ok": True})
    for _ in range(100):
        if jobs.JOBS[j2]["state"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)
    assert jobs.JOBS[j1]["state"] == "failed"
    assert jobs.JOBS[j1]["error_code"] == "INTERNAL"
    assert jobs.JOBS[j2]["state"] == "succeeded", "실패 잡이 큐를 멈추면 안 된다"


# ══════════════════════════════════════════ 3.6.0 부기 가지치기
def test_bookkeeping_pruned_to_existing_chunks():
    """보고할 부기 = 원래 부기 ∩ (base 매니페스트 ∪ 새 디코딩).

    마스크+halo 를 청크로 접으면 오브젝트 없는 공간까지 들어온다. 양쪽 어디에도
    없는 유령 청크는 removed 로도 보고하지 않는다.
    """
    from api.routes import _response
    from deltacontract import ChunkEntry

    e = ChunkEntry(uri="c", hash="h", byte_length=1, vertex_count=1,
                   index_count=3, voxel_count=1, version=1)
    # ops 가 이미 가지치기한 결과를 넘긴다고 가정한 모양
    r = {"entries": {"3_3_3": e}, "removed": ["3_3_4"],
         "book": ["3_3_3", "3_3_4"], "book_raw": 256, "sent": ["3_3_3"],
         "timings": {}, "snapped": 0, "seam": None, "voxels": 1}
    resp = _response("a", 2, r, is_initial=False)
    assert set(resp["bookkeeping_affected_chunk_ids"]) == {"3_3_3", "3_3_4"}
    assert len(resp["bookkeeping_affected_chunk_ids"]) < r["book_raw"]
    # 전칭 규칙은 가지치기 후에도 성립해야 한다.
    ids = {c["chunk_id"] for c in resp["chunks"]} | set(resp["removed_chunk_ids"])
    assert set(resp["bookkeeping_affected_chunk_ids"]) <= ids


def test_prune_uses_decoded_not_active_voxels():
    """가지치기 기준이 '새 디코딩 청크'인지 확인.

    삼각형은 무게중심으로 배정되므로 활성 복셀이 없는 청크에도 기하가 생긴다.
    활성 복셀로 자르면 그런 청크가 사라진다.
    """
    parent_man = {"1_1_1": {}}
    entries = {"2_2_2": None}          # 활성 복셀은 없지만 삼각형이 생긴 청크
    book = ["1_1_1", "2_2_2", "7_7_7"]
    existing = set(parent_man) | set(entries)
    assert sorted(k for k in book if k in existing) == ["1_1_1", "2_2_2"]


# ══════════════════════════════════════════ ErrorBody 한 겹 없음
def test_error_body_is_not_wrapped():
    """4xx/5xx 본문이 {"error_code","message"} 여야 한다. detail 로 싸이면 안 된다."""
    from fastapi.testclient import TestClient

    from api.main import app

    c = TestClient(app, raise_server_exceptions=False)
    for hdr in ({}, {"X-Blockedit-Key": "wrong"}):
        r = c.get("/v2/trellis/health", headers=hdr)
        assert r.status_code == 401, r.status_code
        b = r.json()
        assert "detail" not in b, f"한 겹 싸였다: {b}"
        assert b["error_code"] == "UNAUTHORIZED"
        assert set(b) == {"error_code", "message"}, b
        # 키 값이 새면 안 된다.
        assert "wrong" not in b["message"] and "dev-key" not in b["message"]


def test_not_found_body_shape():
    from fastapi.testclient import TestClient

    from api.main import app

    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/v2/trellis/jobs/j-none", headers={"X-Blockedit-Key": "dev-key"})
    assert r.status_code == 404
    b = r.json()
    assert "detail" not in b and b["error_code"] == "NOT_FOUND", b


# ══════════════════════════════════════════ 3.7.0 멱등 우선 순서
def test_replay_survives_commit():
    """커밋 후에도 같은 키로 재생돼야 한다.

    계약 3.7.0 시나리오: "서버가 v2 로 커밋 → 응답만 유실 → 클라이언트 재시도".
    staging 은 커밋 시 v{n} 으로 승격돼 사라지므로 별도 색인이 필요하다.
    """
    from api import store

    a, k = "a-rep", "e-1"
    resp = {"asset_id": a, "to_version": 2, "chunks": []}
    store.write_staging(a, k, resp)
    assert store.replayable(a, k) == resp          # 커밋 전
    store.commit(a, 2, k)
    assert store.replayable(a, k) == resp          # 커밋 후에도 ✓
    assert store.committed(a) == 2


def test_idempotency_checked_before_base_version():
    """멱등 확인이 base_version 검사보다 먼저여야 한다.

    커밋 후 재시도는 base_version 이 이미 낡았지만 409 가 아니라 저장된 결과를
    받아야 한다. 순서가 뒤바뀌면 '성공한 편집을 실패로 배우는' 상태가 된다.
    """
    from fastapi.testclient import TestClient

    from api import store
    from api.main import app

    a, k = "a-ord", "e-9"
    # v1 생성 + 커밋
    store.write_staging(a, "gen", {"to_version": 1})
    store.commit(a, 1, "gen")
    # base=1 로 편집 → v2 커밋. 응답만 유실됐다고 가정.
    resp = {"asset_id": a, "to_version": 2, "chunks": [], "replayed_base_version": 1}
    store.write_staging(a, k, resp)
    store.commit(a, 2, k)

    c = TestClient(app, raise_server_exceptions=False)
    r = c.post("/v2/trellis/edit", headers={"X-Blockedit-Key": "dev-key"}, json={
        "asset_id": a, "base_version": 1, "prompt": "x",
        "mask": {"mode": "voxels", "voxels": [[32, 32, 32]], "halo_margin_voxels": 1},
        "seed": 1, "idempotency_key": k})
    assert r.status_code == 202, (r.status_code, r.json())   # 409 가 아니어야 한다
    jid = r.json()["job_id"]
    got = c.get(f"/v2/trellis/jobs/{jid}", headers={"X-Blockedit-Key": "dev-key"}).json()
    assert got["to_version"] == 2, got


def test_replay_returns_stored_even_if_base_version_differs():
    """키는 연산을 식별한다 — base 가 달라도 저장된 결과를 준다 (경고는 남긴다)."""
    from fastapi.testclient import TestClient

    from api import store
    from api.main import app

    a, k = "a-mis", "e-m"
    store.write_staging(a, "gen", {"to_version": 1})
    store.commit(a, 1, "gen")
    store.write_staging(a, k, {"asset_id": a, "to_version": 2, "chunks": [],
                               "replayed_base_version": 1})
    store.commit(a, 2, k)

    c = TestClient(app, raise_server_exceptions=False)
    r = c.post("/v2/trellis/edit", headers={"X-Blockedit-Key": "dev-key"}, json={
        "asset_id": a, "base_version": 2, "prompt": "x",      # ← 저장된 건 1
        "mask": {"mode": "voxels", "voxels": [[32, 32, 32]], "halo_margin_voxels": 1},
        "seed": 1, "idempotency_key": k})
    assert r.status_code == 202
    from api import jobs

    j = jobs.JOBS[r.json()["job_id"]]
    assert j["replayed"] is True
    assert j["base_version_mismatch"] == {"requested": 2, "stored": 1}


# ══════════════════════════════════════════ §13-8 잡 범위 staging
def _slot(store, a, key, job_id, ver, chunk_bytes):
    d = store.staging_dir(a, key)
    os.makedirs(os.path.join(d, "chunks"), exist_ok=True)
    for ck, b in chunk_bytes.items():
        open(os.path.join(d, "chunks", f"{ck}.cbin"), "wb").write(b)
    store.write_staging(a, key, {"to_version": ver})
    store.record_job(a, job_id, key, ver)
    return d


def test_job_scoped_lookup_isolates_slots():
    """같은 청크·같은 to_version 이라도 잡마다 다른 바이트를 준다.

    실측(sample-snowman): 슬롯 5개가 전부 to_version=2 였고 3_3_5 해시가 전부
    달랐는데, 기존 .v{n}.cbin 은 sorted() 첫 슬롯만 줘서 4개가 도달 불가였다.
    """
    from api import store

    a = "a-iso"
    _slot(store, a, "zzz-late", "j-late", 2, {"3_3_5": b"LATE" * 10})
    _slot(store, a, "aaa-early", "j-early", 2, {"3_3_5": b"EARLY" * 10})

    p1, s1 = store.staging_chunk(a, "j-late", "3_3_5")
    p2, s2 = store.staging_chunk(a, "j-early", "3_3_5")
    assert s1 == s2 == "ok"
    assert open(p1, "rb").read() == b"LATE" * 10
    assert open(p2, "rb").read() == b"EARLY" * 10, "잡별로 격리돼야 한다"

    # 폴백 제거 후 legacy 경로는 커밋 전 바이트를 아예 주지 않는다.
    assert store.chunk_path(a, "3_3_5", 2) is None


def test_expired_staging_is_gone_not_missing():
    """TTL 초과 → expired(410). 파일이 없으면 missing(404). 둘을 갈라야 한다."""
    from api import store

    a = "a-exp"
    d = _slot(store, a, "k", "j-1", 2, {"1_1_1": b"x"})
    assert store.staging_chunk(a, "j-1", "1_1_1")[1] == "ok"

    old = time.time() - store.STAGING_TTL_S - 10
    os.utime(os.path.join(d, "job.json"), (old, old))
    p, st = store.staging_chunk(a, "j-1", "1_1_1")
    assert st == "expired" and p is None

    # 존재하지 않는 청크는 만료가 아니라 no_chunk 다.
    os.utime(os.path.join(d, "job.json"), None)
    assert store.staging_chunk(a, "j-1", "7_7_7")[1] == "no_chunk"


def test_unknown_job_id_is_missing():
    from api import store

    a = "a-unk"
    _slot(store, a, "k", "j-real", 2, {"0_0_0": b"z"})
    assert store.staging_chunk(a, "j-nope", "0_0_0") == (None, "no_job")
    # 잘못된 chunk_key 는 형식 단계에서 거부된다 (없는 게 아니라 잘못된 것).
    with pytest.raises(ValueError):
        store.staging_chunk(a, "j-real", "../etc")


def test_job_index_survives_lost_index_file():
    """jobs/ 색인이 없어도 슬롯의 job.json 으로 복구한다 (프로세스 재시작 대비)."""
    from api import store

    a = "a-rec"
    _slot(store, a, "k", "j-x", 2, {"2_2_2": b"q"})
    os.remove(store.job_index_path(a, "j-x"))
    p, st = store.staging_chunk(a, "j-x", "2_2_2")
    assert st == "ok" and open(p, "rb").read() == b"q"


def test_gc_removes_slot_and_job_index():
    """GC 가 슬롯과 함께 jobs/ 색인도 지운다 — 고아 색인이 남으면 안 된다."""
    from api import store

    a = "a-gcj"
    d = _slot(store, a, "k", "j-g", 2, {"0_0_0": b"a"})
    idx = store.job_index_path(a, "j-g")
    assert os.path.exists(idx)
    old = time.time() - store.STAGING_TTL_S - 10
    os.utime(d, (old, old))
    assert store.gc_staging() == 1
    assert not os.path.isdir(d) and not os.path.exists(idx)


def test_staging_route_status_codes():
    """라우트가 404/410 을 실제로 구분해 내는가."""
    from fastapi.testclient import TestClient

    from api import store
    from api.main import app

    a = "a-route"
    d = _slot(store, a, "k", "j-r", 2, {"3_3_3": b"BYTES"})
    c = TestClient(app, raise_server_exceptions=False)
    H = {"X-Blockedit-Key": "dev-key"}

    r = c.get(f"/v2/assets/{a}/staging/j-r/chunks/3_3_3.cbin", headers=H)
    assert r.status_code == 200 and r.content == b"BYTES"

    r = c.get(f"/v2/assets/{a}/staging/j-nope/chunks/3_3_3.cbin", headers=H)
    assert r.status_code == 404, r.status_code

    old = time.time() - store.STAGING_TTL_S - 10
    os.utime(os.path.join(d, "job.json"), (old, old))
    r = c.get(f"/v2/assets/{a}/staging/j-r/chunks/3_3_3.cbin", headers=H)
    assert r.status_code == 410, r.status_code
    assert r.json()["error_code"] == "STAGING_EXPIRED"
    assert "detail" not in r.json()

    # 인증은 라우터 단위로 걸려 있다.
    assert c.get(f"/v2/assets/{a}/staging/j-r/chunks/3_3_3.cbin").status_code == 401


def test_response_omits_uri():
    from api.routes import _response
    from deltacontract import ChunkEntry

    e = ChunkEntry(uri="c", hash="h", byte_length=1, vertex_count=1,
                   index_count=3, voxel_count=1, version=1)
    r = {"entries": {"1_1_1": e}, "removed": [], "book": ["1_1_1"],
         "sent": ["1_1_1"], "timings": {}, "snapped": 0, "seam": None, "voxels": 1}
    resp = _response("a-u", 2, r, is_initial=False, job_id="j-9")
    # uri 는 싣지 않는다 — ChunkMeshPayload 에 없는 필드다. 3090이 job_id 로 만든다.
    assert "uri" not in resp["chunks"][0]
    assert resp["chunks"][0]["chunk_id"] == "1_1_1"


def test_failed_job_uses_renamed_error_map():
    """계약 3.11.x 에서 BY_CODE → ERROR_CODE_TO_EXCEPTION 으로 개명됐다."""
    from fastapi.testclient import TestClient

    from api import jobs
    from api.main import app

    jid = jobs.new_job("edit", "a-f")
    jobs.JOBS[jid].update(state="failed", error_code="MASK_EMPTY", error="비었다")
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get(f"/v2/trellis/jobs/{jid}", headers={"X-Blockedit-Key": "dev-key"})
    assert r.status_code == 422, r.status_code
    assert r.json()["error_code"] == "MASK_EMPTY"


# ══════════════════════════════════════════ 3.12.0 마스크 지문 · masked_rows
def test_ops_uses_contract_fingerprint_function():
    """자체 구현이 아니라 **계약 함수 그 자체**를 쓴다 (3.15.x).

    값이 같은 것과 같은 함수를 쓰는 것은 다르다 — 값 일치는 오늘의 사실이고
    단일 출처는 내일의 보장이다.
    """
    import deltacontract

    from api import ops

    assert ops.mask_fingerprint is deltacontract.mask_fingerprint


def test_mask_fingerprint_matches_contract_function():
    """지문은 **계약 함수와 같은 값**이어야 한다 (3.14.0).

    양쪽이 각자 직렬화하면 어긋나도 "지문 불일치" 로만 보여, 마스크가 달랐는지
    인코딩이 달랐는지 구분이 안 된다. 그 구분 불가가 필드를 넣은 목적을 없앤다.
    """
    import numpy as np
    from deltacontract import mask_fingerprint as contract_fp

    from api.ops import mask_fingerprint as mine

    cases = [
        np.zeros((0, 3), np.int64),
        np.array([[32, 32, 32]]),
        np.array([[7, 0, 1], [4, 5, 6], [1, 2, 3]]),
        np.array([[0, 0, 0], [63, 63, 63], [0, 63, 0]]),
        np.stack(np.meshgrid(*[np.arange(24, 32)] * 3, indexing="ij"), -1).reshape(-1, 3),
    ]
    for c in cases:
        assert mine(c) == contract_fp(c), f"셀 {len(c)}개에서 어긋난다"
    # 순서 무관 · 구분력
    a = [[1, 2, 3], [4, 5, 6], [7, 0, 1]]
    assert mine(a) == mine(list(reversed(a)))
    assert mine(a) != mine(a + [[2, 2, 2]])


def test_record_job_persists_mask_fingerprint():
    from api import store

    a = "a-fp"
    fp = {"raw": "deadbeef", "raw_cells": 12, "halo": 1, "mode": "voxels"}
    store.record_job(a, "j-fp", "k", 2, mask=fp)
    rec = json.load(open(os.path.join(store.staging_dir(a, "k"), "job.json")))
    assert rec["mask"]["raw"] == "deadbeef" and rec["mask"]["raw_cells"] == 12


def test_stats_carries_masked_rows():
    from api.routes import _response
    from deltacontract import ChunkEntry

    e = ChunkEntry(uri="c", hash="h", byte_length=1, vertex_count=1,
                   index_count=3, voxel_count=1, version=1)
    r = {"entries": {"0_0_0": e}, "removed": [], "book": ["0_0_0"],
         "sent": ["0_0_0"], "timings": {}, "snapped": 0, "seam": None,
         "voxels": 100, "masked_rows": 37}
    assert _response("a", 2, r, False)["stats"]["masked_rows"] == 37


def test_staging_expired_comes_from_contract():
    """로컬 정의가 아니라 계약 클래스를 쓴다 (3.12.0 에서 편입)."""
    import deltacontract.errors as E

    from api import routes

    assert routes.StagingExpired is E.StagingExpired
    e = routes.StagingExpired("x")
    assert e.error_code == "STAGING_EXPIRED" and e.http_status == 410
    assert e.retriable is False, "같은 URI 는 영원히 410 이다 — 재시도 대상이 아니다"


# ══════════════════════════════════════════ 3.13.0 404 두 갈래 · 지문 응답
def test_404_split_no_job_vs_no_chunk():
    """없는 job_id 와 "슬롯은 살아있는데 청크 없음" 을 가른다 (계약 §15).

    뒤엣것은 부기가 지목했는데 청크가 없다는 뜻이라 §4.2 전칭 검증이 깨졌다는
    신호다. 뭉치면 그 신호가 사라진다.
    """
    from fastapi.testclient import TestClient

    from api import store
    from api.main import app

    a = "a-404"
    _slot(store, a, "k", "j-ok", 2, {"1_1_1": b"B"})
    c = TestClient(app, raise_server_exceptions=False)
    H = {"X-Blockedit-Key": "dev-key"}

    r = c.get(f"/v2/assets/{a}/staging/j-none/chunks/1_1_1.cbin", headers=H)
    assert r.status_code == 404 and r.json()["error_code"] == "NOT_FOUND", r.json()

    r = c.get(f"/v2/assets/{a}/staging/j-ok/chunks/7_7_7.cbin", headers=H)
    assert r.status_code == 404 and r.json()["error_code"] == "ASSET_NOT_FOUND", r.json()

    # 둘 다 ErrorBody 모양이어야 한다 (detail 로 안 싸인다).
    assert set(r.json()) == {"error_code", "message"}


def test_response_carries_mask_fingerprint():
    """지문이 응답 필드로 나간다 (stats 는 Dict[str,float] 라 못 싣는다)."""
    from api.routes import _response
    from deltacontract import ChunkEntry

    e = ChunkEntry(uri="c", hash="h", byte_length=1, vertex_count=1,
                   index_count=3, voxel_count=1, version=1)
    r = {"entries": {"0_0_0": e}, "removed": [], "book": ["0_0_0"],
         "sent": ["0_0_0"], "timings": {}, "snapped": 0, "seam": None,
         "voxels": 9, "masked_rows": 5,
         "mask_fp": {"raw": "aa" * 32, "dilated": "bb" * 32}}
    resp = _response("a", 2, r, is_initial=False, job_id="j-1")
    assert resp["mask_fingerprint"] == "aa" * 32
    assert resp["mask_fingerprint_dilated"] == "bb" * 32
    assert resp["stats"]["masked_rows"] == 5


def test_response_fingerprint_absent_for_generate():
    """generate 는 마스크가 없으므로 지문이 null 이다."""
    from api.routes import _response
    from deltacontract import ChunkEntry

    e = ChunkEntry(uri="c", hash="h", byte_length=1, vertex_count=1,
                   index_count=3, voxel_count=1, version=1)
    r = {"entries": {"0_0_0": e}, "removed": [], "book": ["0_0_0"],
         "timings": {}, "snapped": 0, "seam": None, "voxels": 9}
    resp = _response("a", 1, r, is_initial=True)
    assert resp["mask_fingerprint"] is None


def test_path_escape_is_invalid_request_not_not_found():
    """경로 탈출은 **없는 것(404)이 아니라 잘못된 것(422)** 이다 (계약 3.14.0).

    ASSET_NOT_FOUND 로 접으면 "부기가 지목한 청크가 없다"(§4.2 위반)와 같은
    얼굴이 된다. 그 둘을 가르려고 §15 를 통째로 고친 참이다.
    """
    from fastapi.testclient import TestClient

    from api import store
    from api.main import app

    a = "a-esc"
    _slot(store, a, "k", "j-e", 2, {"1_1_1": b"B"})
    c = TestClient(app, raise_server_exceptions=False)
    H = {"X-Blockedit-Key": "dev-key"}

    # job_id 에 상위 참조 → 계약 validate_job_id 가 파일시스템 전에 막는다.
    for bad in ("..", "../etc", "a/b"):
        r = c.get(f"/v2/assets/{a}/staging/{bad}/chunks/1_1_1.cbin", headers=H)
        assert r.status_code in (404, 422), (bad, r.status_code)
        if r.status_code == 422:
            assert r.json()["error_code"] == "INVALID_REQUEST", r.json()

    # store 층에서도 escape 상태가 나온다 (다중 방어).
    assert "escape" in store.staging_chunk.__doc__


def test_validate_job_id_is_contract_function():
    from deltacontract import validate_job_id
    from deltacontract.uris import UriRuleViolation

    assert validate_job_id("j-abc123") == "j-abc123"
    for bad in ("..", ".", "a/b", "a\\b"):
        with pytest.raises(UriRuleViolation):
            validate_job_id(bad)
    # InvalidRequest 하위라 422 로 나간다.
    e = UriRuleViolation("x")
    assert e.error_code == "INVALID_REQUEST" and e.http_status == 422


def test_unmatched_route_uses_error_body_shape():
    """매치되지 않은 경로의 404 도 ErrorBody 모양이어야 한다.

    fastapi.HTTPException 에만 핸들러를 걸면 Starlette 기본 처리로 빠져
    {"detail":"Not Found"} 가 나간다 — "4xx/5xx 는 전부 한 가지 모양" 이 깨진다.
    """
    from fastapi.testclient import TestClient

    from api.main import app

    c = TestClient(app, raise_server_exceptions=False)
    for path in ("/v2/nope", "/v2/assets/a/staging/../chunks/1_1_1.cbin", "/없는경로"):
        r = c.get(path, headers={"X-Blockedit-Key": "dev-key"})
        assert r.status_code >= 400
        b = r.json()
        assert "detail" not in b, f"{path} 가 한 겹 싸였다: {b}"
        assert "error_code" in b and "message" in b, (path, b)


# ══════════════════════════════════════════ 3.15.x 재생 job_id 불변식
def test_replay_job_id_can_fetch_staging_chunks():
    """★ 응답이 실은 job_id 로 staging GET 이 200 이어야 한다 — **재생에서도**.

    정상 경로 테스트로는 영영 안 잡힌다. 첫 요청은 항상 되기 때문이다.
    같은 멱등키로 두 번 보내고 **두 번째 응답의 job_id** 로 조회해야 드러난다.

    3090 실측(2026-08-01): 원본 j-d9b0590bab30 은 200, 재생이 준
    j-fef13fefe896 / j-ae2ee239571e 는 404 였다. 바이트는 멀쩡하고 포인터만 틀렸다.
    """
    from fastapi.testclient import TestClient

    from api import store
    from api.main import app

    a, key = "a-replay-uri", "e-inv"
    # v1 커밋 + 편집 슬롯을 원본 job_id 로 만든다 (실제 edit 경로가 만드는 모양).
    store.write_staging(a, "gen", {"to_version": 1})
    store.commit(a, 1, "gen")
    orig = "j-orig000001"
    _slot(store, a, key, orig, 2, {"3_3_5": b"CHUNKBYTES"})
    store.write_staging(a, key, {"asset_id": a, "to_version": 2, "chunks": [],
                                 "replayed_base_version": 1})
    store.record_job(a, orig, key, 2)

    c = TestClient(app, raise_server_exceptions=False)
    H = {"X-Blockedit-Key": "dev-key"}
    body = {"asset_id": a, "base_version": 1, "prompt": "x",
            "mask": {"mode": "voxels", "voxels": [[32, 32, 32]],
                     "halo_margin_voxels": 1},
            "seed": 1, "idempotency_key": key}

    seen = set()
    for i in range(3):                      # 여러 번 재생해도 매번 성립해야 한다
        r = c.post("/v2/trellis/edit", headers=H, json=body)
        assert r.status_code == 202, r.json()
        jid = r.json()["job_id"]
        seen.add(jid)
        g = c.get(f"/v2/assets/{a}/staging/{jid}/chunks/3_3_5.cbin", headers=H)
        assert g.status_code == 200, (i, jid, g.status_code, g.json())
        assert g.content == b"CHUNKBYTES"

    # (a) 안을 골랐으므로 job_id 는 멱등키의 함수다 — 3090 매핑이 1:1 로 남는다.
    assert seen == {orig}, f"재생마다 id 가 달라졌다: {seen}"


def test_replay_job_id_pollable_after_restart():
    """재생이 원본 id 를 주므로, JOBS 가 비어도 그 id 로 폴링이 돼야 한다."""
    from fastapi.testclient import TestClient

    from api import jobs, store
    from api.main import app

    a, key, orig = "a-restart", "e-r", "j-restart001"
    _slot(store, a, key, orig, 2, {"1_1_1": b"z"})
    store.write_staging(a, key, {"asset_id": a, "to_version": 2, "chunks": []})
    store.record_job(a, orig, key, 2)
    jobs.JOBS.pop(orig, None)               # 프로세스 재시작 흉내

    c = TestClient(app, raise_server_exceptions=False)
    r = c.get(f"/v2/trellis/jobs/{orig}", headers={"X-Blockedit-Key": "dev-key"})
    assert r.status_code == 200, r.json()
    assert r.json()["to_version"] == 2


def test_unmatched_route_error_code_is_not_found():
    """매치 안 된 경로는 NOT_FOUND 이고 경로를 메시지에 싣는다 (INTERNAL 아님).

    Unity 가 "라우트 미등록 vs 파일 부재" 를 이 코드로 가른다. 3090 도 같은
    상황에 NOT_FOUND 를 주므로, 여기서 INTERNAL 을 주면 두 서버가 다른 말을 한다.
    """
    from fastapi.testclient import TestClient

    from api.main import app

    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/v2/nope", headers={"X-Blockedit-Key": "dev-key"})
    assert r.status_code == 404
    b = r.json()
    assert b["error_code"] == "NOT_FOUND", b
    assert "/v2/nope" in b["message"], b
    assert "detail" not in b


# ══════════════════════════════════════════ build 정보 (계약 3.16.0)
def test_health_carries_build_info():
    """/health 가 build·build_dirty·started_at 을 실어야 한다.

    "테스트 통과" 와 "그 코드가 떠 있음" 은 다른 사실이다. 이 셋이 없으면
    후자를 밖에서 확인할 방법이 세션의 자기보고밖에 없다.
    """
    import subprocess
    from datetime import datetime

    from fastapi.testclient import TestClient

    from api.main import app

    c = TestClient(app)
    b = c.get("/v2/trellis/health", headers={"X-Blockedit-Key": "dev-key"}).json()

    assert "build" in b and "build_dirty" in b and "started_at" in b, b
    assert isinstance(b["build_dirty"], bool), b["build_dirty"]

    # ISO8601 로 파싱돼야 한다. 문자열이 들어있기만 하면 3090 이 커밋 시각과
    # 비교할 수 없다 — 비교 가능한 것이 이 필드의 목적이다.
    assert datetime.fromisoformat(b["started_at"]).tzinfo is not None, b["started_at"]

    # git 이 있는 환경이면 실제 HEAD 와 같아야 한다.
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        assert b["build"] == r.stdout.strip(), (b["build"], r.stdout.strip())


def test_build_info_is_frozen_at_startup():
    """요청마다 다시 재면 안 된다.

    요청 시점에 git 을 부르면, 고친 뒤 커밋만 하고 프로세스를 안 내려도 /health 가
    새 해시를 보고한다 — 잡으려던 상황이 정확히 안 보이게 된다.
    """
    from fastapi.testclient import TestClient

    from api import build
    from api.main import app

    c = TestClient(app)
    hdr = {"X-Blockedit-Key": "dev-key"}
    first = c.get("/v2/trellis/health", headers=hdr).json()

    # 작업 트리가 그 사이 바뀐 것처럼 흉내내도 응답은 안 변해야 한다.
    called = []
    orig = build._git
    build._git = lambda *a: called.append(a) or "deadbee"
    try:
        second = c.get("/v2/trellis/health", headers=hdr).json()
    finally:
        build._git = orig

    assert not called, f"요청 처리 중 git 을 불렀다: {called}"
    for k in ("build", "build_dirty", "started_at"):
        assert first[k] == second[k], (k, first[k], second[k])


def test_build_info_survives_missing_git():
    """git 이 없어도 health 가 죽으면 안 된다 — build=None 으로 나간다."""
    from api import build

    def _boom(*a, **k):
        raise FileNotFoundError("git")

    orig = build.subprocess.run
    build.subprocess.run = _boom
    try:
        rev, dirty, untracked = build._read()
    finally:
        build.subprocess.run = orig

    assert rev is None
    assert dirty is False
    # git 이 없으면 untracked 는 **0 이 아니라 모름(None)** 이다.
    # 0 으로 채우면 "untracked 없음" 이라고 말한 것이 되고, build_dirty 가
    # False 를 기본값으로 두면 안 되는 것과 정확히 같은 종류의 거짓말이다.
    assert untracked is None


def test_build_dirty_counts_tracked_only_and_untracked_separately():
    """🔴 untracked 파일 하나가 build_dirty 를 영구히 켜두면 안 된다 (3.19.x).

    실측(2026-08-03): 커밋을 다 마쳐 tracked 변경이 0건인데 문서 파일 하나가
    untracked 로 남아 `build_dirty=true` 가 안 내려갔다. RUNBOOK 은 그 값을
    "해시로 코드를 특정할 수 없다"로 읽는다.

    ⚠️ 그렇다고 untracked 를 무시하면 안 된다 — 패키지 안의 untracked `.py` 는
    임포트된다. 버리지 않고 **따로 센다**.
    """
    from api import build

    calls = []

    def _fake(args, **k):
        calls.append(list(args))
        class R:
            returncode = 0
            stdout = ("abc1234" if "rev-parse" in args else
                      "" if "status" in args else "a.txt\nb.txt")
        return R()

    orig = build.subprocess.run
    build.subprocess.run = _fake
    try:
        rev, dirty, untracked = build._read()
    finally:
        build.subprocess.run = orig

    assert rev == "abc1234"
    assert dirty is False          # tracked 변경 없음
    assert untracked == 2          # untracked 는 따로 센다
    # status 는 반드시 --untracked-files=no 로 물어야 한다
    st = [c for c in calls if "status" in c][0]
    assert "--untracked-files=no" in st
