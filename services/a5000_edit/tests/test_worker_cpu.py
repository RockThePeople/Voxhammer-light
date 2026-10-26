"""04L 워커 CPU 테스트(GPU·모델 없이): 규약 §3.3 거절 사유, 인증, 렌더 파일, 시점 선택 규칙."""
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
pytest.importorskip("fastapi")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TRELLIS_DATA_ROOT", str(tmp_path / "store"))
    monkeypatch.setenv("VHL_RENDER_ROOT", str(tmp_path / "renders"))
    monkeypatch.setenv("TRELLIS_API_KEY", "k")
    for m in [m for m in sys.modules if m.startswith("services.a5000_edit")]:
        del sys.modules[m]
    from fastapi.testclient import TestClient
    from services.a5000_edit import app as A, store
    aid = "vhl-test"
    vd = Path(store.version_dir(aid, 1))
    vd.mkdir(parents=True)
    (Path(store.asset_dir(aid)) / "committed.json").write_text(json.dumps({"latest": 1}))
    return TestClient(A.app), A, aid


def _png(size=(1024, 1024), color=(255, 255, 255)):
    from PIL import Image
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, format="PNG")
    return b.getvalue()


def _record(A, aid, rid="r-test", base=1, created=None, cells=((10, 10, 10),)):
    from deltacontract import mask_fingerprint
    d = A.RENDER_ROOT / rid
    d.mkdir(parents=True, exist_ok=True)
    rb, mb = _png(color=(1, 2, 3)), _png(color=(0, 0, 0))
    (d / "2d_render.png").write_bytes(rb)
    (d / "2d_mask.png").write_bytes(mb)
    rec = {"render_id": rid, "asset_id": aid, "base_version": base, "view": 2, "render_sha256": hashlib.sha256(rb).hexdigest(),
           "mask_sha256": hashlib.sha256(mb).hexdigest(), "created": created or time.time(),
           "mask_fingerprint": mask_fingerprint(np.asarray(cells)), "renderer": {}}
    (d / "record.json").write_text(json.dumps(rec))
    return rec, rb, mb


def _meta(aid, rec, **kw):
    m = {"asset_id": aid, "base_version": 1, "prompt": "p", "seed": 0, "idempotency_key": "key1",
         "mask": {"mode": "voxels", "voxels": [[10, 10, 10]], "halo_margin_voxels": 2},
         "render_id": rec["render_id"], "render_sha256": rec["render_sha256"], "mask_sha256": rec["mask_sha256"]}
    m.update(kw)
    return json.dumps(m)


H = {"X-Blockedit-Key": "k"}


def test_auth_off_by_default_and_enforced_when_on(client, monkeypatch):
    c, A, aid = client
    r = c.post("/v2/trellis/edit", json={})
    assert r.status_code != 401                     # 기본 끔: 키 없이 통과(본문 검사에서 걸린다)
    monkeypatch.setattr(A, "AUTH", True)
    r = c.post("/v2/trellis/edit", json={})
    assert r.status_code == 401 and r.json()["error_code"] == "UNAUTHORIZED"


def test_json_body_rejected_conditions_required(client):
    c, A, aid = client
    r = c.post("/v2/trellis/edit", json={"asset_id": aid}, headers=H)
    assert r.status_code == 422 and r.json()["error_code"] == "CONDITION_IMAGES_REQUIRED"


def test_missing_image_part(client):
    c, A, aid = client
    rec, rb, mb = _record(A, aid)
    r = c.post("/v2/trellis/edit", headers=H, data={"meta": _meta(aid, rec)},
               files={"2d_render": ("a.png", rb, "image/png"), "2d_mask": ("b.png", mb, "image/png")})
    assert r.status_code == 422 and r.json()["error_code"] == "CONDITION_IMAGES_REQUIRED"
    assert r.json()["detail"]["missing"] == "2d_edit"


def test_bad_condition_image(client):
    c, A, aid = client
    rec, rb, mb = _record(A, aid)
    for bad in (b"not a png", _png(size=(512, 512))):
        r = c.post("/v2/trellis/edit", headers=H, data={"meta": _meta(aid, rec)},
                   files={"2d_render": ("a.png", rb, "image/png"), "2d_mask": ("b.png", mb, "image/png"),
                          "2d_edit": ("c.png", bad, "image/png")})
        assert r.status_code == 400 and r.json()["error_code"] == "BAD_CONDITION_IMAGE"


def test_render_unknown_and_expired(client):
    c, A, aid = client
    rec, rb, mb = _record(A, aid, rid="r-old", created=time.time() - 25 * 3600)
    files = {"2d_render": ("a.png", rb, "image/png"), "2d_mask": ("b.png", mb, "image/png"), "2d_edit": ("c.png", _png(), "image/png")}
    for rid in ("r-none", "r-old"):
        r = c.post("/v2/trellis/edit", headers=H, data={"meta": _meta(aid, {**rec, "render_id": rid})}, files=files)
        assert r.status_code == 404 and r.json()["error_code"] == "RENDER_UNKNOWN"
    assert c.get("/v2/trellis/renders/r-old/2d_render.png", headers=H).json()["error_code"] == "RENDER_UNKNOWN"


def test_render_mismatch(client):
    c, A, aid = client
    rec, rb, mb = _record(A, aid)
    r = c.post("/v2/trellis/edit", headers=H, data={"meta": _meta(aid, rec)},
               files={"2d_render": ("a.png", _png(color=(9, 9, 9)), "image/png"), "2d_mask": ("b.png", mb, "image/png"),
                      "2d_edit": ("c.png", _png(), "image/png")})
    assert r.status_code == 409 and r.json()["error_code"] == "RENDER_MISMATCH"
    r = c.post("/v2/trellis/edit", headers=H, data={"meta": _meta(aid, rec, mask_sha256="0" * 64)},
               files={"2d_render": ("a.png", rb, "image/png"), "2d_mask": ("b.png", mb, "image/png"), "2d_edit": ("c.png", _png(), "image/png")})
    assert r.status_code == 409 and r.json()["error_code"] == "RENDER_MISMATCH"


def test_version_conflict(client):
    c, A, aid = client
    rec, rb, mb = _record(A, aid)
    r = c.post("/v2/trellis/edit", headers=H, data={"meta": _meta(aid, rec, base_version=2)},
               files={"2d_render": ("a.png", rb, "image/png"), "2d_mask": ("b.png", mb, "image/png"), "2d_edit": ("c.png", _png(), "image/png")})
    assert r.status_code == 409 and r.json()["error_code"] == "VERSION_CONFLICT"
    r = c.post("/v2/trellis/render", headers=H, json={"asset_id": aid, "base_version": 5, "mask": {"mode": "voxels", "voxels": [[1, 1, 1]]}})
    assert r.status_code == 409 and r.json()["error_code"] == "VERSION_CONFLICT"


def test_render_view_range_and_unknown_route(client):
    c, A, aid = client
    r = c.post("/v2/trellis/render", headers=H, json={"asset_id": aid, "base_version": 1, "view": 7,
                                                     "mask": {"mode": "voxels", "voxels": [[1, 1, 1]]}})
    assert r.status_code == 422 and r.json()["error_code"] == "INVALID_REQUEST"
    assert c.get("/v2/trellis/nope", headers=H).json()["error_code"] == "NOT_FOUND"


def test_choose_view_rule():
    """투영 면적 최대 시점, 동률이면 낮은 번호(규약 §4 결정 4 수정본)."""
    from editcore.render import choose_view, mask_area_per_view
    cells = np.array([[32, 32, 32]])
    a = mask_area_per_view(cells, {"center": [0, 0, 0], "scale": 1.0})
    v, _ = choose_view(cells, {"center": [0, 0, 0], "scale": 1.0})
    assert len(a) == 5 and v == min(i for i in range(5) if a[i] == max(a))
