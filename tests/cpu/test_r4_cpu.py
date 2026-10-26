"""2번 R4 CPU 테스트: mesh 만 디코딩 → .cbin, R11 부기 승계, manifest 일치, 토글 기본 꺼짐."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
REPO = Path(__file__).resolve().parents[2]
if not (REPO / "third_party" / "ai-ar-v3" / "contract" / "python" / "deltacontract").is_dir():
    pytest.skip("ai-ar-v3 contract 가 없다", allow_module_level=True)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "ai-ar-v3" / "contract" / "python"))


@pytest.fixture(scope="module", autouse=True)
def _cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _grid_mesh(n=24, z=0.0, seed=0):
    """NORMALIZED 공간의 물결 평면(여러 청크에 걸친다) + TRELLIS 형식 vertex_attrs(albedo, [0,1] 법선)."""
    g = np.random.default_rng(seed)
    u = np.linspace(-0.45, 0.45, n)
    X, Y = np.meshgrid(u, u, indexing="ij")
    Z = z + 0.05 * np.sin(6 * X) * np.cos(5 * Y)
    v = np.stack([X, Y, Z], -1).reshape(-1, 3).astype(np.float32)
    f = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, i * n + j + 1, (i + 1) * n + j + 1
            f += [[a, b, c], [b, d, c]]
    attrs = np.concatenate([g.random((len(v), 3)), np.tile([0.5, 0.5, 1.0], (len(v), 1))], 1).astype(np.float32)
    return v, np.asarray(f, np.int64), attrs


class _Pipe:
    def __init__(self, v, f, attrs):
        self.calls = []
        self.mesh = SimpleNamespace(vertices=torch.from_numpy(v), faces=torch.from_numpy(f), vertex_attrs=torch.from_numpy(attrs))

    def decode_slat(self, slat, formats):
        self.calls.append(list(formats))
        return {"mesh": [self.mesh]}


def _slat(cells):
    c = np.concatenate([np.zeros((len(cells), 1), np.int32), np.asarray(cells, np.int32)], 1)
    return SimpleNamespace(coords=torch.from_numpy(c))


def test_toggle_default_off(monkeypatch):
    from editcore.models import r4_enabled
    monkeypatch.delenv("VHL_R4", raising=False)
    assert not r4_enabled()
    for v in ("", "0", "true"):
        monkeypatch.setenv("VHL_R4", v)
        assert not r4_enabled()
    monkeypatch.setenv("VHL_R4", "1")
    assert r4_enabled()


def test_mesh_only_and_partition_covers_mesh():
    import deltacontract as dc
    from editcore.export import mesh_to_chunks
    v, f, attrs = _grid_mesh()
    pipe = _Pipe(v, f, attrs)
    blobs, chunks, info = mesh_to_chunks(pipe, _slat([[10, 10, 32], [40, 40, 32]]))
    assert pipe.calls == [["mesh"]]                     # Gaussian 디코더를 부르지 않는다
    assert info["faces"] == len(f) and info["chunks"] == len(blobs) > 4
    assert sum(chunks[k].index_count for k in chunks) == 3 * len(f)   # 삼각형 누락·중복 없음
    for k, b in blobs.items():                          # 계약 디코더 왕복
        m = dc.decode(b)
        assert m.vertex_count == chunks[k].vertex_count and m.index_count == chunks[k].index_count
        n = np.asarray(m.normals, np.float32)
        assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-3)   # [0,1] 법선 → 2n−1 → 단위


def test_bookkeeping_rule():
    import deltacontract as dc
    from deltacontract.coords import dilate_cells
    from editcore.export import bookkeeping
    cells = np.array([[30, 30, 30], [31, 30, 30]])
    assert bookkeeping(cells, 2) == set(dc.affected_chunks(None, None, dilate_cells(cells, 2)))


def test_inherit_outside_book(tmp_path):
    import deltacontract as dc
    from editcore.export import mesh_to_chunks, write_cbin
    v, f, attrs = _grid_mesh(seed=0)
    pb, pc, _ = mesh_to_chunks(_Pipe(v, f, attrs), _slat([[1, 1, 1]]))
    parent = tmp_path / "parent"
    (parent / "chunks").mkdir(parents=True)
    for k, b in pb.items():
        (parent / "chunks" / f"{k}.cbin").write_bytes(b)
    v2 = v.copy()
    v2[:, 2] += 0.001                                    # 자식: 전체가 조금 다르다 → 새 바이트는 모든 청크가 다름
    cb, cc, _ = mesh_to_chunks(_Pipe(v2, f, attrs), _slat([[1, 1, 1]]))
    keys = sorted(cb)
    book = set(keys[: len(keys) // 3])
    res = write_cbin(tmp_path / "out", cb, cc, book=book, parent_dir=parent, snap=False)   # 승계 규칙만(스냅은 따로 테스트)
    out = tmp_path / "out" / "chunks"
    for k in keys:
        if k not in book and k not in pb:                 # 신규: 부모 없음 ∧ 변경 집합 밖 ∧ 결과 기하 있음 → 새 바이트(chief 결정)
            assert (out / f"{k}.cbin").read_bytes() == cb[k] and k in res["sent"] and k in res["new_outside"]
            continue
        b = (out / f"{k}.cbin").read_bytes()
        if k not in book and k in pb and k not in cb:
            continue
        if k in book:                                     # 변경 집합 안 → 새 바이트
            assert b == cb[k] and k in res["sent"]
        else:
            assert b == pb[k] and k in res["reused"]      # 부기 밖 = 부모 바이트 그대로
    man = json.loads((tmp_path / "out" / "manifest.json").read_text())["chunks"]
    assert set(man) == set(keys) | {k for k in pb if k not in book}
    assert res["new_outside"], "신규 청크가 한 건도 없으면 이 테스트는 신규 규칙을 안 잰다"
    assert res["reused"], "부기 밖 승계가 한 건도 없으면 이 테스트는 아무것도 안 잰다"
    for k, e in man.items():                              # manifest ↔ 실제 바이트 ↔ 계약 디코더
        b = (out / f"{k}.cbin").read_bytes()
        m = dc.decode(b)
        assert e["hash"] == dc.blob_hash(b) and e["byte_length"] == len(b)
        assert e["vertex_count"] == m.vertex_count and e["index_count"] == m.index_count


def test_removed_and_parent_only_chunks(tmp_path):
    from editcore.export import write_cbin
    parent = tmp_path / "p" / "chunks"
    parent.mkdir(parents=True)
    from editcore.export import mesh_to_chunks
    v, f, attrs = _grid_mesh()
    pb, pc, _ = mesh_to_chunks(_Pipe(v, f, attrs), _slat([[1, 1, 1]]))
    for k, b in pb.items():
        (parent / f"{k}.cbin").write_bytes(b)
    keys = sorted(pb)
    gone_in_book, gone_out_book = keys[0], keys[1]
    cb = {k: b for k, b in pb.items() if k not in (gone_in_book, gone_out_book)}
    res = write_cbin(tmp_path / "o", cb, pc, book={gone_in_book}, parent_dir=tmp_path / "p")
    assert res["removed"] == [gone_in_book]                    # 부기 안에서 사라진 부모 청크 → removed
    assert (tmp_path / "o" / "chunks" / f"{gone_out_book}.cbin").read_bytes() == pb[gone_out_book]   # 부기 밖은 남는다


def test_occupancy_changed_chunks():
    """R11 점유 비교: 청크(4³ 셀) 안 점유 셀 집합이 다른 청크만."""
    import deltacontract as dc
    from editcore.export import occupancy_changed_chunks
    parent = np.array([[0, 0, 0], [5, 5, 5], [40, 40, 40]])
    child = np.array([[0, 0, 0], [5, 5, 6], [40, 40, 40], [63, 63, 63]])   # (5,5,5)→(5,5,6) 같은 청크, (63,63,63) 새 청크
    got = occupancy_changed_chunks(parent, child)
    assert got == {dc.chunk_key((1, 1, 1)), dc.chunk_key((15, 15, 15))}
    assert occupancy_changed_chunks(parent, parent) == set()


def test_decode_to_cbin_occupancy_outside_book(tmp_path):
    """부기 밖이라도 점유가 바뀐 청크는 새 바이트(마스크 밖 생성이 화면에 반영), 나머지는 부모 승계."""
    import deltacontract as dc
    from editcore.export import decode_to_cbin, mesh_to_chunks
    v, f, attrs = _grid_mesh()
    pb, _, _ = mesh_to_chunks(_Pipe(v, f, attrs), _slat([[1, 1, 1]]))
    parent = tmp_path / "parent" / "chunks"
    parent.mkdir(parents=True)
    for k, b in pb.items():
        (parent / f"{k}.cbin").write_bytes(b)
    v2 = v.copy()
    v2[:, 2] += 0.0005
    keys = sorted(pb)
    target = keys[len(keys) // 2]                      # 부기 밖이지만 점유가 바뀌는 청크
    cid = dc.parse_chunk_key(target)
    child_cells = np.array([[4 * cid[0], 4 * cid[1], 4 * cid[2]]])
    pipe = _Pipe(v2, f, attrs)
    r = decode_to_cbin(pipe, _slat(child_cells), tmp_path / "out", mask_cells=np.array([[0, 0, 0]]),
                       parent_dir=tmp_path / "parent", parent_cells=np.zeros((0, 3), np.int64))
    assert r["occupancy_only_added"] >= 1 and target in r["sent"]
    assert (tmp_path / "out" / "chunks" / f"{target}.cbin").read_bytes() != pb[target]
    for k in r["reused"]:
        assert (tmp_path / "out" / "chunks" / f"{k}.cbin").read_bytes() == pb[k]


def test_vertex_color_render_glb():
    """R-2 렌더 입력 보정: 기본 재질 추가, COLOR_0 sRGB→선형(float32), 기하 그대로."""
    import io
    import json
    import struct
    import trimesh
    from editcore.render import vertex_color_render_glb
    v, f, _ = _grid_mesh(n=6)
    col = np.tile(np.array([[128, 10, 255, 255]], np.uint8), (len(v), 1))
    data = trimesh.Trimesh(v, f, vertex_colors=col, process=False).export(file_type="glb")
    out = vertex_color_render_glb(data)
    j = json.loads(out[20:20 + struct.unpack("<I", out[12:16])[0]])
    pr = j["meshes"][0]["primitives"][0]
    m = j["materials"][pr["material"]]["pbrMetallicRoughness"]
    assert "metallicFactor" not in m and m["baseColorFactor"] == [1.0, 1.0, 1.0, 1.0]   # to_glb 와 같은 재질(glTF 기본 금속 1.0)
    j0 = json.loads(vertex_color_render_glb(data, metallic=0.0)[20:20 + struct.unpack("<I", vertex_color_render_glb(data, metallic=0.0)[12:16])[0]])
    assert j0["materials"][-1]["pbrMetallicRoughness"]["metallicFactor"] == 0.0
    acc = j["accessors"][pr["attributes"]["COLOR_0"]]
    assert acc["componentType"] == 5126 and acc["type"] == "VEC4"
    g = trimesh.load(io.BytesIO(out), file_type="glb", force="mesh", process=False)
    assert np.array_equal(np.asarray(g.vertices, np.float32), v) and np.array_equal(g.faces, f)
    bv = j["bufferViews"][acc["bufferView"]]
    boff = 20 + struct.unpack("<I", out[12:16])[0] + 8
    c = np.frombuffer(out, np.float32, acc["count"] * 4, boff + bv["byteOffset"]).reshape(-1, 4)
    assert np.allclose(c[0, :3], [0.2158605, 0.0030353, 1.0], atol=1e-5)   # sRGB 128/255, 10/255, 1 → 선형


def test_bookkeeping_precompute_same_result(tmp_path):
    """부기 선행 계산(Future)을 넘겨도 변경·승계·생략 청크 집합이 같다(결과 무영향)."""
    from editcore.export import bookkeeping_async, decode_to_cbin, mesh_to_chunks
    v, f, attrs = _grid_mesh()
    pb, _, _ = mesh_to_chunks(_Pipe(v, f, attrs), _slat([[1, 1, 1]]))
    (tmp_path / "p" / "chunks").mkdir(parents=True)
    for k, b in pb.items():
        (tmp_path / "p" / "chunks" / f"{k}.cbin").write_bytes(b)
    v2 = v.copy(); v2[:, 2] += 0.0005
    mask = np.array([[30, 30, 32], [31, 31, 32]])
    kw = dict(mask_cells=mask, parent_dir=tmp_path / "p", parent_cells=np.array([[1, 1, 1]]))
    a = decode_to_cbin(_Pipe(v2, f, attrs), _slat([[1, 1, 1], [40, 40, 32]]), tmp_path / "a", **kw)
    b = decode_to_cbin(_Pipe(v2, f, attrs), _slat([[1, 1, 1], [40, 40, 32]]), tmp_path / "b", book=bookkeeping_async(mask, 2), **kw)
    for k in ("sent", "reused", "removed", "new_outside"):
        assert a[k] == b[k]


def test_seam_snap_closes_boundary_and_keeps_inherited_bytes(tmp_path):
    """경계 스냅(chief 2026-10-26): 새 청크의 경계 정점만 이웃 승계 청크 정점으로 옮긴다. 승계 바이트는 그대로."""
    import deltacontract as dc
    from editcore.export import mesh_to_chunks, write_cbin
    v, f, attrs = _grid_mesh(n=40)
    pb, _, _ = mesh_to_chunks(_Pipe(v, f, attrs), _slat([[1, 1, 1]]))
    (tmp_path / "p" / "chunks").mkdir(parents=True)
    for k, b in pb.items():
        (tmp_path / "p" / "chunks" / f"{k}.cbin").write_bytes(b)
    v2 = v.copy()
    v2[:, 2] += 0.0005                                     # 0.13 메시 셀 — 이음새가 생긴다
    cb, cc, _ = mesh_to_chunks(_Pipe(v2, f, attrs), _slat([[1, 1, 1]]))
    keys = sorted(cb)
    book = set(keys[: len(keys) // 2])
    off = write_cbin(tmp_path / "a", cb, cc, book=book, parent_dir=tmp_path / "p", snap=False)
    on = write_cbin(tmp_path / "b", cb, cc, book=book, parent_dir=tmp_path / "p")
    assert off["seam"] is None
    s = on["seam"]
    assert s["before"]["seam_faces"] > 0 and s["snapped"] > 0
    assert s["before"]["cells_max"] > 0.05 and s["after"]["cells_max"] < 1e-6      # 틈이 닫힌다
    for k in on["reused"]:                                  # 승계 청크는 부모 바이트 그대로
        assert (tmp_path / "b" / "chunks" / f"{k}.cbin").read_bytes() == pb[k]
    man = json.loads((tmp_path / "b" / "manifest.json").read_text())["chunks"]
    for k, e in man.items():                                # 스냅한 청크도 manifest 와 바이트·계약 디코더 일치
        b = (tmp_path / "b" / "chunks" / f"{k}.cbin").read_bytes()
        assert e["hash"] == dc.blob_hash(b) and e["index_count"] == dc.decode(b).index_count
