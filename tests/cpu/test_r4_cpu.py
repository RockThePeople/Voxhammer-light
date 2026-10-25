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
    res = write_cbin(tmp_path / "out", cb, cc, book=book, parent_dir=parent)
    out = tmp_path / "out" / "chunks"
    for k in keys:
        b = (out / f"{k}.cbin").read_bytes()
        if k not in book and k in pb and k not in cb:
            continue
        if k in book or k not in pb:                      # 부기 안, 또는 부모에 없던 청크 → 새 바이트
            assert b == cb[k] and k in res["sent"]
        else:
            assert b == pb[k] and k in res["reused"]      # 부기 밖 = 부모 바이트 그대로
    man = json.loads((tmp_path / "out" / "manifest.json").read_text())["chunks"]
    assert set(man) == set(keys) | {k for k in pb if k not in book}
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
