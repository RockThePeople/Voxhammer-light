"""디코딩·GLB·R2 저장 — 파일 쓰기는 여기와 service 만.

2번 R4(`VHL_R4=1`): `decode_slat(["mesh"])` 만 → 정점 속성 분리 → 청크 분할 → `.cbin` 인코딩 → R11 부기 승계 →
`chunks/*.cbin` + `manifest.json`. Gaussian 디코더와 `to_glb`(단순화·텍스처 굽기)를 타지 않는다.
계약은 ai-ar-v3 `contract/python/deltacontract`(읽기 전용 import), 좌표는 디코더 native NORMALIZED 공간 그대로(D9-b).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
_CONTRACT = _REPO / "third_party" / "ai-ar-v3" / "contract" / "python"


# origin: voxhammer/edit_pipeline.py:run_edit (624–628행) @ f33cd96+patch
def decode_to_glb(pipeline, slat_tgt, output_path, hooks=None):
    span = (hooks or {}).get("span")
    ctx = span("decode_slat", 7) if span else _null()
    with ctx, torch.no_grad():
        assets = pipeline.decode_slat(slat_tgt, ["gaussian", "mesh"])
    from trellis.utils import postprocessing_utils
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(True)  # 포크와 같이 to_glb(텍스처 굽기)는 grad 를 켠 채로
    try:
        ctx = span("to_glb", 7) if span else _null()
        with ctx:
            glb = postprocessing_utils.to_glb(assets["gaussian"][0], assets["mesh"][0], simplify=0.95, texture_size=1024)
        ctx = span("glb_export", 7) if span else _null()
        with ctx:
            glb.export(str(output_path))
    finally:
        torch.set_grad_enabled(prev)
    return assets


def save_slat(slat, path, normalization, extra=None):
    """R2 형식: coords int32 (N,4), feats float32 (N,8), slat_space=denormalized."""
    from safetensors.torch import save_file
    meta = {"slat_space": "denormalized", "slat_channels": str(slat.feats.shape[1]), "voxel_res": "64",
            "coord_order": "canonical",
            "norm_mean": json.dumps(list(map(float, normalization["mean"]))),
            "norm_std": json.dumps(list(map(float, normalization["std"])))}
    meta.update({k: (v if isinstance(v, str) else json.dumps(v)) for k, v in (extra or {}).items()})
    save_file({"coords": slat.coords.detach().cpu().to(torch.int32).contiguous(),
               "feats": slat.feats.detach().cpu().float().contiguous()}, str(path), metadata=meta)
    return Path(path)


def _contract():
    if str(_CONTRACT) not in sys.path:
        sys.path.insert(0, str(_CONTRACT))
    import deltacontract
    return deltacontract


def mesh_to_chunks(pipeline, slat_tgt, hooks=None):
    """R4 계산부(파일 입출력 없음): mesh 디코딩 → 정점 속성 분리 → 청크 분할 → 인코딩.
    반환: (blobs {키: bytes}, chunk_meshes {키: ChunkMesh}, 메시 정보 dict)."""
    dc = _contract()
    span = (hooks or {}).get("span")
    with (span("decode_mesh", 7) if span else _null()), torch.no_grad():
        mesh = pipeline.decode_slat(slat_tgt, ["mesh"])["mesh"][0]
    with span("chunk_partition", 7) if span else _null():
        verts = mesh.vertices.detach().cpu().numpy()
        faces = mesh.faces.detach().cpu().numpy()
        vattrs = mesh.vertex_attrs.detach().cpu().numpy() if mesh.vertex_attrs is not None else None
        dc.assert_in_normalized_bounds(verts)
        albedo, normal = dc.split_trellis_vertex_attrs(vattrs, normal_encoding="0_1")
        cells = slat_tgt.coords[:, 1:].detach().cpu().numpy()
        chunks = dc.partition_mesh(verts, faces, normals=normal, colors_rgba8=dc.albedo_to_rgba8(albedo), voxel_cells=cells)
    with span("cbin_encode", 7) if span else _null():
        blobs = {k: dc.encode(chunks[k]) for k in sorted(chunks)}
    info = {"vertices": int(verts.shape[0]), "faces": int(faces.shape[0]), "chunks": len(blobs),
            "bytes": int(sum(len(b) for b in blobs.values()))}
    return blobs, chunks, info


def bookkeeping(mask_cells, halo=2):
    """R11 부기: 마스크 ∪ halo 셀을 청크로 접은 키 집합(legacy/a5000/edit_service/api/ops.py:mask_rows 와 같은 규칙).
    해시로 변경 청크를 정하지 않는다."""
    dc = _contract()
    from deltacontract.coords import dilate_cells
    cells = dilate_cells(np.asarray(mask_cells, dtype=np.int64).reshape(-1, 3), halo)
    return set(dc.affected_chunks(None, None, cells))


def occupancy_changed_chunks(parent_cells, child_cells):
    """R11 점유 비교(요구사항 4.4 0번 R11): 청크 안 SLat 점유 셀 집합이 부모와 다른 청크 키 = 대칭차 셀이 속한 청크."""
    _contract()
    from deltacontract.coords import chunk_keys_sorted, voxel_to_chunk
    code = lambda c: c[:, 0] * 4096 + c[:, 1] * 64 + c[:, 2]
    p = np.asarray(parent_cells, dtype=np.int64).reshape(-1, 3)
    c = np.asarray(child_cells, dtype=np.int64).reshape(-1, 3)
    d = np.setxor1d(code(p), code(c))
    if d.size == 0:
        return set()
    return set(chunk_keys_sorted(voxel_to_chunk(np.stack([d // 4096, (d // 64) % 64, d % 64], 1))))


def write_cbin(out_dir, blobs, chunks, *, book=None, parent_dir=None, asset_id="vhl", version=1, hooks=None):
    """R4 쓰기부: 변경 집합(book 인자 = 부기 ∪ 점유 변경) 밖 청크는 부모 바이트를 그대로 복사(R11 승계), 안·부모에 없는 청크는 새 바이트.
    부모에만 있고 새 디코딩에 없는 부기 밖 청크도 부모 바이트로 남긴다(부기 밖은 바뀌지 않는다).
    부기 안에서 기하가 사라진 부모 청크는 removed 로 적는다(계약 3.4.3)."""
    dc = _contract()
    from deltacontract.schemas import ChunkEntry, ChunkManifest
    span = (hooks or {}).get("span")
    out_dir = Path(out_dir)
    (out_dir / "chunks").mkdir(parents=True, exist_ok=True)
    pdir = Path(parent_dir) if parent_dir else None
    parent = {p.stem.split(".")[0]: p for p in sorted((pdir / "chunks" if (pdir / "chunks").is_dir() else pdir).glob("*.cbin"))} if pdir else {}
    book = set(book or [])
    final, sent, reused = {}, [], []
    with span("bookkeeping", 7) if span else _null():
        for k in sorted(set(blobs) | {k for k in parent if k not in book}):
            if pdir is not None and k in parent and k not in book:
                final[k] = ("parent", parent[k].read_bytes())
                reused.append(k)
            elif k in blobs:
                final[k] = ("fresh", blobs[k])
                sent.append(k)
        removed = sorted(k for k in book if k in parent and k not in final)
    with span("cbin_write", 7) if span else _null():
        entries = {}
        for k, (src, b) in final.items():
            (out_dir / "chunks" / f"{k}.cbin").write_bytes(b)
            cm = chunks[k] if src == "fresh" else dc.decode(b)
            entries[k] = ChunkEntry(uri=dc.chunk_uri(asset_id, k, version), hash=dc.blob_hash(b), byte_length=len(b),
                                    vertex_count=int(cm.vertex_count), index_count=int(cm.index_count),
                                    voxel_count=int(cm.voxel_count), version=int(version))
        man = ChunkManifest(asset_id=asset_id, version=int(version), chunks=entries)
        (out_dir / "manifest.json").write_text(man.model_dump_json(indent=2))
    return {"chunks": len(final), "sent": sent, "reused": reused, "removed": removed, "book": sorted(book),
            "sent_bytes": int(sum(len(final[k][1]) for k in sent)), "total_bytes": int(sum(len(v[1]) for v in final.values())),
            "parent_dir": str(pdir) if pdir else None}


def decode_to_cbin(pipeline, slat_tgt, out_dir, *, mask_cells=None, parent_dir=None, parent_cells=None, halo=2,
                   glb_path=None, hooks=None):
    """R4 진입점. `out_dir/chunks/*.cbin`·`manifest.json`·`r4.json`, 그리고 glb_path 가 있으면 `.cbin` 에서 조립한
    정점색 GLB(ai-ar-v3 `server/glbbuild.build_glb`, 렌더·뷰어 입력)를 쓴다.
    변경 청크 = 부기(마스크 ∪ halo) ∪ 점유 변경(parent_cells 가 있으면, R11 — chief 정정 2026-10-26). 나머지는 부모 승계."""
    blobs, chunks, info = mesh_to_chunks(pipeline, slat_tgt, hooks=hooks)
    span = (hooks or {}).get("span")
    book, occ = set(), set()
    with span("bookkeeping", 7) if span else _null():
        if mask_cells is not None and parent_dir:
            book = bookkeeping(mask_cells, halo)
        if parent_cells is not None and parent_dir:
            occ = occupancy_changed_chunks(parent_cells, slat_tgt.coords[:, 1:].detach().cpu().numpy())
    res = write_cbin(out_dir, blobs, chunks, book=book | occ, parent_dir=parent_dir, hooks=hooks)
    exist = set(res["sent"]) | set(res["reused"]) | set(res["removed"])
    res.update({"mesh": info, "rule": "부기 ∪ 점유 변경 (R11)", "book_raw": len(book), "book_existing": len(book & exist),
                "occupancy_changed": len(occ & exist), "occupancy_only_added": len((occ - book) & exist)})
    if glb_path:
        span = (hooks or {}).get("span")
        with span("glb_build", "r4_glb") if span else _null():   # 구간 7 밖(렌더·뷰어 입력용)
            v3 = str(_REPO / "third_party" / "ai-ar-v3")
            if v3 not in sys.path:
                sys.path.insert(0, v3)
            from server.glbbuild import build_glb
            build_glb(Path(out_dir) / "chunks", Path(glb_path))
        res["glb"] = str(glb_path)
    Path(out_dir, "r4.json").write_text(json.dumps(res, indent=1))
    return res


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
