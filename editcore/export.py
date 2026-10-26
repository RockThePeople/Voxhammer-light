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


# ── 청크 경계 이음새: 측정·스냅 (chief 결정 2026-10-26 (a) 경계 스냅 — 현행 서비스 동작 복원) ──────────────────
# origin: legacy/a5000/edit_service/lab/seam.py:{seam_pairs,shared_face_verts,measure_seams,snap_boundary}
# 레거시는 청크 한 변을 8 셀로 가정(CHUNK_SPAN = 8/64)했다. 계약은 D75 로 4 셀이 됐으므로 deltacontract.coords.CHUNK_SIZE 를 쓴다.
MESH_CELL = 1.0 / 256.0
SEAM_MATCH_RADIUS = 4 * MESH_CELL
SNAP_CELLS = 2.0


def _chunk_box(key):
    _contract()
    from deltacontract.coords import CHUNK_SIZE, VOXEL_RES, parse_chunk_key
    span = CHUNK_SIZE / VOXEL_RES
    lo = np.asarray(parse_chunk_key(key), dtype=np.float64) * span - 0.5
    return lo, lo + span


def _shared_face_verts(pos, key_self, key_other, tol=MESH_CELL):
    """두 26-이웃 청크의 공유 경계(면·모서리·꼭짓점)에 tol 이내로 붙은 정점."""
    from deltacontract.coords import parse_chunk_key
    a, b = np.asarray(parse_chunk_key(key_self)), np.asarray(parse_chunk_key(key_other))
    lo, hi = _chunk_box(key_self)
    keep = np.ones(len(pos), bool)
    for ax in range(3):
        d = b[ax] - a[ax]
        if d > 0:
            keep &= np.abs(pos[:, ax] - hi[ax]) <= tol
        elif d < 0:
            keep &= np.abs(pos[:, ax] - lo[ax]) <= tol
    return keep


def _seam_pairs(fresh, inherited):
    """(새 바이트 청크, 승계 청크) 26-이웃 쌍."""
    from deltacontract.coords import parse_chunk_key
    inh = {k: np.asarray(parse_chunk_key(k)) for k in inherited}
    out = []
    for f in fresh:
        cf = np.asarray(parse_chunk_key(f))
        out += [(f, o) for o, co in inh.items() if np.abs(cf - co).max() == 1]
    return out


def measure_seams(fresh_pos, inherited_pos, pairs):
    """이음새 정량화: 새 청크와 승계 청크의 공유 경계 정점끼리 최근접 거리(대응 반경 안 쌍만). 단위 메시 셀(1/256)."""
    from scipy.spatial import cKDTree
    dists, faces, unmatched, total = [], 0, 0, 0
    for f, o in pairs:
        pi, po = fresh_pos.get(f), inherited_pos.get(o)
        if pi is None or po is None or not len(pi) or not len(po):
            continue
        mi, mo = _shared_face_verts(pi, f, o), _shared_face_verts(po, o, f)
        if not mi.any() or not mo.any():
            continue
        faces += 1
        for a, b in ((pi[mi], po), (po[mo], pi)):
            d, _ = cKDTree(b).query(a)
            total += d.size
            hit = d <= SEAM_MATCH_RADIUS
            unmatched += int((~hit).sum())
            if hit.any():
                dists.append(d[hit])
    d = np.concatenate(dists) if dists else np.zeros(0)
    return {"seam_faces": faces, "boundary_vertices": total, "unmatched": unmatched,
            "matched": int(d.size), "cells_max": float(d.max() / MESH_CELL) if d.size else None,
            "cells_median": float(np.median(d) / MESH_CELL) if d.size else None,
            "cells_p99": float(np.percentile(d, 99) / MESH_CELL) if d.size else None,
            "zero_ratio": float((d == 0).mean()) if d.size else None,
            "over_cell": int((d > MESH_CELL).sum())}


def snap_boundary(chunks, fresh, inherited_pos, cells=SNAP_CELLS):
    """새 청크의 경계 정점을 이웃 승계 청크의 대응 경계 정점으로 스냅한다(새 청크 쪽만 — 승계 바이트는 그대로, R11).
    임계값(cells 메시 셀) 밖은 위상이 실제로 다른 곳이라 건드리지 않는다. 바뀐 청크는 canonicalize 후 다시 인코딩한다.
    반환: (바뀐 청크 {키: ChunkMesh}, 통계)."""
    from scipy.spatial import cKDTree
    from deltacontract.chunkbin import canonicalize
    from deltacontract.coords import parse_chunk_key
    thr = cells * MESH_CELL
    by_fresh = {}
    for f, o in _seam_pairs(fresh, inherited_pos):
        by_fresh.setdefault(f, []).append(o)
    changed, moved, n_cand, n_snap = {}, [], 0, 0
    for f, outs in by_fresh.items():
        m = chunks[f]
        pos = np.asarray(m.positions, np.float64).copy()
        ref, sel = [], np.zeros(len(pos), bool)
        for o in outs:
            q = inherited_pos[o]
            ref.append(q[_shared_face_verts(q, o, f)])
            sel |= _shared_face_verts(pos, f, o)
        ref = [r for r in ref if len(r)]
        if not ref or not sel.any():
            continue
        ref = np.concatenate(ref)
        idx = np.flatnonzero(sel)
        n_cand += idx.size
        d, j = cKDTree(ref).query(pos[idx])
        hit = (d <= thr) & (d > 0)
        if not hit.any():
            continue
        moved.append(d[hit])
        n_snap += int(hit.sum())
        pos[idx[hit]] = ref[j[hit]]
        changed[f] = canonicalize(chunk_coord=parse_chunk_key(f), positions=pos.astype(np.float32), indices=m.indices,
                                  normals=m.normals, colors=m.colors, uvs=m.uvs, voxel_count=m.voxel_count)
    mv = np.concatenate(moved) if moved else np.zeros(0)
    return changed, {"snap_cells": cells, "candidates": int(n_cand), "snapped": int(n_snap), "chunks_changed": len(changed),
                     "move_mean_cells": float(mv.mean() / MESH_CELL) if mv.size else 0.0,
                     "move_max_cells": float(mv.max() / MESH_CELL) if mv.size else 0.0}


def write_cbin(out_dir, blobs, chunks, *, book=None, parent_dir=None, asset_id="vhl", version=1, hooks=None, snap=True):
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
    final, sent, reused, new_outside = {}, [], [], []
    with span("bookkeeping", 7) if span else _null():
        for k in sorted(set(blobs) | {k for k in parent if k not in book}):
            if pdir is not None and k in parent and k not in book:
                final[k] = ("parent", parent[k].read_bytes())
                reused.append(k)
            elif k in blobs:
                # 변경 집합(부기 ∪ 점유 변경) 안, 또는 신규 청크(부모에 기하 없음 ∧ 결과에 기하 있음) → 새 바이트.
                # 신규는 삼각형이 무게중심으로 배정돼 점유 셀 없는 이웃 청크에 생긴 결과 메시의 일부다. 부모 바이트가 없어
                # 승계할 대상이 없고, 쓰지 않으면 청크 경계에 구멍이 생긴다 → 변경 청크에 넣는다(chief 결정 2026-10-26:
                # 변경 = 부기 ∪ 점유 변경 ∪ 신규).
                final[k] = ("fresh", blobs[k])
                sent.append(k)
                if pdir is not None and k not in book and k not in parent:
                    new_outside.append(k)
        removed = sorted(k for k in book if k in parent and k not in final)
    seam = None
    if snap and pdir is not None and sent and reused:
        with span("seam_snap", 7) if span else _null():
            inh_pos = {k: np.asarray(dc.decode(final[k][1]).positions, np.float64) for k in reused}
            pairs = _seam_pairs(sent, reused)
            near = {o for _, o in pairs}
            inh_pos = {k: v for k, v in inh_pos.items() if k in near}
            before = measure_seams({k: np.asarray(chunks[k].positions, np.float64) for k in sent}, inh_pos, pairs)
            snapped, st = snap_boundary(chunks, sent, inh_pos)
            chunks = dict(chunks)
            for k, cm in snapped.items():
                chunks[k] = cm
                final[k] = ("fresh", dc.encode(cm))
            after = measure_seams({k: np.asarray(chunks[k].positions, np.float64) for k in sent}, inh_pos, pairs)
            seam = {"before": before, "after": after, **st}
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
    return {"chunks": len(final), "sent": sent, "reused": reused, "removed": removed, "new_outside": new_outside, "book": sorted(book),
            "sent_bytes": int(sum(len(final[k][1]) for k in sent)), "total_bytes": int(sum(len(v[1]) for v in final.values())),
            "parent_dir": str(pdir) if pdir else None, "seam": seam}


def bookkeeping_async(mask_cells, halo=2):
    """부기 선행 계산(chief 검토 2026-10-26, 결과 무영향): 입력이 마스크뿐이라 편집 시작 때 별도 스레드로 먼저 계산한다.
    구간 7 에서는 `decode_to_cbin(book=<Future>)` 로 결과만 받는다. 계산 자체(dilate_cells, E2)는 `bookkeeping` 그대로."""
    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="r4-book")
    fut = ex.submit(bookkeeping, mask_cells, halo)
    ex.shutdown(wait=False)
    return fut


def decode_to_cbin(pipeline, slat_tgt, out_dir, *, mask_cells=None, parent_dir=None, parent_cells=None, halo=2,
                   glb_path=None, hooks=None, book=None, asset_id="vhl", version=1):
    """R4 진입점. `out_dir/chunks/*.cbin`·`manifest.json`·`r4.json`, 그리고 glb_path 가 있으면 `.cbin` 에서 조립한
    정점색 GLB(ai-ar-v3 `server/glbbuild.build_glb`, 렌더·뷰어 입력)를 쓴다.
    변경 청크 = 부기(마스크 ∪ halo) ∪ 점유 변경(parent_cells 가 있으면, R11 — chief 정정 2026-10-26). 나머지는 부모 승계."""
    blobs, chunks, info = mesh_to_chunks(pipeline, slat_tgt, hooks=hooks)
    span = (hooks or {}).get("span")
    pre, book, occ = book, set(), set()
    with span("bookkeeping", 7) if span else _null():
        if pre is not None and parent_dir:      # 선행 계산 결과(Future 또는 집합)를 받기만 한다
            book = set(pre.result() if hasattr(pre, "result") else pre)
        elif mask_cells is not None and parent_dir:
            book = bookkeeping(mask_cells, halo)
        if parent_cells is not None and parent_dir:
            occ = occupancy_changed_chunks(parent_cells, slat_tgt.coords[:, 1:].detach().cpu().numpy())
    res = write_cbin(out_dir, blobs, chunks, book=book | occ, parent_dir=parent_dir, hooks=hooks, asset_id=asset_id, version=version)
    exist = set(res["sent"]) | set(res["reused"]) | set(res["removed"])
    res.update({"mesh": info, "rule": "부기 ∪ 점유 변경 ∪ 신규 (R11, chief 2026-10-26)", "book_raw": len(book), "book_existing": len(book & exist),
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
