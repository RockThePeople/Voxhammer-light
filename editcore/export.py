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


# ── 청크 경계 봉합 띠 (chief 결정 2026-10-26 21:4x (d), 토글 VHL_R4_STITCH, 기본 꺼짐 — 결과가 바뀐다) ─────────────
# 스냅(정점 이동)만으로는 새 청크와 승계 청크의 경계 분할이 달라 열린 틈이 남는다(T-접합·한쪽에만 있는 면 띠, X3 C4 실측:
# 지프 P1 r1 열린 모서리 3,981, 대부분 끝이 열린 짧은 사슬). 조립 메시를 용접해 열린 모서리 덩어리마다 정점을 최적 평면에
# 투영·Delaunay 삼각분할하고, 변이 STITCH_MAX_EDGE 이하인 삼각형만 그 덩어리 모서리를 가장 많이 가진 **새 청크**에 더한다.
# 새 청크 모서리가 하나도 없는 덩어리(승계 청크끼리의 틈)는 건드리지 않는다. 승계 청크 바이트는 그대로(R11).
STITCH_MAX_EDGE = 4 * MESH_CELL          # 지프 P1 r1 실측: 2 셀 → 어두운 화소 486→389, 4 셀 → 290
STITCH_WELD = 64 * 2000          # 용접 격자: 1/2000 복셀 셀


def stitch_enabled():
    import os
    return os.environ.get("VHL_R4_STITCH", "0") == "1"


def assembled_open_edges(meshes):
    """청크 메시들을 이어 붙여 용접한 뒤 열린 모서리(면 하나에만 속한 모서리). 반환 (용접 정점 위치, 열린 모서리 (n,2), 모서리 소유 청크, 대표 원본 (청크, 정점))."""
    keys, P, F, own, src = sorted(meshes), [], [], [], []
    off = 0
    for k in keys:
        m = meshes[k]
        pos = np.asarray(m.positions, np.float64)
        P.append(pos)
        F.append(np.asarray(m.indices, np.int64).reshape(-1, 3) + off)
        own += [k] * (len(m.indices) // 3)
        src += [(k, i) for i in range(len(pos))]
        off += len(pos)
    if not off:
        return np.zeros((0, 3)), np.zeros((0, 2), np.int64), np.array([]), []
    P, F, own = np.concatenate(P), np.concatenate(F), np.asarray(own)
    q = np.round(P * STITCH_WELD).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    FF = inv[F]
    e = np.concatenate([FF[:, [0, 1]], FF[:, [1, 2]], FF[:, [2, 0]]])
    eo = np.concatenate([own, own, own])
    es = np.sort(e, 1)
    ue, idx, cnt = np.unique(es, axis=0, return_index=True, return_counts=True)
    open_ = (cnt == 1) & (ue[:, 0] != ue[:, 1])
    return P[first], ue[open_], eo[idx[open_]], [src[i] for i in first]


def _nearest_fresh(pts, fresh):
    """덩어리 정점이 든 청크와 그 26-이웃 중 새 청크(정점 수가 가장 많은 쪽)."""
    from deltacontract.coords import CHUNK_SIZE, VOXEL_RES, parse_chunk_key
    span = CHUNK_SIZE / VOXEL_RES
    fk = {tuple(parse_chunk_key(k)): k for k in fresh}
    votes = {}
    for c in np.floor((pts + 0.5) / span).astype(int):
        for d in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")).reshape(3, -1).T:
            k = fk.get(tuple(c + d))
            if k:
                votes[k] = votes.get(k, 0) + (3 if not d.any() else 1)
    return max(votes, key=votes.get) if votes else None


def stitch_seams(meshes, fresh, max_edge=STITCH_MAX_EDGE, inherited_only=True):
    """meshes: 최종 청크 {키: ChunkMesh}(새 + 승계), fresh: 새 청크 키. 반환 (바뀐 새 청크 {키: ChunkMesh}, 통계).
    inherited_only: 새 청크 모서리가 없는 덩어리도, 그 근처(26-이웃)에 새 청크가 있으면 그 새 청크에 띠를 붙인다."""
    from collections import defaultdict
    from scipy.spatial import Delaunay
    from deltacontract.chunkbin import canonicalize
    from deltacontract.coords import parse_chunk_key
    fresh = set(fresh)
    V, E, Eown, src = assembled_open_edges(meshes)
    adj = defaultdict(list)
    for i, (a, b) in enumerate(E):
        adj[a].append(i)
        adj[b].append(i)
    seen, add = set(), defaultdict(list)          # 새 청크 → [(정점 위치 3개, 원본 정점 3개)]
    n_comp = n_used = n_tri = 0
    area = 0.0
    for v0 in list(adj):
        if v0 in seen:
            continue
        stack, comp, eids = [v0], [], set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            for i in adj[x]:
                eids.add(i)
                stack.extend(E[i])
        n_comp += 1
        if len(comp) < 3:
            continue
        owners = [Eown[i] for i in eids if Eown[i] in fresh]
        pts = V[comp]
        if owners:
            target = max(set(owners), key=owners.count)
        elif inherited_only:
            target = _nearest_fresh(pts, fresh)
            if target is None:
                continue
        else:
            continue
        c = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        uv = (pts - c) @ vt[:2].T
        try:
            tri = Delaunay(uv, qhull_options="QJ").simplices
        except Exception:  # noqa: BLE001 — 한 줄로 늘어선 점(넓이 0) 등
            continue
        p = pts[tri]
        el = np.linalg.norm(p - np.roll(p, 1, axis=1), axis=2).max(1)
        cr = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        ar = 0.5 * np.linalg.norm(cr, axis=1)
        keep = (el <= max_edge) & (ar > 0)
        if not keep.any():
            continue
        n_used += 1
        for t in tri[keep]:
            add[target].append([comp[j] for j in t])
        n_tri += int(keep.sum())
        area += float(ar[keep].sum())
    changed, total_area = {}, 0.0
    for k, m in meshes.items():
        pp = np.asarray(m.positions, np.float64)
        f = pp[np.asarray(m.indices).reshape(-1, 3)]
        total_area += float(0.5 * np.linalg.norm(np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0]), axis=1).sum())
    for k, tris in add.items():
        m = meshes[k]
        pos = [np.asarray(m.positions, np.float32)]
        nrm = [np.asarray(m.normals, np.float32)] if m.normals is not None else None
        col = [np.asarray(m.colors, np.uint8)] if m.colors is not None else None
        uvs = [np.asarray(m.uvs, np.float32)] if m.uvs is not None else None
        idx = [np.asarray(m.indices, np.uint32)]
        base = len(pos[0])
        for t in tris:
            a = []
            for w in t:                                   # 원본 정점의 속성을 그대로 복사(정점색·법선)
                sk, si = src[w]
                sm = meshes[sk]
                pos.append(np.asarray(sm.positions, np.float32)[si:si + 1])
                if nrm is not None:
                    nrm.append(np.asarray(sm.normals, np.float32)[si:si + 1] if sm.normals is not None else np.zeros((1, 3), np.float32))
                if col is not None:
                    col.append(np.asarray(sm.colors, np.uint8)[si:si + 1] if sm.colors is not None else np.full((1, 4), 255, np.uint8))
                if uvs is not None:
                    uvs.append(np.asarray(sm.uvs, np.float32)[si:si + 1] if sm.uvs is not None else np.zeros((1, 2), np.float32))
                a.append(base)
                base += 1
            P3 = np.concatenate(pos[-3:]).astype(np.float64)
            fn = np.cross(P3[1] - P3[0], P3[2] - P3[0])
            vn = np.concatenate(nrm[-3:]).sum(0) if nrm is not None else fn
            if fn @ vn < 0:                               # 이웃 정점 법선과 같은 쪽을 보게
                a = [a[0], a[2], a[1]]
            idx.append(np.asarray(a, np.uint32))
        changed[k] = canonicalize(chunk_coord=parse_chunk_key(k), positions=np.concatenate(pos), indices=np.concatenate(idx),
                                  normals=np.concatenate(nrm) if nrm is not None else None,
                                  colors=np.concatenate(col) if col is not None else None,
                                  uvs=np.concatenate(uvs) if uvs is not None else None, voxel_count=m.voxel_count)
    return changed, {"open_edges_before": int(len(E)), "components": n_comp, "components_stitched": n_used,
                     "strip_faces": n_tri, "strip_area_ratio": area / total_area if total_area else 0.0,
                     "chunks_changed": len(changed), "max_edge_cells": max_edge / MESH_CELL}


# ── 청크 경계 치마 벽 (chief 22:2x 2-1 시험, 토글 VHL_R4_SKIRT, 기본 꺼짐 — 결과가 바뀐다) ───────────────────────
# 새 청크가 승계 청크와 맞닿는 청크 경계면 근처의 열린 경계 모서리(용접 1/2000 셀 뒤 면 하나에만 속한 모서리)마다, 두 정점을
# 정점 법선 반대쪽(표면 안쪽)으로 SKIRT_DEPTH 옮긴 두 정점을 만들고 사각형(삼각형 2개)을 새 청크에 더한다. 틈 사이로 보이는
# 안쪽(배경·내부)을 벽이 가린다. 색 = 원래 두 정점 색, 법선 = 원래 법선. 승계 청크 바이트는 그대로(R11).
SKIRT_DEPTH = 1.5 * MESH_CELL
SKIRT_PLANE_BAND = MESH_CELL      # 경계면에서 이 거리 안의 모서리(청크는 삼각형 무게중심으로 나뉘어 경계가 평면에 딱 붙지 않는다)


def skirt_enabled():
    import os
    return os.environ.get("VHL_R4_SKIRT", "0") == "1"


def skirt_seams(meshes, fresh, depth=SKIRT_DEPTH, band=SKIRT_PLANE_BAND):
    """meshes: 최종 청크 {키: ChunkMesh}, fresh: 새 청크 키. 반환 (바뀐 새 청크 {키: ChunkMesh}, 통계)."""
    from collections import defaultdict
    from deltacontract.chunkbin import canonicalize
    from deltacontract.coords import parse_chunk_key
    fresh = set(fresh)
    keyc = {k: np.asarray(parse_chunk_key(k)) for k in meshes}
    inh = {tuple(c) for k, c in keyc.items() if k not in fresh}
    V, E, Eown, src = assembled_open_edges(meshes)
    add = defaultdict(list)
    n_edge = 0
    for (a, b), k in zip(E, Eown):
        if k not in fresh:
            continue
        lo, hi = _chunk_box(k)
        mid = (V[a] + V[b]) / 2
        ok = False
        for ax in range(3):
            for side, plane in ((-1, lo[ax]), (1, hi[ax])):
                if abs(mid[ax] - plane) <= band:
                    nb = keyc[k].copy()
                    nb[ax] += side
                    ok |= tuple(nb) in inh
        if ok:
            add[k].append((a, b))
            n_edge += 1
    changed, n_face, area, total = {}, 0, 0.0, 0.0
    for m in meshes.values():
        pp = np.asarray(m.positions, np.float64)
        f = pp[np.asarray(m.indices).reshape(-1, 3)]
        total += float(0.5 * np.linalg.norm(np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0]), axis=1).sum())
    for k, edges in add.items():
        m = meshes[k]
        pos = [np.asarray(m.positions, np.float32)]
        nrm = [np.asarray(m.normals, np.float32)] if m.normals is not None else None
        col = [np.asarray(m.colors, np.uint8)] if m.colors is not None else None
        uvs = [np.asarray(m.uvs, np.float32)] if m.uvs is not None else None
        idx = [np.asarray(m.indices, np.uint32)]
        base = len(pos[0])
        for a, b in edges:
            vs = []
            for w in (a, b):
                sk, si = src[w]
                sm = meshes[sk]
                p0 = np.asarray(sm.positions, np.float64)[si]
                n0 = np.asarray(sm.normals, np.float64)[si] if sm.normals is not None else np.zeros(3)
                nn = n0 / max(np.linalg.norm(n0), 1e-12)
                c0 = np.asarray(sm.colors, np.uint8)[si] if sm.colors is not None else np.full(4, 255, np.uint8)
                vs.append((p0, p0 - depth * nn, n0, c0))
            (pa, qa, na, ca), (pb, qb, nb_, cb) = vs
            P4 = np.stack([pa, pb, qb, qa])
            pos.append(P4.astype(np.float32))
            if nrm is not None:
                nrm.append(np.stack([na, nb_, nb_, na]).astype(np.float32))
            if col is not None:
                col.append(np.stack([ca, cb, cb, ca]))
            if uvs is not None:
                uvs.append(np.zeros((4, 2), np.float32))
            q = [base, base + 1, base + 2, base + 3]                  # 벽은 양면으로 그려진다(Cycles·뷰어 DoubleSide)
            idx.append(np.asarray([q[0], q[1], q[2], q[0], q[2], q[3]], np.uint32))
            area += float(np.linalg.norm(np.cross(P4[1] - P4[0], P4[2] - P4[0])) / 2 + np.linalg.norm(np.cross(P4[2] - P4[0], P4[3] - P4[0])) / 2)
            n_face += 2
            base += 4
        changed[k] = canonicalize(chunk_coord=parse_chunk_key(k), positions=np.concatenate(pos), indices=np.concatenate(idx),
                                  normals=np.concatenate(nrm) if nrm is not None else None,
                                  colors=np.concatenate(col) if col is not None else None,
                                  uvs=np.concatenate(uvs) if uvs is not None else None, voxel_count=m.voxel_count)
    return changed, {"skirt_edges": n_edge, "skirt_faces": n_face, "skirt_area_ratio": area / total if total else 0.0,
                     "chunks_changed": len(changed), "depth_cells": depth / MESH_CELL, "band_cells": band / MESH_CELL}


def write_cbin(out_dir, blobs, chunks, *, book=None, parent_dir=None, asset_id="vhl", version=1, hooks=None, snap=True, stitch=None,
               skirt=None):
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
    if (stitch_enabled() if stitch is None else stitch) and pdir is not None and sent and reused:
        with span("seam_stitch", 7) if span else _null():
            meshes = {k: (chunks[k] if src == "fresh" else dc.decode(b)) for k, (src, b) in final.items()}
            stitched, sst = stitch_seams(meshes, sent)
            chunks = dict(chunks)
            for k, cm in stitched.items():
                chunks[k] = cm
                final[k] = ("fresh", dc.encode(cm))
            meshes.update(stitched)
            sst["open_edges_after"] = int(len(assembled_open_edges(meshes)[1]))
            seam = {**(seam or {}), "stitch": sst}
    if (skirt_enabled() if skirt is None else skirt) and pdir is not None and sent and reused:
        with span("seam_skirt", 7) if span else _null():
            meshes = {k: (chunks[k] if src == "fresh" else dc.decode(b)) for k, (src, b) in final.items()}
            skirted, kst = skirt_seams(meshes, sent)
            chunks = dict(chunks)
            for k, cm in skirted.items():
                chunks[k] = cm
                final[k] = ("fresh", dc.encode(cm))
            seam = {**(seam or {}), "skirt": kst}
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
                   glb_path=None, hooks=None, book=None):
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
    res = write_cbin(out_dir, blobs, chunks, book=book | occ, parent_dir=parent_dir, hooks=hooks)
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
