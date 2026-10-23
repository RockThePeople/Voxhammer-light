"""부기 기반 청크 합성 + 이음새 측정 (계약 3.4.0).

해시를 비교하지 않는다. **부기(마스크+halo)가 무엇이 바뀌었는지를 정의**하고,
부기 밖 청크는 새로 디코딩된 결과를 버리고 부모 바이트를 그대로 승계한다.

그래서 산출물은 "옛 몸통 + 새 머리"가 되고, 매니페스트에 서로 다른 버전의 청크가
섞이는 게 정상이다. 대가는 하나뿐이다 — **옛 바이트와 새 바이트가 만나는 면의
이음새**. 이 모듈이 그걸 잰다.
"""
import os

import numpy as np
from scipy.spatial import cKDTree

from deltacontract import decode, parse_chunk_key

MESH_CELL = 1.0 / 256.0        # 메시 셀 한 변
CHUNK_SPAN = 8.0 / 64.0        # 청크 한 변 (NORMALIZED)
# 대응 정점으로 인정할 최대 거리. 이보다 멀면 "짝이 없는 정점"으로 따로 센다.
MATCH_RADIUS = 4 * MESH_CELL


def adjacent(a, b):
    """두 청크 키가 26-이웃으로 인접한가 (면·모서리·꼭짓점 공유)."""
    ca, cb = np.asarray(parse_chunk_key(a)), np.asarray(parse_chunk_key(b))
    d = np.abs(ca - cb)
    return d.max() == 1


def seam_pairs(book, present):
    """이음새가 생기는 (부기 청크, 비-부기 청크) 인접 쌍."""
    outside = [k for k in present if k not in book]
    return [(i, o) for i in book if i in present for o in outside if adjacent(i, o)]


def chunk_box(key):
    c = np.asarray(parse_chunk_key(key), dtype=np.float64)
    lo = c * CHUNK_SPAN - 0.5
    return lo, lo + CHUNK_SPAN


def shared_face_verts(pos, key_self, key_other, tol=MESH_CELL):
    """두 청크의 공유 경계에 tol 이내로 붙어 있는 정점 마스크.

    26-이웃이므로 공유 요소가 면/모서리/꼭짓점 중 하나다. 자기 AABB 면 중
    상대 방향의 면에 붙은 정점만 고른다.
    """
    a = np.asarray(parse_chunk_key(key_self))
    b = np.asarray(parse_chunk_key(key_other))
    lo, hi = chunk_box(key_self)
    keep = np.ones(len(pos), bool)
    for ax in range(3):
        d = b[ax] - a[ax]
        if d > 0:
            keep &= np.abs(pos[:, ax] - hi[ax]) <= tol
        elif d < 0:
            keep &= np.abs(pos[:, ax] - lo[ax]) <= tol
    return keep


def measure_seams(new_dir, old_dir, book, present, parent_present):
    """이음새 정량화.

    새 바이트를 쓰는 청크(부기 안)와 옛 바이트를 쓰는 청크(부기 밖)가 인접한
    면에서, 양쪽 정점이 서로 얼마나 어긋나는지 잰다. 최근접 **정점** 거리를
    쓴다 — 표면 최근접점의 상계다.
    """
    cache = {}

    def load(d, k):
        p = os.path.join(d, "chunks", f"{k}.cbin")
        if (d, k) not in cache:
            cache[(d, k)] = (decode(open(p, "rb").read()).positions.astype(np.float64)
                             if os.path.exists(p) else None)
        return cache[(d, k)]

    dists, n_faces, unmatched, total_b = [], 0, 0, 0
    for inside, outside in seam_pairs(book, present):
        # inside 는 새 바이트(new_dir), outside 는 승계된 옛 바이트(old_dir).
        if outside not in parent_present:
            continue          # 새로 활성화된 청크 — 승계할 옛 바이트가 없다
        pi = load(new_dir, inside)
        po = load(old_dir, outside)
        if pi is None or po is None or len(pi) == 0 or len(po) == 0:
            continue
        mi = shared_face_verts(pi, inside, outside)
        mo = shared_face_verts(po, outside, inside)
        if not mi.any() or not mo.any():
            continue
        n_faces += 1
        # ⚠️ 대응 정점이 있는 쌍만 잰다.
        #
        # partition_mesh 는 청크 경계를 **삼각형 무게중심**으로 가르므로, 경계면에
        # 걸친 정점은 양쪽 청크에 같은 좌표로 복제된다. 연속인 메시라면 그 복제쌍의
        # 거리가 정확히 0 이다. 그런데 경계면 근처지만 **한쪽에만 존재하는** 정점도
        # 있고(표면이 그 면을 실제로 넘지 않는 곳), 그런 정점까지 최근접 거리를 재면
        # 이음새가 아니라 국소 정점 간격을 재게 된다.
        #
        # 대조군 Z(편집 없이 재디코딩, 바이트 완전 동일)에서 이 지표가 max 7.72 셀을
        # 뱉은 게 그 증거다 — 실제 불연속이 0인데 7.72 가 나왔다. 그래서 대응 반경
        # 안에 짝이 있는 정점만 세고, 짝 없는 정점은 따로 보고한다.
        for a, b in ((pi[mi], po), (po[mo], pi)):
            d, _ = cKDTree(b).query(a)
            total_b += d.size
            hit = d <= MATCH_RADIUS
            unmatched += int((~hit).sum())
            if hit.any():
                dists.append(d[hit])

    base = {"seam_faces": n_faces, "boundary_vertices": total_b,
            "unmatched": unmatched,
            "unmatched_ratio": (unmatched / total_b) if total_b else None}
    if not dists:
        return {**base, "matched": 0, "max": None, "median": None, "p99": None,
                "cells_max": None, "cells_median": None,
                "over_cell": 0, "over_cell_ratio": None}

    d = np.concatenate(dists)
    return {
        **base,
        "matched": int(d.size),
        "max": float(d.max()),
        "median": float(np.median(d)),
        "p99": float(np.percentile(d, 99)),
        "cells_max": float(d.max() / MESH_CELL),
        "cells_median": float(np.median(d) / MESH_CELL),
        "over_cell": int((d > MESH_CELL).sum()),
        "over_cell_ratio": float((d > MESH_CELL).mean()),
    }


def seam_vertices(pos, book, present, parent_present):
    """이음새 면 위 정점 마스크. delta 렌더에 자홍색으로 찍기 위한 것.

    숫자가 작아도 눈에 보이면 문제고, 커도 안 보이면 문제가 아니다.
    """
    from deltacontract import chunk_key, normalized_to_chunk

    keys = np.array([chunk_key(c) for c in normalized_to_chunk(pos)])
    mark = np.zeros(len(pos), bool)
    for inside, outside in seam_pairs(book, present):
        if outside not in parent_present:
            continue
        for k, other in ((inside, outside), (outside, inside)):
            sel = keys == k
            if not sel.any():
                continue
            sub = np.flatnonzero(sel)
            mark[sub[shared_face_verts(pos[sub], k, other)]] = True
    return mark


def snap_boundary(new_dir, old_dir, book, present, parent_present, cells=2.0):
    """부기 청크의 경계 정점을 부모의 대응 정점으로 스냅한다.

    ⚠️ **새 청크 쪽만 건드린다.** 부모 청크는 이미 클라이언트가 갖고 있어서 못 고친다.
    이게 이 방식의 구조적 제약이다.

    위상 불일치(T-junction)는 이걸로 안 고쳐진다. 다만 눈에 보이는 건 틈이지
    T-junction 자체가 아니므로, 틈만 닫아도 시각적으로 충분할 수 있다.

    임계값 밖(= 위상이 실제로 달라진 곳)은 스냅하지 않는다.
    """
    from deltacontract import canonicalize, decode, encode, parse_chunk_key

    thr = cells * MESH_CELL
    pairs = {}
    for inside, outside in seam_pairs(book, present):
        if outside in parent_present:
            pairs.setdefault(inside, []).append(outside)

    n_snapped, moved, n_cand = 0, [], 0
    for inside, outs in pairs.items():
        p = os.path.join(new_dir, "chunks", f"{inside}.cbin")
        if not os.path.exists(p):
            continue
        m = decode(open(p, "rb").read())
        pos = m.positions.astype(np.float64)

        # 상대편(부모) 경계 정점을 모은다.
        ref, sel_any = [], np.zeros(len(pos), bool)
        for o in outs:
            po = os.path.join(old_dir, "chunks", f"{o}.cbin")
            if not os.path.exists(po):
                continue
            q = decode(open(po, "rb").read()).positions.astype(np.float64)
            ref.append(q[shared_face_verts(q, o, inside)])
            sel_any |= shared_face_verts(pos, inside, o)
        ref = [r for r in ref if len(r)]
        if not ref or not sel_any.any():
            continue
        ref = np.concatenate(ref)

        idx = np.flatnonzero(sel_any)
        n_cand += idx.size
        d, j = cKDTree(ref).query(pos[idx])
        hit = d <= thr
        if not hit.any():
            continue
        moved.append(d[hit])
        n_snapped += int(hit.sum())
        pos[idx[hit]] = ref[j[hit]]

        # 위치가 바뀌었으니 정렬 키도 바뀐다 — 반드시 재정규화 후 인코딩한다.
        cm = canonicalize(
            chunk_coord=parse_chunk_key(inside),
            positions=pos.astype(np.float32), indices=m.indices,
            normals=m.normals, colors=m.colors, uvs=m.uvs,
            voxel_count=m.voxel_count,
        )
        with open(p, "wb") as f:
            f.write(encode(cm))

    mv = np.concatenate(moved) if moved else np.zeros(0)
    return {
        "snap_cells": cells,
        "candidates": int(n_cand),
        "snapped": int(n_snapped),
        "snapped_ratio": float(n_snapped / n_cand) if n_cand else None,
        "move_mean": float(mv.mean()) if mv.size else 0.0,
        "move_max": float(mv.max()) if mv.size else 0.0,
        "move_mean_cells": float(mv.mean() / MESH_CELL) if mv.size else 0.0,
    }
