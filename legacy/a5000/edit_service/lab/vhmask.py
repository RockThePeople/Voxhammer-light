"""VoxHammer 트랙용 마스크 생성 — **CPU 전용 · GPU 0런**.

debug view 가 URL 파라미터로 규칙을 바꿔 부른다. 라쏘 없이 숫자로 조정하기 위한 것이다.

🔴 만든 마스크는 **반드시 파일로 남긴다.** W29 에서 실제 쓴 마스크가 안 남아 있어
   원래 실패의 근거에 유보가 붙었다(4단계). 같은 일을 반복하지 않는다.

좌표: voxels.ply 는 ply_to_coords 와 같은 규약 — coord = (pos + 0.5) * 64.
      실측(t1-snowman): **z 가 수직**, x 가 좌우(팔), y 가 깊이.
"""
import hashlib
import json
import os

import numpy as np

VOX = 64
NB6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def read_voxels(ply_path):
    import torch
    import utils3d
    pos = utils3d.io.read_ply(ply_path)[0]
    return ((torch.tensor(pos) + 0.5) * VOX).int().numpy().astype(np.int64)


def code(a):
    a = np.asarray(a, dtype=np.int64)
    return a[:, 0] * 4096 + a[:, 1] * 64 + a[:, 2]


def z_profile(xyz):
    return np.bincount(xyz[:, 2], minlength=VOX)


def auto_neck(xyz, search_lo=0.45, search_hi=0.92):
    """수직 프로파일의 **국소 최소**를 목으로 본다.

    윗부분만 본다 — 눈사람은 아래 잘록(몸통↔밑동)이 더 깊어서 전체 최소를 쓰면
    거기로 잡힌다. 비율 구간은 물체의 z 범위 기준이다.
    """
    p = z_profile(xyz).astype(float)
    nz = np.nonzero(p)[0]
    if not len(nz):
        return None
    lo, hi = nz.min(), nz.max()
    span = hi - lo
    a = int(lo + span * search_lo)
    b = int(lo + span * search_hi)
    if b <= a:
        return None
    seg = p[a:b + 1]
    return int(a + int(np.argmin(seg)))


def dilate(cells, halo):
    """6-이웃 팽창을 halo 회. 격자 밖은 버린다."""
    S = set(map(tuple, cells.tolist()))
    for _ in range(int(halo)):
        add = set()
        for x, y, z in S:
            for dx, dy, dz in NB6:
                p = (x + dx, y + dy, z + dz)
                if 0 <= p[0] < VOX and 0 <= p[1] < VOX and 0 <= p[2] < VOX and p not in S:
                    add.add(p)
        S |= add
    return np.array(sorted(S), dtype=np.int64) if S else np.zeros((0, 3), np.int64)


def largest_component(cells):
    """26-이웃 최대 연결 성분만 남긴다.

    z 임계만 쓰면 눈사람 **나뭇가지 팔 끝**처럼 위쪽에 걸친 딴 부위가 딸려 온다.
    머리는 한 덩어리이므로 최대 성분만 남기면 팔이 떨어진다.
    """
    if not len(cells):
        return cells
    S = set(map(tuple, cells.tolist()))
    nb = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
          if (a, b, c) != (0, 0, 0)]
    best = []
    while S:
        s = S.pop()
        st, comp = [s], [s]
        while st:
            x, y, z = st.pop()
            for dx, dy, dz in nb:
                p = (x + dx, y + dy, z + dz)
                if p in S:
                    S.discard(p)
                    st.append(p)
                    comp.append(p)
        if len(comp) > len(best):
            best = comp
    return np.array(sorted(best), dtype=np.int64)


def build(xyz, mode="neck", z_min=None, halo=0, active_only=True, pad_xy=0,
          largest=0):
    """규칙 → 마스크. 반환에 **선택 볼륨**과 **점유분**을 모두 담는다.

    번짐/빈 공간을 재려면 둘 다 필요하다 — 점유분만 보면 빈 칸이 정의상 0 이라
    아무것도 못 잰다(2단계에서 확인한 항진명제).
    """
    act = set(code(xyz).tolist())
    neck = auto_neck(xyz)
    if mode == "z":
        zc = int(z_min) if z_min is not None else (neck if neck is not None else VOX // 2)
    elif mode == "neck":
        zc = (int(z_min) if z_min is not None
              else ((neck + 1) if neck is not None else VOX // 2))
    else:
        raise ValueError(f"모르는 mode: {mode!r}")

    # 선택 볼륨: z >= zc 인 셀 전체 중, 물체의 xy 바운딩(+pad) 안쪽
    top = xyz[xyz[:, 2] >= zc]
    if not len(top):
        vol = np.zeros((0, 3), np.int64)
    else:
        x0, y0 = top[:, 0].min() - pad_xy, top[:, 1].min() - pad_xy
        x1, y1 = top[:, 0].max() + pad_xy, top[:, 1].max() + pad_xy
        gx, gy, gz = np.meshgrid(
            np.arange(max(0, x0), min(VOX, x1 + 1)),
            np.arange(max(0, y0), min(VOX, y1 + 1)),
            np.arange(zc, VOX), indexing="ij")
        vol = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1).astype(np.int64)

    cells = vol[np.isin(code(vol), np.fromiter(act, np.int64, len(act)))] if active_only else vol
    n_before_largest = int(len(cells))
    if largest:
        cells = largest_component(cells)
    if halo:
        cells = dilate(cells, halo)
        if active_only:
            cells = cells[np.isin(code(cells), np.fromiter(act, np.int64, len(act)))]

    occ_in_vol = int(np.isin(code(vol), np.fromiter(act, np.int64, len(act))).sum()) if len(vol) else 0
    empty = len(vol) - occ_in_vol
    bbox = ([int(v) for v in cells.min(0)] + [int(v) for v in cells.max(0)]) if len(cells) else None
    fp = hashlib.sha256(np.ascontiguousarray(np.sort(code(cells))).tobytes()).hexdigest()[:16] \
        if len(cells) else "empty"
    return {
        "cells": cells, "volume": vol,
        "params": {"mode": mode, "z_min": zc, "halo": int(halo),
                   "active_only": bool(active_only), "pad_xy": int(pad_xy),
                   "largest": int(largest)},
        "auto_neck": neck,
        "stats": {
            "cells": int(len(cells)),
            "volume_cells": int(len(vol)),
            "volume_occupied": occ_in_vol,
            "volume_empty": empty,
            "empty_pct": round(100 * empty / len(vol), 1) if len(vol) else None,
            # ★ 번짐 = 선택 볼륨 중 물체가 아닌 칸의 비율. 5-0 에서 눈으로 본 그것.
            "spill_pct": round(100 * empty / len(vol), 1) if len(vol) else None,
            "active_total": int(len(xyz)),
            "cells_pct_of_active": round(100 * len(cells) / len(xyz), 1) if len(xyz) else None,
            "dropped_by_largest": (n_before_largest - int(len(cells))) if largest else 0,
            "bbox": bbox,
            "dims": [bbox[3] - bbox[0] + 1, bbox[4] - bbox[1] + 1, bbox[5] - bbox[2] + 1]
                    if bbox else None,
            "fingerprint": fp,
        },
    }


def write_ply(path, cells):
    """ply_to_coords 가 읽는 규약 그대로 되돌린다: pos = coord/64 - 0.5."""
    import utils3d
    pos = (cells.astype(np.float32) / VOX) - 0.5
    utils3d.io.write_ply(path, vertices=pos)


def save(outdir, m, asset):
    os.makedirs(outdir, exist_ok=True)
    write_ply(os.path.join(outdir, "voxels_delete.ply"), m["cells"])
    meta = {"asset": asset, "params": m["params"], "auto_neck": m["auto_neck"],
            "stats": m["stats"]}
    with open(os.path.join(outdir, "mask.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def overlay_glb(asset_glb, cells, out_path, color=(220, 40, 40, 190)):
    """자산 메시 + 마스크 셀을 **복셀 큐브**로 얹은 GLB.

    debug 화면이라 큐브로 그린다 — 어디가 잡혔는지는 큐브가 제일 명확하다.
    (상위 프로젝트의 '큐브 금지' 는 사용자 앱 규칙이고 여기엔 해당하지 않는다)
    """
    import trimesh
    scene = trimesh.Scene()
    asset = trimesh.load(asset_glb, force="mesh")
    scene.add_geometry(asset, node_name="asset")
    if len(cells):
        s = 1.0 / VOX
        box = trimesh.creation.box(extents=(s, s, s))
        # voxel(x,y,z) → GLB(Y-up): CLAUDE.md GLB_TO_VOXEL = (x, -z, y) 의 역
        centers = (cells.astype(np.float64) + 0.5) / VOX - 0.5
        gx, gy, gz = centers[:, 0], centers[:, 2], -centers[:, 1]
        tf = np.tile(np.eye(4), (len(cells), 1, 1))
        tf[:, 0, 3], tf[:, 1, 3], tf[:, 2, 3] = gx, gy, gz
        cubes = trimesh.util.concatenate(
            [box.copy().apply_transform(t) for t in tf])
        # 정점색만 넣으면 GLB 로 나갈 때 기본 흰 머티리얼로 보인다 —
        # PBR baseColorFactor 를 직접 준다.
        cubes.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=[c / 255.0 for c in color],
                metallicFactor=0.0, roughnessFactor=0.85,
                alphaMode="BLEND" if color[3] < 255 else "OPAQUE"))
        scene.add_geometry(cubes, node_name="mask")
    scene.export(out_path)
    return out_path
