"""00s S-5 트랙 A 마스크 초안 — 동결 SLat 격자(64³, 복셀 프레임 Z-up) 좌표.

시나리오마다 "대상 부위 점유 셀" 을 선택 상자로 고르고(아래 SEL, 실측 투영으로 정함),
그 점유 셀의 bbox + 여유 2셀을 64³ 안으로 자른 **상자 안 모든 셀(빈 칸 포함)** 을 마스크로 한다.
A-S1(정오표 E9)은 오른팔 옆 빈 공간을 명시 상자로 잡는다. 형식은 ai-ar-v3 `handoff/lasso/*.mask.json` 과 같다:
  mode=voxels, grid_source=slat_coords, halo_margin_voxels=2, meta.mask_fingerprint=deltacontract.mask_fingerprint.
조건 렌더용 mask.glb: 셀 합집합의 노출면(닫힌 표면)을 model.glb 와 같은 변환(VOXEL_TO_GLB → 정규화)으로 만든다.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "ai-ar-v3"))
sys.path.insert(0, str(ROOT / "third_party" / "ai-ar-v3" / "contract" / "python"))
from deltacontract import mask_fingerprint  # noqa: E402
from server.pipeline.frames import VOXEL_TO_GLB  # noqa: E402

RES, MARGIN, HALO = 64, 2, 2
ASSETS = Path(os.environ["ASSETS_DIR"]) / "track_a"

# 선택 상자 [lo, hi] (포함). 근거: SLat 점유 투영(보고서 S-5).
SEL = {
    "A-M1": ("moto", "라이더 머리", [(0, 29, 49), (63, 42, 63)], {}),
    "A-M2": ("moto", "뒷바퀴 전체", [(0, 42, 0), (63, 63, 27)], {}),
    # 정오표 E9: "a snowman holding a straw broom in its right hand". 눈사람 정면 = 코 방향 +y, 위 +z →
    # 오른쪽 = +x (오른손 좌표계). 오른팔 점유 x48–59·y30–31·z39–49, 손 끝 x59. 몸통 최대 x50.
    # 빈 공간 상자: 몸통 바깥 x51 ~ 손 끝+2 x61, 팔 y±(3–4), 바닥 z0 ~ 팔 높이 z49. 여유 2셀은 box_mask 가 더한다.
    "A-S1": ("snowman", "오른팔 옆 빈 공간(바닥~팔 높이, 손 끝 포함)", [(51, 27, 0), (61, 34, 49)], {"explicit_box": True}),
    "A-S2": ("snowman", "당근 코", [(28, 44, 50), (36, 63, 56)], {}),
    "A-C1": ("chair", "등받이", [(0, 44, 46), (63, 63, 63)], {}),
    "A-C2": ("chair", "양쪽 팔걸이", [[(0, 0, 36), (17, 43, 55)], [(46, 0, 36), (63, 43, 55)]], {}),
}


def load_coords(asset):
    from safetensors import safe_open
    with safe_open(str(ASSETS / asset / "gen" / "slat.safetensors"), "np") as f:
        return f.get_tensor("coords")[:, 1:].astype(np.int64)


def dense(lo, hi):
    r = [np.arange(lo[i], hi[i] + 1) for i in range(3)]
    return np.stack(np.meshgrid(*r, indexing="ij"), -1).reshape(-1, 3)


def box_mask(occ, sel, opt):
    lo, hi = np.array(sel[0]), np.array(sel[1])
    part = occ[np.all((occ >= lo) & (occ <= hi), axis=1)]
    if opt.get("explicit_box"):  # 빈 공간 위주: 선택 상자 자체가 대상 영역
        blo = np.clip(lo - MARGIN, 0, RES - 1)
        bhi = np.clip(hi + MARGIN, 0, RES - 1)
        return part, dense(blo, bhi), (blo.tolist(), bhi.tolist())
    if not len(part):
        raise SystemExit(f"선택 상자에 점유 셀이 없다: {sel}")
    blo = np.clip(part.min(0) - MARGIN, 0, RES - 1)
    bhi = np.clip(part.max(0) + MARGIN, 0, RES - 1)
    if "extend_hi_z" in opt:
        bhi[2] = opt["extend_hi_z"]
    return part, dense(blo, bhi), (blo.tolist(), bhi.tolist())


_DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def boundary_mesh(cells):
    occ = {tuple(c) for c in cells.tolist()}
    verts, faces = [], []
    for (i, j, k) in occ:
        for d in _DIRS:
            if (i + d[0], j + d[1], k + d[2]) in occ:
                continue
            ax = int(np.nonzero(d)[0][0])
            o = [i, j, k]
            o[ax] += 1 if d[ax] > 0 else 0
            u, v = [x for x in range(3) if x != ax]
            q = []
            for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                p = list(o)
                p[u] += du
                p[v] += dv
                q.append(p)
            if d[ax] < 0:
                q = q[::-1]
            if ax == 1:
                q = q[::-1]
            b = len(verts)
            verts.extend(q)
            faces.extend([(b, b + 1, b + 2), (b, b + 2, b + 3)])
    return np.asarray(verts, float) / RES - 0.5, np.asarray(faces, np.int64)


def main():
    import trimesh
    out = {}
    for sid, (asset, part_name, sel, opt) in SEL.items():
        occ = load_coords(asset)
        sels = sel if isinstance(sel[0], list) else [sel]
        parts, cells, boxes = [], [], []
        for s in sels:
            p, c, b = box_mask(occ, s, opt)
            parts.append(p); cells.append(c); boxes.append(b)
        cells = np.unique(np.concatenate(cells), axis=0)
        part = np.unique(np.concatenate(parts), axis=0)
        oset = {tuple(x) for x in occ.tolist()}
        n_occ_in = sum(tuple(x) in oset for x in cells.tolist())
        fp = mask_fingerprint(cells)
        gen = json.loads((ASSETS / asset / "gen" / "gen.json").read_text())
        nz = gen["glb_normalize"]
        doc = {"mask": {"mode": "voxels", "voxels": cells.tolist(), "halo_margin_voxels": HALO,
                        "grid_source": "slat_coords"},
               "meta": {"asset_id": f"track_a/{asset}", "scenario": sid, "n_cells": int(len(cells)),
                        "mask_fingerprint": fp, "part": part_name, "selection_boxes": sels,
                        "mask_boxes": boxes, "margin": MARGIN, "part_occupied_cells": int(len(part)),
                        "occupied_cells_in_mask": int(n_occ_in), "asset_occupied_cells": int(len(occ)),
                        "slat_sha256": gen["files_sha256"]["slat.safetensors"]}}
        d = ASSETS / asset / "masks"
        d.mkdir(exist_ok=True)
        (d / f"{sid}.mask.json").write_text(json.dumps(doc))
        v, f = boundary_mesh(cells)
        vg = (VOXEL_TO_GLB.apply(v) - np.array(nz["center_glb_frame"])) * nz["scale"]
        trimesh.Trimesh(vg, f, process=True).export(d / f"{sid}.mask.glb")
        out[sid] = {k: doc["meta"][k] for k in ("asset_id", "part", "n_cells", "mask_fingerprint", "mask_boxes",
                                                 "part_occupied_cells", "occupied_cells_in_mask",
                                                 "asset_occupied_cells")}
        out[sid]["occupied_ratio"] = round(n_occ_in / len(occ), 4)
        print(sid, json.dumps(out[sid]))
    (ROOT / "scenarios" / "track_a_masks.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
