"""moto-b 원저자 경로 입력 준비 (CPU 전용, GPU 미사용).

원저자 경로(`inference.py`)는 정규화된 `model.glb` 와 같은 좌표계의 `mask.glb` 를 받는다.
- `voxhammer/bpy_render.py:231-250` normalize_scene: 장면을 최대 변 1, bbox 중심 원점으로 맞춘다.
- `utils/render_rgb_and_mask.py:34` 조건 렌더는 정규화하지 않는다(`# Not normalize here`).
- `voxhammer/delete_region_voxel.py` process_delete_ply 는 mask.glb 를 그대로 64³ 격자에 대본다.
그래서 model.glb 를 미리 정규화해 두고(150뷰 렌더의 정규화는 항등에 가까워진다), mask.glb 를
같은 변환으로 만든다. 이렇게 하면 세 단계가 한 좌표계를 쓴다.

입력 (읽기 전용):
- ai-ar-v3 `assets/moto-b/parent/*.cbin` 376청크 → `server/glbbuild.cbin_dir_to_colored_mesh`
  (복셀 프레임 Z-up, 정규화 공간) → `frames.VOXEL_TO_GLB` 로 GLB(Y-up) 프레임.
- ai-ar-v3 `handoff/lasso/moto-b.*.mask.json` (grid_source=slat_coords, 64³ 셀).

mask.glb = 마스크 셀(정육면체) 합집합의 노출면만 모은 닫힌 표면. 셀 지문은
`deltacontract.mask_fingerprint` 로 다시 계산해 파일 meta 와 대조한다.

출력: `$ASSETS_DIR/moto-b/{model.glb, masks/<name>.mask.glb, prepare.json}`
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "third_party" / "ai-ar-v3"
sys.path.insert(0, str(V3))
sys.path.insert(0, str(V3 / "contract" / "python"))

from deltacontract import mask_fingerprint  # noqa: E402
from server.glbbuild import cbin_dir_to_colored_mesh  # noqa: E402
from server.pipeline.frames import VOXEL_TO_GLB  # noqa: E402

RES = 64
MASKS = ["rider-head", "rear-wheel", "rear-wheel-full"]
_DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cells_boundary_mesh(cells: np.ndarray):
    """셀 합집합의 노출면(이웃 셀이 없는 면)만 사각형 2개씩으로 낸다. 복셀 프레임, 정규화 공간."""
    occ = {tuple(c) for c in cells.tolist()}
    verts, faces = [], []
    for (i, j, k) in occ:
        for d in _DIRS:
            if (i + d[0], j + d[1], k + d[2]) in occ:
                continue
            ax = int(np.nonzero(d)[0][0])
            o = [i, j, k]
            o[ax] += 1 if d[ax] > 0 else 0
            u, v = [a for a in range(3) if a != ax]
            q = []
            for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                p = list(o)
                p[u] += du
                p[v] += dv
                q.append(p)
            if d[ax] < 0:
                q = q[::-1]
            if (ax == 1):                       # 오른손 법칙으로 바깥 법선이 되도록 축 순환 보정
                q = q[::-1]
            b = len(verts)
            verts.extend(q)
            faces.extend([(b, b + 1, b + 2), (b, b + 2, b + 3)])
    v = np.asarray(verts, np.float64) / RES - 0.5
    return v, np.asarray(faces, np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets_dir", default=os.environ.get("ASSETS_DIR"))
    a = ap.parse_args()
    import trimesh

    out = Path(a.assets_dir) / "moto-b"
    (out / "masks").mkdir(parents=True, exist_ok=True)
    rec = {"source": {"ai_ar_v3_commit": "19ed91f86adb11be6d7aad0c6a146de274923bd2"}}

    verts_vox, faces, colors = cbin_dir_to_colored_mesh(V3 / "assets" / "moto-b" / "parent")
    verts_glb = VOXEL_TO_GLB.apply(verts_vox)
    lo, hi = verts_glb.min(0), verts_glb.max(0)
    center = (lo + hi) / 2
    scale = 1.0 / float((hi - lo).max())
    norm = lambda p: (p - center) * scale  # noqa: E731

    model = trimesh.Trimesh(norm(verts_glb), faces, vertex_colors=colors, process=False)
    model.export(out / "model.glb")
    back = trimesh.load(out / "model.glb", force="mesh")
    rec["model"] = {
        "chunks": len(list((V3 / "assets" / "moto-b" / "parent").glob("*.cbin"))),
        "vertices": int(len(verts_vox)), "faces": int(len(faces)),
        "has_colors": colors is not None,
        "voxel_frame_bbox": [verts_vox.min(0).tolist(), verts_vox.max(0).tolist()],
        "normalize": {"center_glb": center.tolist(), "scale": scale},
        "reloaded_bounds": back.bounds.tolist(),
        "sha256": sha256(out / "model.glb"),
    }

    # 정점이 SLat 셀 안에 드는지(프레임 확인): 정점 → 셀 인덱스 → slat coords 집합 소속 비율
    slat_xyz = np.load(V3 / "handoff" / "lasso" / "moto-b.slat_coords.npy").astype(np.int64)
    sset = set(map(tuple, slat_xyz.tolist()))
    vidx = np.clip(np.floor((verts_vox + 0.5) * RES).astype(np.int64), 0, RES - 1)
    rec["model"]["vertex_cell_in_slat_coords_ratio"] = float(np.mean([tuple(x) in sset for x in vidx]))

    rec["masks"] = {}
    for name in MASKS:
        mp = V3 / "handoff" / "lasso" / f"moto-b.{name}.mask.json"
        d = json.loads(mp.read_text())
        cells = np.asarray(d["mask"]["voxels"], np.int64)
        fp = mask_fingerprint(cells)
        if fp != d["meta"]["mask_fingerprint"]:
            raise SystemExit(f"{name}: 셀 지문 불일치 {fp[:12]} != {d['meta']['mask_fingerprint'][:12]}")
        mv, mf = cells_boundary_mesh(cells)
        mesh = trimesh.Trimesh(norm(VOXEL_TO_GLB.apply(mv)), mf, process=True)
        dst = out / "masks" / f"{name}.mask.glb"
        mesh.export(dst)
        # 정규화 좌표계에서 다시 64³ 로 셀화했을 때의 예상 셀(원저자 경로 voxels_delete 와 대조용)
        centers = (cells + 0.5) / RES - 0.5
        cg = norm(VOXEL_TO_GLB.apply(centers))
        cb = VOXEL_TO_GLB.inverse().apply(cg)            # 정규화된 복셀 프레임 (bpy import 후 Z-up)
        re_cells = np.unique(np.floor((cb + 0.5) * RES).astype(np.int64), axis=0)
        rec["masks"][name] = {
            "file": mp.name, "n_cells": int(len(cells)), "fingerprint": fp,
            "fingerprint_matches_meta": True,
            "halo_margin_voxels_in_file": d["mask"].get("halo_margin_voxels"),
            "mesh_faces": int(len(mesh.faces)), "watertight": bool(mesh.is_watertight),
            "expected_cells_in_render_grid": int(len(re_cells)),
            "mask_glb_sha256": sha256(dst),
        }
    (out / "prepare.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    print(json.dumps(rec, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
