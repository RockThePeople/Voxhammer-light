"""편집 영역을 프리미티브 `mask.glb` 로 만든다 — 원저자 경로가 요구하는 형식.

원저자 `assets/example/mask.glb` 를 뜯어보면 **면 960개짜리 구 하나**다
(반지름 0.1207 · 중심 (-0.050, 0.300, 0.053), model.glb 과 같은 정규화 좌표계).
즉 3D 편집 소프트웨어가 필요한 게 아니라 **프리미티브면 충분하다.**

`inference.py` 는 `filter_method="volume"` 으로 이 메시의 **부피 안에 든** 64³ 격자
복셀을 편집 볼륨으로 삼는다 (`process_delete_ply`). 그래서 메시가 닫힌 부피이기만 하면 된다.

★ 마스크 설계 규칙 (작업 1 1차 실패에서 유도):
    "이 마스크 안에서 프롬프트를 만족시키는 **다른 답**이 있나?"
  1차는 마스크가 모자까지 덮었고, 인페인트는 **모자를 호박으로 갈아끼워** 프롬프트를
  만족시켰다. 머리는 그대로 남았다. 마스크가 과제를 잘못 표현하면 결과도 그렇게 나온다.

⚠️ 좌표계: trimesh 로 내보낸 GLB 를 같은 로더로 되읽어 bounds 가 의도와 맞는지
   **매번 검증**한다 (GLB 내보내기가 회전을 넣는 함정이 있다). 안 맞으면 죽는다.
"""
import argparse
import json
import os
import sys

import numpy as np
import trimesh

# process_delete_ply 는 preset_grid64.ply 를 상대경로로 찾는다 — VoxHammer 를 cwd 로 둔다.
VOXHAMMER = os.environ.get("VOXHAMMER_DIR", "<HOME>/ai-ar-v2/VoxHammer")
if VOXHAMMER not in sys.path:
    sys.path.insert(0, VOXHAMMER)


def build(center, radii, out_path, shape="ellipsoid", count=(32, 16)):
    c = np.asarray(center, float)
    r = np.asarray(radii, float)
    if shape == "box":
        m = trimesh.creation.box(extents=2 * r)
        m.apply_translation(c)
    else:
        m = trimesh.creation.uv_sphere(radius=1.0, count=list(count))
        m.apply_scale(r)
        m.apply_translation(c)
    m.export(out_path)

    back = trimesh.load(out_path)
    back = back.to_geometry() if hasattr(back, "to_geometry") else back
    lo, hi = c - r, c + r
    if not (np.allclose(back.bounds[0], lo, atol=3e-3) and np.allclose(back.bounds[1], hi, atol=3e-3)):
        sys.exit(f"🔴 좌표계가 어긋났다 — 기대 {np.round(lo,4)}..{np.round(hi,4)} "
                 f"실제 {np.round(back.bounds[0],4)}..{np.round(back.bounds[1],4)}")
    return m, back.bounds


def voxelize_and_report(mask_glb, model_glb, work_dir, label=""):
    """원저자 process_delete_ply 로 편집 볼륨을 뽑고, 자산과 겹쳐 통계를 낸다."""
    os.makedirs(work_dir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(VOXHAMMER)
    try:
        from voxhammer.delete_region_voxel import process_delete_ply
        process_delete_ply(mask_glb, work_dir)
    finally:
        os.chdir(cwd)

    V = np.asarray(trimesh.load(os.path.join(work_dir, "voxels_delete.ply")).vertices)
    M = set(map(tuple, np.floor((V + 0.5) * 64).astype(int)))

    g = trimesh.load(model_glb)
    g = g.to_geometry() if hasattr(g, "to_geometry") else g
    P, _ = trimesh.sample.sample_surface(g, 3_000_000)
    A = np.stack([P[:, 0], -P[:, 2], P[:, 1]], 1)          # D9: voxel = (x, -z, y)
    occ = set(map(tuple, np.clip(np.floor((A + 0.5) * 64).astype(int), 0, 63)))

    ins = occ & M
    idx = np.array(sorted(M))
    out = {
        "label": label,
        "edit_volume": len(M),
        "occupied_inside": len(ins),
        "empty_inside": len(M) - len(ins),
        "empty_ratio": round((len(M) - len(ins)) / max(len(M), 1), 4),
        "asset_voxels": len(occ),
        "asset_covered_ratio": round(len(ins) / max(len(occ), 1), 4),
        "grid_bbox_min": idx.min(0).tolist(),
        "grid_bbox_max": idx.max(0).tolist(),
    }
    print(f"  편집 볼륨 {out['edit_volume']:6d}  점유 {out['occupied_inside']:5d}  "
          f"빈칸 {100*out['empty_ratio']:.1f}%  자산의 {100*out['asset_covered_ratio']:.1f}% 를 덮는다")
    print(f"  격자 bbox {out['grid_bbox_min']} .. {out['grid_bbox_max']}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--center", required=True, help="x,y,z (model.glb 좌표계)")
    ap.add_argument("--radii", required=True, help="rx,ry,rz")
    ap.add_argument("--shape", default="ellipsoid", choices=["ellipsoid", "box"])
    ap.add_argument("--work", required=True, help="voxels_delete.ply 를 쓸 디렉터리")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    c = [float(x) for x in a.center.split(",")]
    r = [float(x) for x in a.radii.split(",")]
    _, b = build(c, r, a.out, shape=a.shape)
    print(f"mask.glb → {a.out}\n  중심 {c}  반지름 {r}  bounds {np.round(b[0],4)}..{np.round(b[1],4)}  좌표 왕복 OK")
    st = voxelize_and_report(a.out, a.model, a.work, a.label)
    json.dump(st, open(os.path.join(a.work, "mask_stats.json"), "w"), ensure_ascii=False, indent=2)
