"""표면 셀 기준 점유 IoU (chief 21:4x 4 — 논문용 주의).

기존 R_keep IoU 는 생성 SLat 의 내부 셀까지 세서, 재등록(표면만 복셀화) 경로인 P0·P2 에 불리하다(오두막 생성 48.6k 셀 → 재등록 23.9k).
여기서는 A_origin 메시(생성 텍스처 GLB, to_glb 바깥 표면) 표면에 닿는 64³ 셀 S 만 센다:
  IoU_surf = |A_origin ∩ 결과 ∩ S ∩ R_keep| / |(A_origin ∪ 결과) ∩ S ∩ R_keep|,  R_keep = 누적 마스크 ∪ halo 2 밖.
S = A_origin 표면 2,000,000 점(면적 비례, seed 0)이 든 셀.

    python tools/paper/surface_iou.py x3 [자산 …]    # $RUNS_DIR/paper/x3/score/surface_iou.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party" / "ai-ar-v3" / "contract" / "python")]


def surface_cells(v0_mesh, frame, n=2_000_000):
    from tools.paper_metrics import mesh_in_n, sample
    pts, _ = sample(mesh_in_n(v0_mesh, frame, "v0"), n=n)
    c = frame.points_to_cells(pts)
    c = c[((c >= 0) & (c < 64)).all(1)]
    return np.unique(c[:, 0] * 4096 + c[:, 1] * 64 + c[:, 2])


def iou_surf(parent, child, S, excl_codes):
    code = lambda a: np.unique(a[:, 0] * 4096 + a[:, 1] * 64 + a[:, 2])   # noqa: E731
    keep = np.setdiff1d(S, excl_codes)
    p, c = np.intersect1d(code(parent), keep), np.intersect1d(code(child), keep)
    u = np.union1d(p, c).size
    return {"iou_surf": float(np.intersect1d(p, c).size / u) if u else 1.0, "surf_cells": int(keep.size),
            "parent_surf": int(p.size), "child_surf": int(c.size)}


def x3(assets):
    from tools.paper.x3 import OUT
    from tools.paper.x3_score import cum_mask, frame, gen, slat_cells
    from tools.paper_metrics import excluded_set
    out_p = OUT / "score" / "surface_iou.json"
    res = json.loads(out_p.read_text()) if out_p.exists() else {}
    for a in assets:
        fr = frame(a)
        g, _ = gen(a)
        S = surface_cells(g / "model.glb", fr)
        parent = slat_cells("p1", a, 0)
        res.setdefault(a, {})["S_cells"] = int(S.size)
        for p in ("p0", "p1", "p2"):
            for k in range(1, 6):
                child = slat_cells(p, a, k)
                if child is None:
                    continue
                r = iou_surf(parent, child, S, excluded_set(cum_mask(a, k)))
                res[a][f"{p}_r{k}"] = r
                print(f"surf {a} {p} r{k} iou_surf {r['iou_surf']:.4f} (S∩R_keep {r['surf_cells']})", flush=True)
        out_p.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "x3":
        x3(sys.argv[2:] or ["jeep", "cottage", "adventurer"])
