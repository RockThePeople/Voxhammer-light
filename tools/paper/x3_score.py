"""X3 채점 (ops/tasks/X-paper.md X3 측정 + chief 보정 6, 지시서 main b8acf45).

회차 k = 1…5, 경로 P0·P1·P2 마다:
  own    : 자기 출발점 대비(P0 = 생성 텍스처 GLB, P1·P2 = A_origin SLat 의 R4 .cbin → GLB = p1/<자산>/r0)
  common : 공통 기준(생성 텍스처 GLB) 대비 절대값
  prev   : 직전 회차 대비
  영역   : 1…k 회차 마스크 합집합 ∪ halo 2 밖(누적). CD·NC·F1@0.01, PSNR·SSIM·LPIPS(M).
  편집   : 그 회차 편집 시점 DINO-I(자기 2d_edit)·CLIP-T(프롬프트), 두 정의(paper_metrics).
  점유   : A_origin SLat 대비 누적 마스크 밖 R_keep IoU·소실·신규(경로마다), 직전 회차 대비 신규.
  0회차  : 각 출발점 vs 공통 기준(P0 는 0).
P0 출력은 포크 정규화 좌표(round.json C_k = v0 → 그 좌표) → C_k⁻¹ 로 v0 에 되돌린 GLB(output_v0.glb)로 채점하고,
P0 slat_tgt(정규화 격자)의 셀은 중심을 C_k⁻¹ 로 v0 → raw 격자로 옮겨 점유를 센다.

    python tools/paper/x3_score.py [자산 …]     # $RUNS_DIR/paper/x3/score/<자산>/<경로>_r<k>_<기준>.json, 요약 x3_score.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party" / "ai-ar-v3"), str(ROOT / "third_party" / "ai-ar-v3" / "contract" / "python")]
from tools.paper.x3 import ASSET_NAMES, MASKS, OUT, ROUNDS, gen, round_info, transform_glb  # noqa: E402

SC = OUT / "score"
CACHE = OUT / "_cache"


def frame(asset):
    from tools.paper_metrics import AssetFrame
    g, nz = gen(asset)
    return AssetFrame.from_v0(g / "model.glb", nz["center_glb_frame"], nz["scale"])


def mesh_spec(path, asset, k):
    """회차 결과 메시 → paper_metrics 메시 명세. k = 0 은 출발점."""
    g, _ = gen(asset)
    if path == "common" or (path == "p0" and k == 0):
        return {"mesh": str(g / "model.glb"), "kind": "v0"}
    if path == "p0":
        d = OUT / "p0" / asset / f"r{k}"
        v0 = d / "output_v0.glb"
        if not v0.exists():
            C = np.asarray(json.loads((d / "round.json").read_text())["C"])
            transform_glb(d / "output.glb", np.linalg.inv(C), v0)
        return {"mesh": str(v0), "kind": "v0"}
    base = OUT / ("p1" if k == 0 else path) / asset / f"r{k}"
    return {"mesh": str(base / "output.glb"), "kind": "raw", "vertex_color": True}


def result_views(path, asset, k, edit_view):
    """Bench 16시점 렌더(paper_metrics 캐시와 같은 키)에서 포크 편집 시점·3/4 시점(편집 시점이 1 이면 3, 아니면 1) 두 장."""
    import hashlib
    from tools.paper_metrics import FORK_TO_BENCH, bench_render, n_frame_glb
    fr = frame(asset)
    spec = mesh_spec(path, asset, k)
    key = Path(spec["mesh"]).resolve().as_posix() + spec["kind"] + str(spec.get("vertex_color")) + json.dumps(fr.as_dict())
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    glb = n_frame_glb(spec["mesh"], fr, spec["kind"], spec.get("vertex_color", False), CACHE / "glb" / f"{h}.glb")
    d = bench_render(glb, CACHE / "renders" / h)
    v34 = 3 if edit_view == 1 else 1
    return d / f"render_{FORK_TO_BENCH[edit_view]:04d}.png", d / f"render_{FORK_TO_BENCH[v34]:04d}.png"


def cum_mask(asset, k):
    cells = set()
    for j in range(1, k + 1):
        cells |= {tuple(c) for c in json.loads((MASKS / asset / f"r{j}" / "mask.json").read_text())["mask"]["voxels"]}
    return np.array(sorted(cells), np.int64).reshape(-1, 3)


def slat_cells(path, asset, k):
    """회차 SLat 셀(raw 64³ 격자). P0 는 정규화 격자 → v0 → raw 로 옮긴다."""
    from safetensors import safe_open
    from tools.paper_metrics import glb_to_voxel, voxel_to_glb
    g, nz = gen(asset)
    if k == 0:
        p = g / "slat.safetensors"
    else:
        p = OUT / path / asset / f"r{k}" / "slat_tgt.safetensors"
    if not p.exists():
        return None
    c = safe_open(str(p), "np").get_tensor("coords")[:, 1:].astype(np.int64)
    if path != "p0" or k == 0:
        return c
    C = np.asarray(json.loads((OUT / "p0" / asset / f"r{k}" / "round.json").read_text())["C"])
    q = voxel_to_glb((c + 0.5) / 64 - 0.5)                       # 정규화 격자 셀 중심(그 좌표의 glTF)
    q = (np.linalg.inv(C) @ np.c_[q, np.ones(len(q))].T).T[:, :3]   # → v0
    q = q / nz["scale"] + np.asarray(nz["center_glb_frame"])       # → raw
    return np.unique(np.clip(np.floor((glb_to_voxel(q) + 0.5) * 64).astype(np.int64), 0, 63), axis=0)


def occupancy(parent, child, mask):
    from eval.rkeep import rkeep_metrics
    z = lambda a: np.c_[np.zeros(len(a), np.int64), a]   # noqa: E731  rkeep 는 (N,4) 좌표를 받는다
    r = rkeep_metrics(z(parent), z(child), mask)
    return {k: r[k] for k in ("rkeep_iou", "rkeep_lost", "rkeep_new", "edit_new", "edit_lost")}


def main(names):
    from tools.paper_metrics import score
    g0 = {}
    summ = json.loads((SC / "x3_score.json").read_text()) if (SC / "x3_score.json").exists() else {}
    for a in names or ASSET_NAMES:
        g, nz = gen(a)
        fr_asset = {"v0_mesh": str(g / "model.glb"), "raw_c": nz["center_glb_frame"], "raw_s": nz["scale"], "v0_slat": str(g / "slat.safetensors")}
        origin = slat_cells("p1", a, 0)
        jobs = [("p1", 0, "common", [])]
        for k in range(1, ROUNDS + 1):
            for path in ("p0", "p1", "p2"):
                if not (OUT / path / a / f"r{k}" / "round.json").exists():
                    continue
                jobs += [(path, k, "own", None), (path, k, "prev", None)]
                if path != "p0":
                    jobs.append((path, k, "common", None))
        for path, k, ref, _ in jobs:
            o = SC / a / f"{path}_r{k}_{ref}.json"
            if o.exists():
                summ.setdefault(a, {})[o.stem] = json.loads(o.read_text())
                continue
            mask = cum_mask(a, k) if k else np.zeros((0, 3), np.int64)
            refspec = (mesh_spec("common", a, 0) if ref == "common" else
                       mesh_spec(path, a, 0) if ref == "own" else mesh_spec(path, a, k - 1))
            job = {"asset": fr_asset, "ref": refspec, "res": mesh_spec(path, a, k), "excl_cells": mask.tolist(),
                   "work_dir": str(SC / a / o.stem), "cache_dir": str(CACHE)}
            if k and ref == "own":
                _, info = round_info(a, k)
                job.update(edit_view=info["edit_view"], edit_image=str(OUT / path / a / f"r{k}" / "2d_edit.png"), prompt=info["prompt"])
            r = score(job)
            if k:
                cur = slat_cells(path, a, k)
                prev = slat_cells(path, a, k - 1) if (path != "p0" or k > 1) else origin
                if cur is not None:
                    r["occupancy_vs_origin"] = occupancy(origin, cur, mask)
                    if prev is not None:
                        r["occupancy_vs_prev"] = occupancy(prev, cur, mask)
            r.update(path=path, round=k, ref_tag=ref, res_mesh=job["res"]["mesh"], ref_mesh=refspec["mesh"], cum_mask_cells=int(len(mask)))
            o.parent.mkdir(parents=True, exist_ok=True)
            o.write_text(json.dumps(r, indent=1))
            summ.setdefault(a, {})[o.stem] = r
            p, im = r["preserved"], r["image"]
            ov = r.get("occupancy_vs_origin", {})
            print(f"x3s {a} {path} r{k} {ref} CD {p['cd']:.4f} F1 {p['f1']:.3f} PSNR {im['psnr']:.2f} LPIPS {im['lpips']:.4f} "
                  f"new_out {ov.get('rkeep_new')} DINO {im.get('dino_i')} CLIP {im.get('clip_t')}", flush=True)
            SC.mkdir(parents=True, exist_ok=True)
            (SC / "x3_score.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
