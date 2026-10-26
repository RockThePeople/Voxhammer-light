"""X4 채점 (chief 보정 6, 지시서 main b8acf45): 각 경로를 자기 출발점과 비교 + 공통 기준(생성/데이터셋 텍스처 GLB) 절대값 + 0회차 값.

경로·회차 메시
  P0 r1..5 = $RUNS_DIR/paper/x4/p0/<자산>/r<k>/output.glb (포크 재등록 → to_glb, N 좌표). 출발점 = v0 텍스처 GLB(= 공통 기준).
  P1 r1..5 = $RUNS_DIR/paper/x4/p1/<자산>/r<k>/output.glb (저장 SLat → R4 .cbin → glbbuild 정점색, raw). 출발점 = A_origin SLat 의
           R4 .cbin(트랙 A gen/chunks, 트랙 B 02-r4/parents/<id>/chunks) → glbbuild GLB = r0.
측정: 무편집이라 마스크 없음 — 전체 영역 CD·NC·F1@0.01, PSNR·SSIM·LPIPS(전체 실루엣 합), 점유 IoU·소실·신규(A_origin SLat 대비, 직전 회차 대비).
결과 $RUNS_DIR/paper/x4/score/<자산>/<경로>_r<k>_<기준>.json, 요약 x4_score.json.

    python tools/paper/x4_score.py [자산 …]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party" / "ai-ar-v3")]
from tools.paper.x4 import assets  # noqa: E402

RUNS = Path(os.environ["RUNS_DIR"])
ASSETS = Path(os.environ["ASSETS_DIR"])
X4 = RUNS / "paper" / "x4"
OUT = X4 / "score"


def frame(name, a):
    if a["track"] == "A":
        g = json.loads((ASSETS / "track_a" / name / "gen" / "gen.json").read_text())["glb_normalize"]
        return g["center_glb_frame"], g["scale"]
    return [0, 0, 0], 1.0


def start_glb(name, a):
    d = X4 / "p1" / name / "r0"
    if not (d / "output.glb").exists():
        from server.glbbuild import build_glb
        src = ASSETS / "track_a" / name / "gen" / "chunks" if a["track"] == "A" else RUNS / "02-r4" / "parents" / name / "chunks"
        d.mkdir(parents=True, exist_ok=True)
        build_glb(src, d / "output.glb")
        (d / "source.json").write_text(json.dumps({"chunks": str(src)}))
    return d / "output.glb"


def occ(parent, child):
    from safetensors import safe_open
    from eval.rkeep import rkeep_metrics
    c = lambda p: safe_open(str(p), "np").get_tensor("coords")   # noqa: E731
    r = rkeep_metrics(c(parent), c(child), np.zeros((0, 3), np.int64))
    return {k: r[k] for k in ("rkeep_iou", "rkeep_lost", "rkeep_new", "rkeep_parent_occupied", "rkeep_child_occupied")}


def main(names):
    from tools.paper_metrics import score
    A = assets()
    summ = json.loads((OUT / "x4_score.json").read_text()) if (OUT / "x4_score.json").exists() else {}
    for name in names or list(A):
        a = A[name]
        c, s = frame(name, a)
        v0 = str(a["v0"])
        common = {"mesh": v0, "kind": "v0"}
        p1_start = {"mesh": str(start_glb(name, a)), "kind": "raw", "vertex_color": True}
        jobs = [("p1", 0, "common", p1_start, common)]
        for k in range(1, 6):
            jobs.append(("p0", k, "own=common", {"mesh": str(X4 / "p0" / name / f"r{k}" / "output.glb"), "kind": "n"}, common))
            p1 = {"mesh": str(X4 / "p1" / name / f"r{k}" / "output.glb"), "kind": "raw", "vertex_color": True}
            jobs += [("p1", k, "own", p1, p1_start), ("p1", k, "common", p1, common)]
        for path, k, ref_tag, res, ref in jobs:
            o = OUT / name / f"{path}_r{k}_{ref_tag}.json"
            if o.exists():
                summ.setdefault(name, {})[o.stem] = json.loads(o.read_text())
                continue
            job = {"asset": {"v0_mesh": v0, "raw_c": c, "raw_s": s, "v0_slat": str(a["slat"])}, "ref": ref, "res": res,
                   "excl_cells": [], "whole": False, "images": True,
                   "work_dir": str(OUT / name / o.stem), "cache_dir": str(OUT / "_cache")}
            r = score(job)
            slat_k = X4 / path / name / f"r{k}" / ("slat.safetensors" if path == "p0" else "slat.safetensors")
            if k >= 1 and slat_k.exists():
                r["occupancy_vs_origin"] = occ(a["slat"], slat_k)
                prev = a["slat"] if k == 1 else X4 / path / name / f"r{k - 1}" / "slat.safetensors"
                r["occupancy_vs_prev"] = occ(prev, slat_k)
            r["path"], r["round"], r["ref_tag"], r["res_mesh"], r["ref_mesh"] = path, k, ref_tag, res["mesh"], ref["mesh"]
            o.parent.mkdir(parents=True, exist_ok=True)
            o.write_text(json.dumps(r, indent=1))
            summ.setdefault(name, {})[o.stem] = r
            p, im = r["preserved"], r["image"]
            ov = r.get("occupancy_vs_origin", {})
            print(f"x4s {name} {path} r{k} {ref_tag} CD {p['cd']:.4f} NC {p['nc']:.3f} F1 {p['f1']:.3f} PSNR {im['psnr']:.2f} "
                  f"SSIM {im['ssim']:.4f} LPIPS {im['lpips']:.4f} IoU {ov.get('rkeep_iou', float('nan')):.4f} new {ov.get('rkeep_new')}", flush=True)
            (OUT / "x4_score.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
