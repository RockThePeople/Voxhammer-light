"""X2-A·X2-B 채점 (ops/tasks/X-paper.md X2, 사용자 23:18 X2-A P0 2회 축소, 23:10 X2-B 3건).

a    : X2-A P0(r1·r2) vs v0, tools/paper_metrics.score(X0 과 같은 정의·마스크·편집 시점). P0 출력은 T(v0) = N 좌표(kind n).
       P1 은 새로 채점하지 않는다 — X0 의 E2a c15 r1–r3(P1-glb = c15-glb, P1-cbin = c15-cbin-st 봉합본, 봉합 전 c15-cbin 참고).
tost : 시나리오별 회차 평균(P0 2회, P1 3회)의 쌍 차이에 **단측 비열등성 검정**(chief 보정 7, 지시서 main f97aee7) — 양측 TOST 가 아니다.
       좋은 방향으로 맞춘 g = good·(P1 − P0), H0: g ≤ −한계(E19), H1: g > −한계, 단측 t, n = 14, α = 0.05.
       우위(H1: g > 0) p 값도 함께. 개선 칸 수도 센다. 결과 $RUNS_DIR/paper/x2a/score/tost.json(이름은 그대로)
b    : X2-B 3건 P0·P1(봉합) vs 데이터셋 원본 GLB.

    python tools/paper/x2_score.py a | tost | b
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party" / "ai-ar-v3" / "contract" / "python")]
RUNS = Path(os.environ["RUNS_DIR"])
ASSETS = Path(os.environ["ASSETS_DIR"])
X2A = RUNS / "paper" / "x2a"
X2B = RUNS / "paper" / "x2b"
X0 = RUNS / "paper" / "x0"
P0_RUNS = ("r1", "r2")
# E19 절대 허용치(지시서): 지표 → (한계, 좋은 방향 +1 = 클수록 좋음)
MARGINS = {"psnr": (0.5, 1), "ssim": (0.003, 1), "lpips": (0.005, -1), "dino_i": (0.015, 1), "clip_t": (0.015, 1), "cd": (0.002, -1)}


def p0_cells(slat_path, T, raw_c, raw_s):
    """P0 slat_tgt(정규화 격자) 셀 → raw 64³ (셀 중심을 T⁻¹ 로 v0, 다시 raw)."""
    from safetensors import safe_open
    from tools.paper_metrics import glb_to_voxel, voxel_to_glb
    c = safe_open(str(slat_path), "np").get_tensor("coords")[:, 1:].astype(np.int64)
    q = voxel_to_glb((c + 0.5) / 64 - 0.5)
    q = (np.linalg.inv(T) @ np.c_[q, np.ones(len(q))].T).T[:, :3]
    q = q / raw_s + np.asarray(raw_c)
    return np.unique(np.clip(np.floor((glb_to_voxel(q) + 0.5) * 64).astype(np.int64), 0, 63), axis=0)


def occ(parent, child, mask):
    from eval.rkeep import rkeep_metrics
    z = lambda a: np.c_[np.zeros(len(a), np.int64), a]   # noqa: E731
    r = rkeep_metrics(z(parent), z(child), mask)
    return {k: r[k] for k in ("rkeep_iou", "rkeep_lost", "rkeep_new", "edit_new", "edit_lost")}


def score_a():
    from safetensors import safe_open
    from tools.paper.x0 import scenarios
    from tools.paper_metrics import score
    S = scenarios()
    out_p = X2A / "score" / "x2a_p0.json"
    res = json.loads(out_p.read_text()) if out_p.exists() else {}
    for sid, s in S.items():
        T = np.asarray(json.loads((X2A / "p0" / sid / "prep.json").read_text())["T"])
        cells = json.loads(Path(s["mask"]).read_text())["mask"]["voxels"]
        parent = safe_open(str(s["slat"]), "np").get_tensor("coords")[:, 1:].astype(np.int64)
        for r in P0_RUNS:
            d = X2A / "p0" / sid / r
            key = f"{sid}/{r}"
            if key in res or not (d / "round.json").exists():
                continue
            job = {"asset": {"v0_mesh": str(s["v0"]), "raw_c": s["raw_c"], "raw_s": s["raw_s"], "v0_slat": str(s["slat"])},
                   "ref": {"mesh": str(s["v0"]), "kind": "v0"}, "res": {"mesh": str(d / "output.glb"), "kind": "n"},
                   "excl_cells": cells, "edit_view": s["view"], "edit_image": str(s["cond"] / "2d_edit.png"), "prompt": s["prompt"],
                   "work_dir": str(X2A / "score" / sid / r), "cache_dir": str(X0 / "_cache")}
            o = score(job)
            o["occupancy"] = occ(parent, p0_cells(d / "slat_tgt.safetensors", T, s["raw_c"], s["raw_s"]), np.asarray(cells))
            res[key] = o
            p, im = o["preserved"], o["image"]
            print(f"x2a p0 {sid} {r} CD {p['cd']:.4f} F1 {p['f1']:.3f} PSNR {im['psnr']:.2f} LPIPS {im['lpips']:.4f} "
                  f"DINO {im.get('dino_i'):.3f} CLIP {im.get('clip_t'):.3f} IoU {o['occupancy']['rkeep_iou']:.3f}", flush=True)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(res, indent=1))


def metric(r, m):
    return r["preserved"][m] if m == "cd" else r["image"].get(m)


def tost():
    from scipy import stats
    from tools.paper.x0 import scenarios
    S = list(scenarios())
    p0 = json.loads((X2A / "score" / "x2a_p0.json").read_text())
    x0 = json.loads((X0 / "x0_runs.json").read_text())
    x0s = json.loads((X0 / "x0_runs_st.json").read_text()) if (X0 / "x0_runs_st.json").exists() else {}
    out = {"note": "d = P1 − P0, 시나리오별 회차 평균(P0 r1·r2, P1 E2a c15 r1–r3). 단측 비열등성(chief 보정 7): H0 good·d ≤ −m, "
                   "p < 0.05 면 비열등. 우위 p = H0 good·d ≤ 0. 개선 = 좋은 방향으로 한계를 넘은 시나리오 수.", "n": len(S), "p0_runs": list(P0_RUNS), "p1_runs": ["r1", "r2", "r3"]}
    for p1cfg, src in (("P1-glb", ("c15-glb", x0)), ("P1-cbin(봉합)", ("c15-cbin-st", x0s)), ("P1-cbin(봉합 전, 참고)", ("c15-cbin", x0))):
        cfg, J = src
        rows = {}
        for m, (mg, good) in MARGINS.items():
            d, a_m, b_m, rng0 = [], [], [], []
            for sid in S:
                a = [metric(p0[f"{sid}/{r}"], m) for r in P0_RUNS if f"{sid}/{r}" in p0]
                b = [metric(J[sid][r][cfg], m) for r in ("r1", "r2", "r3") if sid in J and r in J[sid] and cfg in J[sid][r]]
                if not a or not b or None in a or None in b:
                    continue
                a_m.append(np.mean(a)); b_m.append(np.mean(b)); d.append(np.mean(b) - np.mean(a)); rng0.append(max(a) - min(a))
            if len(d) < 3:
                continue
            d = np.asarray(d)
            g = good * d                                            # 좋은 방향 기준 차이(> 0 이면 P1 이 좋다)
            p_ni = stats.ttest_1samp(g, -mg, alternative="greater").pvalue   # 비열등: H0 g ≤ −m
            p_sup = stats.ttest_1samp(g, 0.0, alternative="greater").pvalue  # 우위: H0 g ≤ 0
            ci = stats.t.interval(0.90, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
            rows[m] = {"n": int(len(d)), "margin": mg, "good_direction": "higher" if good > 0 else "lower",
                       "p0_mean": float(np.mean(a_m)), "p1_mean": float(np.mean(b_m)),
                       "d_mean": float(d.mean()), "ci90": [float(ci[0]), float(ci[1])],
                       "p_noninferiority": float(p_ni), "noninferior": bool(p_ni < 0.05),
                       "p_superiority": float(p_sup), "superior": bool(p_sup < 0.05),
                       "p1_better_beyond_margin": int((g > mg).sum()), "p1_worse_beyond_margin": int((g < -mg).sum()),
                       "p0_range_mean": float(np.mean(rng0))}
        out[p1cfg] = rows
        print(p1cfg, {m: (round(v["d_mean"], 4), v["noninferior"], round(v["p_superiority"], 4)) for m, v in rows.items()}, flush=True)
    (X2A / "score" / "tost.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


def score_b():
    from tools.paper_metrics import score
    items = json.loads((ROOT / "scenarios" / "x2b_list.json").read_text())["scenarios"]
    out_p = X2B / "score" / "x2b.json"
    res = json.loads(out_p.read_text()) if out_p.exists() else {}
    for s in items:
        a = ASSETS / "x2b" / s["id"]
        g = json.loads((a / "gen.json").read_text())
        cells = json.loads((a / "mask.json").read_text())["mask"]["voxels"]
        src = ASSETS / "Edit3D-Bench" / "data" / s["dataset"] / s["object"] / "source_model" / "model.glb"
        for path, spec in (("p0", {"kind": "n"}), ("p1", {"kind": "raw", "vertex_color": True})):
            d = X2B / path / s["id"] / "r1"
            key = f"{s['id']}/{path}"
            if key in res or not (d / "round.json").exists():
                continue
            job = {"asset": {"v0_mesh": str(src), "raw_c": [0, 0, 0], "raw_s": 1.0, "v0_slat": str(a / "slat.safetensors")},
                   "ref": {"mesh": str(src), "kind": "v0"}, "res": {"mesh": str(d / "output.glb"), **spec,
                                                                    **({"slat": str(d / "slat_tgt.safetensors")} if path == "p1" else {})},
                   "excl_cells": cells, "edit_view": g["edit_view"], "edit_image": str(a / "cond" / "2d_edit.png"), "prompt": s["prompt"],
                   "work_dir": str(X2B / "score" / s["id"] / path), "cache_dir": str(X2B / "score" / "_cache")}
            if path == "p1":
                job["rkeep"] = {"parent_slat": str(a / "slat.safetensors"), "mask_cells": cells}
            o = score(job)
            res[key] = o
            p, im = o["preserved"], o["image"]
            print(f"x2b {s['id']} {path} CD {p['cd']:.4f} F1 {p['f1']:.3f} PSNR {im['psnr']:.2f} LPIPS {im['lpips']:.4f} "
                  f"DINO {im.get('dino_i'):.3f} CLIP {im.get('clip_t'):.3f}", flush=True)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"a": score_a, "tost": tost, "b": score_b}[sys.argv[1]]()
