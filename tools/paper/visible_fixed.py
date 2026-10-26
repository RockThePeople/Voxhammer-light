"""chief 00:5x 표 보정 1·2 (CPU).

1. 고정 영역(X3): 회차마다 누적 마스크가 커져 비교 영역이 줄어드는 몫을 없애려고, 모든 회차를 "5회차 누적 마스크 ∪ halo 2 밖" 한 영역으로
   CD·F1·PSNR(M)·LPIPS(M)(paper_metrics.score, 렌더는 캐시)와 렌더 L*(c4 와 같은 정의)를 다시 잰다. 기존 누적 영역 값은 그대로 둔다.
2. 보이는 표면(X3·X0·X2-A·X2-B): Bench 16시점 레이캐스팅 첫 교차점만 100k 점(paper_metrics.visible_metrics)으로 CD·NC·F1.
   R4 .cbin 의 가려진 내부 면 몫을 뺀다(오두막 P1 출발점 F1 전체 0.709 → 보이는 표면 0.990, 실측). 기존 전체 표면 값은 참고 열.
P1·P2 메시는 봉합본(X3_STITCHED=1 과 같은 mesh_spec), X0·X2-A 의 P1-cbin 은 c15-cbin-st.

    python tools/paper/visible_fixed.py x3 | x0 | x2a | x2b      # 결과 $RUNS_DIR/paper/visible/<실험>.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("X3_STITCHED", "1")             # x3_score.mesh_spec 가 P1·P2 봉합본을 고르게(가져오기 전에)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party" / "ai-ar-v3" / "contract" / "python")]
RUNS = Path(os.environ["RUNS_DIR"])
ASSETS = Path(os.environ["ASSETS_DIR"])
OUT = RUNS / "paper" / "visible"


def _save(name, res):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))


def _load(name):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def render_L_fixed(asset, path, k, cells5):
    """c4_color.render_lightness 와 같은 정의, 영역만 5회차 누적 마스크 ∪ halo 2 밖으로 고정."""
    from PIL import Image
    from deltacontract.coords import dilate_cells
    from tools.paper.c4_color import lc, render_dir, srgb_to_lab
    from tools.paper.x3_score import frame
    from tools.paper_metrics import BENCH_VIEWS, HALO, silhouette
    fr = frame(asset)
    d = render_dir(asset, path, k)
    dil = np.asarray(dilate_cells(cells5, HALO))
    Ls = []
    for i, vw in enumerate(BENCH_VIEWS):
        a = np.asarray(Image.open(d / f"render_{i:04d}.png").convert("RGBA"), float) / 255.0
        m = (a[..., 3] > 0.5) & ~silhouette(dil, fr, vw, res=a.shape[0])
        Ls.append(lc(srgb_to_lab(a[..., :3][m]))[0])
    L = np.concatenate(Ls)
    return {"L": float(L.mean()), "pixels": int(L.size)}


def x3():
    from tools.paper.x3 import OUT as X3OUT, ROUNDS
    from tools.paper.x3_score import CACHE, cum_mask, gen, mesh_spec
    from tools.paper_metrics import score, visible_metrics
    res = _load("x3")
    for a in ("jeep", "cottage", "adventurer"):
        g, nz = gen(a)
        fa = {"v0_mesh": str(g / "model.glb"), "raw_c": nz["center_glb_frame"], "raw_s": nz["scale"]}
        c5 = cum_mask(a, ROUNDS)
        for p in ("p0", "p1", "p2"):
            for k in range(0, ROUNDS + 1):
                key = f"{a}/{p}_r{k}"
                if key in res or (k and not (X3OUT / p / a / f"r{k}" / "round.json").exists()):
                    continue
                own = mesh_spec(p, a, 0)
                job = {"asset": fa, "ref": own, "res": mesh_spec(p, a, k), "excl_cells": cum_mask(a, k).tolist() if k else [],
                       "work_dir": str(OUT / "x3" / a / f"{p}_r{k}"), "cache_dir": str(CACHE)}
                r = {"visible_cum": visible_metrics(job)}
                fj = {**job, "excl_cells": c5.tolist()}
                fixed = score(fj) if k else None
                r["fixed"] = None if fixed is None else {"preserved": fixed["preserved"],
                                                         "image": {m: fixed["image"][m] for m in ("psnr", "ssim", "lpips")}}
                r["fixed_visible"] = visible_metrics(fj)
                r["fixed_render_L"] = render_L_fixed(a, p, k, c5)
                res[key] = r
                f = r["fixed"]
                print(f"vf x3 {a} {p} r{k} vis CD {r['visible_cum']['cd']:.4f} F1 {r['visible_cum']['f1']:.3f} | fixed "
                      + (f"PSNR {f['image']['psnr']:.2f} " if f else "") + f"L {r['fixed_render_L']['L']:.1f}", flush=True)
                _save("x3", res)


def x0():
    from tools.paper.x0 import configs, scenarios
    from tools.paper_metrics import visible_metrics
    res = _load("x0")
    S = scenarios()
    os.environ["X0_CONFIGS"] = "base,c15-glb,c15-cbin,c15-cbin-st"
    for sid, s in S.items():
        cells = json.loads(Path(s["mask"]).read_text())["mask"]["voxels"]
        for r in ("r1", "r2", "r3"):
            for cfg, (glb, _, vc) in configs(sid, s, r).items():
                key = f"{sid}/{r}/{cfg}"
                if key in res or not Path(glb).exists():
                    continue
                job = {"asset": {"v0_mesh": str(s["v0"]), "raw_c": s["raw_c"], "raw_s": s["raw_s"]},
                       "ref": {"mesh": str(s["v0"]), "kind": "v0"}, "res": {"mesh": str(glb), "kind": "raw"}, "excl_cells": cells}
                res[key] = visible_metrics(job)
                print(f"vf x0 {key} CD {res[key]['cd']:.4f} F1 {res[key]['f1']:.3f}", flush=True)
                _save("x0", res)


def x2a():
    from tools.paper.x0 import scenarios
    from tools.paper_metrics import visible_metrics
    res = _load("x2a")
    for sid, s in scenarios().items():
        cells = json.loads(Path(s["mask"]).read_text())["mask"]["voxels"]
        for r in ("r1", "r2"):
            d = RUNS / "paper" / "x2a" / "p0" / sid / r
            key = f"{sid}/{r}"
            if key in res or not (d / "round.json").exists():
                continue
            job = {"asset": {"v0_mesh": str(s["v0"]), "raw_c": s["raw_c"], "raw_s": s["raw_s"]},
                   "ref": {"mesh": str(s["v0"]), "kind": "v0"}, "res": {"mesh": str(d / "output.glb"), "kind": "n"}, "excl_cells": cells}
            res[key] = visible_metrics(job)
            print(f"vf x2a {key} CD {res[key]['cd']:.4f} F1 {res[key]['f1']:.3f}", flush=True)
            _save("x2a", res)


def x2b():
    from tools.paper_metrics import visible_metrics
    res = _load("x2b")
    for s in json.loads((ROOT / "scenarios" / "x2b_list.json").read_text())["scenarios"]:
        a = ASSETS / "x2b" / s["id"]
        cells = json.loads((a / "mask.json").read_text())["mask"]["voxels"]
        src = ASSETS / "Edit3D-Bench" / "data" / s["dataset"] / s["object"] / "source_model" / "model.glb"
        for p, kind in (("p0", "n"), ("p1", "raw")):
            d = RUNS / "paper" / "x2b" / p / s["id"] / "r1"
            key = f"{s['id']}/{p}"
            if key in res or not (d / "round.json").exists():
                continue
            job = {"asset": {"v0_mesh": str(src)}, "ref": {"mesh": str(src), "kind": "v0"},
                   "res": {"mesh": str(d / "output.glb"), "kind": kind}, "excl_cells": cells}
            res[key] = visible_metrics(job)
            print(f"vf x2b {key} CD {res[key]['cd']:.4f} F1 {res[key]['f1']:.3f}", flush=True)
            _save("x2b", res)


if __name__ == "__main__":
    {"x3": x3, "x0": x0, "x2a": x2a, "x2b": x2b}[sys.argv[1]]()
