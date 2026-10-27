"""X0 재채점 (ops/tasks/X-paper.md X0): 14건 × {base(25 스텝 to_glb), c15 P1-glb, c15 P1-cbin} vs v0, `tools/paper_metrics.score`.

base     = $RUNS_DIR/01-port/<이름>/port-r1 (0-1 이식본 25 스텝, E2a base 와 같은 출처) — to_glb, raw 프레임
c15-glb  = $RUNS_DIR/2026-10-25/<이름>/e2a-c15/r1 (E2a c15) — to_glb, raw 프레임
c15-cbin = 같은 c15 r1 slat_tgt 를 R4(정정 규칙 R11 + 경계 스냅)로 다시 디코딩한 것(tools/paper/x0_cbin.py,
           $RUNS_DIR/paper/x0/cbin/<시나리오>/output.glb) — glbbuild 정점색. P1-glb 와 SLat 이 같고 메시 경로만 다르다.

    python tools/paper/x0.py [시나리오 …]    # 결과 $RUNS_DIR/paper/x0/<시나리오>/<설정>.json, 요약 x0.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ASSETS = Path(os.environ["ASSETS_DIR"])
RUNS = Path(os.environ["RUNS_DIR"])
OUT = RUNS / "paper" / "x0"
E2A_DAY = "2026-10-25"
# 논문 11건판(사용자 10-27 17:27: A-M1·A-M2·A-C1 제외). VHL_PAPER_CASES=11 또는 쉼표 목록이면 scenarios() 를 그 건으로 줄인다(재집계 전용).
CASES11 = ("A-S1", "A-S2", "A-C2", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8")


def case_filter():
    v = os.environ.get("VHL_PAPER_CASES", "")
    return None if not v else set(CASES11 if v == "11" else v.split(","))


def scenarios():
    ta = json.loads((ROOT / "scenarios" / "track_a.json").read_text())["scenarios"]
    tb = json.loads((ROOT / "scenarios" / "track_b.json").read_text())["scenarios"]
    out = {}
    for sid, s in ta.items():
        a = ASSETS / "track_a" / s["asset"]
        g = json.loads((a / "gen" / "gen.json").read_text())["glb_normalize"]
        out[sid] = {"name": f"track_a-{sid}", "v0": a / "gen" / "model.glb", "slat": a / "gen" / "slat.safetensors",
                    "raw_c": g["center_glb_frame"], "raw_s": g["scale"], "mask": a / "masks" / f"{sid}.mask.json",
                    "cond": ASSETS / s["condition_images"]["dir"], "prompt": s["inpaint_prompt"], "view": s["edit_view"]}
    for bid, s in tb.items():
        b = ASSETS / "track_b" / bid
        out[bid] = {"name": f"track_b-{bid}", "v0": ASSETS / "Edit3D-Bench" / "data" / s["dataset"] / s["object"] / "source_model" / "model.glb",
                    "slat": b / "slat.safetensors", "raw_c": [0, 0, 0], "raw_s": 1.0, "mask": b / "mask.json",
                    "cond": b / "cond", "prompt": s["prompt"], "view": s["edit_view"]}
    keep = case_filter()
    return out if keep is None else {k: v for k, v in out.items() if k in keep}


def configs(sid, s, run="r1"):
    """설정 → (메시, slat_tgt, 정점색). run = r1·r2·r3(지시서 14건 × 3회, chief 15:2x 1)."""
    base, c15 = RUNS / "01-port" / s["name"] / f"port-{run}", RUNS / E2A_DAY / s["name"] / "e2a-c15" / run
    cb = OUT / "cbin" / sid if run == "r1" else OUT / "cbin" / sid / run
    cfg = {"base": (base / "output.glb", base / "slat_tgt.safetensors", False),
           "c15-glb": (c15 / "output.glb", c15 / "slat_tgt.safetensors", False),
           "c15-cbin": (cb / "output.glb", c15 / "slat_tgt.safetensors", True),
           # 봉합본(chief 21:4x (d), tools/paper/stitch_offline.py) — 같은 slat_tgt, R4 .cbin + 봉합 띠
           "c15-cbin-st": (OUT / "cbin_stitched" / sid / run / "output.glb", c15 / "slat_tgt.safetensors", True)}
    want = os.environ.get("X0_CONFIGS")
    return {k: v for k, v in cfg.items() if k in want.split(",")} if want else {k: v for k, v in cfg.items() if k != "c15-cbin-st"}


def main(ids):
    from tools.paper_metrics import score
    S = scenarios()
    suf = os.environ.get("X0_SUMMARY_SUFFIX", "")      # 다른 프로세스와 요약 파일이 겹치지 않게(예: _st)
    summary = json.loads((OUT / f"x0{suf}.json").read_text()) if (OUT / f"x0{suf}.json").is_file() else {}
    runs_sum = json.loads((OUT / f"x0_runs{suf}.json").read_text()) if (OUT / f"x0_runs{suf}.json").is_file() else {}
    for sid, run in [(i, r) for i in (ids or list(S)) for r in os.environ.get("X0_RUNS", "r1").split(",")]:
        s = S[sid]
        cells = json.loads(Path(s["mask"]).read_text())["mask"]["voxels"]
        for cfg, (glb, slat, vc) in configs(sid, s, run).items():
            o = OUT / sid / f"{cfg}.json" if run == "r1" else OUT / sid / run / f"{cfg}.json"
            if o.is_file():
                r = json.loads(o.read_text())
                if run == "r1":
                    summary.setdefault(sid, {})[cfg] = r
                runs_sum.setdefault(sid, {}).setdefault(run, {})[cfg] = r
                continue
            if not Path(glb).is_file():
                print(f"x0 {sid} {run} {cfg} 메시 없음 — 건너뜀 ({glb})", flush=True)
                continue
            job = {"asset": {"v0_mesh": str(s["v0"]), "raw_c": s["raw_c"], "raw_s": s["raw_s"], "v0_slat": str(s["slat"])},
                   "ref": {"mesh": str(s["v0"]), "kind": "v0"},
                   "res": {"mesh": str(glb), "kind": "raw", "vertex_color": vc, "slat": str(slat)},
                   "excl_cells": cells, "rkeep": {"parent_slat": str(s["slat"]), "mask_cells": cells},
                   "edit_view": s["view"], "edit_image": str(s["cond"] / "2d_edit.png"), "prompt": s["prompt"],
                   "work_dir": str(OUT / sid / cfg) if run == "r1" else str(OUT / sid / run / cfg), "cache_dir": str(OUT / "_cache")}
            r = score(job)
            r["source"] = str(glb)
            o.parent.mkdir(parents=True, exist_ok=True)
            o.write_text(json.dumps(r, indent=1))
            if run == "r1":
                summary.setdefault(sid, {})[cfg] = r
            runs_sum.setdefault(sid, {}).setdefault(run, {})[cfg] = r
            p, im = r["preserved"], r["image"]
            print(f"x0 {sid} {run} {cfg} CD {p['cd']:.4f} NC {p['nc']:.3f} F1 {p['f1']:.3f} PSNR {im['psnr']:.2f} SSIM {im['ssim']:.4f} "
                  f"LPIPS {im['lpips']:.4f} DINO {im.get('dino_i'):.4f} CLIP {im.get('clip_t'):.4f} | E2a정의 CLIP6 {im.get('clip_t_front6'):.4f} "
                  f"DINO16 {im.get('dino_i_ref16'):.4f}", flush=True)
            (OUT / f"x0{suf}.json").write_text(json.dumps(summary, indent=1))
            (OUT / f"x0_runs{suf}.json").write_text(json.dumps(runs_sum, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
