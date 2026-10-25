"""0-1 후속: B1 이식본 r2·r3 GLB 면 수 +14% 의 원인 가르기.

1. `to_glb` 단순화 **전** — 각 회차 slat_tgt(기준 F-5 r1–r3, 이식 r1–r3)를 mesh 디코더로 다시 디코딩해
   면·정점 수를 센다(디코더 native 메시). 여기서 갈리면 원인은 SLat 값, 같으면 `to_glb`(단순화·텍스처 굽기) 쪽.
2. GLB 면 수(이미 있는 output.glb)를 나란히.
3. Bench 지표: 이식 r1–r3 output.glb 를 Edit3D-Bench render_images(16시점)로 렌더해 eval_main 으로 채점,
   기준(00s2 F-5 bench_eval) 과 나란히.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
RUNS = Path(os.environ["RUNS_DIR"])
ASSETS = Path(os.environ["ASSETS_DIR"])
OUT = RUNS / "01-port" / "b1_followup"
EVAL_PY = os.environ.get("EVAL_PY", str(Path.home() / "vhl-venvs/edit3dbench/bin/python"))
BENCH_SRC = REPO / "third_party" / "VoxHammer" / "Edit3D-Bench"


def decode_counts():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import editcore.models as em
    import torch
    import trimesh
    import trellis.modules.sparse as sp
    from safetensors import safe_open
    ref_dir = Path(open(RUNS / "01-port" / "track_b-B1" / "ref_dir.txt").read().strip().split("=", 1)[1])
    runs = {f"ref-{r}": ref_dir / f"{r}.work" for r in ("r1", "r2", "r3")}
    runs.update({f"port-{r}": RUNS / "01-port" / "track_b-B1" / f"port-{r}" for r in ("r1", "r2", "r3")})
    pipe = em.load_pipeline()
    dec = pipe.models["slat_decoder_mesh"]
    res = {}
    for name, d in runs.items():
        with safe_open(str(d / "slat_tgt.safetensors"), "pt") as f:
            c, ft = f.get_tensor("coords").int().cuda(), f.get_tensor("feats").float().cuda()
        with torch.no_grad():
            m = dec(sp.SparseTensor(feats=ft, coords=c))[0]
        g = trimesh.load(d / "output.glb", force="mesh")
        res[name] = {"slat_cells": int(len(c)), "decoder_vertices": int(m.vertices.shape[0]),
                     "decoder_faces": int(m.faces.shape[0]), "glb_vertices": int(len(g.vertices)), "glb_faces": int(len(g.faces))}
        print(name, res[name], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "decode_counts.json").write_text(json.dumps(res, indent=1))


def bench():
    tb = json.loads((REPO.parent / "vhl-00s2" / "scenarios" / "track_b.json").read_text())["scenarios"]["B1"]
    gt = ASSETS / "Edit3D-Bench" / "data"
    res = {}
    for r in ("r1", "r2", "r3"):
        d = OUT / "pred" / r / tb["dataset"] / tb["object"] / f"prompt_{tb['prompt_id']}"
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "edit.glb").exists():
            (d / "edit.glb").symlink_to(RUNS / "01-port" / "track_b-B1" / f"port-{r}" / "output.glb")
        if not (d / "images" / "render_0015.png").exists():
            (d / "images").mkdir(exist_ok=True)
            subprocess.run([EVAL_PY, "-c", "import sys; sys.path.insert(0,'.'); import render; render.render_images(sys.argv[1], sys.argv[2])",
                            str(d / "edit.glb"), str(d / "images")], cwd=BENCH_SRC, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        od = OUT / "eval" / r
        if not (od / "summary.json").exists():
            od.mkdir(parents=True, exist_ok=True)
            subprocess.run([EVAL_PY, "eval_main.py", "--gt_root", str(gt), "--pred_root", str(OUT / "pred" / r),
                            "--metrics", "psnr", "ssim", "lpips", "dino_i", "chamfer", "clip_t", "--device", "cuda:0",
                            "--output_dir", str(od)], cwd=BENCH_SRC, stdout=open(od / "eval.log", "w"), stderr=subprocess.STDOUT)
        res[r] = {k: v["mean"] for k, v in json.loads((od / "summary.json").read_text())["results"].items()}
        print("port", r, res[r], flush=True)
    ref = json.loads((RUNS / "f5_metrics" / "bench_eval.json").read_text())["B1"]
    (OUT / "bench.json").write_text(json.dumps({"port": res, "ref": {r: ref[r] for r in ("r1", "r2", "r3")},
                                                "v0_no_edit": ref.get("v0_no_edit")}, indent=1))


if __name__ == "__main__":
    {"decode": decode_counts, "bench": bench}[sys.argv[1]]()
