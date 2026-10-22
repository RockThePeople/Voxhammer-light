"""00s2 F-1·F-2 트랙 A 조건 이미지 생성·동결.

시나리오마다 1회:
  1) 포크 조건 렌더(`reference/run_condition_render.py`, EXR 은 bpy 프록시) — 동결 model.glb + 시나리오 mask.glb, 5시점
  2) 편집 시점 = 2D 마스크 화소가 가장 큰 시점(00s S-5 규칙). 3/4 시점 = 1(편집 시점이 1 이면 3)
  3) A5000 bf16 FLUX(`reference/run_inpaint.py` = 현행 inpaint_offload 설정, 1024², 50스텝, guidance 30, seed 0)
결과: `$ASSETS_DIR/track_a/<asset>/cond/<ID>/{2d_render,2d_mask,2d_edit}.png` (+ timing), 렌더 원본은 `$RUNS_DIR/f2/<ID>/render`.
이미 cond/<ID>/2d_edit.png 가 있으면 동결본으로 보고 건너뛴다.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(os.environ["ASSETS_DIR"]) / "track_a"
RUNS = Path(os.environ["RUNS_DIR"]) / "f2"
PY = os.environ.get("VH_PY", str(Path.home() / "ai-ar-v2/VoxHammer/.venv/bin/python"))
REF = str((ROOT.parent / "Voxhammer-light" / "build" / "reference-v2").resolve())
SCEN = {  # ID: (asset, 인페인팅 프롬프트) — 요구사항서 6장 표, A-S1 은 정오표 E9
    "A-M1": ("moto", "a red and gold Iron Man helmet as the rider's head"),
    "A-M2": ("moto", "a thick chunky knobby off-road rear tire"),
    "A-S1": ("snowman", "a snowman holding a straw broom in its right hand"),
    "A-S2": ("snowman", "a small round red ball nose"),
    "A-C1": ("chair", "a tall backrest with vertical wooden slats"),
    "A-C2": ("chair", "the chair without armrests, plain seat sides"),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(ids):
    out = {}
    for sid in ids:
        asset, prompt = SCEN[sid]
        dst = ASSETS / asset / "cond" / sid
        if (dst / "2d_edit.png").exists():
            print(f"{sid}: 동결본 있음 — 건너뜀")
            continue
        rd = RUNS / sid / "render"
        rd.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([PY, str(ROOT / "reference" / "run_condition_render.py"),
                        "--model", str(ASSETS / asset / "gen" / "model.glb"),
                        "--mask", str(ASSETS / asset / "masks" / f"{sid}.mask.glb"),
                        "--out", str(rd), "--reference", REF], check=True,
                       stdout=open(RUNS / sid / "render.log", "w"), stderr=subprocess.STDOUT)
        rt = json.loads((rd / "timing.json").read_text())
        if rt["error"]:
            raise SystemExit(f"{sid} 조건 렌더 실패: {rt['error']}")
        px = [int((np.asarray(Image.open(rd / "images" / f"mask_{v:04d}.png").convert("L")) > 127).sum())
              for v in range(5)]
        ev = int(np.argmax(px))
        tv = 3 if ev == 1 else 1
        ip = RUNS / sid / "inpaint"
        subprocess.run([PY, str(ROOT / "reference" / "run_inpaint.py"),
                        "--image", str(rd / "images" / f"render_{ev:04d}.png"),
                        "--mask", str(rd / "images" / f"mask_{ev:04d}.png"),
                        "--prompt", prompt, "--out", str(ip), "--reference", REF], check=True,
                       stdout=open(RUNS / sid / "inpaint.log", "w"), stderr=subprocess.STDOUT)
        it = json.loads((ip / "timing.json").read_text())
        dst.mkdir(parents=True, exist_ok=True)
        for n in ("2d_render.png", "2d_mask.png", "2d_edit.png"):
            shutil.copy(ip / n, dst / n)
        span = lambda d, n: next(s for s in d["spans"] if s["name"] == n)  # noqa: E731
        rec = {
            "scenario": sid, "asset": asset, "prompt": prompt, "seed": 0,
            "edit_view": ev, "view_34": tv, "mask_px_per_view": px,
            "files_sha256": {n: sha(dst / n) for n in ("2d_render.png", "2d_mask.png", "2d_edit.png")},
            "seg1_condition_render_s": span(rt, "condition_render_total")["seconds"],
            "seg1_parts_s": {"source_5views": span(rt, "render_source_5views")["seconds"],
                             "mask_5views": span(rt, "render_mask_5views")["seconds"],
                             "postprocess": rt["mask_postprocess_seconds"]},
            "seg2_inpaint_s": {"load": span(it, "flux_load")["seconds"], "inpaint": span(it, "flux_inpaint")["seconds"]},
            "inpaint_peak_GiB": span(it, "flux_inpaint")["max_memory_allocated"] / 2 ** 30,
            "flux": {"model_id": it["model_id"], "steps": it["steps"], "guidance": it["guidance"],
                     "resolution": it["resolution"], "dtype": it["dtype"], "offload": it["offload"]},
            "exr_reader": rt.get("exr_reader"),
        }
        (dst / "cond.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        out[sid] = rec
        print(sid, json.dumps({k: rec[k] for k in ("edit_view", "mask_px_per_view", "seg1_condition_render_s", "seg2_inpaint_s")}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(SCEN))
