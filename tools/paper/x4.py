"""X4 무편집 재등록 체인 (ops/tasks/X-paper.md X4). 자산 11종(트랙 A 생성 3, 트랙 B 8) × 5회.

P0 (원저자 경로, 포크 무수정 함수): 매 회 GLB → 150시점 렌더(`inference.run_3d_rendering`) → DINOv2 특징
    (`run_feature_extraction`) → 인코딩(`edit_pipeline.ply_to_coords` + `feats_to_slat`) → `decode_slat(["gaussian","mesh"])`
    → `to_glb(simplify=0.95, texture_size=1024)`(포크 run_edit 끝과 같음). 다음 회차 입력 = 직전 output.glb.
    회차마다 output.glb · slat.safetensors(R2, denormalized) · timing 을 남긴다(시간은 판정에 쓰지 않는다).
P1 (우리 경로): 1회차는 editcore 를 빈 마스크로 실제로 돌려 SLat 비트 동일을 보인다(거절되면 사유를 적고 R4 디코딩만).
    각 회차 출력 = 저장 SLat → R4 디코딩(.cbin, 부모 없음) → glbbuild GLB. 다음 회차는 같은 SLat.

    python tools/paper/x4.py p0 [자산 …]     # 결과 $RUNS_DIR/paper/x4/p0/<자산>/r{1..5}/
    python tools/paper/x4.py p1 [자산 …]     # 결과 $RUNS_DIR/paper/x4/p1/<자산>/r{1..5}/
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF = (ROOT.parent / "Voxhammer-light" / "build" / "reference-v2").resolve()
ASSETS = Path(os.environ["ASSETS_DIR"])
RUNS = Path(os.environ["RUNS_DIR"])
OUT = RUNS / "paper" / "x4"
ROUNDS = 5


def assets():
    tb = json.loads((ROOT / "scenarios" / "track_b.json").read_text())["scenarios"]
    out = {}
    for a in ("moto", "snowman", "chair"):
        out[a] = {"track": "A", "v0": ASSETS / "track_a" / a / "gen" / "model.glb", "slat": ASSETS / "track_a" / a / "gen" / "slat.safetensors"}
    for bid, s in tb.items():
        out[bid] = {"track": "B", "v0": ASSETS / "Edit3D-Bench" / "data" / s["dataset"] / s["object"] / "source_model" / "model.glb",
                    "slat": ASSETS / "track_b" / bid / "slat.safetensors"}
    return out


def p0(names):
    sys.path[:0] = [str(REF), str(ROOT)]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cwd = os.getcwd()
    os.chdir(REF)                                  # 포크가 상대경로(assets/preset/...)를 쓴다
    import torch
    import inference as inf
    import voxhammer.edit_pipeline as ep
    from trellis.utils import postprocessing_utils as pu
    from editcore.export import save_slat
    pipe = inf.TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipe.cuda()
    A = assets()
    for name in names:
        src = A[name]["v0"]
        for k in range(1, ROUNDS + 1):
            d = OUT / "p0" / name / f"r{k}"
            if (d / "output.glb").is_file():
                src = d / "output.glb"
                continue
            if d.exists():
                shutil.rmtree(d)
            (d / "render").mkdir(parents=True)
            t = {}
            t0 = time.perf_counter()
            inf.run_3d_rendering(str(src), str(d / "render"))
            t["render_150views"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            inf.run_feature_extraction(str(d / "render"))
            t["feature_extraction"] = time.perf_counter() - t0
            with torch.no_grad():
                t0 = time.perf_counter()
                coords = ep.ply_to_coords(str(d / "render" / "voxels.ply"))
                slat = ep.feats_to_slat(pipe, str(d / "render" / "features.npz"))
                t["encode"] = time.perf_counter() - t0
                save_slat(slat, d / "slat.safetensors", pipe.slat_normalization,
                          {"source": "x4-p0 re-registration", "round": k, "input": str(src)})
                t0 = time.perf_counter()
                assets_ = pipe.decode_slat(slat, ["gaussian", "mesh"])
                t["decode"] = time.perf_counter() - t0
            torch.set_grad_enabled(True)                # 포크 run_edit 와 같이 to_glb 는 grad 켠 채로
            t0 = time.perf_counter()
            glb = pu.to_glb(assets_["gaussian"][0], assets_["mesh"][0], simplify=0.95, texture_size=1024)
            glb.export(str(d / "output.glb"))
            t["to_glb"] = time.perf_counter() - t0
            torch.set_grad_enabled(False)
            # 150시점 렌더(512² × 150)는 크다 — 특징·복셀만 남기고 이미지는 지운다
            for p in (d / "render").glob("*.png"):
                p.unlink()
            (d / "round.json").write_text(json.dumps({"asset": name, "round": k, "input": str(src), "n_cells": int(coords.shape[0]),
                                                      "seconds": t}, indent=1))
            print(f"x4 p0 {name} r{k} cells {coords.shape[0]} {json.dumps({a: round(b, 1) for a, b in t.items()})}", flush=True)
            del slat, assets_, glb
            torch.cuda.empty_cache()                    # 실험 스크립트(시간 측정 안 함)
            src = d / "output.glb"
    os.chdir(cwd)


def p1(names):
    sys.path[:0] = [str(ROOT), str(ROOT / "third_party" / "ai-ar-v3")]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import numpy as np
    import torch
    import editcore.models as em
    import trellis.modules.sparse as sp
    from safetensors import safe_open
    from editcore.export import decode_to_cbin
    pipe = em.load_pipeline()
    A = assets()
    for name in names:
        with safe_open(str(A[name]["slat"]), "pt") as f:
            c0, f0 = f.get_tensor("coords"), f.get_tensor("feats")
        note = {}
        # 1회차: editcore 를 빈 마스크로 실제로 돌린다(SLat 비트 동일 확인). 거절되면 사유를 적는다.
        try:
            from editcore.pipeline import Editor
            from editcore.service import load_request
            d1 = OUT / "p1" / name / "r1"
            d1.mkdir(parents=True, exist_ok=True)
            (d1 / "mask.json").write_text(json.dumps({"mask": {"mode": "voxels", "voxels": []}}))
            cond = ASSETS / "track_a" / name / "cond" if A[name]["track"] == "A" else ASSETS / "track_b" / name / "cond"
            cond = next(cond.iterdir()) if A[name]["track"] == "A" else cond
            req = load_request(A[name]["slat"], d1 / "mask.json", cond, seed=0)
            with torch.inference_mode():
                res = Editor(pipe, em.derive_constants(pipe)).edit(req)
            same = bool(torch.equal(res.slat_tgt.coords.cpu(), c0) and torch.equal(res.slat_tgt.feats.cpu().float(), f0.float()))
            note = {"empty_mask_edit": "ran", "slat_bit_equal": same}
            del res
        except Exception as e:  # noqa: BLE001
            note = {"empty_mask_edit": "rejected", "reason": f"{type(e).__name__}: {e}"}
        print(f"x4 p1 {name} empty-mask edit: {note}", flush=True)
        slat = sp.SparseTensor(feats=f0.float().cuda(), coords=c0.int().cuda())
        for k in range(1, ROUNDS + 1):
            d = OUT / "p1" / name / f"r{k}"
            d.mkdir(parents=True, exist_ok=True)
            if (d / "output.glb").is_file():
                continue
            with torch.inference_mode():
                w = decode_to_cbin(pipe, slat, d / "cbin", glb_path=d / "output.glb")
            shutil.copyfile(A[name]["slat"], d / "slat.safetensors")
            (d / "round.json").write_text(json.dumps({"asset": name, "round": k, "note": note if k == 1 else None,
                                                      "faces": w["mesh"]["faces"]}, indent=1))
            print(f"x4 p1 {name} r{k} faces {w['mesh']['faces']}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    mode, names = sys.argv[1], sys.argv[2:] or list(assets())
    {"p0": p0, "p1": p1}[mode](names)
