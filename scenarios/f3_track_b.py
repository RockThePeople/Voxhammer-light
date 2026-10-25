"""00s2 F-3 트랙 B 판본 0 과 조건 이미지 (정오표 E10).

phase encode (GPU, 한 프로세스): 선정 8건마다 원저자 경로로 SLat 1회 인코딩 → 판본 0 동결.
  포크 `inference.run_3d_rendering`(150뷰) → `run_feature_extraction`(DINOv2) → `edit_pipeline.feats_to_slat`.
  저장 `$ASSETS_DIR/track_b/<B#>/slat.safetensors`(R2: coords int32 (N,4), feats float32 (N,8),
  slat_space=denormalized — `sample_slat_inverse` 가 (slat−mean)/std 로 시작하므로 인코더 출력은 denormalized 공간).
  점유 손실·IoU: 원본 = 150뷰 렌더의 mesh.ply 를 포크 `voxelize_mesh` 로 64³ 셀화(= voxels.ply),
  재구성 = SLat 을 mesh 디코더로 디코딩한 메시를 같은 함수로 셀화.
phase cond: 포크 조건 렌더(원본 model.glb + 3d_edit_region.glb, 5시점) → 2D 마스크 vs 데이터셋 2d_mask IoU
  (데이터셋 크기로 최근접 리샘플). 최고 IoU ≥ 0.9 → 데이터셋 2d_render/2d_mask/2d_edit 그대로,
  미만 → 그 시점 포크 렌더 → bf16 FLUX(seed 0, 데이터셋 프롬프트).
마스크: S-4 후보표 계산 때의 편집 영역 셀(포크 process_delete_ply, 64³) → handoff/lasso 와 같은 json, halo 2.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = (ROOT.parent / "Voxhammer-light" / "build" / "reference-v2").resolve()
ASSETS = Path(os.environ["ASSETS_DIR"])
RUNS = Path(os.environ["RUNS_DIR"])
BENCH = ASSETS / "Edit3D-Bench" / "data"
OUT = ASSETS / "track_b"
PY = os.environ.get("VH_PY", str(Path.home() / "ai-ar-v2/VoxHammer/.venv/bin/python"))
IOU_MIN = 0.9


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def selection():
    rows = list(csv.DictReader(open(ROOT / "scenarios" / "track_b_selection_draft.csv")))
    return [(f"B{i + 1}", r) for i, r in enumerate(rows)]


def encode():
    sys.path.insert(0, str(REF))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.chdir(REF)
    import torch
    import utils3d
    import inference as inf
    import voxhammer.edit_pipeline as ep
    from voxhammer.extract_feature import voxelize_mesh
    from safetensors.torch import save_file

    t = time.perf_counter()
    pipe = inf.TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipe.cuda()
    t_load = time.perf_counter() - t
    cells = lambda p: np.unique(np.floor((np.asarray(utils3d.io.read_ply(str(p))[0]) + 0.5) * 64).astype(np.int64), axis=0)  # noqa: E731
    for bid, r in selection():
        d = OUT / bid
        if (d / "slat.safetensors").exists():
            print(bid, "판본 0 있음 — 건너뜀")
            continue
        d.mkdir(parents=True, exist_ok=True)
        src = BENCH / r["dataset"] / r["object"] / "source_model" / "model.glb"
        rd = RUNS / "f3" / bid / "render150"
        T = {}
        import gc; gc.collect(); torch.cuda.empty_cache()  # Cycles(GPU) 렌더 전에 torch 캐시 반환 — B3 에서 OOM 실측
        torch.cuda.synchronize(); t = time.perf_counter()
        inf.run_3d_rendering(str(src), str(rd))
        T["render_150views"] = time.perf_counter() - t; t = time.perf_counter()
        inf.run_feature_extraction(str(rd))
        torch.cuda.synchronize(); T["feature_extraction"] = time.perf_counter() - t; t = time.perf_counter()
        with torch.no_grad():
            slat = ep.feats_to_slat(pipe, str(rd / "features.npz"))
            torch.cuda.synchronize(); T["slat_encode"] = time.perf_counter() - t
            mesh = pipe.decode_slat(slat, ["mesh"])["mesh"][0]
        meta = {"slat_space": "denormalized", "slat_channels": str(slat.feats.shape[1]), "voxel_res": "64",
                "coord_order": "canonical", "contract_version": "4", "source": "edit3d-bench original-path v0",
                "norm_mean": json.dumps(list(map(float, pipe.slat_normalization["mean"]))),
                "norm_std": json.dumps(list(map(float, pipe.slat_normalization["std"])))}
        save_file({"coords": slat.coords.detach().cpu().to(torch.int32).contiguous(),
                   "feats": slat.feats.detach().cpu().float().contiguous()}, str(d / "slat.safetensors"), metadata=meta)
        # 점유: 원본(voxels.ply) vs 재구성 메시 셀화
        rec_dir = RUNS / "f3" / bid / "recon_vox"
        rec_dir.mkdir(parents=True, exist_ok=True)
        import trimesh
        trimesh.Trimesh(mesh.vertices.detach().cpu().numpy(), mesh.faces.detach().cpu().numpy(), process=False).export(rec_dir / "mesh.ply")
        voxelize_mesh(str(rec_dir))
        a = {tuple(x) for x in cells(rd / "voxels.ply").tolist()}
        b = {tuple(x) for x in cells(rec_dir / "voxels.ply").tolist()}
        sc = {tuple(x) for x in slat.coords[:, 1:].cpu().numpy().tolist()}
        rec = {"id": bid, "dataset": r["dataset"], "object": r["object"], "prompt_id": int(r["prompt_id"]),
               "prompt": r["prompt"], "source_prompt": r["source_prompt"],
               "source_glb_sha256": sha(src), "slat_sha256": sha(d / "slat.safetensors"), "slat_cells": len(sc),
               "slat_cells_equal_source_voxels": sc == a,
               "occupancy": {"source_cells": len(a), "recon_cells": len(b), "intersection": len(a & b),
                             "iou": len(a & b) / max(len(a | b), 1), "lost": len(a - b), "new": len(b - a),
                             "loss_ratio": len(a - b) / max(len(a), 1)},
               "registration_seconds": T, "pipeline_load_seconds": t_load}
        (d / "gen.json").write_text(json.dumps(rec, indent=1))
        print(bid, json.dumps({k: rec[k] for k in ("slat_cells", "occupancy", "registration_seconds")}), flush=True)


def cond():
    from PIL import Image
    sys.path.insert(0, str(ROOT / "third_party" / "ai-ar-v3" / "contract" / "python"))
    from deltacontract import mask_fingerprint
    for bid, r in selection():
        d = OUT / bid
        p = BENCH / r["dataset"] / r["object"] / f"prompt_{r['prompt_id']}"
        src = BENCH / r["dataset"] / r["object"] / "source_model" / "model.glb"
        # 마스크 json (S-4 셀 캐시)
        cc = RUNS / "track_b_cells" / f"{r['dataset']}__{r['object']}__p{r['prompt_id']}.cells.npy"
        mc = np.load(cc).astype(np.int64)
        fp = mask_fingerprint(mc)
        (d / "mask.json").write_text(json.dumps({
            "mask": {"mode": "voxels", "voxels": mc.tolist(), "halo_margin_voxels": 2, "grid_source": "slat_coords"},
            "meta": {"asset_id": f"track_b/{bid}", "scenario": bid, "n_cells": int(len(mc)), "mask_fingerprint": fp,
                     "source": "Edit3D-Bench 3d_edit_region.glb → 포크 process_delete_ply(volume, 64³)",
                     "region_glb_sha256": sha(p / "3d_edit_region.glb")}}))
        # 포크 조건 렌더 5시점
        rd = RUNS / "f3" / bid / "cond_render"
        if not (rd / "timing.json").exists():
            subprocess.run([PY, str(ROOT / "reference" / "run_condition_render.py"), "--model", str(src),
                            "--mask", str(p / "3d_edit_region.glb"), "--out", str(rd), "--reference", str(REF)],
                           check=True, stdout=open(RUNS / "f3" / bid / "cond_render.log", "w"), stderr=subprocess.STDOUT)
        dm = Image.open(p / "2d_mask.png").convert("L")
        dmb = np.asarray(dm) > 127
        ious = []
        for v in range(5):
            m = Image.open(rd / "images" / f"mask_{v:04d}.png").convert("L").resize(dm.size, Image.NEAREST)
            mb = np.asarray(m) > 127
            ious.append(float((mb & dmb).sum() / max((mb | dmb).sum(), 1)))
        bv = int(np.argmax(ious))
        cdir = d / "cond"
        cdir.mkdir(exist_ok=True)
        if ious[bv] >= IOU_MIN:
            origin = "dataset"
            for n in ("2d_render.png", "2d_mask.png", "2d_edit.png"):
                shutil.copy(p / n, cdir / n)
            reason = f"최고 IoU {ious[bv]:.3f} ≥ {IOU_MIN}"
        else:
            origin = "FLUX"
            ip = RUNS / "f3" / bid / "inpaint"
            if not (ip / "2d_edit.png").exists():
                subprocess.run([PY, str(ROOT / "reference" / "run_inpaint.py"),
                                "--image", str(rd / "images" / f"render_{bv:04d}.png"),
                                "--mask", str(rd / "images" / f"mask_{bv:04d}.png"),
                                "--prompt", r["prompt"], "--out", str(ip), "--reference", str(REF)],
                               check=True, stdout=open(RUNS / "f3" / bid / "inpaint.log", "w"), stderr=subprocess.STDOUT)
            for n in ("2d_render.png", "2d_mask.png", "2d_edit.png"):
                shutil.copy(ip / n, cdir / n)
            reason = f"최고 IoU {ious[bv]:.3f} < {IOU_MIN} → view {bv} 포크 렌더 + FLUX seed 0"
        g = json.loads((d / "gen.json").read_text())
        g.update({"mask": {"n_cells": int(len(mc)), "fingerprint": fp, "file_sha256": sha(d / "mask.json")},
                  "mask_iou_per_view": ious, "best_view": bv, "best_iou": ious[bv],
                  "condition_origin": origin, "condition_reason": reason, "edit_view": bv,
                  "condition_sha256": {n: sha(cdir / n) for n in ("2d_render.png", "2d_mask.png", "2d_edit.png")},
                  "dataset_2d_sha256": {n: sha(p / n) for n in ("2d_render.png", "2d_mask.png", "2d_edit.png")},
                  "seed": 0})
        (d / "gen.json").write_text(json.dumps(g, indent=1, ensure_ascii=False))
        print(bid, origin, f"best view {bv} IoU {ious[bv]:.3f}", [round(x, 3) for x in ious], flush=True)


if __name__ == "__main__":
    {"encode": encode, "cond": cond}[sys.argv[1]]()
