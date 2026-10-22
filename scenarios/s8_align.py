"""00s S-8 출력 정렬 확인 — Bench 1건(GSO/3D_Dollhouse_Lamp): 원본 → 무편집 재구성 → 렌더 → 실루엣 IoU.

재구성은 원저자 경로 그대로(포크 함수 호출): `render_3d_model`(150뷰) → `extract_features`(DINOv2)
→ `slat_encoder`(feats_to_slat) → `decode_slat` → 두 가지 출력:
  (a) 현행: `postprocessing_utils.to_glb(simplify=0.95, texture_size=1024)` → GLB
  (b) R4 이후: mesh 디코더 native → `deltacontract.partition_mesh`/`encode` → `.cbin` → ai-ar-v3 `server/glbbuild.build_glb`
두 GLB 를 Bench `render.py render_images`(16시점) 로 렌더해 GT `source_model/render/render_*.png` 와 알파 실루엣 IoU 를 잰다.
렌더는 별도 프로세스(eval venv)에서 한다. 변환이 필요하면 `eval/` 에 함수로 둔다(지금은 측정만).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = (ROOT.parent / "Voxhammer-light" / "build" / "reference-v2").resolve()
BENCH_SRC = ROOT / "third_party" / "VoxHammer" / "Edit3D-Bench"
GT = Path(os.environ["ASSETS_DIR"]) / "Edit3D-Bench" / "data"
OBJ = "GSO/3D_Dollhouse_Lamp"
OUT = Path(os.environ["RUNS_DIR"]) / "s8_align"
EVAL_PY = os.environ.get("EVAL_PY", str(Path.home() / "vhl-venvs" / "edit3dbench" / "bin" / "python"))


def reconstruct():
    sys.path.insert(0, str(REF))
    sys.path.insert(0, str(ROOT / "third_party" / "ai-ar-v3"))
    sys.path.insert(0, str(ROOT / "third_party" / "ai-ar-v3" / "contract" / "python"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.chdir(REF)
    import numpy as np
    import torch
    import inference as inf
    import voxhammer.edit_pipeline as ep
    from trellis.utils import postprocessing_utils
    from deltacontract.chunkbin import encode
    from deltacontract.partition import albedo_to_rgba8, partition_mesh, split_trellis_vertex_attrs
    from server.glbbuild import build_glb

    rd = OUT / "render150"
    pipe = inf.TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipe.cuda()
    inf.run_3d_rendering(str(GT / OBJ / "source_model" / "model.glb"), str(rd))
    inf.run_feature_extraction(str(rd))
    with torch.no_grad():
        slat = ep.feats_to_slat(pipe, str(rd / "features.npz"))
        dec = pipe.decode_slat(slat, ["gaussian", "mesh"])
    mesh = dec["mesh"][0]
    torch.set_grad_enabled(True)
    (OUT / "a_to_glb").mkdir(parents=True, exist_ok=True)
    postprocessing_utils.to_glb(dec["gaussian"][0], mesh, simplify=0.95, texture_size=1024).export(
        OUT / "a_to_glb" / "edit.glb")
    torch.set_grad_enabled(False)
    v = mesh.vertices.detach().cpu().numpy()
    f = mesh.faces.detach().cpu().numpy()
    albedo, normals = split_trellis_vertex_attrs(mesh.vertex_attrs.detach().cpu().numpy())
    parts = partition_mesh(v, f, normals=normals, colors_rgba8=albedo_to_rgba8(albedo))
    cdir = OUT / "cbin"
    cdir.mkdir(exist_ok=True)
    for k, m in parts.items():
        (cdir / f"{k}.cbin").write_bytes(encode(m))
    (OUT / "b_cbin_glbbuild").mkdir(exist_ok=True)
    build_glb(cdir, OUT / "b_cbin_glbbuild" / "edit.glb")
    src = np.unique(np.floor((__import__("utils3d").io.read_ply(str(rd / "voxels.ply"))[0] + 0.5) * 64).astype(int), axis=0)
    rec = {"voxels_src_cells": int(len(src)), "slat_cells": int(slat.coords.shape[0]),
           "decoder_vertex_bbox": [v.min(0).tolist(), v.max(0).tolist()], "cbin_chunks": len(parts)}
    (OUT / "reconstruct.json").write_text(json.dumps(rec, indent=2))


def render_and_iou():
    import numpy as np
    from PIL import Image
    res = {}
    gt_dir = GT / OBJ / "source_model" / "render"
    for arm in ("a_to_glb", "b_cbin_glbbuild"):
        img_dir = OUT / arm / "images"
        if not img_dir.exists():
            img_dir.mkdir(parents=True)
            subprocess.run([EVAL_PY, "-c", "import sys; sys.path.insert(0,'.'); import render; "
                            "render.render_images(sys.argv[1], sys.argv[2])",
                            str(OUT / arm / "edit.glb"), str(img_dir)], cwd=BENCH_SRC, check=True,
                           stdout=subprocess.DEVNULL)
        ious = []
        for g in sorted(gt_dir.glob("render_*.png")):
            a = np.asarray(Image.open(g).convert("RGBA"))[..., 3] > 127
            b = np.asarray(Image.open(img_dir / g.name).convert("RGBA"))[..., 3] > 127
            ious.append(float((a & b).sum() / max((a | b).sum(), 1)))
        res[arm] = {"silhouette_iou_mean": float(np.mean(ious)), "min": float(np.min(ious)), "per_view": ious}
    gtm = __import__("trimesh").load(GT / OBJ / "source_model" / "model.glb", force="mesh")
    res["gt_bounds"] = gtm.bounds.tolist()
    for arm in ("a_to_glb", "b_cbin_glbbuild"):
        m = __import__("trimesh").load(OUT / arm / "edit.glb", force="mesh")
        res[arm]["bounds"] = m.bounds.tolist()
    (OUT / "align.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: (v if k == "gt_bounds" else {kk: vv for kk, vv in v.items() if kk != "per_view"})
                      for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ["reconstruct"]:
        reconstruct()
    else:
        render_and_iou()
