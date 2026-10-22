"""00s S-2 트랙 A 3D 생성과 동결 (a5000).

TRELLIS-image-large v1(`build/reference-v2/trellis`, TRELLIS.2 아님), 기본 샘플러 설정(pipeline.json:
steps 25, cfg 5.0/3.0 기본값 그대로 — 인자를 넘기지 않는다), 시드 고정. `pipeline.run()` 과 같은 순서
(preprocess_image → get_cond → manual_seed → sample_sparse_structure → sample_slat → decode_slat)를
풀어 써서 SLat 을 그대로 저장한다(run() 은 SLat 을 돌려주지 않는다).

자산별 산출 (`$ASSETS_DIR/track_a/<asset>/gen/`, git 제외 — sha256 만 scenarios/track_a.json):
  slat.safetensors   coords int32 (N,4), feats float32 (N,8), slat_space=denormalized, norm_mean/std
  chunks/*.cbin      mesh 디코더 native 출력(복셀 프레임) → deltacontract.partition_mesh → encode
  model_raw.glb      TRELLIS to_glb(simplify 0.95, texture 1024) — 원본 TRELLIS 예제 설정
  model.glb          model_raw.glb 를 최대 변 1·bbox 중심 원점으로 정규화(조건 렌더·원저자 경로 입력)
  gen.json           모델·시드·샘플러·환경·정규화 값·파일 sha256
한 번 만든 산출물은 다시 만들지 않는다(같은 시드여도 비결정적). gen/ 이 있으면 건너뛴다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--assets", nargs="+", default=["moto", "snowman", "chair"])
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--assets_dir", default=os.environ.get("ASSETS_DIR"))
ap.add_argument("--reference", default=str(ROOT.parent / "Voxhammer-light" / "build" / "reference-v2"))
a = ap.parse_args()

REF = Path(a.reference).resolve()
sys.path.insert(0, str(REF))
sys.path.insert(0, str(ROOT / "third_party" / "ai-ar-v3" / "contract" / "python"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("SPCONV_ALGO", "native")
os.chdir(REF)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from deltacontract.chunkbin import encode  # noqa: E402
from deltacontract.partition import albedo_to_rgba8, partition_mesh, split_trellis_vertex_attrs  # noqa: E402
from trellis.pipelines import TrellisImageTo3DPipeline  # noqa: E402
from trellis.utils import postprocessing_utils  # noqa: E402
import trellis.modules.attention as tattn  # noqa: E402
import trellis.modules.sparse as tsp  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


t0 = time.perf_counter()
pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipe.cuda()
t_load = time.perf_counter() - t0

for name in a.assets:
    base = Path(a.assets_dir) / "track_a" / name
    out = base / "gen"
    if out.exists():
        print(f"{name}: gen/ 있음 — 동결, 건너뜀")
        continue
    out.mkdir(parents=True)
    img = Image.open(base / "input.png")
    times = {}
    torch.cuda.synchronize(); t = time.perf_counter()
    with torch.no_grad():  # 디코더 출력에 grad 가 붙으면 to_glb 의 .numpy() 가 죽는다
        image = pipe.preprocess_image(img)
        cond = pipe.get_cond([image])
        torch.manual_seed(a.seed)
        coords = pipe.sample_sparse_structure(cond, 1, {})
        slat = pipe.sample_slat(cond, coords, {})
        torch.cuda.synchronize(); times["sample"] = time.perf_counter() - t
        t = time.perf_counter()
        dec = pipe.decode_slat(slat, ["gaussian", "mesh"])
    torch.cuda.synchronize(); times["decode"] = time.perf_counter() - t

    meta = {"slat_space": "denormalized", "slat_channels": str(slat.feats.shape[1]), "voxel_res": "64",
            "coord_order": "canonical", "seed": str(a.seed), "contract_version": "4",
            "norm_mean": json.dumps(list(map(float, pipe.slat_normalization["mean"]))),
            "norm_std": json.dumps(list(map(float, pipe.slat_normalization["std"])))}
    save_file({"coords": slat.coords.detach().cpu().to(torch.int32).contiguous(),
               "feats": slat.feats.detach().cpu().float().contiguous()}, str(out / "slat.safetensors"), metadata=meta)

    mesh = dec["mesh"][0]
    v = mesh.vertices.detach().cpu().numpy().astype(np.float32)
    f = mesh.faces.detach().cpu().numpy().astype(np.int64)
    attrs = mesh.vertex_attrs.detach().cpu().numpy() if getattr(mesh, "vertex_attrs", None) is not None else None
    vc = slat.coords[:, 1:].detach().cpu().numpy()
    t = time.perf_counter()
    if attrs is not None:
        albedo, normals = split_trellis_vertex_attrs(attrs)
        parts = partition_mesh(v, f, normals=normals, colors_rgba8=albedo_to_rgba8(albedo), voxel_cells=vc)
    else:
        parts = partition_mesh(v, f, voxel_cells=vc)
    (out / "chunks").mkdir()
    for k, m in parts.items():
        (out / "chunks" / f"{k}.cbin").write_bytes(encode(m))
    times["cbin"] = time.perf_counter() - t

    t = time.perf_counter()
    torch.set_grad_enabled(True)
    glb = postprocessing_utils.to_glb(dec["gaussian"][0], mesh, simplify=0.95, texture_size=1024)
    torch.set_grad_enabled(False)
    glb.export(out / "model_raw.glb")
    times["to_glb"] = time.perf_counter() - t
    m = trimesh.load(out / "model_raw.glb", force="mesh")
    lo, hi = m.bounds
    center, scale = (lo + hi) / 2, 1.0 / float((hi - lo).max())
    sc = trimesh.load(out / "model_raw.glb")
    sc.apply_translation(-center)
    sc.apply_scale(scale)
    sc.export(out / "model.glb")

    files = {"slat.safetensors": sha(out / "slat.safetensors"), "model_raw.glb": sha(out / "model_raw.glb"),
             "model.glb": sha(out / "model.glb")}
    cb = hashlib.sha256()
    for p in sorted((out / "chunks").glob("*.cbin")):
        cb.update(p.name.encode()); cb.update(p.read_bytes())
    rec = {
        "asset": name, "input": {"input.png": sha(base / "input.png")},
        "model": "microsoft/TRELLIS-image-large (v1)", "seed": a.seed,
        "sampler": {"sparse_structure": pipe.sparse_structure_sampler_params, "slat": pipe.slat_sampler_params},
        "attention": {"dense": tattn.BACKEND, "sparse": tsp.ATTN},
        "slat": {"n": int(slat.coords.shape[0]), "feats": list(slat.feats.shape)},
        "mesh": {"vertices": int(len(v)), "faces": int(len(f)), "chunks": len(parts),
                 "chunks_digest": cb.hexdigest()},
        "glb_normalize": {"center_glb_frame": center.tolist(), "scale": scale,
                          "note": "model.glb = (model_raw.glb − center) × scale; to_glb 는 복셀 프레임을 Y-up 으로 돌린다(frames D9-b)"},
        "files_sha256": files, "seconds": times, "pipeline_load_seconds": t_load,
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
    }
    (out / "gen.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: rec[k] for k in ("asset", "slat", "mesh", "seconds")}))
