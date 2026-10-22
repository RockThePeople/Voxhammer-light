"""기준선 계측 래퍼 (00 A-4). 포크(`build/reference-v2`)는 고치지 않는다.

두 경로:
- `--mode original` (0번 시간 기준선): 포크 `inference.run_complete_pipeline` 를 `inference.py __main__`
  과 같은 기본 인자로 그대로 부른다 — 150뷰 렌더(512², CYCLES) → DINOv2 특징(batch 10) →
  mask.glb 부피 복셀화(volume, 1/64) → `run_edit(skip_step=0, re_init=False, cfg=[5,6,0,0])`.
- `--mode w34` (보조 참고값, 00 A-6 확인 통과 후에만): `run_edit(render_dir=None, slat_path=, mask_cells=)`.

계측은 모듈 속성 교체로 한다. `run_complete_pipeline`·`run_edit` 는 호출 시점에 모듈 전역 이름을
찾으므로 바꿔 끼운 함수가 쓰인다. 디코더는 인스턴스 `forward` 만 감싼다. 모든 감싼 함수의 호출 횟수를
JSON `calls` 에 남겨 실제로 그 경로가 돌았는지 확인한다(기대 횟수는 `EXPECTED_CALLS`).

구간 정의 (요구사항서 6장, TEMPLATE 구간 1–8; 1·2 는 `run_condition_render.py`·`run_inpaint.py`):
  registration  자산 등록: 150뷰 렌더, 특징 추출, SLat 인코딩(`feats_to_slat`)  ← 원저자 경로만
  3  전처리     mask.glb 복셀화, ply 읽기, 이미지 전처리(rembg 2회), get_cond 2회, 마스크 텐서
  4  Stage 1    sample_sparse_structure_inverse + denoise(디코더 제외)
  5  Stage 1 디코딩  sparse_structure_decoder
  6  Stage 2    sample_slat_inverse + denoise
  7  디코딩·출력  slat_decoder_gs, slat_decoder_mesh, to_glb, GLB export (재청킹은 이 경로에 없음)
  8  모델 로드  from_pretrained + .cuda() (스크립트 1회 실행 = 매 회차 로드, 큐 없음)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["original", "w34"], default="original")
ap.add_argument("--model", help="정규화된 model.glb (original)")
ap.add_argument("--mask_glb", help="mask.glb (original)")
ap.add_argument("--mask_json", required=True, help="셀 마스크 json (지문 기록, w34 입력)")
ap.add_argument("--slat", help="입력 SLat (w34)")
ap.add_argument("--image_dir", required=True, help="2d_render/2d_mask/2d_edit")
ap.add_argument("--asset", required=True)
ap.add_argument("--run", required=True, help="회차 이름 (r1, r2, ... / profile)")
ap.add_argument("--reuse_render", help="동등성 대조용: 이 render 디렉터리(150뷰·mesh.ply)를 복사해 렌더 단계를 건너뛴다(inference.run_3d_rendering 의 기존 동작). 시간 기준선에는 쓰지 않는다")
ap.add_argument("--profile", action="store_true", help="구간 4·6 torch.profiler 분해 (평균에서 제외)")
ap.add_argument("--runs_dir", default=os.environ.get("RUNS_DIR", "runs"))
ap.add_argument("--reference", default=os.environ.get("REFERENCE_DIR", "build/reference-v2"))
a = ap.parse_args()

T_PROCESS0 = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
REF = (ROOT / a.reference).resolve()
sys.path.insert(0, str(REF))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "ai-ar-v3" / "contract" / "python"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
args_abs = {k: (str(Path(v).resolve()) if isinstance(v, str) and k in
                ("model", "mask_glb", "mask_json", "slat", "image_dir") else v) for k, v in vars(a).items()}
os.chdir(REF)  # 포크가 상대경로(assets/preset/...)를 쓴다

import numpy as np  # noqa: E402
from deltacontract import mask_fingerprint  # noqa: E402

mask_doc = json.loads(Path(args_abs["mask_json"]).read_text())
mask_cells = np.asarray(mask_doc["mask"]["voxels"], np.int64)
mask_fp = mask_fingerprint(mask_cells)
if mask_fp != mask_doc["meta"]["mask_fingerprint"]:
    raise SystemExit(f"셀 지문 불일치: {mask_fp[:12]} != meta {mask_doc['meta']['mask_fingerprint'][:12]}")

out_dir = Path(a.runs_dir).resolve() / _dt.date.today().isoformat() / a.asset / f"{a.mode}-{mask_fp[:12]}"
work = out_dir / f"{a.run}.work"
work.mkdir(parents=True, exist_ok=True)

import torch  # noqa: E402

from reference.meter import Meter, Sampler, ru_maxrss_bytes, tensor_bytes  # noqa: E402
from reference.slat_capture import SlatCapture  # noqa: E402

sampler = Sampler().start()
M = Meter(sampler)
info: dict = {"cache_bytes": {}, "counts": {}, "profile": {}}

# ── import (구간 8 에 포함해 따로 적는다) ────────────────────────────────
with M.span("import_modules", segment=8):
    import inference as inf  # noqa: E402  포크 inference.py
    import voxhammer.edit_pipeline as ep  # noqa: E402
    import trellis.modules.attention as tattn  # noqa: E402
    import trellis.modules.sparse as tsp  # noqa: E402
    from trellis.utils import postprocessing_utils as pu  # noqa: E402
    import trimesh  # noqa: E402

env_log = {
    "ATTN_BACKEND_env": os.environ.get("ATTN_BACKEND"),
    "dense_attention_backend": tattn.BACKEND,
    "sparse_backend": tsp.BACKEND, "sparse_attention_backend": tsp.ATTN,
    "SPCONV_ALGO": os.environ.get("SPCONV_ALGO"),
    "W39_ATTN_MASK_env": os.environ.get("W39_ATTN_MASK"), "W39_ATTN_MASK_effective": ep.W39_ATTN_MASK,
    "edit_pipeline_file": ep.__file__, "inference_file": inf.__file__,
}
print("[env]", json.dumps(env_log), flush=True)
if ep.W39_ATTN_MASK:
    raise SystemExit("W39_ATTN_MASK 가 켜져 있다 — 기준 실행에서는 끈다")


def sub(fn_name, mod, name, seg):
    setattr(mod, fn_name, M.wrap(getattr(mod, fn_name), name, segment=seg))


# 자산 등록 / 전처리 (inference 모듈 전역)
sub("run_3d_rendering", inf, "render_150views", "registration")
sub("run_feature_extraction", inf, "feature_extraction", "registration")
sub("run_voxel_masking", inf, "mask_glb_voxelize", 3)
# edit_pipeline 모듈 전역
sub("feats_to_slat", ep, "slat_encode", "registration")
sub("ply_to_coords", ep, "ply_to_coords", 3)
sub("coords_to_voxel", ep, "coords_to_voxel", 3)
sub("preprocess_image", ep, "preprocess_image", 3)
sub("ply_to_ss_mask", ep, "ply_to_ss_mask", 3)


_orig_slat_mask = ep.ply_to_slat_mask


def ply_to_slat_mask(coords_tgt, coords_mask):
    info["counts"]["coords_tgt"] = int(coords_tgt.shape[0])
    info["counts"]["coords_preserve"] = int(coords_mask.shape[0])
    with M.span("ply_to_slat_mask", segment=3):
        return _orig_slat_mask(coords_tgt, coords_mask)


ep.ply_to_slat_mask = ply_to_slat_mask


def _profiled(fn, name, seg, cache_key=None):
    def run(*args, **kw):
        if a.profile:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                with M.span(name, segment=seg):
                    r = fn(*args, **kw)
            info["profile"][name] = summarize_profile(prof, M.spans[-1]["seconds"])
        else:
            with M.span(name, segment=seg):
                r = fn(*args, **kw)
        if cache_key:
            info["cache_bytes"][cache_key] = {
                "latent": tensor_bytes(r[-2]), "kv": tensor_bytes(r[-1]),
                "latent_entries": len(r[-2]), "kv_entries": len(r[-1])}
        return r
    run.__wrapped__ = fn
    return run


def summarize_profile(prof, wall_s):
    """DtoH·HtoD memcpy, 좌표 매칭(aten::eq/all/argmax), 나머지 — 장치 시간(µs) 합."""
    cats = {"memcpy_DtoH": 0.0, "memcpy_HtoD": 0.0, "memcpy_DtoD": 0.0,
            "match_eq": 0.0, "match_all": 0.0, "match_argmax": 0.0}
    total_dev = 0.0
    cpu_ops = {}
    for e in prof.key_averages():
        dev = float(getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0.0)) or 0.0)
        total_dev += dev
        k = e.key
        if "Memcpy DtoH" in k:
            cats["memcpy_DtoH"] += dev
        elif "Memcpy HtoD" in k:
            cats["memcpy_HtoD"] += dev
        elif "Memcpy DtoD" in k:
            cats["memcpy_DtoD"] += dev
        if k in ("aten::eq", "aten::all", "aten::argmax", "aten::copy_", "aten::to", "aten::_to_copy"):
            cpu_ops[k] = {"cpu_total_us": float(e.cpu_time_total),
                          "device_total_us": float(getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0.0)) or 0.0),
                          "count": int(e.count)}
            if k == "aten::eq":
                cats["match_eq"] = cpu_ops[k]["device_total_us"]
            elif k == "aten::all":
                cats["match_all"] = cpu_ops[k]["device_total_us"]
            elif k == "aten::argmax":
                cats["match_argmax"] = cpu_ops[k]["device_total_us"]
    return {"wall_s": wall_s, "device_self_total_us": total_dev, "categories_device_us": cats,
            "ops": cpu_ops,
            "note": "match_* 는 해당 aten op 의 device_time_total(하위 커널 포함), memcpy 는 커널 self 시간"}


ep.sample_sparse_structure_inverse = _profiled(ep.sample_sparse_structure_inverse, "stage1_inverse", 4, "stage1")
ep.sample_sparse_structure_denoise = _profiled(ep.sample_sparse_structure_denoise, "stage1_denoise", 4)
ep.sample_slat_inverse = _profiled(ep.sample_slat_inverse, "stage2_inverse", 6, "stage2")
ep.sample_slat_denoise = _profiled(ep.sample_slat_denoise, "stage2_denoise", 6)
pu.to_glb = M.wrap(pu.to_glb, "to_glb", segment=7)  # ep.postprocessing_utils 는 같은 모듈 객체
_orig_exports = {cls: cls.export for cls in (trimesh.Trimesh, trimesh.Scene)}
for cls, fn in _orig_exports.items():
    cls.export = M.wrap(fn, "glb_export", segment=7)

# ── 구간 8: 모델 로드 ──────────────────────────────────────────────────
with M.span("model_load", segment=8):
    pipeline = inf.TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()

for key, name, seg in (("sparse_structure_decoder", "stage1_ss_decoder", 5),
                       ("sparse_structure_encoder", "stage1_ss_encoder", 4),
                       ("slat_encoder", "slat_encoder_forward", "registration"),
                       ("slat_decoder_gs", "decode_gaussian", 7),
                       ("slat_decoder_mesh", "decode_mesh", 7)):
    mod = pipeline.models[key]
    mod.forward = M.wrap(mod.forward, name, segment=seg)
pipeline.get_cond = M.wrap(pipeline.get_cond, "get_cond", segment=3)

out_glb = work / "output.glb"
slat_out = work / "slat_tgt.safetensors"
error = None
with SlatCapture(pipeline, slat_out, extra_meta={"mode": a.mode, "mask_fingerprint": mask_fp}) as cap:
    with M.span("edit_total", segment="total"):
        try:
            if a.mode == "original":
                render_dir = work / "render"
                if a.reuse_render:
                    import shutil
                    shutil.copytree(a.reuse_render, render_dir, dirs_exist_ok=True,
                                    ignore=shutil.ignore_patterns("features.npz", "voxels_delete.ply", "mesh_delete.ply"))
                inf.run_complete_pipeline(
                    pipeline=pipeline, input_model_path=args_abs["model"],
                    mask_glb_path=args_abs["mask_glb"], render_dir=str(render_dir),
                    output_path=str(out_glb), image_dir=args_abs["image_dir"], is_text=False,
                    source_prompt="", target_prompt="")
            else:
                with M.span("run_edit", segment="edit"):
                    ep.run_edit(pipeline, render_dir=None, output_path=str(out_glb),
                                image_dir=args_abs["image_dir"], is_text=False, source_prompt="",
                                target_prompt="", slat_path=args_abs["slat"], mask_cells=mask_cells)
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
            import traceback
            traceback.print_exc()
for cls, fn in _orig_exports.items():
    cls.export = fn
sampler.stop()

# ── 경로 확인 ────────────────────────────────────────────────────────
checks = {"decode_slat_calls": cap.n_calls}
if a.mode == "original":
    rd = work / "render"
    feat = rd / "features.npz"
    checks["features_npz_exists"] = feat.exists()
    checks["voxels_ply_exists"] = (rd / "voxels.ply").exists()
    checks["voxels_delete_ply_exists"] = (rd / "voxels_delete.ply").exists()
    if checks["voxels_delete_ply_exists"]:
        import utils3d
        pos = utils3d.io.read_ply(str(rd / "voxels_delete.ply"))[0]
        dcells = np.unique(np.floor((np.asarray(pos) + 0.5) * 64).astype(np.int64), axis=0)
        mset = {tuple(c) for c in mask_cells.tolist()}
        dset = {tuple(c) for c in dcells.tolist()}
        checks["voxels_delete_cells"] = len(dset)
        checks["voxels_delete_vs_mask_json"] = {"intersection": len(mset & dset), "mask_json": len(mset),
                                                "only_in_delete": len(dset - mset), "only_in_mask": len(mset - dset)}
    if checks["voxels_ply_exists"]:
        import utils3d
        pos = utils3d.io.read_ply(str(rd / "voxels.ply"))[0]
        checks["voxels_src_cells"] = int(len(pos))
checks["output_glb_exists"] = out_glb.exists()

EXPECTED_CALLS = {"stage1_inverse": 1, "stage1_denoise": 1, "stage2_inverse": 1, "stage2_denoise": 1,
                  "preprocess_image": 1, "get_cond": 2, "ply_to_ss_mask": 1, "ply_to_slat_mask": 1,
                  "stage1_ss_decoder": 1, "decode_gaussian": 1, "decode_mesh": 1, "to_glb": 1}
if a.mode == "original":
    EXPECTED_CALLS.update({"render_150views": 1, "feature_extraction": 1, "mask_glb_voxelize": 1,
                           "slat_encode": 1})
checks["calls_match_expected"] = {k: (M.calls.get(k, 0), v) for k, v in EXPECTED_CALLS.items()
                                  if M.calls.get(k, 0) != v}

# ── 구간 합계 (자기 시간 = 포함 시간 − 측정된 자식) ─────────────────────
spans = M.spans
by_name: dict = {}
for s in spans:
    by_name.setdefault(s["name"], []).append(s)
child_time: dict = {}
for s in spans:
    if s["parent"]:
        child_time.setdefault(s["parent"], 0.0)
# 같은 이름이 여러 번 중첩될 수 있어 시간 창으로 자식을 찾는다
for s in spans:
    s["self_seconds"] = s["seconds"] - sum(
        c["seconds"] for c in spans
        if c["parent"] == s["name"] and c["t_start"] >= s["t_start"] and c["t_end"] <= s["t_end"])
segments: dict = {}
for s in spans:
    seg = str(s.get("segment"))
    if seg in ("total", "edit"):
        continue
    d = segments.setdefault(seg, {"seconds": 0.0, "max_memory_allocated": 0, "nvidia_smi_mib_max": 0,
                                  "rss_bytes_max": 0, "parts": {}})
    d["seconds"] += s["self_seconds"]
    d["parts"][s["name"]] = d["parts"].get(s["name"], 0.0) + s["self_seconds"]
    for k in ("max_memory_allocated", "nvidia_smi_mib_max", "rss_bytes_max"):
        if s.get(k) is not None:
            d[k] = max(d[k], s[k])
tot = next((s for s in spans if s["name"] == "edit_total"), None)
measured = sum(v["seconds"] for k, v in segments.items() if k not in ("8",))
residual = (tot["seconds"] - measured) if tot else None

slat_sha = hashlib.sha256(slat_out.read_bytes()).hexdigest() if slat_out.exists() else None
res = {
    "schema": "vhl-run/1", "mode": a.mode, "asset": a.asset, "run": a.run, "profile_run": a.profile,
    "mask": {"file": Path(args_abs["mask_json"]).name, "fingerprint": mask_fp, "n_cells": int(len(mask_cells))},
    "inputs": {k: args_abs[k] for k in ("model", "mask_glb", "slat", "image_dir")},
    "image_sha256": {n: hashlib.sha256((Path(args_abs["image_dir"]) / n).read_bytes()).hexdigest()
                     for n in ("2d_render.png", "2d_mask.png", "2d_edit.png")},
    "reuse_render": a.reuse_render,
    "edit_params": {"skip_step": 0, "re_init": False, "cfg": [5.0, 6.0, 0.0, 0.0]},
    "env": env_log, "error": error, "checks": checks, "counts": info["counts"],
    "cache_bytes": info["cache_bytes"], "profile": info["profile"],
    "segments": segments, "edit_total_seconds": tot["seconds"] if tot else None,
    "unattributed_seconds_in_edit_total": residual,
    "process_wall_seconds": time.perf_counter() - T_PROCESS0,
    "ru_maxrss_bytes": ru_maxrss_bytes(), "calls": M.calls, "spans": spans,
    "outputs": {"glb": str(out_glb), "slat_tgt": str(slat_out), "slat_tgt_sha256": slat_sha,
                "slat_capture": cap.summary},
    "host": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
             "cuda": torch.version.cuda, "python": platform.python_version(),
             "repo_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                           capture_output=True, text=True).stdout.strip()},
}
out_json = out_dir / f"{a.run}.json"
out_json.write_text(json.dumps(res, indent=2, default=str))
print(f"[result] {out_json}")
print(f"[result] error={error} checks={json.dumps(checks, default=str)}")
for k in sorted(segments, key=str):
    v = segments[k]
    print(f"  seg {k:13s} {v['seconds']:8.2f}s  peak {v['max_memory_allocated'] / 2**30:6.2f} GiB  "
          f"smi {v['nvidia_smi_mib_max']} MiB  parts {', '.join(f'{n}={t:.1f}' for n, t in v['parts'].items())}")
print(f"  edit_total {res['edit_total_seconds']}  unattributed {residual}")
sys.exit(1 if error else 0)
