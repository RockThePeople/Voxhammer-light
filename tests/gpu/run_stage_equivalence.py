"""0-1 GPU 단계 동등성 (실제 모델, moto-b W34 입력, seed 0). pytest 가 아니라 스크립트(무겁다: 캐시 62 GB).

1. 포크: `torch.manual_seed(seed)` → `run_edit(slat_path, mask_cells)`. 모듈 전역 함수를 감싸 각 단계 출력을 잡는다.
   - P-3: `sample_slat_denoise` 진입 시 CPU RNG 상태가 manual_seed 직후 상태와 같은지(그 사이 전역 RNG 소비 없음)
   - 캐시(dict)는 메모리 때문에 텐서별 sha256 다이제스트로 줄여 보관
2. editcore: `Editor.edit` 를 같은 입력·seed 로. 같은 항목을 잡는다.
3. 비교: Stage 1 inversion noise·잠재·K/V 다이제스트, voxel_tgt, coords_tgt, Stage 2 잠재·K/V 다이제스트,
   출발 잡음 텐서, slat_tgt (보존 셀 비트 동일 / 편집 셀 max|Δ|).
결과 JSON 을 --out 에 쓴다.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import editcore.models as em  # noqa: E402  (환경변수·trellis 경로)

import numpy as np  # noqa: E402
import torch  # noqa: E402


def digest(d):
    """{key: tensor|SparseTensor} → {key: sha256(bytes)} (CPU 텐서 기준)."""
    out = {}
    for k in sorted(d):
        v = d[k]
        h = hashlib.sha256()
        for t in ((v.feats, v.coords) if hasattr(v, "feats") else (v,)):
            t = t.detach().cpu().contiguous()
            h.update(str(t.dtype).encode() + str(tuple(t.shape)).encode())
            h.update(t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.bool else t.numpy().tobytes())
        out[k] = h.hexdigest()
    return out


def tsha(t):
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


class _StopBeforeDecode(Exception):
    pass


def _stop(*a, **k):
    raise _StopBeforeDecode()


def run_fork(pipe, req_paths, seed):
    os.chdir(em.TRELLIS_ROOT)
    import voxhammer.edit_pipeline as ep
    import trellis.modules.sparse as sp
    cap = {}
    orig = {n: getattr(ep, n) for n in ("sample_sparse_structure_inverse", "sample_sparse_structure_denoise",
                                        "sample_slat_inverse", "sample_slat_denoise")}

    def w_s1i(*a, **k):
        noise, lat, kv = orig["sample_sparse_structure_inverse"](*a, **k)
        cap["s1_noise"] = tsha(noise); cap["s1_latent"] = digest(lat); cap["s1_kv"] = digest(kv)
        return noise, lat, kv

    def w_s1d(*a, **k):
        v = orig["sample_sparse_structure_denoise"](*a, **k)
        cap["voxel_tgt"] = tsha(v)
        return v

    def w_s2i(*a, **k):
        lat, kv = orig["sample_slat_inverse"](*a, **k)
        cap["s2_latent"] = digest(lat); cap["s2_kv"] = digest(kv)
        return lat, kv

    def w_s2d(pipeline, cond_tgt, coords_tgt, *a, **k):
        cap["rng_state_at_denoise_equal_seed_state"] = bool(torch.equal(torch.random.get_rng_state(), cap["_seed_state"]))
        # 출발 잡음: 포크 512행과 같은 식을 RNG 상태를 복사한 생성기로 재현해 기록(전역 상태는 건드리지 않음)
        g = torch.Generator().set_state(torch.random.get_rng_state())
        cap["start_noise"] = torch.randn(coords_tgt.shape[0], pipeline.models["slat_flow_model"].in_channels, generator=g)
        cap["coords_tgt"] = coords_tgt.detach().cpu()
        out = orig["sample_slat_denoise"](pipeline, cond_tgt, coords_tgt, *a, **k)
        cap["slat_tgt"] = (out.coords.detach().cpu(), out.feats.detach().cpu())
        cap["counts"] = None
        return out

    ep.sample_sparse_structure_inverse, ep.sample_sparse_structure_denoise = w_s1i, w_s1d
    ep.sample_slat_inverse, ep.sample_slat_denoise = w_s2i, w_s2d
    pipe.decode_slat = _stop  # 디코딩 전에 멈춘다
    torch.manual_seed(seed)
    cap["_seed_state"] = torch.random.get_rng_state()
    cells = np.asarray(json.loads(Path(req_paths["mask"]).read_text())["mask"]["voxels"])
    try:
        with torch.no_grad():
            ep.run_edit(pipe, render_dir=None, output_path="/dev/null", image_dir=req_paths["images"], is_text=False,
                        source_prompt="", target_prompt="", slat_path=req_paths["slat"], mask_cells=cells)
    except _StopBeforeDecode:
        pass
    finally:
        del pipe.decode_slat
        for n, f in orig.items():
            setattr(ep, n, f)
    cap.pop("_seed_state")
    return cap


def run_port(pipe, req_paths, seed):
    from editcore.models import derive_constants
    from editcore.pipeline import Editor
    from editcore.service import load_request
    import editcore.pipeline as P
    ed = Editor(pipe, derive_constants(pipe))
    cap = {}
    o1i, o1d, o2i, o2d = ed.stage1_inverse, ed.stage1_denoise, ed.stage2_inverse, ed.stage2_denoise

    def s1i(*a, **k):
        noise, lat, kv = o1i(*a, **k)
        cap["s1_noise"] = tsha(noise); cap["s1_latent"] = digest(lat); cap["s1_kv"] = digest(kv)
        return noise, lat, kv

    def s1d(*a, **k):
        v = o1d(*a, **k); cap["voxel_tgt"] = tsha(v); return v

    def s2i(*a, **k):
        lat, kv = o2i(*a, **k); cap["s2_latent"] = digest(lat); cap["s2_kv"] = digest(kv); return lat, kv

    def s2d(cond_tgt, coords_tgt, *a, **k):
        gen = a[-1] if a and isinstance(a[-1], torch.Generator) else k["generator"]
        cap["start_noise"] = torch.randn(coords_tgt.shape[0], 8, generator=torch.Generator().set_state(gen.get_state()))
        cap["coords_tgt"] = coords_tgt.detach().cpu()
        out = o2d(cond_tgt, coords_tgt, *a, **k)
        cap["slat_tgt"] = (out.coords.detach().cpu(), out.feats.detach().cpu())
        return out

    ed.stage1_inverse, ed.stage1_denoise, ed.stage2_inverse, ed.stage2_denoise = s1i, s1d, s2i, s2d
    req = load_request(req_paths["slat"], req_paths["mask"], req_paths["images"], seed=seed)
    res = ed.edit(req)
    cap["counts"] = res.counts
    return cap


def compare(f, p, preserve):
    r = {}
    for k in ("s1_noise", "voxel_tgt"):
        r[k + "_equal"] = f[k] == p[k]
    for k in ("s1_latent", "s1_kv", "s2_latent", "s2_kv"):
        same = [kk for kk in f[k] if p[k].get(kk) == f[k][kk]]
        r[k] = {"keys_equal": f[k].keys() == p[k].keys(), "n": len(f[k]), "bit_equal": len(same)}
    r["coords_tgt_equal"] = bool(torch.equal(f["coords_tgt"], p["coords_tgt"]))
    r["start_noise_bit_equal"] = bool(torch.equal(f["start_noise"], p["start_noise"]))
    r["rng_untouched_before_512"] = f["rng_state_at_denoise_equal_seed_state"]
    (cf, ff), (cp, fp) = f["slat_tgt"], p["slat_tgt"]
    r["slat_coords_equal"] = bool(torch.equal(cf, cp))
    if r["slat_coords_equal"]:
        keys = [tuple(x) for x in cf[:, 1:].tolist()]
        pm = np.array([k in preserve for k in keys])
        d = (ff - fp).abs()
        r["preserved"] = {"cells": int(pm.sum()), "bit_equal": bool(d[torch.as_tensor(pm)].max() == 0) if pm.any() else None}
        e = d[torch.as_tensor(~pm)]
        r["edited"] = {"cells": int((~pm).sum()), "max_abs": float(e.max()) if len(e) else 0.0,
                       "mean_abs": float(e.mean()) if len(e) else 0.0, "bit_equal": bool(e.max() == 0) if len(e) else True}
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slat", required=True)
    ap.add_argument("--mask_json", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fork_twice", action="store_true", help="대조: 포크를 두 번 돌려 포크–포크 다이제스트를 비교")
    a = ap.parse_args()
    paths = {"slat": str(Path(a.slat).resolve()), "mask": str(Path(a.mask_json).resolve()),
             "images": str(Path(a.image_dir).resolve())}
    out_path = Path(a.out).resolve()
    pipe = em.load_pipeline()
    Cc = em.derive_constants(pipe)
    const = {"derived": Cc.__dict__, "fork_literals": {"ss_tokens": 4096, "ss_resolution": 16, "voxel_res": 64,
             "ss_head_dim": 64, "patch_grid": 37, "patch_size": 14, "cond_tokens": 1374}}
    const["equal"] = all(getattr(Cc, k) == v for k, v in const["fork_literals"].items())
    import trellis.modules.attention as ta, trellis.modules.sparse as ts
    const["attention"] = {"dense": ta.BACKEND, "sparse": ts.ATTN}
    f = run_fork(pipe, paths, a.seed)
    gc.collect()
    p = run_fork(pipe, paths, a.seed) if a.fork_twice else run_port(pipe, paths, a.seed)
    p.setdefault('counts', None)
    from safetensors import safe_open
    with safe_open(paths["slat"], "np") as sf:
        src = {tuple(x) for x in sf.get_tensor("coords")[:, 1:].tolist()}
    mask = {tuple(x) for x in json.loads(Path(paths["mask"]).read_text())["mask"]["voxels"]}
    r = compare(f, p, src - mask)
    r["counts_port"] = p["counts"]
    r["constants"] = const
    out_path.write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
