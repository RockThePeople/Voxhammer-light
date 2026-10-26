"""04L 상주 편집 엔진 — 모델은 프로세스 시작 때 한 번 올리고, 요청마다 editcore 로 편집·R4 `.cbin`·조건 렌더를 한다.

- 로드(R8a A-1): `editcore.models.load_pipeline()` = TRELLIS-image-large 전체(두 flow, SS 인코더·디코더, SLat 디코더들,
  DINOv2 조건 인코더). 로드 시간을 기록한다. editcore 기본 15/15(E20), R4 경로(`decode_to_cbin`)를 쓴다.
- 요청마다 `torch.inference_mode()`. 편집이 끝나면 결과 텐서·K/V 캐시를 놓고 `torch.cuda.empty_cache()`(계측 구간 밖).
- 파일: 판본 디렉터리(`store.version_dir`) = slat.safetensors(R2) · chunks/*.cbin · manifest.json · asset.json(정규화 xf,
  렌더 메시 종류) · model.glb(생성 판본, 텍스처) 또는 output.glb(R4 판본, glbbuild 정점색).
"""
from __future__ import annotations

import gc
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "third_party" / "ai-ar-v3" / "contract" / "python"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

STATE = {"pipe": None, "constants": None, "load": None}


def _mem():
    import torch
    rss = None
    try:
        with open("/proc/self/statm") as f:
            rss = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except OSError:
        pass
    return {"rss_bytes": rss, "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "cuda_allocated": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else None,
            "cuda_reserved": int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else None}


def load():
    """시작 시 1회. 로드 시간·메모리를 STATE['load'] 에 남긴다."""
    import torch
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("VHL_R4", "1")
    import editcore.models as em
    t0 = time.perf_counter()
    pipe = em.load_pipeline()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    STATE.update(pipe=pipe, constants=em.derive_constants(pipe))
    STATE["load"] = {"seconds": round(t1 - t0, 3), "models": sorted(pipe.models), "steps": em.e2a_steps(),
                     "vhl_r4": em.r4_enabled(), "vhl_ds": em.ds_enabled(), "mem_after": _mem(), "t": time.time()}
    print(f"[04l] 모델 상주 로드 {t1 - t0:.1f}s models={sorted(pipe.models)} steps={em.e2a_steps()} R4={em.r4_enabled()}", flush=True)
    return STATE["load"]


def mask_cells(mask: dict):
    """EditMask(dict) → halo 전 셀 (N,3)."""
    from deltacontract.coords import bbox_to_voxel_cells
    if mask.get("mode") == "bbox":
        return np.asarray(bbox_to_voxel_cells(mask["bbox_min"], mask["bbox_max"]), np.int64).reshape(-1, 3)
    return np.asarray(mask.get("voxels") or [], np.int64).reshape(-1, 3)


def version_meta(vdir: Path) -> dict:
    p = Path(vdir) / "asset.json"
    return json.loads(p.read_text()) if p.is_file() else {"xf": {"center": [0.0, 0.0, 0.0], "scale": 1.0}}


def render_input_glb(vdir: Path, work: Path) -> Path:
    """판본의 조건 렌더 입력 메시. 생성 판본 = 텍스처 GLB(model.glb, 이미 정규화본).
    R4 판본 = .cbin → glbbuild 정점색 GLB → 판본 정규화 (p − c) × s → 렌더 입력 보정 (iv)(editcore/render.py)."""
    import trimesh
    from editcore.render import vertex_color_render_glb
    vdir, work = Path(vdir), Path(work)
    if (vdir / "model.glb").is_file():
        return vdir / "model.glb"
    src = vdir / "output.glb"
    if not src.is_file():
        v3 = str(ROOT / "third_party" / "ai-ar-v3")
        if v3 not in sys.path:
            sys.path.insert(0, v3)
        from server.glbbuild import build_glb
        build_glb(vdir / "chunks", work / "cbin.glb")
        src = work / "cbin.glb"
    xf = version_meta(vdir)["xf"]
    sc = trimesh.load(src)
    sc.apply_translation([-x for x in xf["center"]])
    sc.apply_scale(float(xf["scale"]))
    sc.export(work / "normalized.glb")
    out = work / "render_input.glb"
    out.write_bytes(vertex_color_render_glb((work / "normalized.glb").read_bytes()))
    return out


def render(vdir: Path, cells, view, work: Path) -> dict:
    """조건 렌더(구간 1). view 가 None 이면 마스크 투영 면적 최대 시점."""
    from editcore.render import choose_view, render_condition
    xf = version_meta(vdir)["xf"]
    areas = None
    if view is None:
        view, areas = choose_view(cells, xf)
    t0 = time.perf_counter()
    mesh = render_input_glb(vdir, work)
    t1 = time.perf_counter()
    r = render_condition(mesh, cells, int(view), xf=xf, out_dir=work)
    r["timing"]["input_mesh_s"] = round(t1 - t0, 3)
    r["timing"]["wall_s"] = round(time.perf_counter() - t0, 3)
    r["view_areas"] = areas
    r["view_auto"] = areas is not None
    return r


def edit(parent: Path, staging: Path, *, cells, halo: int, seed: int, asset_id: str, to_version: int) -> dict:
    """editcore 편집(W34 입력 = 부모 판본 slat_tgt) → R2 SLat 저장 → R4 .cbin(부기 ∪ 점유 변경 ∪ 신규, R11 승계, 경계 스냅)."""
    import torch
    from reference.meter import Meter, Sampler
    from editcore.export import bookkeeping_async, decode_to_cbin, save_slat
    from editcore.pipeline import Editor
    from editcore.service import load_request
    pipe, C = STATE["pipe"], STATE["constants"]
    parent, staging = Path(parent), Path(staging)
    mem0 = _mem()
    (staging / "mask.json").write_text(json.dumps({"mask": {"mode": "voxels", "voxels": cells.tolist(),
                                                             "halo_margin_voxels": halo, "grid_source": "slat_coords"}}))
    sampler = Sampler().start()
    meter = Meter(sampler)
    hooks = {"span": (lambda n, s: meter.span(n, segment=s))}
    with torch.inference_mode():
        book = bookkeeping_async(cells, halo)
        with meter.span("load_request", segment=3):
            req = load_request(parent / "slat.safetensors", staging / "mask.json", staging, seed=seed)
        ed = Editor(pipe, C)
        res = ed.edit(req, hooks=hooks)
        save_slat(res.slat_tgt, staging / "slat.safetensors", pipe.slat_normalization,
                  {"seed": seed, "source": "04l-worker", "steps": ed.steps})
        with open(parent / "slat.safetensors", "rb"):
            pass
        from safetensors import safe_open
        with safe_open(str(parent / "slat.safetensors"), "np") as f:
            pc = f.get_tensor("coords")[:, 1:]
        w = decode_to_cbin(pipe, res.slat_tgt, staging, mask_cells=cells, parent_dir=parent, parent_cells=pc, halo=halo,
                           glb_path=staging / "output.glb", hooks=hooks, book=book, asset_id=asset_id, version=to_version)
        counts, steps = res.counts, dict(ed.steps)
        cache_entries = {k: {"latent": len(v[0]), "kv": len(v[1])} for k, v in res.caches.items()}
    sampler.stop()
    del res, req
    gc.collect()
    mem1 = _mem()
    torch.cuda.empty_cache()   # 편집 뒤 캐시 해제(계측 구간 밖)
    mem2 = _mem()
    (staging / "asset.json").write_text(json.dumps({**version_meta(parent), "render_mesh": "output.glb (R4 glbbuild)",
                                                    "parent": str(parent)}))
    segs = {}
    for s in meter.spans:
        d = segs.setdefault(str(s["segment"]), {"seconds": 0.0, "parts": {}, "max_memory_allocated": 0, "rss_bytes_max": 0})
        d["seconds"] += s["seconds"]
        d["parts"][s["name"]] = d["parts"].get(s["name"], 0.0) + s["seconds"]
        for k in ("max_memory_allocated", "rss_bytes_max"):
            if s.get(k) is not None:
                d[k] = max(d[k], s[k])
    timing = {"segments": segs, "counts": counts, "steps": steps, "cache_entries": cache_entries,
              "mem_before": mem0, "mem_after_edit": mem1, "mem_after_release": mem2}
    (staging / "timing.json").write_text(json.dumps(timing, indent=1, default=str))
    return {"r4": w, "timing": timing}
