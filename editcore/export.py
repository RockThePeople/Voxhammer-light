"""디코딩·GLB·R2 저장 — 파일 쓰기는 여기와 service 만."""
from __future__ import annotations

import json
from pathlib import Path

import torch


# origin: voxhammer/edit_pipeline.py:run_edit (624–628행) @ f33cd96+patch
def decode_to_glb(pipeline, slat_tgt, output_path, hooks=None):
    span = (hooks or {}).get("span")
    ctx = span("decode_slat", 7) if span else _null()
    with ctx, torch.no_grad():
        assets = pipeline.decode_slat(slat_tgt, ["gaussian", "mesh"])
    from trellis.utils import postprocessing_utils
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(True)  # 포크와 같이 to_glb(텍스처 굽기)는 grad 를 켠 채로
    try:
        ctx = span("to_glb", 7) if span else _null()
        with ctx:
            glb = postprocessing_utils.to_glb(assets["gaussian"][0], assets["mesh"][0], simplify=0.95, texture_size=1024)
        ctx = span("glb_export", 7) if span else _null()
        with ctx:
            glb.export(str(output_path))
    finally:
        torch.set_grad_enabled(prev)
    return assets


def save_slat(slat, path, normalization, extra=None):
    """R2 형식: coords int32 (N,4), feats float32 (N,8), slat_space=denormalized."""
    from safetensors.torch import save_file
    meta = {"slat_space": "denormalized", "slat_channels": str(slat.feats.shape[1]), "voxel_res": "64",
            "coord_order": "canonical",
            "norm_mean": json.dumps(list(map(float, normalization["mean"]))),
            "norm_std": json.dumps(list(map(float, normalization["std"])))}
    meta.update({k: (v if isinstance(v, str) else json.dumps(v)) for k, v in (extra or {}).items()})
    save_file({"coords": slat.coords.detach().cpu().to(torch.int32).contiguous(),
               "feats": slat.feats.detach().cpu().float().contiguous()}, str(path), metadata=meta)
    return Path(path)


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
