"""입력 파일 읽기(0-1). 상주 워커(R8a, 4번)는 여기에 붙는다 — 0-1 에서는 비워 둔다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .pipeline import EditRequest


def load_request(slat_path, mask_json, image_dir, seed=0, cfg=(5.0, 6.0, 0.0, 0.0)) -> EditRequest:
    from safetensors import safe_open
    with safe_open(str(slat_path), "pt") as f:
        coords, feats, meta = f.get_tensor("coords"), f.get_tensor("feats"), dict(f.metadata() or {})
    cells = np.asarray(json.loads(Path(mask_json).read_text())["mask"]["voxels"], dtype=np.int64)
    d = Path(image_dir)
    return EditRequest(slat_coords=coords, slat_feats=feats, mask_cells=cells,
                       img_src=Image.open(d / "2d_render.png"), img_tgt=Image.open(d / "2d_edit.png"),
                       img_mask=Image.open(d / "2d_mask.png"), seed=seed, cfg=tuple(cfg), slat_meta=meta)
