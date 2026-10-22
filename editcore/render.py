"""조건 렌더 — 0-1 에서는 포크 렌더러(`utils/render_rgb_and_mask.py`) 호출 래퍼만 둔다.

이 장비의 opencv 는 OpenEXR 없이 빌드돼 있어 0번에서 만든 계측 래퍼(`reference/run_condition_render.py`,
EXR 만 bpy 로 읽는 cv2 프록시)를 별도 프로세스로 부른다. R10a(3번)에서 교체한다.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .models import REPO, TRELLIS_ROOT


def render_condition(model_glb, mask_glb, out_dir, python=None) -> Path:
    py = python or os.environ.get("VHL_RENDER_PY", sys.executable)
    subprocess.run([py, str(REPO / "reference" / "run_condition_render.py"), "--model", str(model_glb),
                    "--mask", str(mask_glb), "--out", str(out_dir), "--reference", str(TRELLIS_ROOT)], check=True)
    return Path(out_dir) / "images"
