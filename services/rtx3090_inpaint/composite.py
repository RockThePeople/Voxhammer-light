"""R9 마스크 밖 합성과 차이 통계 (torch 없이 numpy·PIL 만, CPU 테스트 대상).

규칙(05-R9 R-2): `2d_edit` 의 마스크 밖 픽셀은 원본 렌더로 덮는다. 2진 마스크, 경계 페더 없음.
마스크 규약은 포크와 같다: 흰색(>127) = 다시 그릴 영역.
"원본 렌더" 는 모델이 실제로 본 RGB 다 — RGBA 면 포크 `utils/inpaint.py` 와 같이 흰 배경에 합성한 RGB.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

MASK_THRESHOLD = 127


def render_rgb(image: Image.Image) -> Image.Image:
    # origin: utils/inpaint.py:inpaint (RGBA → 흰 배경 RGB) @ f33cd96+patch
    if image.mode == "RGBA":
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        bg.paste(image, (0, 0), image)
        return bg.convert("RGB")
    return image.convert("RGB")


def mask_bool(mask: Image.Image) -> np.ndarray:
    """True = 다시 그릴 영역."""
    return np.asarray(mask.convert("L")) > MASK_THRESHOLD


def composite(edit: Image.Image, render: Image.Image, mask: Image.Image) -> Image.Image:
    e = np.array(edit.convert("RGB"))
    r = np.asarray(render_rgb(render))
    m = mask_bool(mask)
    if e.shape != r.shape or m.shape != e.shape[:2]:
        raise ValueError(f"크기 불일치: edit {e.shape}, render {r.shape}, mask {m.shape}")
    e[~m] = r[~m]
    return Image.fromarray(e)


def outside_diff(edit: Image.Image, render: Image.Image, mask: Image.Image) -> dict:
    """마스크 밖에서 edit 과 원본 렌더의 차이. 합성 전에는 VAE 왕복 흔들림, 합성 후에는 0 이어야 한다."""
    e = np.asarray(edit.convert("RGB")).astype(np.int16)
    r = np.asarray(render_rgb(render)).astype(np.int16)
    m = mask_bool(mask)
    d = np.abs(e - r).max(axis=-1)[~m]
    n = int(d.size)
    return {
        "outside_pixels": n,
        "mask_pixels": int(m.sum()),
        "outside_identical_frac": float((d == 0).mean()) if n else None,
        "outside_gt16_frac": float((d > 16).mean()) if n else None,
        "outside_max_abs": int(d.max()) if n else None,
        "outside_mean_abs": float(d.mean()) if n else None,
    }
