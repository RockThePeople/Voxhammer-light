"""조건 이미지 전처리 — 공동 bbox crop, 518² 리사이즈, 배경 제거(rembg). 파일 입출력·세션 생성 없음.

rembg 세션은 호출자가 만들어 넘긴다(포크는 함수 안에서 이미지마다 `new_session("u2net")` 을 만든다 —
세션 재사용은 순서표 7번 R8b 의 일이므로 0-1 에서는 호출자(pipeline)가 이미지마다 새 세션을 만든다).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .models import COND_IMAGE_SIZE


# origin: voxhammer/edit_pipeline.py:image_rgb @ f33cd96+patch
def image_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    image = image.convert("RGBA")
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3])
    return background


# origin: voxhammer/edit_pipeline.py:preprocess_image @ f33cd96+patch
def preprocess_images(img_src: Image.Image, img_tgt: Image.Image, img_mask: Image.Image, session_src, session_tgt):
    import rembg
    img_src = image_rgb(img_src)
    img_tgt = image_rgb(img_tgt)
    img_mask = img_mask.convert("L")
    max_size = max(img_src.size)
    scale = min(1, 1024 / max_size)
    resize_size = (int(img_src.width * scale), int(img_src.height * scale))
    if scale < 1:
        img_src = img_src.resize(resize_size, Image.Resampling.LANCZOS)
        img_tgt = img_tgt.resize(resize_size, Image.Resampling.LANCZOS)
        img_mask = img_mask.resize(resize_size, Image.Resampling.LANCZOS)
    pre_img_src = rembg.remove(img_src, session=session_src)
    pre_img_tgt = rembg.remove(img_tgt, session=session_tgt)
    pre_img_src_np = np.array(pre_img_src)
    pre_img_tgt_np = np.array(pre_img_tgt)
    alpha_src = pre_img_src_np[:, :, 3]
    alpha_tgt = pre_img_tgt_np[:, :, 3]
    bbox_src = np.argwhere(alpha_src > 0.8 * 255)
    bbox_tgt = np.argwhere(alpha_tgt > 0.8 * 255)
    # 공동 bbox: 원본·편집 이미지의 알파 bbox 합집합을 1.2 배 정사각형으로
    bbox = (
        min(np.min(bbox_src[:, 1]), np.min(bbox_tgt[:, 1])),
        min(np.min(bbox_src[:, 0]), np.min(bbox_tgt[:, 0])),
        max(np.max(bbox_src[:, 1]), np.max(bbox_tgt[:, 1])),
        max(np.max(bbox_src[:, 0]), np.max(bbox_tgt[:, 0])),
    )
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.2)
    bbox = (center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2)
    s = COND_IMAGE_SIZE
    pre_img_src = pre_img_src.crop(bbox).resize((s, s), Image.Resampling.LANCZOS)
    pre_img_tgt = pre_img_tgt.crop(bbox).resize((s, s), Image.Resampling.LANCZOS)
    pre_img_mask = img_mask.crop(bbox).resize((s, s), Image.Resampling.LANCZOS)
    pre_img_src = np.array(pre_img_src).astype(np.float32) / 255
    pre_img_tgt = np.array(pre_img_tgt).astype(np.float32) / 255
    pre_img_src = pre_img_src[:, :, :3] * pre_img_src[:, :, 3:4]
    pre_img_tgt = pre_img_tgt[:, :, :3] * pre_img_tgt[:, :, 3:4]
    pre_img_src = Image.fromarray((pre_img_src * 255).astype(np.uint8))
    pre_img_tgt = Image.fromarray((pre_img_tgt * 255).astype(np.uint8))
    return pre_img_src, pre_img_tgt, pre_img_mask
