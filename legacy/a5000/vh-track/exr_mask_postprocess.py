"""render_rgb_and_mask.py 의 후처리 단계를 그대로 재현한다 — EXR 만 bpy 로 읽는다.

왜 필요한가 (실측):
  이 기계의 OpenCV 는 `cv2.getBuildInformation()` 에 **OpenEXR: NO** 다 (cv2 5.0.0).
  그래서 `load_depth()` 의 `cv2.imread(*.exr)` 가 **None 을 돌려주고**
  `[..., 0]` 에서 TypeError 로 죽는다. 렌더(5뷰 × 2회)는 이미 끝난 뒤다.

  ⚠️ 원저자 코드에는 별개의 잠재 결함이 하나 더 있다 —
     `import cv2` 가 7행, `os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"` 이 17행이다.
     OpenCV 는 그 변수를 **import 시점에** 읽으므로 너무 늦다. 다만 이 기계에서는
     빌드 자체에 EXR 이 없어서 순서를 고쳐도 안 된다 — 확인함.

VoxHammer 소스는 수정하지 않는다 (D51 7줄 최소성 유지). 렌더 산출물만 받아 후처리한다.

깊이 비교식·임계값·출력 이름은 원저자와 **글자 그대로** 같다:
    mask  = ~(depth > 1000.0);  depth[~mask] = 10000.0
    diff  = (model_depth - mask_depth > 1e-4)      # 부호 있음 = 가림 고려
    diff[mask_mask <= 0.5] = 0
"""
import os
import sys
import glob
import shutil

import numpy as np


def read_exr_depth(path):
    """EXR 첫 채널을 읽는다. Blender 는 아래→위로 저장하므로 뒤집는다."""
    import bpy

    img = bpy.data.images.load(path)
    try:
        w, h, ch = img.size[0], img.size[1], img.channels
        buf = np.empty(w * h * ch, dtype=np.float32)
        img.pixels.foreach_get(buf)
        return buf.reshape(h, w, ch)[::-1, :, 0].copy()
    finally:
        bpy.data.images.remove(img)


def load_depth(path):
    depth = read_exr_depth(path)
    mask = ~(depth > 1000.0)
    depth[mask < 0.5] = 10000.0
    return depth, mask


def main(out_dir):
    import cv2

    model_dir = os.path.join(out_dir, "rgb")
    mask_dir = os.path.join(out_dir, "mask_3d")
    render_dir = os.path.join(out_dir, "images")
    for d in (model_dir, mask_dir):
        if not os.path.isdir(d):
            sys.exit(f"🔴 {d} 가 없다 — render_rgb_and_mask.py 의 렌더 단계가 먼저 돌아야 한다")
    os.makedirs(render_dir, exist_ok=True)

    md = sorted(glob.glob(os.path.join(model_dir, "*.exr")))
    kd = sorted(glob.glob(os.path.join(mask_dir, "*.exr")))
    if len(md) != len(kd):
        sys.exit(f"🔴 깊이 파일 수가 다르다: {len(md)} vs {len(kd)}")

    for i, (a, b) in enumerate(zip(md, kd)):
        model_depth, _ = load_depth(a)
        mask_depth, mask_mask = load_depth(b)
        diff = (model_depth - mask_depth > 1e-4).astype(np.uint8)
        diff[mask_mask <= 0.5] = 0
        cv2.imwrite(os.path.join(render_dir, f"mask_{i:04d}.png"), diff * 255)
        print(f"  mask_{i:04d}.png  마스크 화소 {int(diff.sum()):7d} ({100*diff.mean():5.2f}%)")

    for f in sorted(glob.glob(os.path.join(model_dir, "*.png"))) + [
        os.path.join(model_dir, "meta.json")
    ]:
        if os.path.exists(f):
            shutil.copy(f, os.path.join(render_dir, os.path.basename(f)))
    print(f"→ {render_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
