"""구간 1 조건 렌더 계측 — 포크 `utils/render_rgb_and_mask.py` 를 수정 없이 감싼다 (00 A-5, A-5b).

`process(model, mask, out)` 는 `render_model` 을 원본·마스크로 두 번 부르고 2D 마스크 후처리를 한다.
모듈 속성 `render_model` 을 감싸 두 렌더를 따로 재고, 나머지(EXR 읽기·마스크·합성)를 후처리로 친다.
포크는 상대경로(`assets/preset/...`)를 쓰므로 cwd 를 기준 사본으로 옮긴다.

`OPENCV_IO_ENABLE_OPENEXR` 는 cv2 import 전에 정해야 하므로 인자로 받아 프로세스 시작 때 설정한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--mask", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--exr", choices=["on", "off"], default="on", help="OPENCV_IO_ENABLE_OPENEXR=1 설정 여부")
ap.add_argument("--reference", default=os.environ.get("REFERENCE_DIR", "build/reference-v2"))
a = ap.parse_args()

if a.exr == "on":
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
else:
    os.environ.pop("OPENCV_IO_ENABLE_OPENEXR", None)

ROOT = Path(__file__).resolve().parents[1]
REF = (ROOT / a.reference).resolve() if not os.path.isabs(a.reference) else Path(a.reference)
model, mask, out = (str(Path(p).resolve()) for p in (a.model, a.mask, a.out))
sys.path.insert(0, str(REF))
sys.path.insert(0, str(ROOT))
os.chdir(REF)

from reference.meter import Meter, Sampler, ru_maxrss_bytes  # noqa: E402

sampler = Sampler().start()
meter = Meter(sampler)
t_import = time.perf_counter()
import bpy  # noqa: E402
import utils.render_rgb_and_mask as rr  # noqa: E402
t_import = time.perf_counter() - t_import

import cv2  # noqa: E402
import numpy as np  # noqa: E402

CV2_HAS_EXR = any("OpenEXR" in l and "YES" in l for l in cv2.getBuildInformation().splitlines())


class _Cv2ExrViaBpy:
    """이 env 의 opencv 5.0.0 은 OpenEXR 없이 빌드됐다(`getBuildInformation`: OpenEXR NO) —
    `OPENCV_IO_ENABLE_OPENEXR=1` 을 줘도 `imread(.exr)` 가 None 이다(00 A-5 실측).
    포크의 `load_depth` 는 그대로 두고, 모듈이 보는 `cv2` 만 이 프록시로 바꾼다:
    `.exr` 는 bpy 로 읽어 cv2 와 같은 모양(위→아래 행, BGRA, float32)으로 돌려주고
    나머지 호출은 전부 진짜 cv2 로 넘긴다."""

    def __init__(self, real):
        self._real = real
        self.exr_reads = 0

    def __getattr__(self, k):
        return getattr(self._real, k)

    def imread(self, path, flags=None):
        if not str(path).lower().endswith(".exr"):
            return self._real.imread(path) if flags is None else self._real.imread(path, flags)
        img = bpy.data.images.load(str(path), check_existing=False)
        try:
            w, h = img.size
            a = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, img.channels)[::-1]
        finally:
            bpy.data.images.remove(img)
        self.exr_reads += 1
        if a.shape[2] >= 3:
            a = a[..., [2, 1, 0] + list(range(3, a.shape[2]))]
        return np.ascontiguousarray(a)


exr_reader = "cv2"
if not CV2_HAS_EXR:
    rr.cv2 = _Cv2ExrViaBpy(cv2)
    exr_reader = "bpy (cv2 proxy)"

names = iter(["render_source_5views", "render_mask_5views"])
orig_render = rr.render_model


def render_model(input_model, output_dir):
    with meter.span(next(names)):
        r = orig_render(input_model, output_dir)
    scn = bpy.context.scene
    meter.spans[-1].update(engine=scn.render.engine,
                           cycles_samples=getattr(scn.cycles, "samples", None),
                           cycles_device=getattr(scn.cycles, "device", None),
                           resolution=[scn.render.resolution_x, scn.render.resolution_y])
    return r


rr.render_model = render_model
os.makedirs(out, exist_ok=True)
error = None
with meter.span("condition_render_total"):
    try:
        rr.process(model, mask, out)
    except Exception as e:  # 기록하고 끝낸다 — EXR off 비교에서 실패가 예상된다
        error = f"{type(e).__name__}: {e}"
rr.render_model = orig_render
sampler.stop()

spans = {s["name"]: s for s in meter.spans}
tot = spans["condition_render_total"]["seconds"]
post = tot - sum(spans[n]["seconds"] for n in ("render_source_5views", "render_mask_5views") if n in spans)
img_dir = Path(out) / "images"
files = {}
if img_dir.is_dir():
    for p in sorted(img_dir.iterdir()):
        files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
res = {
    "model": model, "mask": mask, "exr_env": a.exr,
    "cv2_version": cv2.__version__, "cv2_has_openexr": CV2_HAS_EXR, "exr_reader": exr_reader,
    "exr_reads": getattr(rr.cv2, "exr_reads", None),
    "OPENCV_IO_ENABLE_OPENEXR": os.environ.get("OPENCV_IO_ENABLE_OPENEXR"),
    "import_seconds": t_import, "spans": meter.spans, "calls": meter.calls,
    "mask_postprocess_seconds": post, "error": error, "outputs_sha256": files,
    "ru_maxrss_bytes": ru_maxrss_bytes(),
}
Path(out, "timing.json").write_text(json.dumps(res, indent=2, default=str))
print(json.dumps({k: res[k] for k in ("exr_env", "error", "mask_postprocess_seconds")}, default=str))
for s in meter.spans:
    print(f"  {s['name']:26s} {s['seconds']:8.2f}s  samples={s.get('cycles_samples')} dev={s.get('cycles_device')}")
