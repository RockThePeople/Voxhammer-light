"""구간 2 인페인팅 계측 — 현행 A5000 경로와 같은 설정 (00 A-5).

포크 `utils/inpaint.py` 의 `inpaint()` 를 기본값 그대로 호출한다
(1024², guidance 30, 50 스텝, max_sequence_length 512, CPU generator seed 0 = 정오표 E5).
FluxFillPipeline 은 bf16 으로 올리고 `.to("cuda")` 대신 `enable_model_cpu_offload()` 를 쓴다
(가중치 합이 24 GB 를 넘어 통째로 올리면 OOM — 00a A6). 이 오프로딩은 텐서 위치만 바꾼다.
장비 간 전송은 이 경로에 없다(같은 장비) → 0 으로 기록.

출력 디렉터리에 `2d_render.png`·`2d_mask.png`·`2d_edit.png` 와 `timing.json` 을 둔다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True, help="조건 렌더 RGB (render_000N.png)")
ap.add_argument("--mask", required=True, help="2D 마스크 (mask_000N.png)")
ap.add_argument("--prompt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model_id", default="black-forest-labs/FLUX.1-Fill-dev")
ap.add_argument("--reference", default=os.environ.get("REFERENCE_DIR", "build/reference-v2"))
a = ap.parse_args()

ROOT = Path(__file__).resolve().parents[1]
REF = (ROOT / a.reference).resolve()
sys.path.insert(0, str(REF))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
from diffusers import FluxFillPipeline  # noqa: E402

from reference.meter import Meter, Sampler, ru_maxrss_bytes  # noqa: E402
from utils.inpaint import inpaint  # noqa: E402  포크 함수 그대로

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
sampler = Sampler().start()
meter = Meter(sampler)

with meter.span("flux_load"):
    pipe = FluxFillPipeline.from_pretrained(a.model_id, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
with meter.span("flux_inpaint"):
    inpaint(pipeline=pipe, image_path=a.image, mask_path=a.mask,
            output_path=str(out / "2d_edit.png"), prompt=a.prompt)
sampler.stop()

shutil.copy(a.image, out / "2d_render.png")
shutil.copy(a.mask, out / "2d_mask.png")
sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("2d_*.png"))}
res = {"model_id": a.model_id, "prompt": a.prompt, "seed": 0, "steps": 50, "guidance": 30,
       "resolution": 1024, "dtype": "bfloat16", "offload": "enable_model_cpu_offload",
       "transfer_seconds": 0.0, "spans": meter.spans, "outputs_sha256": sha,
       "ru_maxrss_bytes": ru_maxrss_bytes(), "torch": torch.__version__,
       "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE")}
(out / "timing.json").write_text(json.dumps(res, indent=2, default=str))
for s in meter.spans:
    print(f"  {s['name']:14s} {s['seconds']:8.2f}s  peak {s['max_memory_allocated'] / 2**30:.2f} GiB  "
          f"smi {s.get('nvidia_smi_mib_max')} MiB  rss {(s.get('rss_bytes_max') or 0) / 2**30:.1f} GiB")
