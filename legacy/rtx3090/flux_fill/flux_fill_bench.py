"""FLUX.1-Fill-dev INT4 편집 소요 시간 계측. 적재/추론을 분리하고 워밍업 후 반복 측정한다.

⚠️ 첫 호출은 CUDA 커널 컴파일·캐시 때문에 느리다. 그걸 '편집 시간' 이라고 부르면
   실제보다 부풀려진다. 그래서 워밍업 1회를 버리고 그 뒤 N회를 잰다.
"""
import argparse, time, statistics as st
import torch
from PIL import Image
from diffusers import FluxFillPipeline
from nunchaku import NunchakuFluxTransformer2DModelV2
from nunchaku.utils import get_precision

p = argparse.ArgumentParser()
p.add_argument("--image", required=True); p.add_argument("--mask", required=True)
p.add_argument("--prompt", required=True); p.add_argument("--out", required=True)
p.add_argument("--repeat", type=int, default=3)
p.add_argument("--steps", type=int, nargs="+", default=[30])
a = p.parse_args()

def sync(): torch.cuda.synchronize()

T = {}
t = time.perf_counter()
prec = get_precision()
tr = NunchakuFluxTransformer2DModelV2.from_pretrained(
    f"nunchaku-ai/nunchaku-flux.1-fill-dev/svdq-{prec}_r32-flux.1-fill-dev.safetensors")
sync(); T["① transformer 적재 (INT4 6.77GB)"] = time.perf_counter()-t

t = time.perf_counter()
pipe = FluxFillPipeline.from_pretrained("camenduru/FLUX.1-Fill-dev-ungated",
                                        transformer=tr, torch_dtype=torch.bfloat16)
sync(); T["② 나머지 부품 적재 (VAE·T5·CLIP)"] = time.perf_counter()-t

t = time.perf_counter(); pipe.to("cuda"); sync()
T["③ GPU 전송"] = time.perf_counter()-t

img = Image.open(a.image).convert("RGB"); msk = Image.open(a.mask).convert("L")
kw = dict(prompt=a.prompt, image=img, mask_image=msk, height=img.height, width=img.width,
          guidance_scale=30.0, max_sequence_length=512)

print("═" * 66)
for k, v in T.items(): print(f"  {k:38s} {v:7.2f} s")
print(f"  {'── 적재 합계 (1회만)':38s} {sum(T.values()):7.2f} s")
print("═" * 66)

res = None
for steps in a.steps:
    # 워밍업 — 버린다
    pipe(**kw, num_inference_steps=steps,
         generator=torch.Generator("cuda").manual_seed(0)); sync()
    xs = []
    for i in range(a.repeat):
        t = time.perf_counter()
        out = pipe(**kw, num_inference_steps=steps,
                   generator=torch.Generator("cuda").manual_seed(42)).images[0]
        sync(); xs.append(time.perf_counter()-t)
    m = st.mean(xs)
    print(f"  {steps:2d} steps · {a.repeat}회 → 각 " + " / ".join(f"{x:.2f}s" for x in xs)
          + f"  │ 평균 {m:5.2f}s · 스텝당 {m/steps*1000:5.0f}ms")
    if res is None: res = out
print("═" * 66)
print(f"  peak VRAM {torch.cuda.max_memory_allocated()/2**30:.1f} GiB / "
      f"{torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GiB")

import numpy as np
o, s_, mk = np.array(res.convert("RGB")), np.array(img), np.array(msk) > 127
o[~mk] = s_[~mk]
Image.fromarray(o).save(a.out)
print(f"  저장 {a.out} (마스크 밖 {int((~mk).sum()):,} 화소 원본 복원)")
