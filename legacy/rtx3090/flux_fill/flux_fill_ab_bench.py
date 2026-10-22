"""FLUX.1-Fill-dev — 원본 bf16 vs nunchaku INT4 를 같은 프로토콜로 계측.

⚠️ 공정성 주의
  원본 bf16 transformer 는 **22.2 GiB** 다. T5(9.5GB)·VAE 까지 얹으면 3090(24GiB)에
  안 들어간다. 그래서 원본은 `enable_model_cpu_offload()` 로 돌린다 — 디노이즈 동안엔
  transformer 가 GPU 에 상주하므로 **스텝당 시간은 공정한 연산 비교**지만, 호출 총시간엔
  매 호출 22 GiB PCIe 전송이 끼어 있다. 그래서 둘을 분리해 잰다.
"""
import argparse, time, statistics as st
import torch
from PIL import Image
from diffusers import FluxFillPipeline, FluxTransformer2DModel

BASE = "camenduru/FLUX.1-Fill-dev-ungated"
p = argparse.ArgumentParser()
p.add_argument("--backend", choices=["int4", "bf16"], required=True)
p.add_argument("--image", required=True); p.add_argument("--mask", required=True)
p.add_argument("--prompt", required=True); p.add_argument("--out")
p.add_argument("--steps", type=int, default=30); p.add_argument("--repeat", type=int, default=3)
a = p.parse_args()
sync = torch.cuda.synchronize

t0 = time.perf_counter()
if a.backend == "int4":
    from nunchaku import NunchakuFluxTransformer2DModelV2
    from nunchaku.utils import get_precision
    tr = NunchakuFluxTransformer2DModelV2.from_pretrained(
        f"nunchaku-ai/nunchaku-flux.1-fill-dev/svdq-{get_precision()}_r32-flux.1-fill-dev.safetensors")
else:
    tr = FluxTransformer2DModel.from_pretrained(BASE, subfolder="transformer", torch_dtype=torch.bfloat16)
nb = sum(q.numel()*q.element_size() for q in tr.parameters()) + \
     sum(q.numel()*q.element_size() for q in tr.buffers())
pipe = FluxFillPipeline.from_pretrained(BASE, transformer=tr, torch_dtype=torch.bfloat16)
if a.backend == "int4":
    pipe.to("cuda"); mode = "전체 GPU 상주"
else:
    pipe.enable_model_cpu_offload(); mode = "model_cpu_offload (VRAM 부족)"
sync(); load = time.perf_counter()-t0

img = Image.open(a.image).convert("RGB"); msk = Image.open(a.mask).convert("L")
steps_t = []
def cb(pipe_, i, t_, kw):
    sync(); steps_t.append(time.perf_counter()); return kw
kw = dict(prompt=a.prompt, image=img, mask_image=msk, height=img.height, width=img.width,
          guidance_scale=30.0, max_sequence_length=512, num_inference_steps=a.steps,
          callback_on_step_end=cb)

pipe(**kw, generator=torch.Generator("cuda").manual_seed(0)); sync()   # 워밍업 버림
tot, per = [], []
for _ in range(a.repeat):
    steps_t.clear(); t = time.perf_counter()
    out = pipe(**kw, generator=torch.Generator("cuda").manual_seed(42)).images[0]
    sync(); tot.append(time.perf_counter()-t)
    per.append((steps_t[-1]-steps_t[0])/(len(steps_t)-1))

print(f"backend={a.backend}  가중치 {nb/2**30:6.2f} GiB  적재 {load:5.1f}s  [{mode}]")
print(f"  호출 총시간 {a.steps}스텝 : " + " / ".join(f"{x:.2f}s" for x in tot) + f"  평균 {st.mean(tot):6.2f}s")
print(f"  디노이즈 스텝당           : {st.mean(per)*1000:6.0f} ms  → 30스텝 환산 {st.mean(per)*30:5.2f}s")
print(f"  peak VRAM {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
if a.out:
    import numpy as np
    o, s_, mk = np.array(out.convert("RGB")), np.array(img), np.array(msk) > 127
    o[~mk] = s_[~mk]; Image.fromarray(o).save(a.out); print(f"  저장 {a.out}")
