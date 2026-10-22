"""FLUX.1-Fill-dev (nunchaku SVDQuant INT4) 인페인트 워커. `flux-fill` conda 환경 전용.

zimage/trellis/birefnet 과 **다른 환경**에서 subprocess 로 실행된다 (같은 프로세스에서
섞지 않는다 — 이 리포의 기존 워커 규약과 같다).

부품 출처
  transformer : nunchaku-ai/nunchaku-flux.1-fill-dev  svdq-int4_r32  (비게이트)
  나머지      : camenduru/FLUX.1-Fill-dev-ungated      (비게이트 미러)
                ⚠️ 원본 black-forest-labs/FLUX.1-Fill-dev 는 게이트다. 미러의 vae·
                text_encoder·text_encoder_2 sha256 을 **서로 무관한 미러 3곳과 대조**해
                전부 일치하는 것을 확인하고 채택했다. 토큰이 생기면 원본으로 바꿔라.

마스크 규약: **흰색(255) = 다시 그릴 영역**, 검은색 = 보존 영역.
"""
import argparse, os, time
import torch
from PIL import Image
from diffusers import FluxFillPipeline
from nunchaku import NunchakuFluxTransformer2DModelV2
from nunchaku.utils import get_precision

BASE = os.environ.get("FLUX_FILL_BASE", "camenduru/FLUX.1-Fill-dev-ungated")
QUANT = os.environ.get("FLUX_FILL_TRANSFORMER", "nunchaku-ai/nunchaku-flux.1-fill-dev")

p = argparse.ArgumentParser()
p.add_argument("--image", required=True)
p.add_argument("--mask", required=True, help="흰색=다시 그릴 영역")
p.add_argument("--prompt", required=True)
p.add_argument("--out", required=True)
p.add_argument("--steps", type=int, default=30)
p.add_argument("--guidance", type=float, default=30.0)   # Fill-dev 권장값
p.add_argument("--seed", type=int, default=42)
p.add_argument("--offload", action="store_true", help="VRAM 부족 시 CPU 오프로드")
p.add_argument("--no-composite", dest="composite", action="store_false",
               help="마스크 밖을 원본으로 되돌리지 않는다 (기본은 되돌린다)")
a = p.parse_args()

t0 = time.time()
prec = get_precision()                       # sm_86(3090) → int4
tr = NunchakuFluxTransformer2DModelV2.from_pretrained(
    f"{QUANT}/svdq-{prec}_r32-flux.1-fill-dev.safetensors")
pipe = FluxFillPipeline.from_pretrained(BASE, transformer=tr, torch_dtype=torch.bfloat16)
if a.offload:
    pipe.enable_model_cpu_offload()
else:
    pipe.to("cuda")
print(f"[{time.time()-t0:.0f}s] 적재 완료 · precision={prec}", flush=True)

img = Image.open(a.image).convert("RGB")
msk = Image.open(a.mask).convert("L")
assert msk.size == img.size, f"마스크 크기가 다르다: {msk.size} vs {img.size}"

t1 = time.time()
out = pipe(
    prompt=a.prompt, image=img, mask_image=msk,
    height=img.height, width=img.width,
    guidance_scale=a.guidance, num_inference_steps=a.steps, max_sequence_length=512,
    generator=torch.Generator("cuda").manual_seed(a.seed),
).images[0]
# 🔴 마스크 밖 보존. FluxFill 은 잠재공간 전체를 디코드하므로 **손대지 않은 영역도
# VAE 왕복으로 전부 미세하게 바뀐다** — 실측 1.72% 화소가 흔들렸고 바이트 동일은 0.01%
# 였다. "마스크 밖은 바이트 단위로 불변" 이 이 리포의 요구(보존 B)이므로 원본을 되돌린다.
if a.composite:
    import numpy as np
    o, s_, mk = np.array(out.convert("RGB")), np.array(img), np.array(msk) > 127
    o[~mk] = s_[~mk]
    out = Image.fromarray(o)
    print(f"합성: 마스크 밖 {int((~mk).sum()):,} 화소를 원본으로 되돌렸다 (바이트 동일)")
out.save(a.out)
print(f"[{time.time()-t1:.0f}s] 추론 완료")
print(f"FLUXFILL_OK {a.out}  peak_vram={torch.cuda.max_memory_allocated()/2**30:.1f}GiB  total={time.time()-t0:.0f}s")
