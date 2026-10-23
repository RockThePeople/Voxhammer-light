"""utils/inpaint.py 를 이 카드(24.56 GB)에서 돌리기 위한 구동기 — 파라미터는 하나도 안 바꾼다.

왜 필요한가 (실측):
    FluxFillPipeline 가중치 합계 34.25 GB  >  이 카드 24.56 GB
      transformer 23.81 · text_encoder_2(T5) 9.52 · text_encoder 0.25 · vae 0.34
    원저자 `inpaint.py` 는 `.to("cuda")` 로 통째로 올린다 → transformer 를 올린 뒤
    T5 를 올리다 OOM. 실측: "Tried to allocate 18.00 MiB ... 23.50 GiB in use".
    아슬아슬한 것이 아니라 10 GB 부족이다.

무엇을 바꾸는가:
    `.to("cuda")` 대신 `enable_model_cpu_offload()`.
    ⚠️ 이것은 **텐서가 어느 메모리에 앉아 있는가**만 바꾼다. 필요한 모듈만 GPU 로 올렸다 내린다.
    dtype(bf16) · 해상도(1024²) · guidance(30) · 스텝(50) · 시드(0) 전부 그대로이고
    연산 순서·정밀도가 같으므로 **산출 이미지가 달라지지 않는다.** 느려질 뿐이다.
    ⚠️ `.to("cuda")` 와 같이 쓰면 안 된다 — 오프로딩 훅이 깨진다. 여기서는 부르지 않는다.

인페인트 본체는 원저자 `utils.inpaint.inpaint` 를 **그대로 호출**한다. 인자를 하나도 안 넘겨
기본값(height/width/guidance/steps/max_seq/seed)이 전부 원저자 값으로 남게 한다.
VoxHammer 소스는 수정하지 않는다 (D51 7줄 최소성 유지).
"""
import argparse
import os
import shutil
import sys
import time

VOXHAMMER = os.environ.get("VOXHAMMER_DIR", "<HOME>/ai-ar-v2/VoxHammer")

ap = argparse.ArgumentParser()
ap.add_argument("--image_path", required=True)
ap.add_argument("--mask_path", required=True)
ap.add_argument("--output_dir", required=True)
ap.add_argument("--prompt", required=True)
ap.add_argument("--sequential", action="store_true",
                help="①로도 OOM 일 때만. enable_sequential_cpu_offload() 로 내린다 (훨씬 느리다)")
args = ap.parse_args()

os.chdir(VOXHAMMER)
sys.path.insert(0, VOXHAMMER)

import torch  # noqa: E402
from diffusers import FluxFillPipeline  # noqa: E402
from utils.inpaint import inpaint  # noqa: E402  ← 원저자 함수 그대로

os.makedirs(args.output_dir, exist_ok=True)

t0 = time.time()
pipeline = FluxFillPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16
)
if args.sequential:
    pipeline.enable_sequential_cpu_offload()
    mode = "sequential_cpu_offload"
else:
    pipeline.enable_model_cpu_offload()
    mode = "model_cpu_offload"
t_load = time.time() - t0
print(f"[시간] 적재 {t_load:.1f}s  ({mode} · .to('cuda') 미사용)", flush=True)

t1 = time.time()
inpaint(
    pipeline=pipeline,
    image_path=args.image_path,
    mask_path=args.mask_path,
    output_path=os.path.join(args.output_dir, "2d_edit.png"),
    prompt=args.prompt,
)  # ← 그 외 인자 없음. 1024² · guidance 30 · 50스텝 · max_seq 512 · seed 0 (원저자 기본값)
t_inf = time.time() - t1

shutil.copy(args.image_path, os.path.join(args.output_dir, "2d_render.png"))
shutil.copy(args.mask_path, os.path.join(args.output_dir, "2d_mask.png"))

peak = torch.cuda.max_memory_allocated() / 2**30
print(f"[시간] 인페인트 {t_inf:.1f}s   총 {t_load + t_inf:.1f}s")
print(f"[VRAM] PyTorch 최대 할당 {peak:.2f} GiB")
print(f"DONE {args.output_dir}")
