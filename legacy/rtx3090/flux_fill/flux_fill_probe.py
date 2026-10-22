"""FLUX Fill 한 번 실행하며 ① VRAM 피크 ② 오프로드 실태 ③ 순수 동작시간을 함께 잰다."""
import argparse, json, os, subprocess, threading, time
import torch
from PIL import Image
from diffusers import FluxFillPipeline, FluxTransformer2DModel

BASE="camenduru/FLUX.1-Fill-dev-ungated"
ap=argparse.ArgumentParser()
ap.add_argument("--backend", choices=["bf16","int4"], required=True)
ap.add_argument("--image", required=True); ap.add_argument("--mask", required=True)
ap.add_argument("--prompt", required=True); ap.add_argument("--steps", type=int, default=30)
a=ap.parse_args()

# ── 외부 관측: nvidia-smi 로 프로세스 VRAM 을 50ms 간격 샘플링 (CUDA 컨텍스트·단편화 포함)
stop=threading.Event(); samples=[]
def sampler():
    me=str(os.getpid())
    while not stop.is_set():
        try:
            o=subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=2).stdout
            for line in o.strip().splitlines():
                pid,mb=[x.strip() for x in line.split(",")]
                if pid==me: samples.append((time.perf_counter(), int(mb)))
        except Exception: pass
        time.sleep(0.05)
threading.Thread(target=sampler,daemon=True).start()

sync=torch.cuda.synchronize
t_load=time.perf_counter()
if a.backend=="int4":
    from nunchaku import NunchakuFluxTransformer2DModelV2
    from nunchaku.utils import get_precision
    tr=NunchakuFluxTransformer2DModelV2.from_pretrained(
        f"nunchaku-ai/nunchaku-flux.1-fill-dev/svdq-{get_precision()}_r32-flux.1-fill-dev.safetensors")
else:
    tr=FluxTransformer2DModel.from_pretrained(BASE, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe=FluxFillPipeline.from_pretrained(BASE, transformer=tr, torch_dtype=torch.bfloat16)
if a.backend=="int4": pipe.to("cuda"); mode="상주 (오프로드 없음)"
else:                 pipe.enable_model_cpu_offload(); mode="model_cpu_offload"
sync(); load=time.perf_counter()-t_load

img=Image.open(a.image).convert("RGB"); msk=Image.open(a.mask).convert("L")
marks={}
def cb(p_,i,t_,kw):
    sync(); marks.setdefault("first_step",time.perf_counter()); marks["last_step"]=time.perf_counter(); return kw
kw=dict(prompt=a.prompt,image=img,mask_image=msk,height=img.height,width=img.width,
        guidance_scale=30.0,max_sequence_length=512,num_inference_steps=a.steps,callback_on_step_end=cb)

pipe(**kw, generator=torch.Generator("cuda").manual_seed(0)); sync()   # 워밍업
torch.cuda.reset_peak_memory_stats(); samples.clear(); marks.clear()

t0=time.perf_counter()
out=pipe(**kw, generator=torch.Generator("cuda").manual_seed(42)).images[0]
sync(); t1=time.perf_counter()
stop.set(); time.sleep(0.2)

pre  = marks["first_step"]-t0                     # 호출~첫 스텝: 텍스트 인코딩 + (오프로드면) 가중치 적재
den  = marks["last_step"]-marks["first_step"]     # 디노이즈 본체
post = t1-marks["last_step"]                      # VAE 디코딩 + (오프로드면) 회수
peak_t = torch.cuda.max_memory_allocated()/2**30
peak_r = torch.cuda.max_memory_reserved()/2**30
peak_n = max(m for _,m in samples)/1024 if samples else float("nan")
low_n  = min(m for _,m in samples)/1024 if samples else float("nan")
print(json.dumps({
    "backend":a.backend,"mode":mode,"steps":a.steps,"load_s":round(load,2),
    "total_s":round(t1-t0,3),"pre_s":round(pre,3),"denoise_s":round(den,3),"post_s":round(post,3),
    "per_step_ms":round(den/(a.steps-1)*1000,1),
    "peak_torch_alloc_GiB":round(peak_t,2),"peak_torch_reserved_GiB":round(peak_r,2),
    "peak_nvidia_smi_GiB":round(peak_n,2),"min_nvidia_smi_GiB":round(low_n,2),
    "gpu_total_GiB":round(torch.cuda.get_device_properties(0).total_memory/2**30,2),
    "samples":len(samples)}, ensure_ascii=False))
