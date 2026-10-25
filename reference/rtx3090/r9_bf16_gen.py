"""05 R9 원인 분리 실험(PR #7 chief 2): 3090 에서 bf16 FLUX.1-Fill-dev 로 트랙 A 6건 × N회.

목적: (a) 장비·커널 차이 = 3090 bf16 vs A5000 bf16 동결본, (b) 양자화 차이 = 3090 INT4 vs 3090 bf16,
(c) 마스크 밖 합성 효과 = 합성 전(`2d_edit_raw`) vs 합성 후(`2d_edit`).

- 파이프라인 구성은 A5000 기준 경로(`reference/run_inpaint.py`)와 같다: bf16, `enable_model_cpu_offload()`.
  bf16 transformer 22.17 GiB + T5 9.5 GB 가 24 GB 를 넘어 상주할 수 없다(00a 기록, README).
- 인페인팅은 포크 `utils/inpaint.py:inpaint()` 그대로(1024², 50스텝, guidance 30, CPU generator, seed 0).
- 가중치: transformer 외 부품은 원본 `black-forest-labs/FLUX.1-Fill-dev` 스냅샷. bf16 transformer 는 원본 저장소에서
  받지 않았으므로 미러(`camenduru/FLUX.1-Fill-dev-ungated`)의 `transformer/` 를 쓴다 — 원본과 파일 sha256 이 같다
  (00 B-4 HEAD 대조 23/23, 정오표 E11).
- 마스크 밖 합성은 R9 워커와 같은 함수(`services/rtx3090_inpaint/composite.py`).

    <flux-fill python> reference/rtx3090/r9_bf16_gen.py --cond $ASSETS_DIR/conditions/track_a --repeat 3
산출: RUNS_DIR/r9-bf16-<시각>/<ID>/run<k>/{2d_render,2d_mask,2d_edit,2d_edit_raw}.png, gen.json
"""
import argparse, datetime, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF = (ROOT / os.environ.get("REFERENCE_DIR", "build/reference-v2")).resolve()
sys.path.insert(0, str(REF))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ap = argparse.ArgumentParser()
ap.add_argument("--cond", required=True)
ap.add_argument("--repeat", type=int, default=3)
ap.add_argument("--base", default="black-forest-labs/FLUX.1-Fill-dev")
ap.add_argument("--transformer_repo", default="camenduru/FLUX.1-Fill-dev-ungated")
a = ap.parse_args()

if subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                  capture_output=True, text=True).stdout.strip():
    sys.exit("다른 GPU 작업이 있다 — 측정하지 않는다")

import torch  # noqa: E402
from diffusers import FluxFillPipeline, FluxTransformer2DModel  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from PIL import Image  # noqa: E402

from reference.meter import Meter, Sampler  # noqa: E402
from services.rtx3090_inpaint.composite import composite, outside_diff  # noqa: E402
from utils.inpaint import inpaint  # noqa: E402  포크 함수 그대로


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


cond = Path(a.cond)
man = json.loads((cond / "manifest.json").read_text())["scenarios"]
out = Path(os.environ.get("RUNS_DIR") or ROOT / "runs") / f"r9-bf16-{datetime.datetime.now():%Y%m%d-%H%M%S}"
out.mkdir(parents=True)
sampler = Sampler().start()
meter = Meter(sampler)
with meter.span("load"):
    # 오프라인에서 subfolder 지정이 hub API 를 부르므로 로컬 스냅샷 경로로 연다
    tr_dir = Path(snapshot_download(a.transformer_repo, local_files_only=True, allow_patterns=["transformer/*"])) / "transformer"
    tr = FluxTransformer2DModel.from_pretrained(str(tr_dir), torch_dtype=torch.bfloat16)
    pipe = FluxFillPipeline.from_pretrained(a.base, transformer=tr, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
res = {"item": "05 R9 bf16 cause split (3090)", "base": a.base,
       "base_snapshot": Path(snapshot_download(a.base, local_files_only=True, allow_patterns=["model_index.json"])).name,
       "transformer_repo": a.transformer_repo,
       "transformer_snapshot": Path(snapshot_download(a.transformer_repo, local_files_only=True,
                                                      allow_patterns=["transformer/config.json"])).name,
       "dtype": "bfloat16", "offload": "enable_model_cpu_offload", "torch": torch.__version__,
       "settings": {"steps": 50, "guidance": 30, "resolution": 1024, "max_sequence_length": 512, "generator": "cpu"},
       "load": meter.spans[-1], "repeat": a.repeat, "scenarios": {}}
print(f"[load] {meter.spans[-1]['seconds']:.1f}s", flush=True)
for sid, v in sorted(man.items()):
    runs = []
    for k in range(a.repeat):
        d = out / sid / f"run{k}"
        d.mkdir(parents=True)
        shutil.copy(cond / sid / "2d_render.png", d / "2d_render.png")
        shutil.copy(cond / sid / "2d_mask.png", d / "2d_mask.png")
        m = Meter(sampler)
        with m.span("inpaint"):
            inpaint(pipeline=pipe, image_path=str(d / "2d_render.png"), mask_path=str(d / "2d_mask.png"),
                    output_path=str(d / "2d_edit_raw.png"), prompt=v["prompt"], seed=v["seed"])
        ren, msk, raw = (Image.open(d / f"{n}.png") for n in ("2d_render", "2d_mask", "2d_edit_raw"))
        comp = composite(raw, ren, msk)
        comp.save(d / "2d_edit.png")
        sp = m.spans[-1]
        runs.append({"inpaint_s": sp["seconds"], "max_alloc_GiB": sp["max_memory_allocated"] / 2**30,
                     "smi_mib_max": sp.get("nvidia_smi_mib_max"), "rss_bytes_max": sp.get("rss_bytes_max"),
                     "sha256": {n: sha(d / f"{n}.png") for n in ("2d_render", "2d_mask", "2d_edit", "2d_edit_raw")},
                     "outside_diff": {"before": outside_diff(raw, ren, msk), "after": outside_diff(comp, ren, msk)}})
        print(f"{sid} run{k} {sp['seconds']:.1f}s peak {runs[-1]['max_alloc_GiB']:.2f}GiB smi {runs[-1]['smi_mib_max']}", flush=True)
    res["scenarios"][sid] = {"prompt": v["prompt"], "seed": v["seed"], "input_sha256": v["sha256"], "runs": runs}
sampler.stop()
path = out / "gen.json"
path.write_text(json.dumps(res, indent=1, default=str))
print("RESULT", path, sha(path))
