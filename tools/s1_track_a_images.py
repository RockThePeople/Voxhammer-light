"""00s S-1: 트랙 A 입력 이미지 생성 (rtx3090).

기존 3090 생성 경로를 그대로 쓴다: Z-Image-Turbo 워커(`zimage_gen.py`, 1024², 9스텝, guidance 0)
→ BiRefNet 워커(`birefnet_seg.py`) 배경 제거. 워커는 수정하지 않고 subprocess 로 순차 호출한다
(ai-ar-v3 server/t2i.py 와 같은 방식). 프롬프트만 요구사항서 6장 템플릿으로 만든다.

산출 (ASSETS_DIR/track_a/<asset>/, git 제외):
    image_rgb.png   Z-Image 출력
    input.png       BiRefNet RGBA (S-2 TRELLIS 입력)
    params.json     모델·시드·스텝·프롬프트 전문·환경 버전·sha256
git 에는 sha256 과 파라미터만 `scenarios/track_a_inputs.json` 로 남긴다.

환경변수: T2I_TOOLS_DIR, ZIMAGE_PYTHON, BIREFNET_PYTHON, ASSETS_DIR

    python3 tools/s1_track_a_images.py --asset moto --seed 42
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 요구사항서 6장 「트랙 A 자산 생성 포맷」 원문 그대로
TEMPLATE = ("{subject}, full body, 3/4 view, product photography, studio lighting, "
            "plain white background, centered, entire subject fully visible with margin, "
            "no text, no watermark")
SUBJECTS = {
    "moto": ("a man riding a black motorcycle", ["A-M1", "A-M2"]),
    "snowman": ("a snowman with a carrot nose, stick arms and a red scarf", ["A-S1", "A-S2"]),
    "chair": ("a wooden armchair with armrests", ["A-C1", "A-C2"]),
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def versions(py, mods):
    code = ("import importlib,sys,json;d={'python':sys.version.split()[0]}\n"
            "for m in %r:\n"
            "  try: d[m]=importlib.import_module(m).__version__\n"
            "  except Exception as e: d[m]=None\n"
            "import torch;d['torch_cuda']=torch.version.cuda;print(json.dumps(d))" % (mods,))
    return json.loads(subprocess.run([py, "-c", code], capture_output=True, text=True,
                                     check=True).stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", choices=SUBJECTS, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--note", default="", help="시드를 바꾼 이유 등")
    a = ap.parse_args()

    env = {n: os.environ.get(n, "").strip()
           for n in ("T2I_TOOLS_DIR", "ZIMAGE_PYTHON", "BIREFNET_PYTHON", "ASSETS_DIR")}
    miss = [n for n, v in env.items() if not v]
    if miss:
        sys.exit(f"환경변수가 없다: {', '.join(miss)}")
    tools = Path(env["T2I_TOOLS_DIR"])
    zw, bw = tools / "zimage_gen.py", tools / "birefnet_seg.py"

    subject, scen = SUBJECTS[a.asset]
    prompt = TEMPLATE.replace("{subject}", subject)
    out = Path(env["ASSETS_DIR"]) / "track_a" / a.asset / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    rgb, rgba = out / "image_rgb.png", out / "input.png"

    subprocess.run([env["ZIMAGE_PYTHON"], str(zw), "--prompt", prompt, "--out", str(rgb),
                    "--seed", str(a.seed)], check=True)
    subprocess.run([env["BIREFNET_PYTHON"], str(bw), "--in", str(rgb), "--out", str(rgba)],
                   check=True)

    params = {
        "item": "00s S-1", "asset": a.asset, "scenarios": scen,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "t2i": {"model": "Tongyi-MAI/Z-Image-Turbo", "width": 1024, "height": 1024,
                "num_inference_steps": 9, "guidance_scale": 0.0, "dtype": "bfloat16",
                "seed": a.seed, "generator": "cuda",
                "worker": "zimage_gen.py", "worker_sha256": sha256(zw)},
        "bg_removal": {"model": "ZhengPeng7/BiRefNet", "dtype": "float16", "resize": 1024,
                       "worker": "birefnet_seg.py", "worker_sha256": sha256(bw)},
        "subject": subject, "prompt_template": TEMPLATE, "prompt": prompt,
        "env": {"zimage": versions(env["ZIMAGE_PYTHON"], ["torch", "diffusers", "transformers"]),
                "birefnet": versions(env["BIREFNET_PYTHON"], ["torch", "torchvision", "transformers"])},
        "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                               "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
        "files": {"image_rgb.png": sha256(rgb), "input.png": sha256(rgba)},
        "note": a.note,
    }
    (out / "params.json").write_text(json.dumps(params, indent=1, ensure_ascii=False))
    print(json.dumps({"asset": a.asset, "seed": a.seed, **params["files"]}))


if __name__ == "__main__":
    main()
