"""X3 자산 이미지 생성 (rtx3090, 지시서 X-paper 2절 X3): 계획 원문 프롬프트 3개 × 시드 0–3.

기존 3090 생성 경로 그대로: Z-Image-Turbo 워커(`zimage_gen.py`, 1024², 9스텝, guidance 0)
→ BiRefNet 워커(`birefnet_seg.py`). 워커는 수정하지 않고 순차 subprocess 로 부른다(00s S-1 과 같다).
프롬프트는 `ops/paper/PLAN.md` 「X3 자산과 프롬프트」 원문을 **그대로** 쓴다(템플릿을 씌우지 않는다 —
원문이 이미 구도·배경을 적은 전체 프롬프트다).

산출 (git 제외): runs/paper/x3/assets/<asset>/seed<N>/{image_rgb.png, input.png, params.json}
              runs/paper/x3/assets/index.json (전 파일 sha256), sheet.png (원본 | 배경 제거, 12장)

환경변수: T2I_TOOLS_DIR, ZIMAGE_PYTHON, BIREFNET_PYTHON (ai-ar-v3 서버와 같은 이름), RUNS_DIR(기본 ./runs)

    python3 tools/x3_assets.py              # 3 × 4 = 12장
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

# ops/paper/PLAN.md 원문 그대로 (2026-10-26, main 60ebce5)
PROMPTS = {
    "jeep": "A compact off-road jeep, three-quarter front view, centered, full object visible, chunky all-terrain tires, "
            "a black front winch bumper, a side air-intake snorkel on the right front pillar, a spare tire mounted on the "
            "rear door, olive green paint, simple clean 3D game asset style, pure white background, no shadow",
    "cottage": "A small stylized wooden cottage, three-quarter front view, centered, full object visible, a stone chimney "
               "on the roof, a plain wooden front door, two square windows on the front wall, a small front porch with a "
               "wooden railing, brown shingled roof, simple clean 3D game asset style, pure white background, no shadow",
    "adventurer": "A chibi adventurer figurine standing, three-quarter front view, centered, full body visible, wearing a "
                  "brown leather hat, holding a wooden sword in the right hand, a round wooden shield on the left arm, brown "
                  "leather boots, green tunic, simple clean 3D game asset style, pure white background, no shadow",
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--assets", nargs="+", default=list(PROMPTS))
    a = ap.parse_args()

    env = {n: os.environ.get(n, "").strip() for n in ("T2I_TOOLS_DIR", "ZIMAGE_PYTHON", "BIREFNET_PYTHON")}
    miss = [n for n, v in env.items() if not v]
    if miss:
        sys.exit(f"환경변수가 없다: {', '.join(miss)}")
    busy = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    if busy:
        sys.exit(f"다른 GPU 작업이 있다(INT4 워커를 먼저 내릴 것): {busy}")
    tools = Path(env["T2I_TOOLS_DIR"])
    zw, bw = tools / "zimage_gen.py", tools / "birefnet_seg.py"
    root = Path(os.environ.get("RUNS_DIR") or ROOT / "runs") / "paper" / "x3" / "assets"

    index = {"item": "X-paper X3 자산 이미지", "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
             "t2i": {"model": "Tongyi-MAI/Z-Image-Turbo", "size": 1024, "steps": 9, "guidance": 0.0, "dtype": "bfloat16",
                     "worker_sha256": sha256(zw)},
             "bg_removal": {"model": "ZhengPeng7/BiRefNet", "dtype": "float16", "worker_sha256": sha256(bw)},
             "prompt_source": "ops/paper/PLAN.md X3 (원문 그대로, 템플릿 없음)", "assets": {}}
    for asset in a.assets:
        index["assets"][asset] = {"prompt": PROMPTS[asset], "seeds": {}}
        for seed in a.seeds:
            out = root / asset / f"seed{seed}"
            out.mkdir(parents=True, exist_ok=True)
            rgb, rgba = out / "image_rgb.png", out / "input.png"
            subprocess.run([env["ZIMAGE_PYTHON"], str(zw), "--prompt", PROMPTS[asset], "--out", str(rgb),
                            "--seed", str(seed)], check=True, capture_output=True)
            subprocess.run([env["BIREFNET_PYTHON"], str(bw), "--in", str(rgb), "--out", str(rgba)],
                           check=True, capture_output=True)
            files = {"image_rgb.png": sha256(rgb), "input.png": sha256(rgba)}
            (out / "params.json").write_text(json.dumps({"asset": asset, "seed": seed, "prompt": PROMPTS[asset],
                                                         "files": files}, indent=1, ensure_ascii=False))
            index["assets"][asset]["seeds"][str(seed)] = files
            print(asset, seed, files["input.png"][:12], flush=True)
    (root / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False))
    print("INDEX", root / "index.json", sha256(root / "index.json"))


if __name__ == "__main__":
    main()
