"""05 R9 R-4 지표: INT4(3090) 대 bf16 기준(A5000 동결본) 2D 인페인팅 비교 + 비교 시트.

지표 정의는 Edit3D-Bench `eval_modules` 를 따른다(포크 파일은 호출하지 않고 같은 방식으로 다시 쓴다 —
그 모듈은 DINO·데이터로더까지 한꺼번에 올려서 이 비교에는 무겁다).
- 512² 로 줄인다(이미지 bilinear, 마스크 nearest). 마스크 안 = mask > 0.
- PSNR: 마스크 안 화소만, [0,1] 범위. LPIPS: alex, spatial=True, 마스크 안 평균.
- CLIP-T: `openai/clip-vit-base-patch32`, 이미지·프롬프트 특징 정규화 후 코사인. 전체 이미지.
비교 대상(마스크 안):
- int4_vs_bf16: INT4 합성 전 출력 각 회 vs bf16 기준 `2d_edit`
- int4_self: INT4 반복끼리(3쌍) — 비결정성 바닥값
- edit_vs_render: 원본 렌더 대비 변화량(효능). bf16·INT4 각각
마스크 밖: INT4 합성 후 `2d_edit` 와 원본 렌더의 바이트 동일 비율, bf16 기준(합성 없음)의 같은 값.

    <eval venv python> reference/rtx3090/r9_compare_eval.py --cond <조건 디렉터리> --gen runs/r9-r4-<시각>
"""
import argparse, hashlib, itertools, json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from services.rtx3090_inpaint.composite import outside_diff, render_rgb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cond", required=True)
ap.add_argument("--gen", required=True)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
cond, gen = Path(a.cond), Path(a.gen)
S = 512

import lpips  # noqa: E402
from transformers import CLIPModel, CLIPProcessor  # noqa: E402

LP = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(a.device).eval()
CM = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(a.device).eval()
CP = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def img512(p_or_im):
    im = p_or_im if isinstance(p_or_im, Image.Image) else Image.open(p_or_im)
    return render_rgb(im).resize((S, S), Image.BILINEAR)


def t(im):
    return torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)[None].to(a.device)


def inside(mask_path):
    return np.asarray(Image.open(mask_path).convert("L").resize((S, S), Image.NEAREST)) > 0


def psnr(x, y, m):
    d = (np.asarray(x, np.float32) / 255 - np.asarray(y, np.float32) / 255)[m]
    mse = float((d ** 2).mean())
    return float("inf") if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))


@torch.no_grad()
def lp(x, y, m):
    mp = LP(t(x) * 2 - 1, t(y) * 2 - 1)[0, 0].cpu().numpy()
    return float(mp[m].mean())


@torch.no_grad()
def clip_t(im, text):
    ii = CP(images=im, return_tensors="pt").to(a.device)
    ti = CP(text=text, return_tensors="pt", padding=True, truncation=True).to(a.device)
    f1, f2 = CM.get_image_features(**ii), CM.get_text_features(**ti)
    f1, f2 = f1 / f1.norm(dim=-1, keepdim=True), f2 / f2.norm(dim=-1, keepdim=True)
    return float(torch.cosine_similarity(f1, f2, dim=-1))


g = json.loads((gen / "gen.json").read_text())
out = {"item": "05 R9 R-4 eval", "gen_json_sha256": hashlib.sha256((gen / "gen.json").read_bytes()).hexdigest(),
       "definitions": __doc__, "scenarios": {}}
for sid, v in sorted(g["scenarios"].items()):
    c = cond / sid
    m = inside(c / "2d_mask.png")
    ren, ref = img512(c / "2d_render.png"), img512(c / "2d_edit.png")
    runs = sorted((gen / sid).glob("run*"))
    raw = [img512(r / "2d_edit_raw.png") for r in runs]
    comp = [Image.open(r / "2d_edit.png") for r in runs]
    msk_full = Image.open(c / "2d_mask.png")
    r = {
        "prompt": v["prompt"], "mask_inside_px_512": int(m.sum()),
        "int4_vs_bf16": {"psnr": [psnr(x, ref, m) for x in raw], "lpips": [lp(x, ref, m) for x in raw]},
        "int4_self": {"psnr": [psnr(raw[i], raw[j], m) for i, j in itertools.combinations(range(len(raw)), 2)],
                      "lpips": [lp(raw[i], raw[j], m) for i, j in itertools.combinations(range(len(raw)), 2)]},
        "edit_vs_render": {"bf16_lpips": lp(ref, ren, m), "int4_lpips": [lp(x, ren, m) for x in raw],
                           "bf16_psnr": psnr(ref, ren, m), "int4_psnr": [psnr(x, ren, m) for x in raw]},
        "clip_t": {"render": clip_t(ren, v["prompt"]), "bf16": clip_t(ref, v["prompt"]),
                   "int4": [clip_t(img512(x), v["prompt"]) for x in comp]},
        "outside": {"int4_composited_identical": [outside_diff(x, Image.open(c / "2d_render.png"), msk_full)["outside_identical_frac"] for x in comp],
                    "bf16_identical": outside_diff(Image.open(c / "2d_edit.png"), Image.open(c / "2d_render.png"), msk_full)["outside_identical_frac"],
                    "bf16_gt16": outside_diff(Image.open(c / "2d_edit.png"), Image.open(c / "2d_render.png"), msk_full)["outside_gt16_frac"]},
        "time": {"int4_inpaint_s": [x["inpaint_s"] for x in v["runs"]], "int4_peak_GiB": [x["max_alloc_GiB"] for x in v["runs"]]},
    }
    out["scenarios"][sid] = r
    f = lambda xs: f"{np.mean(xs):.3f}"
    print(f"{sid}: vs_bf16 psnr {f(r['int4_vs_bf16']['psnr'])} lpips {f(r['int4_vs_bf16']['lpips'])} | self psnr {f(r['int4_self']['psnr'])} "
          f"lpips {f(r['int4_self']['lpips'])} | clipT render {r['clip_t']['render']:.3f} bf16 {r['clip_t']['bf16']:.3f} int4 {f(r['clip_t']['int4'])} "
          f"| out int4 {min(r['outside']['int4_composited_identical'])} bf16 {r['outside']['bf16_identical']:.3f}", flush=True)

    # 비교 시트: 원본 | 마스크 | bf16 | INT4 run0..2  (512 → 384 축소)
    W = 384
    tiles = [ren, Image.open(c / "2d_mask.png").convert("RGB").resize((S, S)), ref] + [img512(x) for x in comp]
    sheet = Image.new("RGB", (W * len(tiles), W + 28), "white")
    from PIL import ImageDraw
    d = ImageDraw.Draw(sheet)
    for i, (tile, name) in enumerate(zip(tiles, ["render", "mask", "bf16 (A5000)"] + [f"INT4 run{k}" for k in range(len(comp))])):
        sheet.paste(tile.resize((W, W)), (i * W, 28))
        d.text((i * W + 6, 6), f"{sid} {name}", fill="black")
    sheet.save(gen / f"sheet_{sid}.png")

path = gen / "eval.json"
path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
print("RESULT", path, hashlib.sha256(path.read_bytes()).hexdigest())
