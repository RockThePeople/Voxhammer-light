"""05 R9 R-4·원인 분리 지표: A5000 bf16 동결본(기준) 대 3090 결과 묶음들의 2D 인페인팅 비교 + 비교 시트.

지표 정의는 Edit3D-Bench `eval_modules` 를 따른다(포크 모듈은 DINO·데이터로더까지 올려 무거워 같은 방식으로 다시 쓴다).
- 512² 로 줄인다(이미지 bilinear, 마스크 nearest). 마스크 안 = mask > 0.
- PSNR: 마스크 안 화소만, [0,1]. LPIPS: alex, spatial=True, 마스크 안 평균.
- CLIP-T: `openai/clip-vit-base-patch32`, 정규화 특징 코사인, 전체 이미지(합성 후 `2d_edit`).
묶음(--set 이름=디렉터리, 여러 개)마다:
- vs_a5000: 합성 전 출력 각 회 vs A5000 bf16 `2d_edit` (마스크 안)
- self: 반복끼리(3쌍) — 같은 시드 비결정성 바닥값. raw 바이트가 몇 종류인지도 센다
- edit_vs_render: 원본 렌더 대비 변화량(효능, 마스크 안 LPIPS)
- outside: 원본 렌더와 마스크 밖 바이트 동일 비율 — 합성 전 / 합성 후
- composite_effect: 합성 전 vs 합성 후 전체 이미지 LPIPS(마스크 밖 합성이 바꾼 양)
묶음 쌍마다(cross): 마스크 안 PSNR·LPIPS, 모든 회차 조합 평균.

    <eval venv python> reference/rtx3090/r9_compare_eval.py --cond <조건 디렉터리> \
        --set int4=runs/r9-r4-<시각> --set bf16_3090=runs/r9-bf16-<시각> --out runs/r9-split-<시각>
"""
import argparse, hashlib, itertools, json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from services.rtx3090_inpaint.composite import outside_diff, render_rgb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cond", required=True)
ap.add_argument("--set", action="append", required=True, help="이름=gen 디렉터리 (gen.json 과 <ID>/run<k>/ 가 있는 곳)")
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
cond, out = Path(a.cond), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
SETS = dict(x.split("=", 1) for x in a.set)
SETS = {k: Path(v) for k, v in SETS.items()}
S = 512

import lpips  # noqa: E402
from transformers import CLIPModel, CLIPProcessor  # noqa: E402

LP = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(a.device).eval()
CM = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(a.device).eval()
CP = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def img512(p):
    return render_rgb(p if isinstance(p, Image.Image) else Image.open(p)).resize((S, S), Image.BILINEAR)


def t(im):
    return torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)[None].to(a.device)


def psnr(x, y, m):
    d = (np.asarray(x, np.float32) / 255 - np.asarray(y, np.float32) / 255)[m]
    mse = float((d ** 2).mean())
    return float("inf") if mse == 0 else float(20 * np.log10(1.0 / np.sqrt(mse)))


@torch.no_grad()
def lp(x, y, m=None):
    mp = LP(t(x) * 2 - 1, t(y) * 2 - 1)[0, 0].cpu().numpy()
    return float(mp.mean() if m is None else mp[m].mean())


@torch.no_grad()
def clip_t(im, text):
    ii = CP(images=im, return_tensors="pt").to(a.device)
    ti = CP(text=text, return_tensors="pt", padding=True, truncation=True).to(a.device)
    f1, f2 = CM.get_image_features(**ii), CM.get_text_features(**ti)
    f1, f2 = f1 / f1.norm(dim=-1, keepdim=True), f2 / f2.norm(dim=-1, keepdim=True)
    return float(torch.cosine_similarity(f1, f2, dim=-1))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


man = json.loads((cond / "manifest.json").read_text())["scenarios"]
res = {"item": "05 R9 eval", "definitions": __doc__,
       "sets": {k: {"dir": str(v), "gen_json_sha256": sha(v / "gen.json")} for k, v in SETS.items()},
       "scenarios": {}}
for sid, v in sorted(man.items()):
    c = cond / sid
    m = np.asarray(Image.open(c / "2d_mask.png").convert("L").resize((S, S), Image.NEAREST)) > 0
    ren_full, msk_full = Image.open(c / "2d_render.png"), Image.open(c / "2d_mask.png")
    ren, ref = img512(c / "2d_render.png"), img512(c / "2d_edit.png")
    r = {"prompt": v["prompt"], "mask_inside_px_512": int(m.sum()),
         "a5000_bf16": {"edit_vs_render_lpips": lp(ref, ren, m), "clip_t": clip_t(ref, v["prompt"]),
                        "outside_identical_raw": outside_diff(Image.open(c / "2d_edit.png"), ren_full, msk_full)["outside_identical_frac"],
                        "outside_gt16_raw": outside_diff(Image.open(c / "2d_edit.png"), ren_full, msk_full)["outside_gt16_frac"]},
         "render_clip_t": clip_t(ren, v["prompt"]), "sets": {}, "cross": {}}
    raws = {}
    for name, g in SETS.items():
        runs = sorted((g / sid).glob("run*"))
        raw = [img512(x / "2d_edit_raw.png") for x in runs]
        comp_full = [Image.open(x / "2d_edit.png") for x in runs]
        raw_full = [Image.open(x / "2d_edit_raw.png") for x in runs]
        raws[name] = raw
        pairs = list(itertools.combinations(range(len(raw)), 2))
        r["sets"][name] = {
            "vs_a5000": {"psnr": [psnr(x, ref, m) for x in raw], "lpips": [lp(x, ref, m) for x in raw]},
            "self": {"psnr": [psnr(raw[i], raw[j], m) for i, j in pairs], "lpips": [lp(raw[i], raw[j], m) for i, j in pairs],
                     "raw_distinct_bytes": len({sha(x / "2d_edit_raw.png") for x in runs}), "n": len(runs)},
            "edit_vs_render_lpips": [lp(x, ren, m) for x in raw],
            "clip_t": [clip_t(img512(x), v["prompt"]) for x in comp_full],
            "outside": {"identical_raw": [outside_diff(x, ren_full, msk_full)["outside_identical_frac"] for x in raw_full],
                        "gt16_raw": [outside_diff(x, ren_full, msk_full)["outside_gt16_frac"] for x in raw_full],
                        "identical_composited": [outside_diff(x, ren_full, msk_full)["outside_identical_frac"] for x in comp_full]},
            "composite_effect_lpips_full": [lp(img512(x), img512(y)) for x, y in zip(raw_full, comp_full)],
        }
    for x, y in itertools.combinations(SETS, 2):
        pr = list(itertools.product(raws[x], raws[y]))
        r["cross"][f"{x}__{y}"] = {"psnr": float(np.mean([psnr(p, q, m) for p, q in pr])),
                                   "lpips": float(np.mean([lp(p, q, m) for p, q in pr]))}
    res["scenarios"][sid] = r
    mm = lambda xs: f"{np.mean(xs):.3f}"
    print(sid, " | ".join(f"{n}: vsA5000 lpips {mm(s['vs_a5000']['lpips'])} self {mm(s['self']['lpips'])} distinct {s['self']['raw_distinct_bytes']}/{s['self']['n']}"
                          for n, s in r["sets"].items()), "| cross", {k: round(v2["lpips"], 3) for k, v2 in r["cross"].items()}, flush=True)

    # 시트: render | mask | A5000 bf16 | 각 묶음 run0..2 (합성 후)
    W = 256
    tiles = [(ren, "render"), (msk_full.convert("RGB").resize((S, S)), "mask"), (ref, "A5000 bf16")]
    for name, g in SETS.items():
        tiles += [(img512(x / "2d_edit.png"), f"{name} r{k}") for k, x in enumerate(sorted((g / sid).glob("run*")))]
    sheet = Image.new("RGB", (W * len(tiles), W + 22), "white")
    d = ImageDraw.Draw(sheet)
    for i, (tile, name) in enumerate(tiles):
        sheet.paste(tile.resize((W, W)), (i * W, 22))
        d.text((i * W + 4, 5), f"{sid} {name}", fill="black")
    sheet.save(out / f"sheet_{sid}.png")

path = out / "eval.json"
path.write_text(json.dumps(res, indent=1, ensure_ascii=False))
print("RESULT", path, sha(path))
