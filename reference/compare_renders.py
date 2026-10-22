"""A-5b 조건 렌더 실행 간 차이표 + 5시점 썸네일(마스크 겹침).

입력: 같은 (model, mask) 로 돌린 run 디렉터리 3개(각각 images/render_000i.png, mask_000i.png).
RGB 는 채널별 0–255 정수 차의 최대·평균(알파 포함 4채널이면 RGB 만), 마스크는 다른 픽셀 수·비율.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load(p, mode):
    return np.asarray(Image.open(p).convert(mode)).astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json", required=True)
    ap.add_argument("--thumb", help="5시점 썸네일 PNG (run1 기준)")
    a = ap.parse_args()
    runs = [Path(r) / "images" for r in a.runs]
    rows = []
    for v in range(5):
        for (i, ri), (j, rj) in itertools.combinations(enumerate(runs, 1), 2):
            x, y = load(ri / f"render_{v:04d}.png", "RGB"), load(rj / f"render_{v:04d}.png", "RGB")
            d = np.abs(x - y)
            mx, my = load(ri / f"mask_{v:04d}.png", "L") > 127, load(rj / f"mask_{v:04d}.png", "L") > 127
            nd = int((mx != my).sum())
            rows.append({"view": v, "pair": f"{i}-{j}", "rgb_max": int(d.max()), "rgb_mean": float(d.mean()),
                         "rgb_diff_pixels": int((d.max(-1) > 0).sum()),
                         "mask_diff_pixels": nd, "mask_diff_ratio": nd / mx.size,
                         "mask_pixels_run_i": int(mx.sum())})
    Path(a.json).write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(f"v{r['view']} {r['pair']}  rgb max {r['rgb_max']:3d} mean {r['rgb_mean']:.4f} "
              f"px {r['rgb_diff_pixels']:7d} | mask diff {r['mask_diff_pixels']:6d} ({r['mask_diff_ratio']:.5f}) "
              f"mask px {r['mask_pixels_run_i']}")
    if a.thumb:
        tiles = []
        for v in range(5):
            rgb = Image.open(runs[0] / f"render_{v:04d}.png").convert("RGBA")
            bg = Image.new("RGBA", rgb.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, rgb).convert("RGB")
            m = np.asarray(Image.open(runs[0] / f"mask_{v:04d}.png").convert("L")) > 127
            arr = np.asarray(img).copy()
            arr[m] = (0.45 * arr[m] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
            t = Image.fromarray(arr).resize((256, 256), Image.LANCZOS)
            ImageDraw.Draw(t).text((6, 6), f"view {v}  mask px {int(m.sum())}", fill=(0, 0, 0))
            tiles.append(t)
        sheet = Image.new("RGB", (256 * 5, 256), "white")
        for k, t in enumerate(tiles):
            sheet.paste(t, (256 * k, 0))
        sheet.save(a.thumb)


if __name__ == "__main__":
    main()
