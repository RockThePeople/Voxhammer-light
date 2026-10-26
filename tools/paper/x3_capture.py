"""X3 캡처 시트(지시서 X3 "캡처"): 자산마다 1·3·5회차 × (편집 시점 | 3/4 시점) × P0 | P1 | P2.

렌더는 x3_score.result_views(Bench render.py, 같은 카메라·조명, P1·P2 정점색 (iv) 보정)를 그대로 쓴다.
결과 $RUNS_DIR/paper/x3/capture/<자산>.png

    python tools/paper/x3_capture.py [자산 …]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
from tools.paper.x3 import ASSET_NAMES, OUT, round_info  # noqa: E402
from tools.paper.x3_score import result_views  # noqa: E402

PATHS = ("p0", "p1", "p2")


def sheet(asset, rounds=(1, 3, 5)):
    from PIL import Image, ImageDraw
    S = 280
    W, H = S * 6, (S + 24) * len(rounds) + 30
    sh = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(sh)
    for j, lab in enumerate(("edit view", "3/4 view")):
        for i, p in enumerate(PATHS):
            dr.text((j * 3 * S + i * S + 6, 8), f"{p.upper()} — {lab}", fill="black")
    for r, k in enumerate(rounds):
        _, info = round_info(asset, k)
        y = 30 + r * (S + 24)
        dr.text((6, y + 2), f"r{k}: {info['part']} — {info['prompt'][:90]}", fill="red")
        for i, p in enumerate(PATHS):
            for j, img in enumerate(result_views(p, asset, k, info["edit_view"])):
                im = Image.open(img).convert("RGBA")
                bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                bg.alpha_composite(im)
                sh.paste(bg.convert("RGB").crop((112, 112, 912, 912)).resize((S, S)), (j * 3 * S + i * S, y + 20))
    d = OUT / "capture"
    d.mkdir(parents=True, exist_ok=True)
    sh.save(d / f"{asset}.png")
    print("capture", d / f"{asset}.png", flush=True)


if __name__ == "__main__":
    for a in sys.argv[1:] or ASSET_NAMES:
        sheet(a)
