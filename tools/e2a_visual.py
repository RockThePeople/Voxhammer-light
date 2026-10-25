"""1 E2a 육안 시트: 14건마다 편집 시점·3/4 시점에서 원본 | base | c15 | c12 (r1).

렌더는 tools/e2a_metrics.py render 가 만든 Bench 16시점(CPU Cycles, 고도 15°·30° × 방위 8)을 그대로 쓴다.
F-5 시점 v(고도 15°, 방위 [90,45,0,315,270][v]−90)는 Bench 16시점의 고도 15° 줄에서 방위가 같은 번호로 옮긴다.
시트: docs/reports/img/01-E2a-visual.png
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(os.environ["ASSETS_DIR"])
OUT = Path(os.environ["RUNS_DIR"]) / "e2a_metrics"
CONFIGS = ("base", "c15", "c12")
F5_TO_BENCH = {0: 2, 1: 1, 2: 0, 3: 7, 4: 6}
TAGS = {"A-C2": "weak condition image (E12)", "A-S1": "out-of-mask generation, known defect (E12)"}
T = 224


def items():
    ta = json.loads((ROOT / "scenarios" / "track_a.json").read_text())["scenarios"]
    tb = json.loads((ROOT / "scenarios" / "track_b.json").read_text())["scenarios"]
    for sid, s in ta.items():
        yield {"id": sid, "src": OUT / "trackA_gt" / "data" / "TrackA" / sid / "source_model" / "render",
               "pred": lambda c, sid=sid: OUT / "pred" / c / "r1" / sid / "TrackA" / sid / "prompt_1" / "images",
               "views": (s["edit_view"], s["view_34"]), "text": s["inpaint_prompt"]}
    for bid, s in tb.items():
        rel = Path(s["dataset"]) / s["object"]
        yield {"id": bid, "src": ASSETS / "Edit3D-Bench" / "data" / rel / "source_model" / "render",
               "pred": lambda c, bid=bid, rel=rel, p=s["prompt_id"]: OUT / "pred" / c / "r1" / bid / rel / f"prompt_{p}" / "images",
               "views": (s["edit_view"], 3 if s["edit_view"] == 1 else 1), "text": s["prompt"]}


def tile(png, caption):
    im = Image.open(png).convert("RGBA")
    t = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im).convert("RGB").resize((T, T), Image.LANCZOS)
    ImageDraw.Draw(t).text((4, 4), caption, fill=(200, 0, 0))
    return t


def main():
    rows = []
    for it in items():
        tiles = []
        for k, v in enumerate(it["views"]):
            name = f"render_{F5_TO_BENCH[v]:04d}.png"
            vt = "edit" if k == 0 else "3/4"
            tiles.append(tile(it["src"] / name, f"{vt} v{v} source"))
            tiles += [tile(it["pred"](c) / name, f"{vt} v{v} {c}") for c in CONFIGS]
        row = Image.new("RGB", (len(tiles) * T, T + 20), "white")
        label = f"{it['id']}  {it['text'][:70]}" + (f"  [{TAGS[it['id']]}]" if it["id"] in TAGS else "")
        ImageDraw.Draw(row).text((4, 4), label, fill=(0, 0, 0))
        for i, t in enumerate(tiles):
            row.paste(t, (T * i, 20))
        rows.append(row)
        print("visual", it["id"], flush=True)
    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
    for i, r in enumerate(rows):
        sheet.paste(r, (0, r.height * i))
    dst = ROOT / "docs" / "reports" / "img" / "01-E2a-visual.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst, optimize=True)
    print("sheet", dst)


if __name__ == "__main__":
    main()
