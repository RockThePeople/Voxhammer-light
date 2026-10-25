"""1 E2a 육안 시트: 14건마다 편집 시점·3/4 시점에서 원본 | base | c15 | c12 (r1).

렌더는 tools/e2a_metrics.py render 가 만든 Bench 16시점(CPU Cycles, 고도 15°·30° × 방위 8)을 그대로 쓴다.
F-5 시점 v(고도 15°, 방위 [90,45,0,315,270][v]−90)는 Bench 16시점의 고도 15° 줄에서 방위가 같은 번호로 옮긴다.
B7 은 c12 에서 마스크 ∪ halo 에 붙어 늘어난 큰 성분 3개(§4a)의 확대 컷을 한 줄 더 둔다: 성분 셀 중심을 Bench 카메라
(반지름 1.8, 원점 주시, Z 위, 센서 32 mm·렌즈 35 mm → 화각 49.1°)로 투영해 그 범위를 자르고, 마지막 칸에 투영 위치를
빨간 점으로 겹친다(가림은 무시).
시트: docs/reports/img/01-E2a-visual.png
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import math

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(os.environ["ASSETS_DIR"])
OUT = Path(os.environ["RUNS_DIR"]) / "e2a_metrics"
CONFIGS = ("base", "c15", "c12")
F5_TO_BENCH = {0: 2, 1: 1, 2: 0, 3: 7, 4: 6}
TAGS = {"A-C2": "weak condition image (E12)", "A-S1": "out-of-mask generation, known defect (E12)"}
T = 224
HEADER = ("Renderer: Edit3D-Bench render.py (Blender Cycles on CPU, 16 views at elev 15/30 deg, 1024 px), "
          "the same renders used for the Bench metrics. Not the fork Cycles renderer used for the 00s2 F-5 sheet. Run r1.")
BENCH_AZ = [0, 45, 90, 135, 180, 225, 270, 315]


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


def b7_components(k=3):
    """B7 c12 의 R_keep 증가 셀(§4a) 중 마스크 ∪ halo 에 맞닿은 큰 성분 k 개의 셀 좌표."""
    import sys
    from safetensors import safe_open
    from scipy import ndimage
    sys.path.insert(0, str(ROOT / "third_party" / "ai-ar-v3" / "contract" / "python"))
    from deltacontract.coords import dilate_cells
    runs = Path(os.environ["RUNS_DIR"])
    co = lambda p: safe_open(str(p), "np").get_tensor("coords")[:, 1:].astype(int)

    def vol(c):
        v = np.zeros((64, 64, 64), bool)
        v[tuple(c.T)] = True
        return v
    src = vol(co(ASSETS / "track_b" / "B7" / "slat.safetensors"))
    excl = vol(dilate_cells(np.array(json.loads((ASSETS / "track_b" / "B7" / "mask.json").read_text())["mask"]["voxels"]), 2))
    base = vol(co(runs / "01-port" / "track_b-B7" / "port-r1" / "slat_tgt.safetensors"))
    c12 = vol(co(runs / "2026-10-25" / "track_b-B7" / "e2a-c12" / "r1" / "slat_tgt.safetensors"))
    inc = (c12 & ~src & ~excl) & ~(base & ~src & ~excl)
    lab, n = ndimage.label(inc, structure=np.ones((3, 3, 3)))
    touch = ndimage.binary_dilation(excl, structure=np.ones((3, 3, 3)))
    comps = [np.argwhere(lab == i) for i in range(1, n + 1) if (touch & (lab == i)).any()]
    return sorted(comps, key=len, reverse=True)[:k]


def project(cells, view, W):
    """TRELLIS 셀 → Bench 카메라 view 의 픽셀 (x, y). 결과 GLB 가 Bench 프레임과 정렬돼 있어 셀 중심 (i+0.5)/64−0.5 를 그대로 쓴다."""
    P = (cells + 0.5) / 64 - 0.5
    el, az = math.radians(15 if view < 8 else 30), math.radians(BENCH_AZ[view % 8] - 90)
    cam = 1.8 * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    f = -cam / np.linalg.norm(cam)
    r = np.cross(f, [0, 0, 1])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    d = P - cam
    t = 16 / 35
    return np.stack([(d @ r / (d @ f) / t * 0.5 + 0.5) * W, (0.5 - d @ u / (d @ f) / t * 0.5) * W], 1)


def b7_zoom_row(it, width):
    comps = b7_components()
    cells = np.concatenate(comps)
    tz = width // 6
    tiles = []
    for k, v in enumerate(it["views"]):
        name = f"render_{F5_TO_BENCH[v]:04d}.png"
        vt = "edit" if k == 0 else "3/4"
        ims = {c: Image.alpha_composite(Image.new("RGBA", Image.open(it["pred"](c) / name).size, "white"),
                                        Image.open(it["pred"](c) / name).convert("RGBA")).convert("RGB") for c in ("base", "c12")}
        W = ims["c12"].width
        xy = project(cells, F5_TO_BENCH[v], W)
        (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
        half = max(x1 - x0, y1 - y0) / 2 + W * 0.04
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        box = tuple(int(round(b)) for b in (cx - half, cy - half, cx + half, cy + half))
        zoom = tz / (box[2] - box[0])
        for c, over in (("base", False), ("c12", False), ("c12", True)):
            t = ims[c].crop(box).resize((tz, tz), Image.LANCZOS)
            dr = ImageDraw.Draw(t)
            if over:
                for comp in comps:
                    for px, py in project(comp, F5_TO_BENCH[v], W):
                        qx, qy = (px - box[0]) * zoom, (py - box[1]) * zoom
                        dr.ellipse((qx - 1.5, qy - 1.5, qx + 1.5, qy + 1.5), fill=(230, 0, 0))
            dr.text((4, 4), f"{vt} v{v} {c} zoom x{zoom * W / T:.1f}" + (" + 3 comps" if over else ""), fill=(200, 0, 0))
            tiles.append(t)
    row = Image.new("RGB", (width, tz + 20), "white")
    ImageDraw.Draw(row).text((4, 4), "B7 zoom (c12): red dots = projected cells of the 3 largest new components touching "
                             f"mask+halo ({', '.join(str(len(c)) for c in comps)} cells; occlusion ignored)", fill=(0, 0, 0))
    for i, t in enumerate(tiles):
        row.paste(t, (tz * i, 20))
    return row


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
        if it["id"] == "B7":
            rows.append(b7_zoom_row(it, row.width))
        print("visual", it["id"], flush=True)
    head = Image.new("RGB", (rows[0].width, 22), "white")
    ImageDraw.Draw(head).text((4, 5), HEADER, fill=(0, 0, 160))
    rows.insert(0, head)
    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    dst = ROOT / "docs" / "reports" / "img" / "01-E2a-visual.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst, optimize=True)
    print("sheet", dst)


if __name__ == "__main__":
    main()
