"""00s2 F-5 육안 비교 렌더(요구사항서 6장): 14건마다 편집 시점·3/4 시점, 같은 조명으로 원본 | 편집 결과(1회차).

렌더는 포크 `utils/render_rgb_and_mask.render_model`(Cycles, 1024², 고도 15°, 방위 5시점, 같은 환경맵) 그대로.
- 트랙 A: 원본 = gen/model.glb(정규화본). 결과 = r1 output.glb(to_glb) 에 gen.json 의 같은 정규화((p−c)×s)를 적용한 사본.
- 트랙 B: 원본 = Bench source_model/model.glb, 결과 = r1 output.glb (S-8: 두 출력 모두 Bench 프레임과 정렬, 변환 불필요).
시트: docs/reports/img/00s2-f5-visual.png (행 = 시나리오, 열 = 편집 시점 원본|결과, 3/4 시점 원본|결과).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT.parent / "Voxhammer-light"
REF = (MAIN / "build" / "reference-v2").resolve()
ASSETS = Path(os.environ["ASSETS_DIR"])
RUNS = Path(os.environ["RUNS_DIR"])
DAY = os.environ.get("F5_DAY", "2026-10-22")
OUT = RUNS / "f5_visual"
PY = os.environ.get("VH_PY", str(Path.home() / "ai-ar-v2/VoxHammer/.venv/bin/python"))

RENDER = r"""
import os, sys
ref, glb, out = sys.argv[1:4]
sys.path.insert(0, ref); os.chdir(ref)
import utils.render_rgb_and_mask as rr
rr.render_model(glb, out)
"""


def items():
    ta = json.loads((ROOT / "scenarios" / "track_a.json").read_text())
    tb = json.loads((ROOT / "scenarios" / "track_b.json").read_text())
    for sid, s in ta["scenarios"].items():
        a = s["asset"]
        g = json.loads((ASSETS / "track_a" / a / "gen" / "gen.json").read_text())
        yield {"id": sid, "src": ASSETS / "track_a" / a / "gen" / "model.glb",
               "res": RUNS / DAY / f"track_a-{sid}" / f"w34-{s['mask']['fingerprint'][:12]}" / "r1.work" / "output.glb",
               "norm": g["glb_normalize"], "edit_view": s["edit_view"], "view_34": s["view_34"], "label": f"{sid} {s['inpaint_prompt'][:40]}"}
    for bid, s in tb["scenarios"].items():
        yield {"id": bid, "src": ASSETS / "Edit3D-Bench" / "data" / s["dataset"] / s["object"] / "source_model" / "model.glb",
               "res": RUNS / DAY / f"track_b-{bid}" / f"w34-{s['mask']['fingerprint'][:12]}" / "r1.work" / "output.glb",
               "norm": None, "edit_view": s["edit_view"], "view_34": 3 if s["edit_view"] == 1 else 1,
               "label": f"{bid} {s['prompt'][:40]}"}


def main():
    import trimesh
    from PIL import Image, ImageDraw
    rows = []
    for it in items():
        d = OUT / it["id"]
        d.mkdir(parents=True, exist_ok=True)
        res = it["res"]
        if it["norm"]:
            sc = trimesh.load(res)
            sc.apply_translation([-x for x in it["norm"]["center_glb_frame"]])
            sc.apply_scale(it["norm"]["scale"])
            res = d / "result_normalized.glb"
            sc.export(res)
        for tag, glb in (("src", it["src"]), ("res", res)):
            if not (d / tag / "render_0004.png").exists():
                subprocess.run([PY, "-c", RENDER, str(REF), str(glb), str(d / tag)], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tiles = []
        for v in (it["edit_view"], it["view_34"]):
            for tag in ("src", "res"):
                im = Image.open(d / tag / f"render_{v:04d}.png").convert("RGBA")
                bg = Image.new("RGBA", im.size, "white")
                t = Image.alpha_composite(bg, im).convert("RGB").resize((256, 256), Image.LANCZOS)
                ImageDraw.Draw(t).text((4, 4), f"{'edit' if v == it['edit_view'] else '3/4'} v{v} {'원본' if tag == 'src' else '결과'}", fill=(200, 0, 0))
                tiles.append(t)
        row = Image.new("RGB", (4 * 256, 276), "white")
        ImageDraw.Draw(row).text((4, 2), it["label"], fill=(0, 0, 0))
        for i, t in enumerate(tiles):
            row.paste(t, (256 * i, 20))
        rows.append(row)
        print("visual", it["id"], flush=True)
    sheet = Image.new("RGB", (4 * 256, 276 * len(rows)), "white")
    for i, r in enumerate(rows):
        sheet.paste(r, (0, 276 * i))
    dst = ROOT / "docs" / "reports" / "img" / "00s2-f5-visual.png"
    sheet.save(dst, optimize=True)
    print("sheet", dst)


if __name__ == "__main__":
    main()
