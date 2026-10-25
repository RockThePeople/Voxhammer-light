"""2번 R4 3-8 출력 품질 확인 (계측 구간 밖).

phase color : r1 결과를 포크 Cycles 로 네 방식 렌더(편집 시점·3/4 시점) →
              (i) 기존 텍스처 GLB(c15, to_glb) — checks/render/<ID>/a 재사용
              (ii) R4 정점색 + 비금속 재질 + COLOR_0 sRGB→선형(현재 보정)
              (iii) R4 정점색 + 비금속 재질만(변환 없음)
              (iv) R4 정점색 + to_glb 와 같은 재질(metallicFactor 생략 = 1.0) + sRGB→선형
              R4 는 정정 규칙 재조립 결과(output_r11.glb). 편집 시점을 2d_edit(인페인팅 결과)과, 마스크 밖을 2d_render 와 비교:
              객체 알파 안 평균 L*(CIELAB, D65) 차이, 채도 C* 차이, RGB 평균 절대 차이. 시트 docs/reports/img/02-R4-color.png
phase bench : 선택 방식(R4_MODE, 기본 iv)으로 r1–r3 를 E2a 와 같은 Bench 16시점(tools/e2a_metrics._render, CPU Cycles)으로 렌더해
              $RUNS_DIR/e2a_metrics/pred/r4/<r>/<ID>/… 에 둔다. 지표는 `E2A_CONFIGS=r4 tools/e2a_metrics.py eval`.
phase crops : 1024 px 포크 렌더(편집 시점)에서 세부 영역을 잘라 기존(i) | R4(선택 방식) 시트 docs/reports/img/02-R4-detail.png
결과: $RUNS_DIR/02-r4/checks/quality/{color.json, <ID>/{ii,iii,iv}/}
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests" / "gpu"))
from r4_checks import OUT, PY, REF, RENDER, scenarios  # noqa: E402

Q = OUT / "checks" / "quality"
MODES = {"ii": dict(metallic=0.0, linearize=True), "iii": dict(metallic=0.0, linearize=False),
         "iv": dict(metallic=None, linearize=True)}
LABEL = {"i": "(i) textured GLB (to_glb)", "ii": "(ii) VC + non-metal + sRGB->lin", "iii": "(iii) VC + non-metal, no conv",
         "iv": "(iv) VC + to_glb material + sRGB->lin"}
ASSETS = Path(os.environ["ASSETS_DIR"])


def cond_dir(s):
    if s["track"] == "A":
        a = json.loads((REPO / "scenarios" / "track_a.json").read_text())["scenarios"][s["id"]]["asset"]
        return ASSETS / "track_a" / a / "cond" / s["id"]
    return ASSETS / "track_b" / s["id"] / "cond"


def r4_glb(s, r="r1"):
    return OUT / s["name"] / "r4-on" / r / "output_r11.glb"


def prepared(s, src, mode, dst):
    """트랙 A 정규화(F-5·E2a 와 같게) → 렌더 입력 보정(editcore/render.py) → dst."""
    import trimesh
    from editcore.render import vertex_color_render_glb
    g = src
    if s["norm"]:
        sc = trimesh.load(src)
        sc.apply_translation([-x for x in s["norm"]["center_glb_frame"]])
        sc.apply_scale(s["norm"]["scale"])
        g = dst.with_name(dst.stem + "_raw.glb")
        sc.export(g)
    dst.write_bytes(vertex_color_render_glb(Path(g).read_bytes(), **MODES[mode]))
    return dst


def srgb_to_lab(rgb):
    c = rgb.astype(np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return L, np.hypot(a, b)


def stats(img_rgba, ref_rgb, sel):
    """sel 화소에서 렌더 vs 기준: ΔL*(렌더 − 기준), ΔC*, RGB 평균 절대 차이."""
    x, y = img_rgba[..., :3][sel], ref_rgb[sel]
    Lx, Cx = srgb_to_lab(x)
    Ly, Cy = srgb_to_lab(y)
    return {"dL": float(Lx.mean() - Ly.mean()), "dC": float(Cx.mean() - Cy.mean()),
            "rgb_mean_abs": float(np.abs(x.astype(int) - y.astype(int)).mean()), "L_render": float(Lx.mean()), "L_ref": float(Ly.mean()),
            "C_render": float(Cx.mean()), "C_ref": float(Cy.mean()), "px": int(sel.sum())}


def color():
    from PIL import Image, ImageDraw
    res, rows = {}, []
    for s in scenarios():
        d = Q / s["id"]
        for mode in MODES:
            out = d / mode
            out.mkdir(parents=True, exist_ok=True)
            if not (out / "render_0004.png").exists():
                glb = prepared(s, r4_glb(s), mode, out / "render_input.glb")
                subprocess.run([PY, "-c", RENDER, str(REF), str(glb), str(out)], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cd = cond_dir(s)
        v = s["edit_view"]
        edit = np.asarray(Image.open(cd / "2d_edit.png").convert("RGB"))
        rend = np.asarray(Image.open(cd / "2d_render.png").convert("RGBA"))
        mask = np.asarray(Image.open(cd / "2d_mask.png").convert("L")) > 127
        imgs = {"i": np.asarray(Image.open(OUT / "checks" / "render" / s["id"] / "a" / f"render_{v:04d}.png").convert("RGBA"))}
        for mode in MODES:
            imgs[mode] = np.asarray(Image.open(d / mode / f"render_{v:04d}.png").convert("RGBA"))
        row = {}
        for k, im in imgs.items():
            obj = (im[..., 3] > 0) & (rend[..., 3] > 0)
            row[k] = {"vs_2d_edit": stats(im, edit, obj), "outside_mask_vs_2d_render": stats(im, rend[..., :3], obj & ~mask)}
        res[s["id"]] = row
        print(s["id"], {k: (round(x["vs_2d_edit"]["dL"], 1), round(x["vs_2d_edit"]["dC"], 1), round(x["vs_2d_edit"]["rgb_mean_abs"], 1),
                           round(x["outside_mask_vs_2d_render"]["rgb_mean_abs"], 1)) for k, x in row.items()}, flush=True)
        T = 200
        tiles = []
        ref_t = Image.fromarray(edit).resize((T, T), Image.LANCZOS)
        ImageDraw.Draw(ref_t).text((4, 4), "2d_edit (inpaint)", fill=(200, 0, 0))
        tiles.append(ref_t)
        for k in ("i", "ii", "iii", "iv"):
            im = Image.fromarray(imgs[k])
            t = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im).convert("RGB").resize((T, T), Image.LANCZOS)
            ImageDraw.Draw(t).text((4, 4), LABEL[k], fill=(200, 0, 0))
            tiles.append(t)
        r_ = Image.new("RGB", (5 * T, T + 16), "white")
        ImageDraw.Draw(r_).text((4, 2), f"{s['id']} edit view v{v}", fill=(0, 0, 0))
        for i, t in enumerate(tiles):
            r_.paste(t, (T * i, 16))
        rows.append(r_)
    sheet = Image.new("RGB", (5 * 200, 20 + sum(x.height for x in rows)), "white")
    ImageDraw.Draw(sheet).text((4, 4), "Renderer: fork render_rgb_and_mask (Blender Cycles), r1. VC = R4 vertex-color GLB (output_r11).", fill=(0, 0, 160))
    y = 20
    for x in rows:
        sheet.paste(x, (0, y)); y += x.height
    sheet.save(REPO / "docs" / "reports" / "img" / "02-R4-color.png", optimize=True)
    (Q / "color.json").write_text(json.dumps(res, indent=1))


def bench():
    sys.path.insert(0, str(REPO / "tools"))
    import e2a_metrics as em2
    mode = os.environ.get("R4_MODE", "iv")
    by_id = {s["id"]: s for s in em2.scenarios()}
    for s in scenarios():
        b = by_id[s["id"]]
        for r in ("r1", "r2", "r3"):
            src = r4_glb(s, r)
            if b["track"] == "B":
                bb = b["bench"]
                d = em2.OUT / "pred" / "r4" / r / s["id"] / bb["dataset"] / bb["object"] / f"prompt_{bb['prompt_id']}"
            else:
                d = em2.OUT / "pred" / "r4" / r / s["id"] / "TrackA" / s["id"] / "prompt_1"
            d.mkdir(parents=True, exist_ok=True)
            if not (d / "edit.glb").exists():
                prepared(s, src, mode, d / "edit.glb")
            em2._render(d / "edit.glb", d / "images")
            print("bench rendered", s["id"], r, flush=True)


# 세부 영역(1024 px 편집 시점 렌더 좌표 x0, y0, x1, y1) — 시나리오마다 무늬·세부가 있는 곳
CROPS = {"B1": [("basket weave", (240, 600, 700, 830)), ("watermelon stripes", (420, 450, 710, 640))],
         "B6": [("face", (420, 230, 640, 450)), ("torso", (360, 420, 720, 660))],
         "B8": [("roof", (370, 200, 710, 470)), ("wall/door", (360, 640, 720, 880))],
         "A-M1": [("helmet", (380, 220, 590, 410))],
         "B2": [("toppings", (250, 600, 710, 800))],
         "A-C1": [("backrest wood grain", (290, 180, 750, 360)), ("seat", (290, 450, 750, 600))]}


def crops():
    from PIL import Image, ImageDraw
    mode = os.environ.get("R4_MODE", "iv")
    rows = []
    for s in scenarios():
        if s["id"] not in CROPS:
            continue
        v = s["edit_view"]
        A = Image.open(OUT / "checks" / "render" / s["id"] / "a" / f"render_{v:04d}.png").convert("RGBA")
        B = Image.open(Q / s["id"] / mode / f"render_{v:04d}.png").convert("RGBA")
        A = Image.alpha_composite(Image.new("RGBA", A.size, "white"), A).convert("RGB")
        B = Image.alpha_composite(Image.new("RGBA", B.size, "white"), B).convert("RGB")
        for name, box in CROPS[s["id"]]:
            ca, cb = A.crop(box), B.crop(box)
            sc = 380 / max(ca.width, ca.height)
            ca = ca.resize((int(ca.width * sc), int(ca.height * sc)), Image.LANCZOS)
            cb = cb.resize(ca.size, Image.LANCZOS)
            r_ = Image.new("RGB", (2 * ca.width + 8, ca.height + 18), "white")
            ImageDraw.Draw(r_).text((4, 3), f"{s['id']} {name} (edit view v{v}, 1024 px crop {box})  left: textured GLB  right: R4 ({mode})", fill=(0, 0, 0))
            r_.paste(ca, (0, 18)); r_.paste(cb, (ca.width + 8, 18))
            rows.append(r_)
    W = max(x.width for x in rows)
    sheet = Image.new("RGB", (W, sum(x.height for x in rows)), "white")
    y = 0
    for x in rows:
        sheet.paste(x, (0, y)); y += x.height
    sheet.save(REPO / "docs" / "reports" / "img" / "02-R4-detail.png", optimize=True)
    print("crops", len(rows))


if __name__ == "__main__":
    {"color": color, "bench": bench, "crops": crops}[sys.argv[1]]()
