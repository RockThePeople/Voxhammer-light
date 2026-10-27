"""paper-results 실험별 캡처 빌더(chief 10-27). 새 편집·새 채점 없음 — 기존 출력 메시를 capture.Plan 으로 같은 시점에서 찍는다.

P1·P2(정점색 R4 메시)는 output.glb 를 N 좌표로 옮긴 뒤 (iv) 보정(editcore/render.py:vertex_color_render_glb)을 건다 — view.glb 와 같은 보정.
view.glb 를 그대로 trimesh 로 옮기면 보정 재질·선형 float 정점색이 빠져 검게 렌더된다(실측, 10-27).
모델 1개당 shaded.png(고정 조명) · albedo.png(조명 없음) · crop.png(편집 영역: 3D 마스크 bbox 투영 + 여유 15%, X4 제외).
편집 영역이 시점에서 보이지 않으면(원본 메시 레이캐스팅에서 마스크 셀에 닿는 화소 < OCC_MIN) "가려짐" 목록에 적는다(시점은 늘리지 않는다).
기존 출력이 없으면 만들지 않고 "없음" 목록에 적는다. 결과 = <out>/<실험>/<자산·시나리오>/<경로>/<회차>/, 기록 = <out>/_build/<실험>.json

    python tools/paper/capture/build.py <out> x3|stitch|x4|x0|x2a|x2b
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tools" / "paper" / "capture"), str(ROOT / "third_party" / "ai-ar-v3" / "contract" / "python")]
import capture as C  # noqa: E402

RUNS = Path(os.environ["RUNS_DIR"])
ASSETS = Path(os.environ["ASSETS_DIR"])
P = RUNS / "paper"
OCC_MIN = 50          # 256² 레이캐스팅에서 마스크(∪1셀) 에 닿는 화소가 이보다 적으면 "가려짐"
LOG = {"missing": [], "occluded": [], "models": [], "silhouette": []}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mask_cells(path):
    return np.asarray(json.loads(Path(path).read_text())["mask"]["voxels"], np.int64).reshape(-1, 3)


def add(pl, asset, mesh, kind, vc, dest, src_note, exp):
    if not Path(mesh).exists():
        LOG["missing"].append({"exp": exp, "dest": str(dest), "mesh": str(mesh)})
        return None
    pl.add(asset, mesh, kind, vc, dest)
    LOG["models"].append({"exp": exp, "dest": str(dest), "asset": asset, "mesh": str(mesh), "kind": kind, "vertex_color": vc,
                          "sha256": sha(mesh), "note": src_note})
    return dest


def crop(reg, cam, asset, cells, shaded, out):
    """3D 마스크 bbox(raw 셀) 8 꼭짓점을 N 으로 옮겨 같은 카메라로 투영 → 15% 여유 정사각형 → 1024² 로."""
    from PIL import Image
    from tools.paper_metrics import project, voxel_to_glb
    fr = C.frame(reg, asset)
    lo, hi = cells.min(0), cells.max(0) + 1
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)
    pn = fr.to_n(voxel_to_glb(corners / 64.0 - 0.5), "raw")
    xy = project(pn, cam["elev"], cam["az"], 1024)
    (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
    side = max(x1 - x0, y1 - y0) * 1.30                          # 양쪽 15% 여유
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(side, 64)
    box = [int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(cx + side / 2)), int(round(cy + side / 2))]
    im = Image.open(shaded).convert("RGB")
    canvas = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), (255, 255, 255))   # 이미지 밖은 흰 배경
    canvas.paste(im.crop((max(box[0], 0), max(box[1], 0), min(box[2], 1024), min(box[3], 1024))), (max(-box[0], 0), max(-box[1], 0)))
    canvas.resize((1024, 1024), Image.LANCZOS).save(out)
    return box


def occluded(reg, cam, asset, cells, label, exp):
    """원본 메시를 그 시점에서 레이캐스팅해 마스크(∪1셀) 셀에 닿는 화소 수."""
    import open3d as o3d
    from deltacontract.coords import dilate_cells
    from tools.paper_metrics import mesh_in_n, view_rays
    fr = C.frame(reg, asset)
    m = mesh_in_n(reg[asset]["v0"], fr, "v0")
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(np.asarray(m.vertices, np.float32)), o3d.core.Tensor(np.asarray(m.faces, np.uint32)))
    o, d = view_rays((cam["elev"], cam["az"]), 256)
    h = sc.cast_rays(o3d.core.Tensor(np.concatenate([np.broadcast_to(o, d.shape), d], 1).astype(np.float32)))
    t = h["t_hit"].numpy()
    ok = np.isfinite(t)
    pts = o[None] + t[ok, None] * d[ok]
    c = fr.points_to_cells(pts)
    dil = np.asarray(dilate_cells(cells, 1), np.int64)
    code = lambda a: a[:, 0] * 4096 + a[:, 1] * 64 + a[:, 2]   # noqa: E731
    n = int(np.isin(code(c), code(dil)).sum())
    if n < OCC_MIN:
        LOG["occluded"].append({"exp": exp, "item": label, "asset": asset, "visible_px_256": n})
    return n


def silhouette(pl, key_a, key_b):
    from PIL import Image
    A = np.asarray(Image.open(C.CACHE / "renders" / key_a / "shaded_alpha.png")) > 127
    B = np.asarray(Image.open(C.CACHE / "renders" / key_b / "shaded_alpha.png")) > 127
    rgb = np.full(A.shape + (3,), 255, np.uint8)
    rgb[A & ~B] = (220, 40, 40)
    rgb[B & ~A] = (40, 170, 220)
    rgb[A & B] = (150, 150, 150)
    return float((A & B).sum() / max((A | B).sum(), 1)), Image.fromarray(rgb)


def key_of(pl, dest):
    for k, it in pl.items.items():
        if Path(dest) in it["dest"]:
            return k
    return None


def grid_sheet(rows, cols, cells, out, title, S=256):
    """rows/cols 라벨, cells[(r, c)] = png 경로(없으면 빈 칸) — 라벨은 시트에만."""
    from PIL import Image, ImageDraw
    W, H = 160 + S * len(cols), 40 + 24 + S * len(rows)
    sh = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sh)
    d.text((6, 6), title, fill="black")
    for j, c in enumerate(cols):
        d.text((160 + j * S + 4, 40), c, fill="black")
    for i, r in enumerate(rows):
        d.text((6, 64 + i * S + S // 2), r, fill="black")
        for j in range(len(cols)):
            p = cells.get((i, j))
            if p and Path(p).exists():
                sh.paste(Image.open(p).convert("RGB").resize((S, S), Image.LANCZOS), (160 + j * S, 64 + i * S))
            else:
                d.text((160 + j * S + S // 3, 64 + i * S + S // 2), "-", fill="gray")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sh.save(out)


def copy_cond(src_dir, dst_dir, prompt=None, names=("2d_render.png", "2d_mask.png", "2d_edit.png")):
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        if (Path(src_dir) / n).exists():
            shutil.copyfile(Path(src_dir) / n, dst_dir / n)
        else:
            LOG["missing"].append({"exp": "cond", "dest": str(dst_dir / n), "mesh": str(Path(src_dir) / n)})
    if prompt is not None:
        (dst_dir / "prompt.txt").write_text(prompt + "\n")


# ── X3 ──────────────────────────────────────────────────────────────────────────
def x3(out):
    from tools.paper.x3 import round_info
    from tools.paper.x3_score import mesh_spec
    pl = C.Plan()
    reg, cam = pl.reg, pl.cam
    base = out / "x3"
    st = P / "x3" / "stitched"
    for a in ("jeep", "cottage", "adventurer"):
        add(pl, a, reg[a]["v0"], "v0", False, base / a / "original" / "r0", "생성 텍스처 GLB(A_origin)", "x3")
        for k in range(1, 6):
            p0 = mesh_spec("p0", a, k)                               # output_v0.glb = 포크 출력에 C_k⁻¹(round.json)
            add(pl, a, p0["mesh"], "v0", False, base / a / "p0" / f"r{k}", "P0 to_glb, round.json C_k⁻¹ 로 v0 로 되돌림", "x3")
            for p in ("p1", "p2"):
                add(pl, a, st / p / a / f"r{k}" / "output.glb", "raw", True, base / a / p / f"r{k}", "봉합본 output.glb + (iv) 보정(= view.glb)", "x3")
    pl.run("x3")
    for a in ("jeep", "cottage", "adventurer"):
        for k in range(1, 6):
            md, info = round_info(a, k)
            cells = mask_cells(md / "mask.json")
            occluded(reg, cam[a], a, cells, f"x3/{a}/r{k}", "x3")
            for p in ("p0", "p1", "p2"):
                d = base / a / p / f"r{k}"
                if (d / "shaded.png").exists():
                    crop(reg, cam[a], a, cells, d / "shaded.png", d / "crop.png")
                copy_cond(P / "x3" / p / a / f"r{k}", base / a / "cond" / f"r{k}" / p, info["prompt"])
            o = base / a / "original" / "r0"
            crop(reg, cam[a], a, cells, o / "shaded.png", o / f"crop_r{k}.png")
        shutil.copyfile(C.CAMS, base / a / "cameras.json")
        # 시트: 경로 × 회차(음영·알베도·크롭)
        for m in ("shaded", "albedo", "crop"):
            cells_ = {}
            rows = ["original r0", "P0", "P1 (stitched)", "P2 (stitched)"]
            for j, k in enumerate(range(1, 6)):
                cells_[(0, j)] = base / a / "original" / "r0" / (f"crop_r{k}.png" if m == "crop" else f"{m}.png")
                for i, p in enumerate(("p0", "p1", "p2"), 1):
                    cells_[(i, j)] = base / a / p / f"r{k}" / f"{m}.png"
            grid_sheet(rows, [f"r{k}" for k in range(1, 6)], cells_, out / "sheets" / f"x3_{a}_{m}.png", f"X3 {a} - {m}")
        # 실루엣: 원본 vs 각 경로 r1
        ko = key_of(pl, base / a / "original" / "r0")
        for p in ("p0", "p1", "p2"):
            iou, im = silhouette(pl, ko, key_of(pl, base / a / p / "r1"))
            im.save(out / "sheets" / f"silhouette_x3_{a}_{p}_r1.png")
            LOG["silhouette"].append({"exp": "x3", "asset": a, "path": p, "round": "r1", "alpha_iou": iou})
    # 오두막 r5 굴뚝 실패 시트
    cells_, rows = {}, ["P0", "P1 (stitched)", "P2 (stitched)"]
    for i, p in enumerate(("p0", "p1", "p2")):
        for j, n in enumerate(("2d_render.png", "2d_edit.png")):
            cells_[(i, j)] = base / "cottage" / "cond" / "r5" / p / n
        cells_[(i, 2)] = base / "cottage" / p / "r4" / "shaded.png"
        cells_[(i, 3)] = base / "cottage" / p / "r5" / "shaded.png"
        cells_[(i, 4)] = base / "cottage" / p / "r5" / "crop.png"
    grid_sheet(rows, ["2d_render (input)", "2d_edit", "r4 result", "r5 result", "r5 crop"], cells_,
               out / "sheets" / "x3_cottage_r5_chimney.png", "X3 cottage r5 - 'a plain brown shingled roof with no chimney'")
    return pl


# ── 봉합 ────────────────────────────────────────────────────────────────────────
def stitch(out):
    from tools.paper.x3 import round_info
    pl = C.Plan()
    reg, cam = pl.reg, pl.cam
    base = out / "stitch"
    for a in ("jeep", "cottage", "adventurer"):
        for p in ("p1", "p2"):
            for k in range(1, 6):
                add(pl, a, P / "x3" / p / a / f"r{k}" / "output.glb", "raw", True, base / a / f"{p}_prestitch" / f"r{k}",
                    "봉합 전(스냅만) view.glb", "stitch")
    for v, lab in (("skirt", "skirt"), ("both", "stitch_skirt")):
        for k in range(1, 6):
            add(pl, "jeep", P / "x3" / "seamvar" / v / "p1" / "jeep" / f"r{k}" / "output.glb", "raw", True, base / "jeep" / f"p1_{lab}" / f"r{k}",
                f"10-1절 시험 조합 {lab}", "stitch")
    for k in range(1, 6):
        add(pl, "jeep", P / "x3" / "c4" / "fulldecode" / "jeep" / f"r{k}" / "output.glb", "raw", True, base / "jeep" / "p1_fulldecode" / f"r{k}",
            "승계 없이 통째 디코딩(C4 3)", "stitch")
    pl.run("stitch")
    for a in ("jeep", "cottage", "adventurer"):
        for k in range(1, 6):
            cells = mask_cells(round_info(a, k)[0] / "mask.json")
            for d in (base / a).glob(f"*/r{k}"):
                if (d / "shaded.png").exists():
                    crop(reg, cam[a], a, cells, d / "shaded.png", d / "crop.png")
        shutil.copyfile(C.CAMS, base / a / "cameras.json")
    vars_ = ["p1_prestitch", "p1_stitch(x3)", "p1_skirt", "p1_stitch_skirt", "p1_fulldecode"]
    cells_ = {}
    for i, v in enumerate(vars_):
        for j, k in enumerate(range(1, 6)):
            src = out / "x3" / "jeep" / "p1" / f"r{k}" if v == "p1_stitch(x3)" else base / "jeep" / v / f"r{k}"
            cells_[(i, j)] = src / "shaded.png"
    grid_sheet(vars_, [f"r{k}" for k in range(1, 6)], cells_, out / "sheets" / "stitch_jeep_p1_variants.png", "jeep P1 seam variants - shaded")
    return pl


# ── X4 ──────────────────────────────────────────────────────────────────────────
def x4(out):
    pl = C.Plan()
    reg = pl.reg
    base = out / "x4"
    shas = {}
    for a in ("moto", "snowman", "chair", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"):
        add(pl, a, reg[a]["v0"], "v0", False, base / a / "p0" / "r0", "원본(r0)", "x4")
        for k in range(1, 6):
            add(pl, a, P / "x4" / "p0" / a / f"r{k}" / "output.glb", "n", False, base / a / "p0" / f"r{k}", "P0 to_glb(N 좌표, x4_score 와 같음)", "x4")
        hs = [sha(P / "x4" / "p1" / a / f"r{k}" / "output.glb") for k in range(1, 6)]
        shas[a] = {"p1_output_sha256": hs, "identical": len(set(hs)) == 1}
        add(pl, a, P / "x4" / "p1" / a / "r1" / "output.glb", "raw", True, base / a / "p1" / "r1-5", "P1 view.glb(r1–r5 같은 SLat 재디코딩, 1회 촬영)", "x4")
    pl.run("x4")
    rows = list(shas)
    for m in ("shaded", "albedo"):
        cells_ = {}
        for i, a in enumerate(rows):
            for j, k in enumerate(range(0, 6)):
                cells_[(i, j)] = base / a / "p0" / f"r{k}" / f"{m}.png"
            cells_[(i, 6)] = base / a / "p1" / "r1-5" / f"{m}.png"
        grid_sheet(rows, ["original", "P0 r1", "P0 r2", "P0 r3", "P0 r4", "P0 r5", "P1 r1-5"], cells_, out / "sheets" / f"x4_{m}.png", f"X4 - {m}", S=200)
    for a in rows:
        iou, im = silhouette(pl, key_of(pl, base / a / "p0" / "r0"), key_of(pl, base / a / "p0" / "r1"))
        im.save(out / "sheets" / f"silhouette_x4_{a}_p0_r1.png")
        LOG["silhouette"].append({"exp": "x4", "asset": a, "path": "p0", "round": "r1", "alpha_iou": iou})
        iou, im = silhouette(pl, key_of(pl, base / a / "p0" / "r0"), key_of(pl, base / a / "p1" / "r1-5"))
        im.save(out / "sheets" / f"silhouette_x4_{a}_p1.png")
        LOG["silhouette"].append({"exp": "x4", "asset": a, "path": "p1", "round": "r1-5", "alpha_iou": iou})
        shutil.copyfile(C.CAMS, base / a / "cameras.json")
    LOG["x4_p1_sha"] = shas
    return pl


# ── X0 · X2-A · X2-B ────────────────────────────────────────────────────────────
def _asset_of(sid):
    if sid.startswith("A-"):
        return json.loads((ROOT / "scenarios" / "track_a.json").read_text())["scenarios"][sid]["asset"]
    return sid


def x0(out):
    from tools.paper.x0 import scenarios
    pl = C.Plan()
    reg, cam = pl.reg, pl.cam
    base = out / "x0"
    S = scenarios()
    E = RUNS / "2026-10-25"
    for sid, s in S.items():
        a = _asset_of(sid)
        add(pl, a, reg[a]["v0"], "v0", False, base / sid / "original", "원본 v0", "x0")
        add(pl, a, RUNS / "01-port" / s["name"] / "port-r1" / "output.glb", "raw", False, base / sid / "base25" / "r1", "25 스텝 to_glb(0-1 이식본 r1)", "x0")
        add(pl, a, E / s["name"] / "e2a-c15" / "r1" / "output.glb", "raw", False, base / sid / "c15_glb" / "r1", "15 스텝 to_glb(E2a c15 r1)", "x0")
        add(pl, a, P / "x0" / "cbin_stitched" / sid / "r1" / "output.glb", "raw", True, base / sid / "c15_cbin" / "r1",
            "15 스텝 R4 .cbin + 봉합(채택), 같은 slat_tgt", "x0")
    pl.run("x0")
    rows = list(S)
    for sid, s in S.items():
        a = _asset_of(sid)
        cells = mask_cells(s["mask"])
        occluded(reg, cam[a], a, cells, f"x0/{sid}", "x0")
        for d in [base / sid / "original"] + [base / sid / c / "r1" for c in ("base25", "c15_glb", "c15_cbin")]:
            if (d / "shaded.png").exists():
                crop(reg, cam[a], a, cells, d / "shaded.png", d / "crop.png")
        copy_cond(s["cond"], base / sid / "cond", s["prompt"])
        shutil.copyfile(C.CAMS, base / sid / "cameras.json")
        for c in ("base25", "c15_glb", "c15_cbin"):
            iou, im = silhouette(pl, key_of(pl, base / sid / "original"), key_of(pl, base / sid / c / "r1"))
            im.save(out / "sheets" / f"silhouette_x0_{sid}_{c}.png")
            LOG["silhouette"].append({"exp": "x0", "asset": sid, "path": c, "round": "r1", "alpha_iou": iou})
    for m in ("shaded", "crop"):
        cells_ = {}
        for i, sid in enumerate(rows):
            for j, c in enumerate(("original", "base25", "c15_glb", "c15_cbin")):
                cells_[(i, j)] = (base / sid / c if c == "original" else base / sid / c / "r1") / f"{m}.png"
        grid_sheet(rows, ["original", "25 to_glb", "15 to_glb (P1-glb)", "15 cbin+stitch (P1-cbin)"], cells_, out / "sheets" / f"x0_{m}.png", f"X0 r1 - {m}", S=200)
    return pl


def x2a(out):
    from tools.paper.x0 import scenarios
    pl = C.Plan()
    reg, cam = pl.reg, pl.cam
    base = out / "x2a"
    S = scenarios()
    E = RUNS / "2026-10-25"
    for sid, s in S.items():
        a = _asset_of(sid)
        add(pl, a, reg[a]["v0"], "v0", False, base / sid / "original", "원본 v0", "x2a")
        for r in ("r1", "r2"):
            add(pl, a, P / "x2a" / "p0" / sid / r / "output.glb", "n", False, base / sid / "p0" / r, "P0 to_glb(T(v0) = N 좌표, prep.json T)", "x2a")
        for r in ("r1", "r2", "r3"):
            add(pl, a, E / s["name"] / "e2a-c15" / r / "output.glb", "raw", False, base / sid / "p1_glb" / r, "P1-glb(E2a c15)", "x2a")
            add(pl, a, P / "x0" / "cbin_stitched" / sid / r / "output.glb", "raw", True, base / sid / "p1_cbin" / r, "P1-cbin(R4 + 봉합)", "x2a")
    pl.run("x2a")
    for sid, s in S.items():
        a = _asset_of(sid)
        cells = mask_cells(s["mask"])
        occluded(reg, cam[a], a, cells, f"x2a/{sid}", "x2a")
        for d in [base / sid / "original"] + list((base / sid).glob("p*/r*")):
            if (d / "shaded.png").exists():
                crop(reg, cam[a], a, cells, d / "shaded.png", d / "crop.png")
        copy_cond(s["cond"], base / sid / "cond", s["prompt"])
        shutil.copyfile(C.CAMS, base / sid / "cameras.json")
        for pth in ("p0", "p1_glb", "p1_cbin"):
            iou, im = silhouette(pl, key_of(pl, base / sid / "original"), key_of(pl, base / sid / pth / "r1"))
            im.save(out / "sheets" / f"silhouette_x2a_{sid}_{pth}_r1.png")
            LOG["silhouette"].append({"exp": "x2a", "asset": sid, "path": pth, "round": "r1", "alpha_iou": iou})
    cols = ["original", "P0 r1", "P0 r2", "P1-glb r1", "P1-glb r2", "P1-glb r3", "P1-cbin r1", "P1-cbin r2", "P1-cbin r3"]
    src = [("original", None), ("p0", "r1"), ("p0", "r2"), ("p1_glb", "r1"), ("p1_glb", "r2"), ("p1_glb", "r3"), ("p1_cbin", "r1"), ("p1_cbin", "r2"), ("p1_cbin", "r3")]
    for m in ("shaded", "crop"):
        cells_ = {}
        for i, sid in enumerate(S):
            for j, (pth, r) in enumerate(src):
                cells_[(i, j)] = (base / sid / pth if r is None else base / sid / pth / r) / f"{m}.png"
        grid_sheet(list(S), cols, cells_, out / "sheets" / f"x2a_{m}.png", f"X2-A - {m}", S=180)
    return pl


def x2b(out):
    pl = C.Plan()
    reg, cam = pl.reg, pl.cam
    base = out / "x2b"
    items = json.loads((ROOT / "scenarios" / "x2b_list.json").read_text())["scenarios"]
    for s in items:
        a = s["id"]
        add(pl, a, reg[a]["v0"], "v0", False, base / a / "original", "데이터셋 원본 GLB", "x2b")
        add(pl, a, P / "x2b" / "p0" / a / "r1" / "output.glb", "n", False, base / a / "p0", "P0 to_glb(N)", "x2b")
        add(pl, a, P / "x2b" / "p1" / a / "r1" / "output.glb", "raw", True, base / a / "p1", "P1 R4 + 봉합 view.glb", "x2b")
    pl.run("x2b")
    cells_ = {}
    for i, s in enumerate(items):
        a = s["id"]
        cells = mask_cells(ASSETS / "x2b" / a / "mask.json")
        occluded(reg, cam[a], a, cells, f"x2b/{a}", "x2b")
        for j, pth in enumerate(("original", "p0", "p1")):
            d = base / a / pth
            if (d / "shaded.png").exists():
                crop(reg, cam[a], a, cells, d / "shaded.png", d / "crop.png")
            cells_[(i, j)] = d / "shaded.png"
        copy_cond(ASSETS / "x2b" / a / "cond", base / a / "cond", s["prompt"])
        shutil.copyfile(C.CAMS, base / a / "cameras.json")
        for pth in ("p0", "p1"):
            iou, im = silhouette(pl, key_of(pl, base / a / "original"), key_of(pl, base / a / pth))
            im.save(out / "sheets" / f"silhouette_x2b_{a}_{pth}.png")
            LOG["silhouette"].append({"exp": "x2b", "asset": a, "path": pth, "round": "r1", "alpha_iou": iou})
    grid_sheet([s["id"] for s in items], ["original", "P0", "P1"], cells_, out / "sheets" / "x2b_shaded.png", "X2-B - shaded")
    return pl


if __name__ == "__main__":
    out, exp = Path(sys.argv[1]), sys.argv[2]
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    {"x3": x3, "stitch": stitch, "x4": x4, "x0": x0, "x2a": x2a, "x2b": x2b}[exp](out)
    (out / "_build").mkdir(exist_ok=True)
    (out / "_build" / f"{exp}.json").write_text(json.dumps(LOG, ensure_ascii=False, indent=1))
    print(exp, "models", len(LOG["models"]), "missing", len(LOG["missing"]), "occluded", len(LOG["occluded"]),
          "silhouette min", min([s["alpha_iou"] for s in LOG["silhouette"]] or [None]) if LOG["silhouette"] else None, flush=True)
