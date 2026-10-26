"""X3 회차별 마스크 초안 (ops/tasks/X-paper.md X3, 사용자 지시 12:37: seed 0 이미지, 원문 1→5 순서·프롬프트 그대로).

지시서대로 "v0 편집 시점 렌더에서 2D 영역을 잡고 깊이로 셀을 고른다": 회차마다 편집 시점(포크 0–4)의 1024² v0 렌더 위
화소 상자를 잡고, 점유 셀 중심을 같은 카메라로 투영해 상자 안이면서 그 화소의 가장 앞 셀 깊이 + `depth` 셀 이내인 셀(보이는 표면층)을
고른다(`flat` 이면 상자 전체에서 가장 앞 깊이 기준 — 창·문처럼 뚫린 벽 부위에서 안쪽이 잡히지 않게). 마스크 = 고른 셀 ∪ 여유
2셀 팽창(deltacontract.dilate_cells, E2 와 같은 26-이웃). 추가 회차는 `extend` 로 고른 셀 bbox(+여유)를 빈 공간 쪽으로 늘린 상자를 더한다.
(트랙 A 처럼 bbox 전체를 마스크하면 5회 누적이 점유 40% 를 넘어서 — 지프 0.46·모험가 0.94 실측 — 셀 팽창으로 바꿨다.) 화소 상자는 inspect 의 눈금 렌더(grid_v<n>.png)를 보고 정한다. 부위가 생성 결과에 없으면 원문 부위 설명의 위치에 잡고
`absent` 를 적는다.
원문 규칙: 5회차 누적 마스크가 점유 셀의 40% 를 넘으면 그 회차 부위를 줄인다.

    python tools/paper/x3_masks.py inspect [자산 …]   # $RUNS_DIR/paper/x3/inspect/<자산>.png
    python tools/paper/x3_masks.py build   [자산 …]   # $RUNS_DIR/x3/<자산>/r<k>/ (T1 뷰어 항목 x3/<자산>/r<k>)

회차 폴더: model.glb(v0 심볼릭 링크) · mask_r<k>.glb(셀 노출면, 뷰어 겹치기) · mask.json(트랙 A 형식) · view.png(편집 시점 v0 렌더
+ 마스크 실루엣) · info.json(프롬프트·부위·셀 수·누적 점유 비율·absent).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scenarios"), str(ROOT / "third_party" / "ai-ar-v3"),
                str(ROOT / "third_party" / "ai-ar-v3" / "contract" / "python")]
ASSETS = Path(os.environ["ASSETS_DIR"]) / "x3" / "track_a"
RUNS = Path(os.environ["RUNS_DIR"])
OUT = RUNS / "x3"
RES, MARGIN, HALO, LIMIT = 64, 2, 2, 0.40

# 원문(ops/paper/PLAN.md X3) 편집 1→5. (종류, 부위, 프롬프트)
EDITS = {
    "jeep": [("교체", "앞 범퍼", "a heavy chrome steel bull bar front bumper on the jeep"),
             ("교체·외관", "보닛", "the jeep hood painted bright red with two white racing stripes"),
             ("교체", "뒤 스페어 타이어", "a rear rack holding two red jerry cans on the back of the jeep"),
             ("추가", "지붕 위", "a black roof rack with a brown leather travel bag strapped on top of the jeep roof"),
             ("제거", "스노클(오른쪽 앞 기둥)", "the smooth right front pillar of the jeep with no snorkel, olive green paint")],
    "cottage": [("교체", "현관문", "an arched red wooden front door with iron hinges"),
                ("교체·외관", "왼쪽 창", "a round stained glass window with blue and yellow panes"),
                ("추가", "오른쪽 창 아래", "a wooden flower box with red flowers under the window"),
                ("교체", "현관 난간", "a white picket fence railing around the porch"),
                ("제거", "굴뚝", "a plain brown shingled roof with no chimney")],
    "adventurer": [("교체", "모자", "a tall blue pointed wizard hat with yellow stars"),
                   ("교체", "오른손의 칼", "a glowing brass lantern held in the right hand"),
                   ("추가", "오른쪽 어깨 위", "a small blue bird perched on the right shoulder"),
                   ("교체·외관", "부츠", "red sneakers with white soles"),
                   ("제거", "왼팔의 방패", "the bare left arm in a green tunic sleeve, no shield")],
}

# 회차별 {view: 포크 시점, px: [[x0,y0,x1,y1], …](1024² 화소), depth: 표면층 두께(셀), extend: {"+z": n, …}, absent: 메모}
# 화소 상자 근거: $RUNS_DIR/paper/x3/inspect/<자산>_grid_v<n>.png (2026-10-26 a5000 육안).
# 지프: 앞 = 복셀 −y, 차의 오른쪽 = −x. 포크 시점 3 = 앞·오른쪽 3/4, 4 = 오른쪽 옆.
# 오두막: 앞(문) = +x, 시점 0 정면, 1 = 앞·왼쪽 3/4. 앞벽 창은 문 오른쪽 하나뿐 → "왼쪽 창" = 시점 1 왼쪽 옆벽 창, "오른쪽 창" = 앞벽 창.
# 모험가: 시점 1 정면(칼 = 오른손, 화면 왼쪽), 시점 0 = 방패 정면.
SEL = {
    "jeep": [{"view": 3, "px": [[650, 580, 860, 705]], "depth": 6},
             {"view": 3, "px": [[515, 468, 800, 560]], "depth": 3},
             {"view": 4, "px": [[183, 428, 248, 552]], "depth": 10},
             {"view": 3, "px": [[350, 318, 650, 348]], "depth": 3, "extend": {"+z": 8}},
             {"view": 3, "px": [[462, 365, 505, 478]], "depth": 4, "extend": {"-x": 3},
              "absent": "seed 0 지프에 스노클이 없다 — 원문 위치(오른쪽 앞 기둥)에 마스크, 프롬프트 그대로"}],
    "cottage": [{"view": 0, "px": [[350, 612, 492, 858]], "depth": 3},
                {"view": 1, "px": [[245, 590, 338, 750]], "depth": 4},
                {"view": 0, "px": [[550, 778, 725, 830]], "depth": 3, "flat": True, "extend": {"+x": 4}},
                {"view": 1, "px": [[410, 795, 500, 930]], "depth": 6},
                {"view": 0, "px": [[375, 185, 465, 340]], "depth": 14}],
    # 모험가 모자(챙 포함 전체)만으로 점유 0.537 → 원문 40% 규칙으로 1회차 = 모자 윗부분(띠 위 크라운), 2회차 = 칼날·코등이(손 제외)
    "adventurer": [{"view": 1, "px": [[380, 185, 690, 272]], "depth": 30,
                    "reduced": "40% 규칙: 모자 전체(챙 포함, 점유 0.537) → 띠 위 크라운 윗부분"},
                   {"view": 1, "px": [[225, 425, 372, 615]], "depth": 10, "reduced": "40% 규칙: 칼 + 손 → 칼날·코등이"},
                   {"view": 1, "px": [[418, 538, 468, 562]], "depth": 3, "extend": {"+z": 5}},
                   {"view": 1, "px": [[425, 722, 655, 835]], "depth": 10},
                   {"view": 0, "px": [[522, 566, 738, 792]], "depth": 4}],
}


def load_coords(asset):
    from safetensors import safe_open
    with safe_open(str(ASSETS / asset / "gen" / "slat.safetensors"), "np") as f:
        return f.get_tensor("coords")[:, 1:].astype(np.int64)


def v0_renders(asset):
    """v0 model.glb(정규화본 = N 좌표) 를 Bench render.py 16시점으로 렌더(포크 5시점은 FORK_TO_BENCH)."""
    from tools.paper_metrics import bench_render
    return bench_render(ASSETS / asset / "gen" / "model.glb", RUNS / "paper" / "x3" / "v0_render" / asset)


def cam_project(cells, nz, view, res=1024):
    """셀 중심 → (화면 x, y, 깊이). paper_metrics.project 와 같은 카메라(Bench/포크, 반지름 1.8)."""
    import math
    from tools.paper_metrics import CAM_R, FOV_T, FORK_VIEWS, voxel_to_glb
    g = (voxel_to_glb((np.asarray(cells, float) + 0.5) / RES - 0.5) - np.array(nz["center_glb_frame"])) * nz["scale"]
    b = np.stack([g[:, 0], -g[:, 2], g[:, 1]], 1)
    el, az = math.radians(FORK_VIEWS[view][0]), math.radians(FORK_VIEWS[view][1] - 90)
    cam = CAM_R * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    f = -cam / np.linalg.norm(cam)
    r = np.cross(f, [0, 0, 1]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    d = b - cam
    z = d @ f
    return (d @ r / z / FOV_T * 0.5 + 0.5) * res, (0.5 - d @ u / z / FOV_T * 0.5) * res, z


def pick_cells(occ, nz, view, boxes, depth, res=1024, flat=False):
    """화소 상자 안 + 가장 앞 셀 깊이 + depth 셀 이내(보이는 표면층)인 점유 셀. flat: 상자 전체 최소 깊이 기준."""
    x, y, z = cam_project(occ, nz, view, res)
    cell = nz["scale"] / RES
    g = 4                                                  # 4 px 칸 z-버퍼(셀 하나가 ~6 px)
    ix, iy = np.clip((x // g).astype(int), 0, res // g - 1), np.clip((y // g).astype(int), 0, res // g - 1)
    zb = np.full((res // g, res // g), np.inf)
    np.minimum.at(zb, (iy, ix), z)
    # 이웃 칸 최소(셀 크기 > 칸) — 구멍 없이 앞면을 잡는다
    from scipy.ndimage import minimum_filter
    zb = minimum_filter(zb, size=3)
    inb = np.zeros(len(occ), bool)
    for x0, y0, x1, y1 in boxes:
        inb |= (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    front = z <= (z[inb].min() if flat and inb.any() else zb[iy, ix]) + depth * cell
    return occ[inb & front]


def inspect(asset):
    """시트: 위 = 포크 5시점 v0 렌더, 아래 = 점유 셀 직교 투영 3장(위 x–y, 앞 x–z, 옆 y–z; 8셀 눈금)."""
    from PIL import Image, ImageDraw
    from tools.paper_metrics import FORK_TO_BENCH
    occ = load_coords(asset)
    d = v0_renders(asset)
    S = 360
    sheet = Image.new("RGB", (S * 5, S + 3 * 0 + 400), "white")
    for v in range(5):
        im = Image.open(d / f"render_{FORK_TO_BENCH[v]:04d}.png").convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); bg.alpha_composite(im)
        sheet.paste(bg.convert("RGB").resize((S, S)), (v * S, 0))
        ImageDraw.Draw(sheet).text((v * S + 6, 6), f"fork view {v}", fill="black")
    sc = 6
    for n, (a, b, lab) in enumerate(((0, 1, "top x→ y↑ (z max)"), (0, 2, "front x→ z↑ (y min first)"), (1, 2, "side y→ z↑ (x min first)"))):
        c = 3 - a - b
        img = np.full((RES * sc, RES * sc, 3), 255, np.uint8)
        order = np.argsort(occ[:, c]) if n == 0 else np.argsort(-occ[:, c])   # 위: 높은 z 가 위에, 앞·옆: 작은 y/x 가 앞
        for p in occ[order]:
            t = p[c] / (RES - 1)
            col = (int(40 + 200 * t), int(60 + 120 * (1 - t)), int(200 - 150 * t))
            x, y = p[a] * sc, (RES - 1 - p[b]) * sc
            img[y:y + sc, x:x + sc] = col
        pi = Image.fromarray(img)
        dr = ImageDraw.Draw(pi)
        for g in range(0, RES + 1, 8):
            dr.line([(g * sc, 0), (g * sc, RES * sc)], fill=(200, 200, 200))
            dr.line([(0, (RES - g) * sc), (RES * sc, (RES - g) * sc)], fill=(200, 200, 200))
            dr.text((g * sc + 1, RES * sc - 11), str(g), fill="black")
            dr.text((1, (RES - g) * sc - 11), str(g), fill="black")
        dr.text((4, 4), lab, fill="black")
        sheet.paste(pi.resize((384, 384)), (n * 600 + 10, S + 8))
    p = RUNS / "paper" / "x3" / "inspect" / f"{asset}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(p)
    for v in range(5):                                     # 화소 상자를 잡을 눈금 렌더(64 px 눈금)
        im = Image.open(d / f"render_{FORK_TO_BENCH[v]:04d}.png").convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); bg.alpha_composite(im)
        g = bg.convert("RGB"); dr = ImageDraw.Draw(g)
        for t in range(0, 1025, 64):
            dr.line([(t, 0), (t, 1024)], fill=(255, 0, 255) if t % 256 == 0 else (200, 200, 200))
            dr.line([(0, t), (1024, t)], fill=(255, 0, 255) if t % 256 == 0 else (200, 200, 200))
            dr.text((t + 2, 2), str(t), fill="black"); dr.text((2, t + 2), str(t), fill="black")
        g.save(p.parent / f"{asset}_grid_v{v}.png")
    print(asset, "occupied", len(occ), "sheet", p)


def build(asset):
    import trimesh
    from PIL import Image
    from deltacontract import mask_fingerprint
    from deltacontract.coords import dilate_cells
    from s5_masks import boundary_mesh, dense
    from server.pipeline.frames import VOXEL_TO_GLB
    from tools.paper_metrics import FORK_TO_BENCH, FORK_VIEWS, AssetFrame, silhouette
    occ = load_coords(asset)
    oset = {tuple(x) for x in occ.tolist()}
    gen = json.loads((ASSETS / asset / "gen" / "gen.json").read_text())
    nz = gen["glb_normalize"]
    fr = AssetFrame.from_v0(ASSETS / asset / "gen" / "model.glb", nz["center_glb_frame"], nz["scale"])
    rd = v0_renders(asset)
    cum = set()
    summary = []
    for k, ((kind, part_name, prompt), opt) in enumerate(zip(EDITS[asset], SEL[asset]), 1):
        view = opt["view"]
        part = pick_cells(occ, nz, view, opt["px"], opt.get("depth", 6), flat=opt.get("flat", False))
        if not len(part):
            raise SystemExit(f"{asset} r{k}: 화소 상자에 보이는 점유 셀이 없다 {opt['px']}")
        cells = [np.asarray(dilate_cells(part, MARGIN), np.int64).reshape(-1, 3)]
        boxes = []
        if opt.get("extend"):
            blo, bhi = np.clip(part.min(0) - MARGIN, 0, RES - 1), np.clip(part.max(0) + MARGIN, 0, RES - 1)
            for ax_s, n in opt["extend"].items():
                ax = "xyz".index(ax_s[1])
                if ax_s[0] == "+":
                    blo[ax], bhi[ax] = part[:, ax].max(), min(RES - 1, part[:, ax].max() + n)
                else:
                    blo[ax], bhi[ax] = max(0, part[:, ax].min() - n), part[:, ax].min()
            cells.append(dense(blo, bhi)); boxes.append((blo.tolist(), bhi.tolist()))
        cells = np.unique(np.concatenate(cells), axis=0)
        n_occ_in = sum(tuple(x) in oset for x in cells.tolist())
        cum |= {tuple(x) for x in cells.tolist()}
        cum_occ = sum(c in oset for c in cum) / len(occ)
        d = OUT / asset / f"r{k}"
        d.mkdir(parents=True, exist_ok=True)
        doc = {"mask": {"mode": "voxels", "voxels": cells.tolist(), "halo_margin_voxels": HALO, "grid_source": "slat_coords"},
               "meta": {"asset_id": f"x3/{asset}", "round": k, "kind": kind, "part": part_name, "prompt": prompt,
                        "n_cells": int(len(cells)), "mask_fingerprint": mask_fingerprint(cells), "selection_px": opt["px"], "selection_depth": opt.get("depth", 6),
                        "extend_boxes": boxes, "margin": MARGIN, "flat": opt.get("flat", False), "part_occupied_cells": int(len(part)),
                        "occupied_cells_in_mask": int(n_occ_in), "asset_occupied_cells": int(len(occ)),
                        "slat_sha256": gen["files_sha256"]["slat.safetensors"], "edit_view": view, "absent": opt.get("absent"), "reduced": opt.get("reduced")}}
        (d / "mask.json").write_text(json.dumps(doc))
        v, f = boundary_mesh(cells)
        vg = (VOXEL_TO_GLB.apply(v) - np.array(nz["center_glb_frame"])) * nz["scale"]
        trimesh.Trimesh(vg, f, process=True).export(d / f"mask_r{k}.glb")
        if not (d / "model.glb").exists():
            (d / "model.glb").symlink_to(ASSETS / asset / "gen" / "model.glb")
        # 편집 시점 v0 렌더 + 이번 회차 마스크(빨강)·이전 회차 누적(파랑) 실루엣
        im = Image.open(rd / f"render_{FORK_TO_BENCH[view]:04d}.png").convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); bg.alpha_composite(im)
        a = np.asarray(bg.convert("RGB")).astype(np.float32)
        prev = np.array(sorted(cum - {tuple(x) for x in cells.tolist()}), np.int64).reshape(-1, 3)
        for cs, col in ((prev, (60, 90, 255)), (cells, (255, 40, 40))):
            if len(cs):
                s = silhouette(cs, fr, FORK_VIEWS[view], res=a.shape[0], kind_frame="v0")
                a[s] = 0.55 * a[s] + 0.45 * np.array(col)
        Image.fromarray(a.astype(np.uint8)).save(d / "view.png")
        info = {"asset": asset, "round": k, "kind": kind, "part": part_name, "prompt": prompt, "edit_view": view,
                "n_cells": int(len(cells)), "occupied_in_mask": int(n_occ_in), "part_occupied_cells": int(len(part)),
                "cumulative_occupied_ratio": round(cum_occ, 4), "limit": LIMIT, "absent": opt.get("absent"), "reduced": opt.get("reduced"),
                "mask_fingerprint": doc["meta"]["mask_fingerprint"], "extend_boxes": boxes, "selection_px": opt["px"]}
        (d / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
        summary.append(info)
        print(asset, f"r{k}", part_name, "cells", len(cells), "occ_in", n_occ_in, "cum_occ", round(cum_occ, 3), opt.get("absent") or "", flush=True)
    if summary[-1]["cumulative_occupied_ratio"] > LIMIT:
        print(asset, f"누적 점유 {summary[-1]['cumulative_occupied_ratio']:.3f} > {LIMIT} — 원문 규칙: 부위를 줄인다", flush=True)
    (OUT / asset / "masks.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    cmd, names = sys.argv[1], sys.argv[2:] or list(EDITS)
    for n in names:
        {"inspect": inspect, "build": build}[cmd](n)
