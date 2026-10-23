"""W43 — DebugView 상세 패널 생성기.

⚠️ **경계**: DebugView 프런트는 맥북, 서버는 3090 담당이다 (CLAUDE.md).
   라우터는 다른 리포(`ai-ar-prototype/server/lab/debug.py`)에 있다.
   그래서 **서버 코드를 한 줄도 안 고치고**, PIPELINE §6 이 문서화한 확장점인
   `track.json` 의 `process` / `images` 배열에 패널 PNG 를 등록한다.
   (레이아웃·해시 표시를 UI 자체에 넣으려면 그건 맥북/3090 작업이다.)

만드는 것 (런마다)
    cond_triptych.png    2d_render · 2d_mask · 2d_edit 을 나란히 + **각 이미지의 생성 시각·sha**
                         ⇒ 옛 조건을 재사용했는지 한눈에 보인다
    mask_overlay.png     자산 위에 3D 마스크 투영 하이라이트 + 셀 수 · 온-오브젝트 비율
    mask_consistency.png 🔴 2d_mask 와 3D 마스크 투영의 diff (빨강=2d만 · 초록=공통 · 파랑=투영만)
    result_color.png     부모 · Baseline · Ours 를 **같은 방위각**으로 (색 있음)
    result_geom.png      🔴 같은 셋을 **색 벗긴 기하-only** 로  ← P3 이 이것 때문에 뒤집혔다
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import trimesh
from PIL import Image, ImageDraw

T = os.environ.get("TRACK_ROOT", "<HOME>/ai-ar-data/vh-track")
TOOLS = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def stamp(p):
    if not os.path.exists(p):
        return "MISSING"
    m = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))
    return f"{m}  sha {hashlib.sha256(open(p,'rb').read()).hexdigest()[:12]}"


def strip(src, dst):
    g = trimesh.load(src)
    g = g.to_geometry() if hasattr(g, "to_geometry") else g
    g.visual = trimesh.visual.ColorVisuals(
        mesh=g, face_colors=np.tile([185, 185, 185, 255], (len(g.faces), 1)))
    g.export(dst)
    return dst


def thumb(glb, out, az, size=340):
    subprocess.run([PY, f"{TOOLS}/make_thumb.py", glb, out, "--azimuth", str(az),
                    "--size", str(size)], capture_output=True)
    return out


def grid(items, out, cw, ch, pad=10, head=22):
    """items = [(label, PIL.Image|path)]"""
    n = len(items)
    s = Image.new("RGB", (cw * n + pad * (n - 1), ch + head), (255, 255, 255))
    d = ImageDraw.Draw(s)
    for i, (lbl, im) in enumerate(items):
        if isinstance(im, str):
            im = Image.open(im)
        im = im.convert("RGB")
        if im.size != (cw, ch):
            im = im.resize((cw, ch))
        d.text((i * (cw + pad) + 3, 5), lbl, fill=(0, 0, 0))
        s.paste(im, (i * (cw + pad), head))
    s.save(out)
    return out


def build(tag, cond_dir, mask_json, proj_mask, out_dir, parent_glb, base_glb, ours_glb, az, n_parent):
    os.makedirs(out_dir, exist_ok=True)
    R = {k: f"{cond_dir}/2d_{k}.png" for k in ("render", "mask", "edit")}

    # ── 1. 조건 3장 + 시각·해시 ──────────────────────────────────────────────
    grid([(f"2d_{k}   {stamp(R[k])}", R[k]) for k in ("render", "mask", "edit")],
         f"{out_dir}/cond_triptych.png", 340, 340)

    cells = np.asarray(json.load(open(mask_json)), int)
    proj = np.array(Image.open(proj_mask).convert("L")) > 127
    m2d = np.array(Image.open(R["mask"]).convert("L")) > 127
    rgb = np.array(Image.open(R["render"]).convert("RGB")).astype(float)

    # ── 2. 마스크 오버레이 + 통계 ───────────────────────────────────────────
    ov = (rgb * (1 - 0.5 * proj[..., None]) + np.array([60, 120, 255]) * 0.5 * proj[..., None])
    im = Image.fromarray(ov.astype(np.uint8)).resize((520, 520))
    d = ImageDraw.Draw(im)
    # 🔴 온-오브젝트는 **3D 로** 센다. 2D 밝기 임계는 t2s 에서 못 쓴다 —
    #    검은 오토바이가 검은 배경 위에 있어 실루엣이 안 잡히고 비율이 161.9% 로 튄다(실측).
    on = len(cells) / max(n_parent, 1)
    d.text((6, 6), f"3D mask cells {len(cells)} / parent {n_parent}", fill=(255, 255, 0))
    d.text((6, 20), f"on-object {on*100:.1f}%   projected px {int(proj.sum())}", fill=(255, 255, 0))
    im.save(f"{out_dir}/mask_overlay.png")

    # ── 3. 🔴 2d_mask 대 3D 투영 일치 검사 ──────────────────────────────────
    iou = (m2d & proj).sum() / max((m2d | proj).sum(), 1)
    diff = Image.fromarray((np.dstack([m2d & ~proj, m2d & proj, proj & ~m2d]).astype(np.uint8) * 255))
    d = ImageDraw.Draw(diff)
    d.text((6, 6), f"IoU {iou:.4f}   2d-only {int((m2d&~proj).sum())}   proj-only {int((proj&~m2d).sum())}",
           fill=(255, 255, 255))
    d.text((6, 20), "red = 2d_mask only   green = both   blue = 3D projection only", fill=(255, 255, 255))
    diff.resize((520, 520)).save(f"{out_dir}/mask_consistency.png")

    # ── 4. 결과 렌더 두 벌 — 같은 방위각 ────────────────────────────────────
    for kind in ("color", "geom"):
        items = []
        for lbl, g in [("parent", parent_glb), ("Baseline", base_glb), ("Ours", ours_glb)]:
            if not (g and os.path.exists(g)):
                continue
            src = g if kind == "color" else strip(g, f"{out_dir}/_geom_{lbl}.glb")
            items.append((f"{lbl} ({kind})", thumb(src, f"{out_dir}/_{kind}_{lbl}.png", az)))
        if items:
            grid(items, f"{out_dir}/result_{kind}.png", 340, 340)
    return iou, len(cells), on


if __name__ == "__main__":
    VIEW = {"t1s": "0002", "t2s": "0001"}
    AZ = {"t1s": 3.1416, "t2s": 3.1416}
    for tag in sys.argv[1:] or ["t1s", "t2s"]:
        asset = {"t1s": "t1s-snowman", "t2s": "t2s-motorcycle"}[tag]
        for wave, cond, mj, pm in [
            ("w42", f"{T}/{tag}-inpaint", f"{T}/{tag}-mask/w42_cells.json",
             f"{T}/{tag}-cond-w42/images/mask_{VIEW[tag]}.png"),
            ("w43", f"{T}/{tag}-inpaint-w42", f"{T}/{tag}-mask/w42_cells.json",
             f"{T}/{tag}-cond-w42/images/mask_{VIEW[tag]}.png")]:
            if not os.path.exists(f"{cond}/2d_edit.png"):
                print(f"skip {wave} {tag} — 조건 없음")
                continue
            out = f"{T}/{wave}-{tag}-panels"
            from safetensors import safe_open as _so
            with _so(f"{T}/{asset}/slat.safetensors", "pt") as _f:
                n_par = int(_f.get_tensor("coords").shape[0])
            iou, n, on = build(tag, cond, mj, pm, out, f"{T}/{asset}/model.glb",
                               f"{T}/{wave}-{tag}-baseline-r1/output.glb",
                               f"{T}/{wave}-{tag}-ours-r1/output.glb", AZ[tag], n_par)
            json.dump({
                "title": f"{wave} · {tag} · 점유셀 마스크",
                "verdict": f"2d_mask↔3D투영 IoU {iou:.4f} · 마스크 셀 {n} · 온오브젝트 {on*100:.1f}%",
                "process": {"render": f"{cond}/2d_render.png".replace(T, ".."),
                            "mask": f"{cond}/2d_mask.png".replace(T, ".."),
                            "edit": f"{cond}/2d_edit.png".replace(T, "..")},
                "images": [{"label": "조건 3장 (시각·해시)", "path": "cond_triptych.png"},
                           {"label": "3D 마스크 오버레이", "path": "mask_overlay.png"},
                           {"label": "🔴 2d_mask ↔ 3D 투영 일치", "path": "mask_consistency.png"},
                           {"label": "결과 (색)", "path": "result_color.png"},
                           {"label": "🔴 결과 (색 벗긴 기하)", "path": "result_geom.png"}],
            }, open(f"{out}/track.json", "w"), ensure_ascii=False, indent=2)
            print(f"{wave} {tag}: IoU {iou:.4f} · 셀 {n} · 온오브젝트 {on*100:.1f}%  → {out}")
