"""2번 R4 GPU 검증 (계측 구간 밖).

phase parents : 트랙 B 부모 `.cbin` — 판본 0 SLat(입력 SLat)을 R4 와 같은 mesh-only 경로로 디코딩·분할(부기 없음).
                트랙 A 는 생성 때 만든 `gen/chunks` 를 부모로 쓴다(여기서 만들지 않음).
phase decode  : R-3.2 같은 slat_tgt(E2a c15 r1)에서 `["mesh"]` 2회 · `["gaussian","mesh"]` 1회 → mesh 정점·면 비트 동일 여부.
                R-3.4 메시 밀도: 기존 GLB(c15 r1 output.glb, to_glb 단순화) 면 수 vs mesh 디코더 면 수.
phase render  : R-2 렌더 입력 교체 — 포크 Cycles 렌더러(`utils/render_rgb_and_mask.render_model`)로 같은 시점에서
                (a) 기존 텍스처 GLB(c15 r1) (b) R4 정점색 GLB(r4-on r1, glbbuild) 를 렌더 → 편집 시점 RGB·알파 차이.
phase decode2 : R-3.2 판정 기준(chief 2026-10-26): 위상(면) 동일 + 정점·속성 |Δ| ≤ 1e-6 이면 동일. 같은 프로세스 2회와
                `["mesh"]` 단독 vs `["gaussian","mesh"]` 의 max|Δ| 를 적는다.
phase render2 : R-2 보정(editcore/render.py:vertex_color_render_glb — 흰 비금속 재질 + COLOR_0 sRGB→선형) 뒤 다시 비교.
                b1 = r4-on r1 정점색 GLB 를 보정해 렌더. (a) 기존 텍스처 GLB 대비 RGB max·mean, b0(보정 전) 대비 2D 마스크(알파) 비트 동일,
                시트 docs/reports/img/02-R4-render.png (편집 시점·3/4 시점: 기존 텍스처 | 보정 정점색).
결과: $RUNS_DIR/02-r4/{parents/<B#>/, checks/decode.json, checks/decode2.json, checks/render/<ID>/{a,b,b1}/, checks/render{,2}.json}
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
import editcore.models as em  # noqa: E402

ASSETS = Path(os.environ["ASSETS_DIR"])
RUNS = Path(os.environ["RUNS_DIR"])
OUT = RUNS / "02-r4"
E2A_DAY = os.environ.get("E2A_DAY", "2026-10-25")
PY = os.environ.get("VH_PY", sys.executable)
REF = (REPO.parent / "Voxhammer-light" / "build" / "reference-v2").resolve()


def scenarios():
    ta = json.loads((REPO / "scenarios" / "track_a.json").read_text())["scenarios"]
    tb = json.loads((REPO / "scenarios" / "track_b.json").read_text())["scenarios"]
    out = []
    for sid, s in ta.items():
        a = s["asset"]
        out.append({"id": sid, "name": f"track_a-{sid}", "track": "A", "edit_view": s["edit_view"],
                    "slat": ASSETS / "track_a" / a / "gen" / "slat.safetensors", "parent": ASSETS / "track_a" / a / "gen" / "chunks",
                    "norm": json.loads((ASSETS / "track_a" / a / "gen" / "gen.json").read_text())["glb_normalize"]})
    for bid, s in tb.items():
        out.append({"id": bid, "name": f"track_b-{bid}", "track": "B", "edit_view": s["edit_view"],
                    "slat": ASSETS / "track_b" / bid / "slat.safetensors", "parent": OUT / "parents" / bid / "chunks", "norm": None})
    return out


def load_slat(path):
    import torch
    import trellis.modules.sparse as sp
    from safetensors import safe_open
    with safe_open(str(path), "pt") as f:
        return sp.SparseTensor(feats=f.get_tensor("feats").float().cuda(), coords=f.get_tensor("coords").int().cuda())


def parents():
    from editcore.export import decode_to_cbin
    pipe = em.load_pipeline()
    for s in scenarios():
        if s["track"] != "B" or (s["parent"].parent / "manifest.json").exists():
            continue
        r = decode_to_cbin(pipe, load_slat(s["slat"]), s["parent"].parent, glb_path=s["parent"].parent / "parent.glb")
        print("parent", s["id"], r["chunks"], "chunks", r["mesh"]["faces"], "faces", flush=True)


def decode():
    import torch
    import trimesh
    pipe = em.load_pipeline()
    res = {}
    for s in scenarios():
        d = RUNS / E2A_DAY / s["name"] / "e2a-c15" / "r1"
        slat = load_slat(d / "slat_tgt.safetensors")
        with torch.no_grad():
            m1 = pipe.decode_slat(slat, ["mesh"])["mesh"][0]
            m2 = pipe.decode_slat(slat, ["mesh"])["mesh"][0]
            both = pipe.decode_slat(slat, ["gaussian", "mesh"])["mesh"][0]
        eq = lambda a, b: bool(torch.equal(a.vertices, b.vertices) and torch.equal(a.faces, b.faces)
                               and (a.vertex_attrs is None or torch.equal(a.vertex_attrs, b.vertex_attrs)))
        glb = trimesh.load(d / "output.glb", force="mesh", process=False)
        res[s["id"]] = {"mesh_only_twice_bit_equal": eq(m1, m2), "mesh_only_vs_with_gaussian_bit_equal": eq(m1, both),
                        "mesh_vertices": int(m1.vertices.shape[0]), "mesh_faces": int(m1.faces.shape[0]),
                        "glb_faces_to_glb": int(len(glb.faces)), "glb_vertices": int(len(glb.vertices)),
                        "face_ratio_mesh_over_glb": float(m1.faces.shape[0] / max(len(glb.faces), 1))}
        print(s["id"], res[s["id"]], flush=True)
        del m1, m2, both
        torch.cuda.empty_cache()   # 계측 구간 밖(검증 스크립트)
    (OUT / "checks").mkdir(parents=True, exist_ok=True)
    (OUT / "checks" / "decode.json").write_text(json.dumps(res, indent=1))


def decode2():
    import torch
    pipe = em.load_pipeline()
    res = {}
    for s in scenarios():
        slat = load_slat(RUNS / E2A_DAY / s["name"] / "e2a-c15" / "r1" / "slat_tgt.safetensors")
        with torch.no_grad():
            m1 = pipe.decode_slat(slat, ["mesh"])["mesh"][0]
            m2 = pipe.decode_slat(slat, ["mesh"])["mesh"][0]
            both = pipe.decode_slat(slat, ["gaussian", "mesh"])["mesh"][0]

        def cmp(a, b):
            if a.faces.shape != b.faces.shape or a.vertices.shape != b.vertices.shape:
                return {"faces_equal": False, "same_shape": False, "equal": False}
            fv = float((a.vertices - b.vertices).abs().max())
            fa = float((a.vertex_attrs - b.vertex_attrs).abs().max()) if a.vertex_attrs is not None else 0.0
            fe = bool(torch.equal(a.faces, b.faces))
            return {"faces_equal": fe, "vert_max": fv, "attr_max": fa, "bit_equal": bool(fe and fv == 0 and fa == 0),
                    "equal": bool(fe and fv <= 1e-6 and fa <= 1e-6)}
        res[s["id"]] = {"mesh_twice": cmp(m1, m2), "mesh_vs_with_gaussian": cmp(m1, both), "faces": int(m1.faces.shape[0])}
        print(s["id"], res[s["id"]], flush=True)
        del m1, m2, both
        torch.cuda.empty_cache()   # 검증 스크립트(계측 구간 밖)
    (OUT / "checks" / "decode2.json").write_text(json.dumps(res, indent=1))


def render2():
    import trimesh
    from PIL import Image, ImageDraw
    from editcore.render import vertex_color_render_glb
    res, rows = {}, []
    for s in scenarios():
        d = OUT / "checks" / "render" / s["id"]
        src = OUT / s["name"] / "r4-on" / "r1" / "output.glb"
        (d / "b1").mkdir(parents=True, exist_ok=True)
        g = src
        if s["norm"]:   # 트랙 A: 원본과 같은 정규화(F-5·E2a 시트와 같게)
            sc = trimesh.load(src)
            sc.apply_translation([-x for x in s["norm"]["center_glb_frame"]])
            sc.apply_scale(s["norm"]["scale"])
            g = d / "b1" / "normalized_raw.glb"
            sc.export(g)
        fixed = d / "b1" / "render_input.glb"
        fixed.write_bytes(vertex_color_render_glb(Path(g).read_bytes()))
        if not (d / "b1" / "render_0004.png").exists():
            subprocess.run([PY, "-c", RENDER, str(REF), str(fixed), str(d / "b1")], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        v34 = 3 if s["edit_view"] == 1 else 1
        L = lambda t, v: np.asarray(Image.open(d / t / f"render_{v:04d}.png").convert("RGBA")).astype(int)
        row = {}
        for tag, v in (("edit", s["edit_view"]), ("34", v34)):
            a, b0, b1 = L("a", v), L("b", v), L("b1", v)
            both = (a[..., 3] > 0) & (b1[..., 3] > 0)
            diff = np.abs(a[..., :3] - b1[..., :3])[both]
            row[tag] = {"view": v, "rgb_max": int(diff.max()), "rgb_mean": float(diff.mean()),
                        "mask_bit_equal_b0_b1": bool(np.array_equal(b0[..., 3], b1[..., 3])),
                        "alpha_iou_a_b1": float(both.sum() / max(((a[..., 3] > 0) | (b1[..., 3] > 0)).sum(), 1)),
                        "rgb_mean_before_fix": float(np.abs(a[..., :3] - b0[..., :3])[(a[..., 3] > 0) & (b0[..., 3] > 0)].mean())}
        res[s["id"]] = row
        print(s["id"], row, flush=True)
        T = 256
        tiles = []
        for tag, v in (("edit", s["edit_view"]), ("3/4", v34)):
            for t, lab in (("a", "textured GLB (to_glb)"), ("b1", "R4 vertex-color GLB (fixed)")):
                im = Image.open(d / t / f"render_{v:04d}.png").convert("RGBA")
                tile = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im).convert("RGB").resize((T, T), Image.LANCZOS)
                ImageDraw.Draw(tile).text((4, 4), f"{tag} v{v} {lab}", fill=(200, 0, 0))
                tiles.append(tile)
        r = Image.new("RGB", (4 * T, T + 18), "white")
        ImageDraw.Draw(r).text((4, 3), f"{s['id']}  rgb mean diff edit {row['edit']['rgb_mean']:.1f} (before fix {row['edit']['rgb_mean_before_fix']:.1f})", fill=(0, 0, 0))
        for i, t in enumerate(tiles):
            r.paste(t, (T * i, 18))
        rows.append(r)
    head = Image.new("RGB", (4 * 256, 20), "white")
    ImageDraw.Draw(head).text((4, 4), "Renderer: fork utils/render_rgb_and_mask.render_model (Blender Cycles). r1. R4 input fixed by editcore/render.py.", fill=(0, 0, 160))
    sheet = Image.new("RGB", (4 * 256, 20 + sum(x.height for x in rows)), "white")
    sheet.paste(head, (0, 0))
    y = 20
    for x in rows:
        sheet.paste(x, (0, y)); y += x.height
    dst = REPO / "docs" / "reports" / "img" / "02-R4-render.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst, optimize=True)
    (OUT / "checks" / "render2.json").write_text(json.dumps(res, indent=1))


RENDER = r"""
import os, sys
ref, glb, out = sys.argv[1:4]
sys.path.insert(0, ref); os.chdir(ref)
import utils.render_rgb_and_mask as rr
rr.render_model(glb, out)
"""


def render():
    import trimesh
    from PIL import Image
    res = {}
    for s in scenarios():
        d = OUT / "checks" / "render" / s["id"]
        srcs = {"a": RUNS / E2A_DAY / s["name"] / "e2a-c15" / "r1" / "output.glb",
                "b": OUT / s["name"] / "r4-on" / "r1" / "output.glb"}
        for tag, glb in srcs.items():
            if not glb.is_file():
                continue
            (d / tag).mkdir(parents=True, exist_ok=True)
            if s["norm"]:   # 트랙 A: 원본(정규화본)과 같은 좌표로(F-5·E2a 시트와 같은 정규화)
                sc = trimesh.load(glb)
                sc.apply_translation([-x for x in s["norm"]["center_glb_frame"]])
                sc.apply_scale(s["norm"]["scale"])
                glb = d / tag / "normalized.glb"
                sc.export(glb)
            if not (d / tag / "render_0004.png").exists():
                subprocess.run([PY, "-c", RENDER, str(REF), str(glb), str(d / tag)], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not all((d / t / f"render_{s['edit_view']:04d}.png").exists() for t in "ab"):
            continue
        a, b = [np.asarray(Image.open(d / t / f"render_{s['edit_view']:04d}.png").convert("RGBA")).astype(int) for t in "ab"]
        ma, mb = a[..., 3] > 0, b[..., 3] > 0
        both = ma & mb
        diff = np.abs(a[..., :3] - b[..., :3])[both]
        gray = lambda x: float(np.mean(np.std(x[..., :3][x[..., 3] > 0], axis=1)))   # 화소별 RGB 채널 표준편차 평균 — 회색이면 0 에 가깝다
        res[s["id"]] = {"view": s["edit_view"], "rgb_max": int(diff.max()) if diff.size else None,
                        "rgb_mean": float(diff.mean()) if diff.size else None,
                        "alpha_iou": float(both.sum() / max((ma | mb).sum(), 1)), "alpha_diff_px": int((ma ^ mb).sum()),
                        "chroma_a": gray(a), "chroma_b": gray(b)}
        print(s["id"], res[s["id"]], flush=True)
    (OUT / "checks" / "render.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"parents": parents, "decode": decode, "render": render, "decode2": decode2, "render2": render2}[sys.argv[1]]()
