"""2번 R4 GPU 검증 (계측 구간 밖).

phase parents : 트랙 B 부모 `.cbin` — 판본 0 SLat(입력 SLat)을 R4 와 같은 mesh-only 경로로 디코딩·분할(부기 없음).
                트랙 A 는 생성 때 만든 `gen/chunks` 를 부모로 쓴다(여기서 만들지 않음).
phase decode  : R-3.2 같은 slat_tgt(E2a c15 r1)에서 `["mesh"]` 2회 · `["gaussian","mesh"]` 1회 → mesh 정점·면 비트 동일 여부.
                R-3.4 메시 밀도: 기존 GLB(c15 r1 output.glb, to_glb 단순화) 면 수 vs mesh 디코더 면 수.
phase render  : R-2 렌더 입력 교체 — 포크 Cycles 렌더러(`utils/render_rgb_and_mask.render_model`)로 같은 시점에서
                (a) 기존 텍스처 GLB(c15 r1) (b) R4 정점색 GLB(r4-on r1, glbbuild) 를 렌더 → 편집 시점 RGB·알파 차이.
결과: $RUNS_DIR/02-r4/{parents/<B#>/, checks/decode.json, checks/render/<ID>/{a,b}/, checks/render.json}
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
    {"parents": parents, "decode": decode, "render": render}[sys.argv[1]]()
