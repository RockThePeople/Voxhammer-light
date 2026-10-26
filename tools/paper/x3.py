"""X3 연속 5회 편집 (ops/tasks/X-paper.md X3, chief 보정 6). 자산 3(jeep·cottage·adventurer, seed 0 이미지 → A_origin) × 5회 × P0·P1·P2.

각 경로는 자기 직전 결과에서 다음 회차를 편집한다(A_origin → A_1 → … → A_5). 마스크는 승인본($RUNS_DIR/x3/<자산>/r<k>/, 사용자
2026-10-26 14:52) — v0 64³ 셀, 세 경로에 같다. 회차 폴더 $RUNS_DIR/paper/x3/<경로>/<자산>/r<k>/ 에 2d_render·2d_mask·2d_edit·
output.glb(·slat_tgt·cbin)·round.json. r0 = 출발점(P0: 생성 텍스처 GLB, P1·P2: A_origin SLat 의 R4 .cbin → GLB).

P0 (원저자): 회차마다 직전 GLB → 포크 조건 렌더(편집 시점) → A5000 bf16 FLUX(`reference/run_inpaint.py` 그대로) →
    `reference/run_reference.py --mode original --seed 0`(150시점 재등록 + 25 스텝 + to_glb, 포크 무수정).
    포크는 입력 GLB 를 bbox 로 정규화(normalize_scene)해 인코딩하고 마스크 GLB 는 정규화하지 않고 복셀화한다(delete_region_voxel).
    그래서 회차마다 직전 GLB 를 미리 정규화한 in.glb(포크 정규화가 항등이 되게)와, 같은 누적 변환 C_k(v0 → 이번 입력 좌표)로 옮긴
    마스크 GLB 를 넘긴다. C_k 는 round.json 에 남겨 채점 때 v0 로 되돌린다. 조건 렌더도 같은 in.glb·마스크로 한다.
P1 (우리): editcore CLI `VHL_R4=1`, seed 0, 15/15. 입력 = 직전 slat_tgt(1회차 A_origin SLat), 부모 .cbin = 직전 cbin(1회차 gen/chunks).
    조건 렌더 입력 = 직전 정점색 GLB 를 raw → v0 로 옮기고 (iv) 보정. 인페인팅 3090 INT4(`/inpaint?wait=1`, seed 0).
P2 (절제): P1 과 같고, 편집 전에 직전 결과를 150시점 재등록해 입력 SLat 을 만든다(포크 run_3d_rendering → 특징 → 인코딩).
    재등록 렌더 입력 = 직전 정점색 GLB(raw 프레임) + (iv) 보정 + 격자 꼭짓점 표지 삼각형 2개(glTF (±0.5)³) — 포크 normalize_scene 이
    항등(bbox 정확히 [−0.5, 0.5]³)이 되어 재등록 SLat 이 raw 64³ 격자에 그대로 놓인다(마스크 셀·R4 부모 승계를 P1 과 같게).
    표지가 만드는 격자 꼭짓점 셀은 특징 추출 전에 voxels.ply 에서 뺀다. 첫 실행은 150시점 중 3장을 시트로 남긴다.

    python tools/paper/x3.py run p0|p1|p2 [자산 …]
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party" / "ai-ar-v3"), str(ROOT / "third_party" / "ai-ar-v3" / "contract" / "python")]
ASSETS = Path(os.environ["ASSETS_DIR"]) / "x3" / "track_a"
RUNS = Path(os.environ["RUNS_DIR"])
MASKS = RUNS / "x3"
OUT = RUNS / "paper" / "x3"
PY = os.environ.get("VH_PY", str(Path.home() / "ai-ar-v2/VoxHammer/.venv/bin/python"))
REF = (ROOT.parent / "Voxhammer-light" / "build" / "reference-v2").resolve()
INT4 = os.environ.get("X3_INT4_URL", "http://117.16.244.34:8082")
ASSET_NAMES = ("jeep", "cottage", "adventurer")
ROUNDS = 5


def sh(cmd, log, env=None):
    with open(log, "w") as f:
        r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})}, cwd=ROOT)
    if r.returncode:
        raise SystemExit(f"실패({r.returncode}): {' '.join(map(str, cmd))} — {log}")


def gen(asset):
    g = ASSETS / asset / "gen"
    return g, json.loads((g / "gen.json").read_text())["glb_normalize"]


def round_info(asset, k):
    d = MASKS / asset / f"r{k}"
    return d, json.loads((d / "info.json").read_text())


def affine(scale, center):
    """p → (p − center) · scale 의 4×4."""
    M = np.eye(4)
    M[:3, :3] *= scale
    M[:3, 3] = -np.asarray(center, float) * scale
    return M


def glb_bbox(path):
    import trimesh
    sc = trimesh.load(str(path))
    b = sc.bounds
    return b[0], b[1]


def transform_glb(src, M, dst, vertex_color=False, markers=False):
    """장면 변환(재질·정점색 유지) → 필요하면 격자 꼭짓점 표지 → 필요하면 (iv) 보정."""
    import trimesh
    sc = trimesh.load(str(src))
    if not isinstance(sc, trimesh.Scene):
        sc = trimesh.Scene(sc)
    sc.apply_transform(M)
    if markers:   # glTF (±m)³ 꼭짓점에 아주 작은 삼각형 — 포크 normalize_scene 을 중심 0·배율 1/(2m) 로(메시가 ±0.5 를 살짝 넘으면 m > 0.5)
        e = 1e-4
        m = max(0.5, float(np.abs(sc.bounds).max()))
        for s in (-m, m):
            c = np.full(3, s)
            v = np.array([c, c - np.sign(s) * np.array([e, 0, 0]), c - np.sign(s) * np.array([0, e, 0])])
            sc.add_geometry(trimesh.Trimesh(v, [[0, 1, 2]], process=False), node_name=f"marker{int(s > 0)}")
    tmp = Path(dst).with_suffix(".tmp.glb")
    sc.export(str(tmp))
    if vertex_color:
        from editcore.render import vertex_color_render_glb
        Path(dst).write_bytes(vertex_color_render_glb(tmp.read_bytes()))
        tmp.unlink()
    else:
        tmp.replace(dst)
    return Path(dst)


def condition(d, model_glb, mask_glb, view):
    """포크 조건 렌더(5시점) → 편집 시점 render/mask 를 2d_render·2d_mask 로."""
    cr = d / "cond_render"
    sh([PY, ROOT / "reference" / "run_condition_render.py", "--model", model_glb, "--mask", mask_glb, "--out", cr,
        "--reference", REF], d / "cond_render.log")
    shutil.copy(cr / "images" / f"render_{view:04d}.png", d / "2d_render.png")
    shutil.copy(cr / "images" / f"mask_{view:04d}.png", d / "2d_mask.png")


def inpaint_bf16(d, prompt):
    ip = d / "inpaint"
    sh([PY, ROOT / "reference" / "run_inpaint.py", "--image", d / "2d_render.png", "--mask", d / "2d_mask.png",
        "--prompt", prompt, "--out", ip, "--reference", REF], d / "inpaint.log")
    shutil.copy(ip / "2d_edit.png", d / "2d_edit.png")
    return {"device": "a5000", "model": "FLUX.1-Fill-dev bf16 + cpu offload", "timing": json.loads((ip / "timing.json").read_text()).get("sha256")}


def inpaint_int4(d, prompt):
    import urllib.request
    import uuid
    bnd = uuid.uuid4().hex
    parts = []
    for name, val, fn in (("image", (d / "2d_render.png").read_bytes(), "2d_render.png"), ("mask", (d / "2d_mask.png").read_bytes(), "2d_mask.png"),
                          ("prompt", prompt.encode(), None), ("seed", b"0", None)):
        h = f'--{bnd}\r\nContent-Disposition: form-data; name="{name}"' + (f'; filename="{fn}"\r\nContent-Type: image/png' if fn else "") + "\r\n\r\n"
        parts += [h.encode(), val, b"\r\n"]
    body = b"".join(parts) + f"--{bnd}--\r\n".encode()
    req = urllib.request.Request(f"{INT4}/inpaint?wait=1", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={bnd}"})
    r = json.loads(urllib.request.urlopen(req, timeout=1200).read())
    if r.get("state") != "done":
        raise SystemExit(f"INT4 실패: {r}")
    (d / "2d_edit.png").write_bytes(urllib.request.urlopen(f"{INT4}/jobs/{r['job_id']}/2d_edit.png", timeout=120).read())
    (d / "inpaint_int4.json").write_text(json.dumps(r, indent=1))
    return {"device": "rtx3090", "model": "INT4 worker", "job_id": r["job_id"], "sha256": r.get("result", {}).get("sha256")}


def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ── P0 ─────────────────────────────────────────────────────────────────────────
def run_p0(asset):
    g, nz = gen(asset)
    base = OUT / "p0" / asset
    r0 = base / "r0"
    r0.mkdir(parents=True, exist_ok=True)
    if not (r0 / "output.glb").exists():
        shutil.copy(g / "model.glb", r0 / "output.glb")
        (r0 / "round.json").write_text(json.dumps({"start": "생성 텍스처 GLB(gen/model.glb, v0 좌표)", "C": np.eye(4).tolist()}, indent=1))
    C = np.eye(4)                                  # v0 → 직전 출력 좌표
    for k in range(1, ROUNDS + 1):
        d = base / f"r{k}"
        md, info = round_info(asset, k)
        prev = base / f"r{k - 1}"
        if (d / "round.json").exists():
            C = np.asarray(json.loads((d / "round.json").read_text())["C"])
            continue
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        t0 = time.time()
        lo, hi = glb_bbox(prev / "output.glb")
        T = affine(1.0 / float((hi - lo).max()), (lo + hi) / 2)   # 포크 normalize_scene 과 같은 변환을 미리 건다
        C = T @ C
        transform_glb(prev / "output.glb", T, d / "in.glb")
        transform_glb(md / f"mask_r{k}.glb", C, d / "mask.glb")
        condition(d, d / "in.glb", d / "mask.glb", info["edit_view"])
        inp = inpaint_bf16(d, info["prompt"])
        img = d / "image"
        img.mkdir()
        for n in ("2d_render.png", "2d_mask.png", "2d_edit.png"):
            shutil.copy(d / n, img / n)
        sh([PY, ROOT / "reference" / "run_reference.py", "--mode", "original", "--model", d / "in.glb", "--mask_glb", d / "mask.glb",
            "--mask_json", md / "mask.json", "--image_dir", img, "--asset", f"x3-{asset}-p0", "--run", f"r{k}", "--seed", 0,
            "--runs_dir", d / "ref", "--reference", REF], d / "reference.log")
        work = next((d / "ref").glob(f"*/x3-{asset}-p0/*/r{k}.work"))
        shutil.copy(work / "output.glb", d / "output.glb")
        if (work / "slat_tgt.safetensors").exists():
            shutil.copy(work / "slat_tgt.safetensors", d / "slat_tgt.safetensors")
        shutil.rmtree(work / "render", ignore_errors=True)   # 150시점 렌더(크다)
        rec = {"path": "P0", "asset": asset, "round": k, "prompt": info["prompt"], "edit_view": info["edit_view"],
               "mask_fingerprint": info["mask_fingerprint"], "C": C.tolist(), "T": T.tolist(), "inpaint": inp,
               "files_sha256": {n: sha(d / n) for n in ("2d_render.png", "2d_mask.png", "2d_edit.png", "output.glb")},
               "wall_s": time.time() - t0, "note": "C = v0 → 이번 회차 입력·출력 좌표(포크 정규화를 미리 건 in.glb 좌표)"}
        (d / "round.json").write_text(json.dumps(rec, indent=1))
        print(f"x3 p0 {asset} r{k} done {rec['wall_s']:.0f}s", flush=True)


# ── P1·P2 ──────────────────────────────────────────────────────────────────────
def start_r0(asset, base):
    """출발점 = A_origin SLat 의 R4 .cbin(gen/chunks) → glbbuild 정점색 GLB."""
    g, _ = gen(asset)
    r0 = base / "r0"
    if (r0 / "output.glb").exists():
        return r0
    r0.mkdir(parents=True, exist_ok=True)
    from server.glbbuild import build_glb
    build_glb(g / "chunks", r0 / "output.glb")
    (r0 / "cbin").mkdir(exist_ok=True)
    if not (r0 / "cbin" / "chunks").exists():
        (r0 / "cbin" / "chunks").symlink_to(g / "chunks")
    if not (r0 / "slat_tgt.safetensors").exists():
        (r0 / "slat_tgt.safetensors").symlink_to(g / "slat.safetensors")
    (r0 / "round.json").write_text(json.dumps({"start": "A_origin SLat 의 R4 .cbin(gen/chunks) → glbbuild GLB, raw 프레임"}, indent=1))
    return r0


def rereg(glb_raw, out_slat, rdir, sheet=None):
    """P2 재등록(별도 프로세스, 포크 함수): 정점색 raw GLB → (iv)·꼭짓점 표지 → 150시점 → 복셀(꼭짓점 셀 제외) → 특징 → 인코딩."""
    sh([PY, Path(__file__).resolve(), "_rereg", glb_raw, out_slat, rdir, sheet or ""], Path(rdir).parent / "rereg.log")


def _rereg(glb_raw, out_slat, rdir, sheet):
    rdir = Path(rdir)
    rdir.mkdir(parents=True, exist_ok=True)
    inp = transform_glb(glb_raw, np.eye(4), rdir.parent / "rereg_in.glb", vertex_color=True, markers=True)
    sys.path[:0] = [str(REF)]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cwd = os.getcwd()
    os.chdir(REF)
    import torch
    import utils3d
    import inference as inf
    import voxhammer.edit_pipeline as ep
    from voxhammer.extract_feature import voxelize_mesh
    from editcore.export import save_slat
    t = {}
    t0 = time.perf_counter()
    inf.run_3d_rendering(str(inp), str(rdir))
    t["render_150views"] = time.perf_counter() - t0
    voxelize_mesh(str(rdir))
    pos = np.asarray(utils3d.io.read_ply(str(rdir / "voxels.ply"))[0])
    cells = np.floor((pos + 0.5) * 64).astype(np.int64)
    corner = np.all((cells == 0) | (cells == 63), axis=1)
    utils3d.io.write_ply(str(rdir / "voxels.ply"), pos[~corner])
    t0 = time.perf_counter()
    inf.run_feature_extraction(str(rdir))
    t["feature_extraction"] = time.perf_counter() - t0
    pipe = inf.TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipe.cuda()
    with torch.no_grad():
        slat = ep.feats_to_slat(pipe, str(rdir / "features.npz"))
        save_slat(slat, out_slat, pipe.slat_normalization, {"source": "x3-p2 re-registration (raw frame, corner markers)", "input": str(glb_raw)})
    if sheet:
        from PIL import Image
        ims = [Image.open(rdir / f"{i:03d}.png").convert("RGBA") for i in (0, 50, 100)]
        S = Image.new("RGBA", (512 * 3, 512), (255, 255, 255, 255))
        for i, im in enumerate(ims):
            S.alpha_composite(im.resize((512, 512)), (512 * i, 0))
        S.convert("RGB").save(sheet)
    for p in rdir.glob("*.png"):
        p.unlink()
    (rdir / "rereg.json").write_text(json.dumps({"corner_cells_removed": int(corner.sum()), "cells": int((~corner).sum()), "seconds": t}))
    os.chdir(cwd)


def run_p12(asset, path):
    g, nz = gen(asset)
    base = OUT / path / asset
    start_r0(asset, base)
    to_v0 = affine(nz["scale"], nz["center_glb_frame"])   # raw → v0 저장 좌표
    for k in range(1, ROUNDS + 1):
        d = base / f"r{k}"
        md, info = round_info(asset, k)
        prev = base / f"r{k - 1}"
        if (d / "round.json").exists():
            continue
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        t0 = time.time()
        slat_in = prev / "slat_tgt.safetensors"
        rr = None
        if path == "p2":
            first = not (OUT / "p2" / "rereg_sheet.png").exists()
            rereg(prev / "output.glb", d / "slat_in.safetensors", d / "rereg", OUT / "p2" / "rereg_sheet.png" if first else None)
            slat_in = d / "slat_in.safetensors"
            rr = json.loads((d / "rereg" / "rereg.json").read_text())
            shutil.rmtree(d / "rereg", ignore_errors=True)
        transform_glb(prev / "output.glb", to_v0, d / "cond_in.glb", vertex_color=True)
        condition(d, d / "cond_in.glb", md / f"mask_r{k}.glb", info["edit_view"])
        inp = inpaint_int4(d, info["prompt"])
        img = d / "image"
        img.mkdir()
        for n in ("2d_render.png", "2d_mask.png", "2d_edit.png"):
            shutil.copy(d / n, img / n)
        pc = prev / "cbin" / "chunks"
        sh([PY, "-m", "editcore.cli", "--slat", slat_in, "--mask_json", md / "mask.json", "--image_dir", img,
            "--out", d / "output.glb", "--slat_out", d / "slat_tgt.safetensors", "--seed", 0,
            "--parent_cbin", pc, "--cbin_out", d / "cbin"], d / "edit.log", env={"VHL_R4": "1", "HF_HUB_OFFLINE": "1"})
        r4 = json.loads((d / "cbin" / "r4.json").read_text())
        rec = {"path": path.upper(), "asset": asset, "round": k, "prompt": info["prompt"], "edit_view": info["edit_view"],
               "mask_fingerprint": info["mask_fingerprint"], "slat_in": str(slat_in), "parent_cbin": str(pc), "inpaint": inp,
               "rereg": rr, "r4": {"sent": len(r4.get("sent", [])), "reused": len(r4.get("reused", [])), "seam": r4.get("seam", {}).get("snapped")},
               "files_sha256": {n: sha(d / n) for n in ("2d_render.png", "2d_mask.png", "2d_edit.png", "output.glb")},
               "wall_s": time.time() - t0}
        (d / "round.json").write_text(json.dumps(rec, indent=1))
        print(f"x3 {path} {asset} r{k} done {rec['wall_s']:.0f}s sent {rec['r4']['sent']} reused {rec['r4']['reused']}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "_rereg":
        _rereg(*sys.argv[2:6])
    elif sys.argv[1] == "run":
        path, names = sys.argv[2], sys.argv[3:] or list(ASSET_NAMES)
        for n in names:
            run_p0(n) if path == "p0" else run_p12(n, path)
