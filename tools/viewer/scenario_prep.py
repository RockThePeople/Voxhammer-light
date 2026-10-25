#!/usr/bin/env python3
"""T1 시나리오 뷰어 미리 계산 (CPU 전용, GPU 미사용, 입력 파일은 읽기만).

시나리오 16건(트랙 A 6 + 트랙 B 8 + moto-b 2)마다 `RUNS_DIR/.viewer_cache/scenario/` 에:
- `index.json`                     : 시나리오 목록(썸네일·머리말)
- `<ID>/meta.json`                 : 머리말·조건 이미지·GLB 목록(설정 × 회차)·좌표 변환·마스크·halo·지표 표
- `<ID>/<설정>/<회차>.json`         : 결과 slat_tgt vs 입력 SLat 점유 차이(마스크∪halo 안 신규·R_keep 신규·소실),
                                      R_keep 신규의 26-이웃 성분(셀 수·마스크∪halo 까지 거리·9a 대상)
- `alignment.json`                 : V-2 복셀–메시 정렬(원본 SLat 셀 중심 → 원본 메시 표면 거리, 1셀 이내 비율)

좌표: 셀 (i,j,k) → TRELLIS 복셀 프레임 ((i,j,k)+0.5)/64 − 0.5 → glTF(Y-up) (x, z, −y) → 시나리오 정규화 (p − c) × s.
화면 변환은 브라우저 `common.js:voxelToWorld` 한 곳에서만 하고, 여기서는 같은 식을 정렬 검사에만 쓴다(`cells_to_world`).
정규화: 트랙 A = gen.json `glb_normalize`, moto-b = prepare.json `model.normalize`, 트랙 B = 항등(Bench 프레임, 00s2 S-8).

실행(VoxHammer venv: numpy·scipy·safetensors·trimesh):
    python tools/viewer/scenario_prep.py [--only A-S1,B7] [--skip-align]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "third_party" / "ai-ar-v3" / "contract" / "python"))


def load_dotenv(path: Path) -> dict:
    """`server.py:load_dotenv` 와 같다(ai-ar-v3 의 `server` 패키지와 이름이 겹쳐 import 하지 않는다)."""
    env = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


ENV = {**load_dotenv(REPO / ".env"), **load_dotenv(REPO.parent / "Voxhammer-light" / ".env"), **os.environ}
ASSETS = Path(ENV["ASSETS_DIR"])
RUNS = Path(ENV["RUNS_DIR"])
CACHE = RUNS / ".viewer_cache" / "scenario"
E2A_DAY = ENV.get("E2A_DAY", "2026-10-25")
FORK_VIEWS = 5
ALIGN_SAMPLES = 2_000_000
E12 = {"A-C2": "조건 이미지 약함 (E12)", "A-S1": "마스크 밖 생성 — 알려진 결함 (E12)"}
# 설정 이름 → (설명, 결과 폴더(회차 → 폴더) 찾는 함수). 이후 항목 결과는 여기에 한 줄씩 더한다.
CONFIGS = {
    "base": "25/25 (0-1 이식본)",
    "c15": "15/15 (E2a 채택, E20 이후 기준)",
    "c12": "12/12",
    "s1-12": "12/25 (원인 분리)",
    "s2-12": "25/12 (원인 분리)",
}
E19_TOL = {"psnr": -0.5, "ssim": -0.003, "lpips": 0.005, "dino_i": -0.015, "clip_t": -0.015, "chamfer": 0.002}
E19_AVG = {"psnr": -0.2, "dino_i": -0.005, "clip_t": -0.005}
HIGHER_BETTER = {"psnr": True, "ssim": True, "dino_i": True, "clip_t": True, "lpips": False, "chamfer": False}


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.is_file() else ""


def scenarios() -> list[dict]:
    ta = json.loads((REPO / "scenarios" / "track_a.json").read_text())["scenarios"]
    tb = json.loads((REPO / "scenarios" / "track_b.json").read_text())["scenarios"]
    out = []
    for sid, s in ta.items():
        a = s["asset"]
        g = json.loads((ASSETS / "track_a" / a / "gen" / "gen.json").read_text())["glb_normalize"]
        out.append({"id": sid, "track": "A", "name": f"track_a-{sid}", "dataset": "트랙 A (TRELLIS 생성)", "object": a,
                    "prompt": s["inpaint_prompt"], "prompt_kind": "인페인팅 프롬프트",
                    "edit_view": s["edit_view"], "view_34": s["view_34"], "mask_meta": s["mask"],
                    "slat": ASSETS / "track_a" / a / "gen" / "slat.safetensors",
                    "mask": ASSETS / "track_a" / a / "masks" / f"{sid}.mask.json",
                    "cond": ASSETS / "track_a" / a / "cond" / sid, "src_glb": ASSETS / "track_a" / a / "gen" / "model.glb",
                    "xf": {"center": g["center_glb_frame"], "scale": g["scale"], "src": "gen.json glb_normalize"}})
    for bid, s in tb.items():
        rel = Path(s["dataset"]) / s["object"]
        out.append({"id": bid, "track": "B", "name": f"track_b-{bid}", "dataset": s["dataset"], "object": s["object"],
                    "prompt": s["prompt"], "prompt_kind": "데이터셋 프롬프트", "source_prompt": s.get("source_prompt"),
                    "edit_view": s["edit_view"], "view_34": 3 if s["edit_view"] == 1 else 1, "mask_meta": s["mask"],
                    "slat": ASSETS / "track_b" / bid / "slat.safetensors", "mask": ASSETS / "track_b" / bid / "mask.json",
                    "cond": ASSETS / "track_b" / bid / "cond",
                    "src_glb": ASSETS / "Edit3D-Bench" / "data" / rel / "source_model" / "model.glb",
                    "xf": {"center": [0.0, 0.0, 0.0], "scale": 1.0, "src": "항등 (Bench 프레임, 00s2 S-8)"},
                    "bench": {"dataset": s["dataset"], "object": s["object"], "prompt_id": s["prompt_id"]}})
    prep = json.loads((ASSETS / "moto-b" / "prepare.json").read_text())["model"]["normalize"]
    for m, view in (("rider-head", None), ("rear-wheel", None)):
        mj = REPO / "third_party" / "ai-ar-v3" / "handoff" / "lasso" / f"moto-b.{m}.mask.json"
        cond = RUNS / "2026-10-22" / "moto-b" / "inpaint" / f"{m}-run1"
        cj = cond / "timing.json"
        meta = json.loads(mj.read_text())
        out.append({"id": f"moto-b-{m}", "track": "moto-b", "name": f"moto-b-{m}", "dataset": "ai-ar-v3 moto-b (0번 기준선)",
                    "object": "moto-b", "prompt": _moto_prompt(cond), "prompt_kind": "인페인팅 프롬프트",
                    "edit_view": (ev := _moto_view(cond)), "view_34": (3 if ev == 1 else 1) if ev is not None else None,
                    "mask_meta": {"n_cells": len(meta["mask"]["voxels"]), "fingerprint": meta.get("meta", {}).get("mask_fingerprint", "")},
                    "slat": Path(ENV.get("MOTO_B_SLAT", str(Path.home() / "ai-ar-data/assets/v3-moto-b/v1/slat.safetensors"))),
                    "mask": mj, "cond": cond, "src_glb": ASSETS / "moto-b" / "model.glb",
                    "xf": {"center": prep["center_glb"], "scale": prep["scale"], "src": "moto-b prepare.json model.normalize"},
                    "cond_meta": cj})
    return out


def _moto_prompt(cond: Path) -> str:
    for name in ("timing.json", "cond.json", "prompt.txt"):
        p = cond / name
        if p.is_file() and p.suffix == ".json":
            d = json.loads(p.read_text())
            for k in ("prompt", "inpaint_prompt"):
                if isinstance(d.get(k), str):
                    return d[k]
    return "(moto-b 0번 인페인팅 조건)"


def _moto_view(cond: Path):
    """moto-b 는 편집 시점 번호가 메타에 없다 → 인페인팅에 쓴 2d_render 와 5시점 조건 렌더의 알파 일치로 찾는다."""
    from PIL import Image
    m = cond.name.rsplit("-run", 1)[0]
    multi = RUNS / "2026-10-22" / "moto-b" / "cond" / f"{m}-run1" / "images"
    a = np.asarray(Image.open(cond / "2d_render.png").convert("RGBA"))[..., 3] > 0
    best = None
    for v in range(FORK_VIEWS):
        p = multi / f"render_{v:04d}.png"
        if p.is_file():
            b = np.asarray(Image.open(p).convert("RGBA"))[..., 3] > 0
            iou = (a & b).sum() / max((a | b).sum(), 1)
            if best is None or iou > best[0]:
                best = (iou, v)
    return best[1] if best and best[0] > 0.99 else None


def run_dirs(s: dict, cfg: str) -> dict:
    """회차 → 결과 폴더(output.glb·slat_tgt.safetensors). 없는 회차는 빠진다."""
    out = {}
    for r in ("r1", "r2", "r3"):
        d = RUNS / "01-port" / s["name"] / f"port-{r}" if cfg == "base" else RUNS / E2A_DAY / s["name"] / f"e2a-{cfg}" / r
        if (d / "output.glb").is_file() and (d / "slat_tgt.safetensors").is_file():
            out[r] = d
    return out


def coords(p: Path) -> np.ndarray:
    from safetensors import safe_open
    with safe_open(str(p), "np") as f:
        return f.get_tensor("coords")[:, 1:].astype(np.int64)


def vol(c: np.ndarray) -> np.ndarray:
    v = np.zeros((64, 64, 64), bool)
    v[tuple(c.T)] = True
    return v


def flat(v: np.ndarray) -> list[int]:
    return np.argwhere(v).astype(int).ravel().tolist()


def cells_to_world(cells: np.ndarray, xf: dict) -> np.ndarray:
    """`common.js:voxelToWorld` 와 같은 식(정렬 검사 전용)."""
    p = (cells + 0.5) / 64.0 - 0.5
    g = np.stack([p[:, 0], p[:, 2], -p[:, 1]], 1)
    return (g - np.asarray(xf["center"])) * xf["scale"]


def diff(s: dict, src: np.ndarray, mask: np.ndarray, excl: np.ndarray, slat_tgt: Path) -> dict:
    """§4a 와 같은 계산: R_keep 신규 = 결과 점유 ∧ 원본 비어 있음 ∧ 마스크∪halo 2 밖."""
    from scipy import ndimage
    S = np.ones((3, 3, 3))
    res = vol(coords(slat_tgt))
    new = res & ~src
    new_in, new_out = new & excl, new & ~excl
    lost_in, lost_out = src & ~res & excl, src & ~res & ~excl
    cheb = ndimage.distance_transform_cdt(~excl, metric="chessboard")
    euc = ndimage.distance_transform_edt(~excl)
    lab_all, _ = ndimage.label(new, structure=S)   # 9a(E17): 모든 새 셀을 성분으로 → 마스크∪halo 와 닿지 않는 성분 제거
    touch = ndimage.binary_dilation(excl, structure=S)
    keep = set(np.unique(lab_all[touch & new]).tolist()) - {0}
    lab, n = ndimage.label(new_out, structure=S)
    comps = []
    for i in range(1, n + 1):
        P = lab == i
        la = set(np.unique(lab_all[P]).tolist()) - {0}
        comps.append({"cells": flat(P), "n": int(P.sum()), "dist_cheb": int(cheb[P].min()),
                      "dist_euclid": round(float(euc[P].min()), 2), "touch": bool(int(cheb[P].min()) <= 1),
                      "removed_by_9a": not (la & keep)})
    comps.sort(key=lambda c: -c["n"])
    for k, c in enumerate(comps):
        c["id"] = k
    return {"new_in": flat(new_in), "new_out": flat(new_out), "lost_in": flat(lost_in), "lost_out": flat(lost_out),
            "counts": {"new_in": int(new_in.sum()), "new_out": int(new_out.sum()), "lost_in": int(lost_in.sum()),
                       "lost_out": int(lost_out.sum()), "result_cells": int(res.sum()), "source_cells": int(src.sum())},
            "components": comps, "removed_by_9a_cells": int(sum(c["n"] for c in comps if c["removed_by_9a"])),
            "slat_tgt": str(slat_tgt), "slat_tgt_sha16": sha16(slat_tgt)}


def metrics_table(s: dict) -> dict:
    """설정별 R_keep·지표·시간·E19 판정. 출처 JSON 경로·sha 를 같이 둔다."""
    M = RUNS / "e2a_metrics"
    srcs = {k: M / f"{k}.json" for k in ("rkeep", "eval", "time_mem")}
    J = {k: json.loads(p.read_text()) if p.is_file() else {} for k, p in srcs.items()}
    rows = {}
    for cfg in CONFIGS:
        rk = J["rkeep"].get("metrics", {}).get(cfg, {}).get(s["id"])
        ev = J["eval"].get(cfg, {}).get(s["id"])
        tm = J["time_mem"].get(cfg, {}).get(s["id"])
        if not (rk or ev or tm):
            continue
        row = {}
        if rk:
            row["rkeep_lost"], row["rkeep_new"] = rk["rkeep_lost"][0], rk["rkeep_new"][0]
            row["edit_amount"] = rk["edit_new"][0] + rk["edit_lost"][0]
        if ev:
            row["metrics"] = {m: v[0] for m, v in ev["mean_std"].items() if v}
            row["metric_runs"] = sorted(ev["runs"])
        if tm:
            row["seg4_7"] = tm["seg4_7"]
        rows[cfg] = row
    # E19: 기준은 c15(E20 이후). c15 자신은 base(25) 대비(채택 판정과 같은 비교).
    for cfg, row in rows.items():
        ref = "base" if cfg == "c15" else "c15"
        if cfg == ref or ref not in rows or "metrics" not in row or "metrics" not in rows[ref]:
            continue
        j = {}
        for m, v in row["metrics"].items():
            if m not in E19_TOL or m not in rows[ref]["metrics"]:
                continue
            d = v - rows[ref]["metrics"][m]
            bad = d < E19_TOL[m] if HIGHER_BETTER[m] else d > E19_TOL[m]
            j[m] = {"delta": d, "exceed": bool(bad), "better": bool(d > 0 if HIGHER_BETTER[m] else d < 0)}
        row["e19"] = {"ref": ref, "cells": j}
    return {"rows": rows, "sources": {k: {"path": str(p), "sha256_16": sha16(p)} for k, p in srcs.items()},
            "e19_tolerance": E19_TOL, "e19_note": "기준 c15(E20 이후). c15 행은 base(25) 대비. 시나리오별 허용치만(14건 평균은 보고서)."}


def alignment(s: dict, src_cells: np.ndarray) -> dict:
    """V-2 복셀–메시: 원본 SLat 셀 중심(같은 변환) → 원본 메시 표면 최근접 거리, 1셀(= scale/64) 이내 비율."""
    import trimesh
    from scipy.spatial import cKDTree
    mesh = trimesh.load(s["src_glb"], force="mesh", process=False)
    pts = cells_to_world(src_cells, s["xf"])
    cell = s["xf"]["scale"] / 64.0
    # rtree 없이: 표면을 고르게 샘플링한 점까지의 최근접 거리(샘플 간격만큼 크게 잡히는 근사, 간격은 결과에 기록)
    np.random.seed(0)
    surf, _ = trimesh.sample.sample_surface(mesh, ALIGN_SAMPLES)
    dist, _ = cKDTree(surf).query(pts)
    spacing = float(np.sqrt(mesh.area / ALIGN_SAMPLES))
    r = dist / cell
    extra = {}
    chunks = Path(s["src_glb"]).parent / "chunks"
    if chunks.is_dir():
        # 트랙 A: to_glb(단순화·가려진 면 제거) 이전 디코더 메시(cbin 청크)로도 같은 변환을 검사한다 → 변환 오류와 메시 손실을 가른다
        if str(REPO / "third_party" / "ai-ar-v3") not in sys.path:
            sys.path.insert(0, str(REPO / "third_party" / "ai-ar-v3"))
        from server.glbbuild import cbin_dir_to_colored_mesh
        from server.pipeline.frames import VOXEL_TO_GLB
        v, f, _ = cbin_dir_to_colored_mesh(chunks)
        dm = trimesh.Trimesh((VOXEL_TO_GLB.apply(np.asarray(v, float)) - np.asarray(s["xf"]["center"])) * s["xf"]["scale"], f, process=False)
        np.random.seed(0)
        dsurf, _ = trimesh.sample.sample_surface(dm, ALIGN_SAMPLES)
        rd = cKDTree(dsurf).query(pts)[0] / cell
        extra = {"decoder_mesh_within_1cell": float((rd <= 1.0).mean()), "decoder_mesh_median_cells": float(np.median(rd)),
                 "decoder_mesh_faces": int(len(f)), "glb_to_decoder_area_ratio": float(mesh.area / dm.area)}
    return {**extra, "cells": int(len(pts)), "within_1cell": float((r <= 1.0).mean()), "within_0.5cell": float((r <= 0.5).mean()),
            "median_cells": float(np.median(r)), "p95_cells": float(np.percentile(r, 95)), "max_cells": float(r.max()),
            "mesh_faces": int(len(mesh.faces)), "surface_samples": ALIGN_SAMPLES, "sample_spacing_cells": spacing / cell,
            "src_glb": str(s["src_glb"])}


def build(s: dict, do_align: bool) -> dict:
    from deltacontract.coords import dilate_cells
    t0 = time.time()
    d = CACHE / s["id"]
    d.mkdir(parents=True, exist_ok=True)
    src_c = coords(s["slat"])
    src = vol(src_c)
    mask_cells = np.asarray(json.loads(Path(s["mask"]).read_text())["mask"]["voxels"], dtype=np.int64)
    mask = vol(mask_cells)
    excl = vol(np.asarray(dilate_cells(mask_cells, 2)))
    runs = {}
    for cfg in CONFIGS:
        rd = run_dirs(s, cfg)
        if not rd:
            continue
        runs[cfg] = {}
        for r, rdir in rd.items():
            out = d / cfg / f"{r}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            st = rdir / "slat_tgt.safetensors"
            stale = not out.is_file() or out.stat().st_mtime < st.stat().st_mtime
            if stale:
                out.write_text(json.dumps(diff(s, src, mask, excl, st)))
            runs[cfg][r] = {"glb": str(rdir / "output.glb"), "vox": f"{cfg}/{r}.json", "normalize": s["track"] != "B"}
    cond = {n: str(s["cond"] / f"{n}.png") for n in ("2d_render", "2d_mask", "2d_edit") if (s["cond"] / f"{n}.png").is_file()}
    meta = {k: s[k] for k in ("id", "track", "dataset", "object", "prompt", "prompt_kind", "edit_view", "view_34")}
    meta.update({
        "source_prompt": s.get("source_prompt"), "e12": E12.get(s["id"]),
        "mask": {"n_cells": int(mask.sum()), "fingerprint": s["mask_meta"].get("fingerprint", ""),
                 "file": str(s["mask"]), "file_sha16": sha16(Path(s["mask"]))},
        "mask_cells": flat(mask), "halo_cells": flat(excl & ~mask), "source_cells_n": int(src.sum()),
        "xf": s["xf"], "src_glb": str(s["src_glb"]), "cond": cond, "configs": {c: CONFIGS[c] for c in runs}, "runs": runs,
        "metrics": metrics_table(s), "camera": {"radius": 1.8, "elev_deg": 15, "azimuth_deg": [90, 45, 0, 315, 270],
                                                  "azimuth_offset_deg": -90, "fov_deg": 49.13, "note": "포크 render_rgb_and_mask·bpyrenderer 기본(센서 32 mm, 렌즈 35 mm)"},
    })
    if do_align:
        meta["alignment"] = alignment(s, src_c)
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    print(f"{s['id']}: configs {list(runs)} runs {sum(len(v) for v in runs.values())}"
          + (f" align {meta['alignment']['within_1cell']:.4f}" if do_align else "") + f" ({time.time() - t0:.1f}s)", flush=True)
    return {"id": s["id"], "track": s["track"], "title": f"{s['id']}  {s['prompt'][:48]}", "e12": E12.get(s["id"]),
            "thumb": "2d_edit" if "2d_edit" in cond else None, "seconds": round(time.time() - t0, 2),
            "alignment": meta.get("alignment")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-align", action="store_true")
    a = ap.parse_args()
    only = {x for x in a.only.split(",") if x}
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    idx_p = CACHE / "index.json"
    old = {x["id"]: x for x in json.loads(idx_p.read_text())["scenarios"]} if idx_p.is_file() else {}
    for s in scenarios():
        if only and s["id"] not in only:
            continue
        old[s["id"]] = build(s, not a.skip_align)
    order = [s["id"] for s in scenarios()]
    items = [old[i] for i in order if i in old]
    idx_p.write_text(json.dumps({"scenarios": items, "built": time.strftime("%F %T"), "seconds": round(time.time() - t0, 1)},
                                ensure_ascii=False))
    al = {x["id"]: x["alignment"] for x in items if x.get("alignment")}
    if al:
        (CACHE / "alignment.json").write_text(json.dumps(al, indent=1))
    print(f"done {len(items)} scenarios in {time.time() - t0:.1f}s → {CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
