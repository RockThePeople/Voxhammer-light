"""00s2 F-5 기준선 집계 — 시간·메모리·바닥값·품질 지표.

phase floor  : 시나리오마다 3회 → 3쌍 compare_slat_runs(Stage 1 coords, 보존 셀 비트 동일, 편집 셀 max|Δ|·mean|Δ|),
               R_keep(eval/rkeep.py, 부모 = 입력 SLat coords, 자식 = slat_tgt coords), 구간 3–8 평균±표준편차, 메모리 4종.
phase bench_render : 트랙 B 회차별 output.glb 를 Edit3D-Bench 예측 구조(<pred>/rK/<dataset>/<object>/prompt_<id>/edit.glb)로
               연결하고 포크 Bench `render.py render_images`(16시점, 영상 생략)로 렌더 (eval venv, CPU Cycles).
phase bench_eval : eval_main.py — PSNR·SSIM·LPIPS(마스크 밖), DINO-I, CLIP-T, Chamfer. FID·FVD 제외.
결과: scenarios/baseline_14.json (요약), RUNS_DIR/f5_metrics/ (원자료).
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT.parent / "Voxhammer-light"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MAIN))
ASSETS = Path(os.environ["ASSETS_DIR"])
RUNS = Path(os.environ["RUNS_DIR"])
DAY = os.environ.get("F5_DAY")
OUT = RUNS / "f5_metrics"
EVAL_PY = os.environ.get("EVAL_PY", str(Path.home() / "vhl-venvs/edit3dbench/bin/python"))
BENCH_SRC = ROOT / "third_party" / "VoxHammer" / "Edit3D-Bench"


def scenarios():
    ta = json.loads((ROOT / "scenarios" / "track_a.json").read_text())
    tb = json.loads((ROOT / "scenarios" / "track_b.json").read_text())
    out = []
    for sid, s in ta["scenarios"].items():
        a = s["asset"]
        out.append({"id": sid, "track": "A", "name": f"track_a-{sid}", "slat": ASSETS / "track_a" / a / "gen" / "slat.safetensors",
                    "mask": ASSETS / "track_a" / a / "masks" / f"{sid}.mask.json", "fp": s["mask"]["fingerprint"][:12]})
    for bid, s in tb["scenarios"].items():
        out.append({"id": bid, "track": "B", "name": f"track_b-{bid}", "slat": ASSETS / "track_b" / bid / "slat.safetensors",
                    "mask": ASSETS / "track_b" / bid / "mask.json", "fp": s["mask"]["fingerprint"][:12], "bench": s})
    for s in out:
        s["dir"] = RUNS / DAY / s["name"] / f"w34-{s['fp']}"
    return out


def floor():
    from reference.compare_slat_runs import compare
    from eval.rkeep import rkeep_metrics
    from safetensors import safe_open
    res = {}
    for s in scenarios():
        runs = [r for r in ("r1", "r2", "r3") if (s["dir"] / f"{r}.json").exists()]
        J = {r: json.loads((s["dir"] / f"{r}.json").read_text()) for r in runs}
        slat = {r: s["dir"] / f"{r}.work" / "slat_tgt.safetensors" for r in runs}
        pairs = {f"{a}-{b}": compare(slat[a], slat[b], s["slat"], s["mask"]) for a, b in itertools.combinations(runs, 2)}
        with safe_open(str(s["slat"]), "np") as f:
            parent = f.get_tensor("coords")
        mask = np.asarray(json.loads(Path(s["mask"]).read_text())["mask"]["voxels"])
        rk = {}
        for r in runs:
            with safe_open(str(slat[r]), "np") as f:
                rk[r] = rkeep_metrics(parent, f.get_tensor("coords"), mask)
        seg = {}
        for k in ("3", "4", "5", "6", "7", "8"):
            v = [J[r]["segments"][k]["seconds"] for r in runs if k in J[r]["segments"]]
            if v:
                seg[k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                          "alloc_GiB": max(J[r]["segments"][k]["max_memory_allocated"] for r in runs) / 2 ** 30,
                          "smi_MiB": max((J[r]["segments"][k]["nvidia_smi_mib_max"] or 0) for r in runs),
                          "rss_GiB": max((J[r]["segments"][k]["rss_bytes_max"] or 0) for r in runs) / 2 ** 30}
        tot = [J[r]["edit_total_seconds"] for r in runs]
        s47 = [sum(J[r]["segments"].get(k, {}).get("seconds", 0) for k in ("4", "5", "6", "7")) for r in runs]
        rk_mean = {k: [float(np.mean([rk[r][k] for r in runs])), float(np.std([rk[r][k] for r in runs], ddof=1)) if len(runs) > 1 else 0.0]
                   for k in ("rkeep_iou", "rkeep_lost", "rkeep_new", "edit_new", "edit_lost", "halo_ring_new", "halo_ring_lost")}
        res[s["id"]] = {
            "runs": runs, "errors": [J[r]["error"] for r in runs if J[r]["error"]],
            "calls_ok": all(not J[r]["checks"]["calls_match_expected"] for r in runs),
            "counts": J[runs[0]]["counts"] if runs else None,
            "segments": seg, "edit_total": [float(np.mean(tot)), float(np.std(tot, ddof=1)) if len(tot) > 1 else 0.0],
            "seg4_7": [float(np.mean(s47)), float(np.std(s47, ddof=1)) if len(s47) > 1 else 0.0],
            "cache_bytes": J[runs[0]]["cache_bytes"] if runs else None,
            "floor_pairs": {p: {"coords_equal": v["coords_equal"], "preserved_bit_equal": v["preserved"].get("bit_equal"),
                                "edited_cells": v["edited"].get("cells"), "edited_max": v["edited"].get("max_abs_diff"),
                                "edited_mean": v["edited"].get("mean_abs_diff")} for p, v in pairs.items()},
            "rkeep": rk_mean, "rkeep_runs": rk,
            "json_sha256": {r: __import__("hashlib").sha256((s["dir"] / f"{r}.json").read_bytes()).hexdigest() for r in runs},
        }
        print(s["id"], "floor max", [round(v["edited"].get("max_abs_diff", 0), 4) for v in pairs.values()],
              "rkeep iou", round(rk_mean["rkeep_iou"][0], 4), "edit_new", rk_mean["edit_new"][0], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "floor.json").write_text(json.dumps(res, indent=1, default=str))


def v0_decode():
    """트랙 B 바닥값: 판본 0 SLat 을 편집 없이 디코딩 → 편집 출력과 같은 to_glb(0.95, 1024) → OUT/v0/<B#>/output.glb.
    인코딩·디코딩(왕복) 오차를 편집 효과와 분리한다. 측정 구간 밖(시간 기록 안 함)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import editcore.models as em
    from editcore.export import decode_to_glb
    import torch
    import trellis.modules.sparse as sp
    from safetensors import safe_open
    pipe = em.load_pipeline()
    for s in scenarios():
        if s["track"] != "B":
            continue
        dst = OUT / "v0" / s["id"] / "output.glb"
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with safe_open(str(s["slat"]), "pt") as f:
            c, ft = f.get_tensor("coords").int().cuda(), f.get_tensor("feats").float().cuda()
        decode_to_glb(pipe, sp.SparseTensor(feats=ft, coords=c), dst)
        print("v0 decoded", s["id"], flush=True)


def bench_render():
    for s in scenarios():
        if s["track"] != "B":
            continue
        b = s["bench"]
        for r in ("r1", "r2", "r3", "v0"):
            d = OUT / "bench_pred" / r / s["id"] / b["dataset"] / b["object"] / f"prompt_{b['prompt_id']}"
            d.mkdir(parents=True, exist_ok=True)
            glb = d / "edit.glb"
            if not glb.exists():
                glb.symlink_to(OUT / "v0" / s["id"] / "output.glb" if r == "v0" else s["dir"] / f"{r}.work" / "output.glb")
            if not (d / "images" / "render_0015.png").exists():
                (d / "images").mkdir(exist_ok=True)
                subprocess.run([EVAL_PY, "-c", "import sys; sys.path.insert(0,'.'); import render; render.render_images(sys.argv[1], sys.argv[2])",
                                str(glb), str(d / "images")], cwd=BENCH_SRC, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("rendered", s["id"], r, flush=True)


def bench_eval():
    """건별·회차별로 eval_main 을 따로 돌린다(포크 결과 JSON 의 all_scores 에 항목 ID 가 없어서)."""
    gt = ASSETS / "Edit3D-Bench" / "data"
    res = {}
    for s in scenarios():
        if s["track"] != "B":
            continue
        for r in ("r1", "r2", "r3", "v0"):
            od = OUT / "bench_eval" / s["id"] / r
            if not (od / "summary.json").exists():
                od.mkdir(parents=True, exist_ok=True)
                subprocess.run([EVAL_PY, "eval_main.py", "--gt_root", str(gt), "--pred_root", str(OUT / "bench_pred" / r / s["id"]),
                                "--metrics", "psnr", "ssim", "lpips", "dino_i", "chamfer", "clip_t", "--device", "cuda:0",
                                "--output_dir", str(od)], cwd=BENCH_SRC, stdout=open(od / "eval.log", "w"), stderr=subprocess.STDOUT)
            sm = json.loads((od / "summary.json").read_text())["results"] if (od / "summary.json").exists() else {}
            res.setdefault(s["id"], {})[r] = {k: v["mean"] for k, v in sm.items()}
        m = res[s["id"]]
        agg = {}
        for k in ("psnr", "ssim", "lpips", "dino_i", "chamfer", "clip_t"):
            v = [m[r].get(k) for r in ("r1", "r2", "r3") if r in m and m[r].get(k) is not None]
            agg[k] = [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0] if v else None
        res[s["id"]]["mean_std"] = agg
        res[s["id"]]["v0_no_edit"] = m.get("v0")
        print(s["id"], {k: (round(v[0], 4) if v else None) for k, v in agg.items()}, flush=True)
    (OUT / "bench_eval.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    {"floor": floor, "v0_decode": v0_decode, "bench_render": bench_render, "bench_eval": bench_eval}[sys.argv[1]]()
