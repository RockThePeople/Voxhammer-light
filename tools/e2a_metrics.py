"""1번 E2a 집계 — 설정(base·c12·c15·…) × 14건 × 3회.

base = 0-1 종단 비교의 이식본 r1–r3(`$RUNS_DIR/01-port/<시나리오>/port-r*`, 25 스텝) — 재실행 없이 품질 지표만 새로 계산.
설정 c12·c15 등 = `$RUNS_DIR/<E2A_DAY>/<시나리오>/e2a-<설정>/r*`.

phase time   : 구간 3·4·6·7·8, 4–7 합, alloc·RSS 피크, ss_kv·slat_kv 바이트·항목 수 (timing JSON)
phase rkeep  : R_keep(eval/rkeep.py) + E18 판정(소실 0, 신규 ≤ max(base×1.2, base+20)·합계 ≤ base×1.1, 편집량 < base 50% → 효능 저하)
phase render : 트랙 B 8건 GLB → Edit3D-Bench render_images(16시점) / 트랙 A 6건 GLB(원본 정규화 적용) → 같은 렌더
phase eval   : 트랙 B eval_main(PSNR·SSIM·LPIPS 마스크 밖, DINO-I, CLIP-T, Chamfer), 트랙 A 는 같은 코드로 DINO-I·CLIP-T
              (트랙 A 정답: 동결 model.glb 를 같은 렌더로 16시점, metadata 의 prompt = 인페인팅 프롬프트)
결과: $RUNS_DIR/e2a_metrics/*.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ASSETS = Path(os.environ["ASSETS_DIR"])
RUNS = Path(os.environ["RUNS_DIR"])
DAY = os.environ.get("E2A_DAY", "2026-10-25")
OUT = RUNS / "e2a_metrics"
EVAL_PY = os.environ.get("EVAL_PY", str(Path.home() / "vhl-venvs/edit3dbench/bin/python"))
BENCH_SRC = ROOT / "third_party" / "VoxHammer" / "Edit3D-Bench"
SCEN_DIR = ROOT / "scenarios" if (ROOT / "scenarios" / "track_b.json").exists() else ROOT.parent / "vhl-00s2" / "scenarios"
CONFIGS = [c for c in os.environ.get("E2A_CONFIGS", "base,c15,c12").split(",") if c]
# render·eval 만 좁힌다(예: s1-12·s2-12 는 트랙 B r1 만): E2A_RUNS=r1 E2A_TRACKS=B
RUN_IDS = [r for r in os.environ.get("E2A_RUNS", "r1,r2,r3").split(",") if r]
TRACKS = [t for t in os.environ.get("E2A_TRACKS", "A,B").split(",") if t]


def scenarios():
    ta = json.loads((SCEN_DIR / "track_a.json").read_text())
    tb = json.loads((SCEN_DIR / "track_b.json").read_text())
    out = []
    for sid, s in ta["scenarios"].items():
        a = s["asset"]
        g = json.loads((ASSETS / "track_a" / a / "gen" / "gen.json").read_text())
        out.append({"id": sid, "track": "A", "name": f"track_a-{sid}", "asset": a, "prompt": s["inpaint_prompt"],
                    "slat": ASSETS / "track_a" / a / "gen" / "slat.safetensors", "mask": ASSETS / "track_a" / a / "masks" / f"{sid}.mask.json",
                    "norm": g["glb_normalize"], "src_glb": ASSETS / "track_a" / a / "gen" / "model.glb"})
    for bid, s in tb["scenarios"].items():
        out.append({"id": bid, "track": "B", "name": f"track_b-{bid}", "bench": s,
                    "slat": ASSETS / "track_b" / bid / "slat.safetensors", "mask": ASSETS / "track_b" / bid / "mask.json"})
    return out


def run_dir(s, cfg, r):
    if cfg == "base":
        return RUNS / "01-port" / s["name"] / f"port-{r}", RUNS / "01-port" / s["name"] / f"port-{r}.json"
    d = RUNS / DAY / s["name"] / f"e2a-{cfg}"
    return d / r, d / f"{r}.json"


def ms(v):
    v = [x for x in v if x is not None]
    return [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0] if v else None


def time_mem():
    res = {}
    for cfg in CONFIGS:
        for s in scenarios():
            J = [json.loads(run_dir(s, cfg, r)[1].read_text()) for r in ("r1", "r2", "r3") if run_dir(s, cfg, r)[1].exists()]
            if not J:
                continue
            seg = {k: ms([j["segments"].get(k, {}).get("seconds") for j in J]) for k in ("3", "4", "6", "7", "8")}
            s47 = ms([sum(j["segments"].get(k, {}).get("seconds", 0) for k in ("4", "6", "7")) for j in J])
            res.setdefault(cfg, {})[s["id"]] = {
                "n": len(J), "segments": seg, "seg4_7": s47,
                "alloc_GiB": max(max(v["max_memory_allocated"] for v in j["segments"].values()) for j in J) / 2 ** 30,
                "rss_GiB": max(max((v["rss_bytes_max"] or 0) for v in j["segments"].values()) for j in J) / 2 ** 30,
                "ss_kv_GB": J[0]["cache_bytes"]["stage1"]["kv"]["cpu"] / 1e9, "slat_kv_GB": J[0]["cache_bytes"]["stage2"]["kv"]["cpu"] / 1e9,
                "cache_entries": J[0].get("cache_entries"), "steps": J[0].get("steps"),
            }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "time_mem.json").write_text(json.dumps(res, indent=1))
    for cfg, d in res.items():
        a = [v["seg4_7"][0] for v in d.values()]
        print(cfg, "seg4-7 mean over", len(a), "=", round(float(np.mean(a)), 1), "ss_kv", round(next(iter(d.values()))["ss_kv_GB"], 1), "GB")


def rkeep():
    from eval.rkeep import rkeep_metrics
    from safetensors import safe_open
    res = {}
    for s in scenarios():
        with safe_open(str(s["slat"]), "np") as f:
            parent = f.get_tensor("coords")
        mask = np.asarray(json.loads(Path(s["mask"]).read_text())["mask"]["voxels"])
        for cfg in CONFIGS:
            rk = []
            for r in ("r1", "r2", "r3"):
                p = run_dir(s, cfg, r)[0] / "slat_tgt.safetensors"
                if p.exists():
                    with safe_open(str(p), "np") as f:
                        rk.append(rkeep_metrics(parent, f.get_tensor("coords"), mask))
            if rk:
                res.setdefault(cfg, {})[s["id"]] = {k: ms([x[k] for x in rk]) for k in
                                                    ("rkeep_iou", "rkeep_lost", "rkeep_new", "edit_new", "edit_lost")}
    # E18 판정
    base = res.get("base", {})
    verdict = {}
    for cfg, d in res.items():
        if cfg == "base":
            continue
        rows, tot_c, tot_b = {}, 0.0, 0.0
        for sid, v in d.items():
            b = base[sid]
            nb, nc = b["rkeep_new"][0], v["rkeep_new"][0]
            eb = b["edit_new"][0] + b["edit_lost"][0]
            ec = v["edit_new"][0] + v["edit_lost"][0]
            tot_c += nc
            tot_b += nb
            rows[sid] = {"rkeep_lost": v["rkeep_lost"][0], "lost_ok": v["rkeep_lost"][0] == 0,
                         "rkeep_new": nc, "base_new": nb, "new_limit": max(nb * 1.2, nb + 20), "new_ok": nc <= max(nb * 1.2, nb + 20),
                         "edit_amount": ec, "base_edit_amount": eb, "efficacy_drop": ec < 0.5 * eb,
                         "rkeep_iou": v["rkeep_iou"][0], "base_iou": b["rkeep_iou"][0]}
        verdict[cfg] = {"rows": rows, "new_total": tot_c, "base_new_total": tot_b, "total_ok": tot_c <= tot_b * 1.1}
    (OUT / "rkeep.json").write_text(json.dumps({"metrics": res, "e18": verdict}, indent=1))
    for cfg, v in verdict.items():
        bad = [k for k, r in v["rows"].items() if not (r["lost_ok"] and r["new_ok"])]
        eff = [k for k, r in v["rows"].items() if r["efficacy_drop"]]
        print(cfg, "new total", v["new_total"], "/ base", v["base_new_total"], "ok", v["total_ok"], "| fail", bad, "| efficacy drop", eff)


RENDER = "import sys; sys.path.insert(0,'.'); import render; render.render_images(sys.argv[1], sys.argv[2])"


def _render(glb, img_dir):
    if (img_dir / "render_0015.png").exists():
        return
    img_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([EVAL_PY, "-c", RENDER, str(glb), str(img_dir)], cwd=BENCH_SRC, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _norm_glb(src, dst, norm):
    import trimesh
    if dst.exists():
        return dst
    sc = trimesh.load(src)
    sc.apply_translation([-x for x in norm["center_glb_frame"]])
    sc.apply_scale(norm["scale"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    sc.export(dst)
    return dst


def render():
    gtA = OUT / "trackA_gt" / "data"
    meta = []
    for s in scenarios():
        if s["track"] not in TRACKS:
            continue
        if s["track"] == "A":
            # 트랙 A 정답(원본 16시점)과 metadata
            g = gtA / "TrackA" / s["id"]
            _render(s["src_glb"], g / "source_model" / "render")
            (g / "prompt_1" / "render").mkdir(parents=True, exist_ok=True)
            from PIL import Image
            for i in range(16):
                mp = g / "prompt_1" / "render" / f"mask_{i:04d}.png"
                if not mp.exists():
                    Image.new("L", (512, 512), 0).save(mp)  # 마스크 밖 지표는 트랙 A 에서 쓰지 않는다
            meta.append({"dataset": "TrackA", "source_model": s["id"], "source_prompt": s["asset"], "prompt_1": s["prompt"]})
        for cfg in CONFIGS:
            for r in RUN_IDS:
                glb = run_dir(s, cfg, r)[0] / "output.glb"
                if not glb.exists():
                    continue
                if s["track"] == "B":
                    b = s["bench"]
                    d = OUT / "pred" / cfg / r / s["id"] / b["dataset"] / b["object"] / f"prompt_{b['prompt_id']}"
                    d.mkdir(parents=True, exist_ok=True)
                    if not (d / "edit.glb").exists():
                        (d / "edit.glb").symlink_to(glb)
                else:
                    d = OUT / "pred" / cfg / r / s["id"] / "TrackA" / s["id"] / "prompt_1"
                    _norm_glb(glb, d / "edit.glb", s["norm"])
                _render(d / "edit.glb", d / "images")
                print("rendered", cfg, r, s["id"], flush=True)
    if meta:
        (gtA / "metadata.json").write_text(json.dumps(meta, indent=1))


def evaluate():
    # 이미 있는 설정의 결과는 두고, 이번 CONFIGS 의 시나리오만 덮어쓴다
    res = json.loads((OUT / "eval.json").read_text()) if (OUT / "eval.json").exists() else {}
    for s in scenarios():
        if s["track"] not in TRACKS:
            continue
        gt = ASSETS / "Edit3D-Bench" / "data" if s["track"] == "B" else OUT / "trackA_gt" / "data"
        metrics = ["psnr", "ssim", "lpips", "dino_i", "chamfer", "clip_t"] if s["track"] == "B" else ["dino_i", "clip_t"]
        for cfg in CONFIGS:
            vals = {}
            for r in RUN_IDS:
                pred = OUT / "pred" / cfg / r / s["id"]
                if not pred.exists():
                    continue
                od = OUT / "eval" / cfg / s["id"] / r
                if not (od / "summary.json").exists():
                    od.mkdir(parents=True, exist_ok=True)
                    subprocess.run([EVAL_PY, "eval_main.py", "--gt_root", str(gt), "--pred_root", str(pred), "--metrics", *metrics,
                                    "--device", "cuda:0", "--output_dir", str(od)], cwd=BENCH_SRC,
                                   stdout=open(od / "eval.log", "w"), stderr=subprocess.STDOUT)
                if (od / "summary.json").exists():
                    vals[r] = {k: v["mean"] for k, v in json.loads((od / "summary.json").read_text())["results"].items()}
            if vals:
                res.setdefault(cfg, {})[s["id"]] = {"runs": vals, "mean_std": {m: ms([vals[r].get(m) for r in vals]) for m in metrics}}
        print("eval", s["id"], {c: {m: round(v[0], 4) for m, v in res[c][s["id"]]["mean_std"].items() if v} for c in res if s["id"] in res[c]}, flush=True)
    (OUT / "eval.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"time": time_mem, "rkeep": rkeep, "render": render, "eval": evaluate}[sys.argv[1]]()
