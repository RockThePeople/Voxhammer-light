"""W41 반복 측정 집계 — 항목별 평균·표준편차·최소/최대, 팔별로.

부모 대비 지표는 **slat 격자**에서 잰다 (D28: 정본은 slat 격자).
부모 = 생성 SLat, 자식 = 각 런이 포획한 편집 SLat.
⇒ Baseline 과 Ours 를 **같은 격자·같은 방식**으로 비교한다. 지금까지는 Baseline 의
   산출 SLat 이 없어 GLB 샘플링으로 쟀다.
"""
import glob
import json
import os
import re
import statistics as st
import sys

import numpy as np
import utils3d
from safetensors import safe_open

TRACK = os.environ.get("TRACK_ROOT", "<HOME>/ai-ar-data/vh-track")


def coords_of(p):
    with safe_open(p, "pt") as f:
        return set(map(tuple, f.get_tensor("coords").numpy()[:, 1:].tolist()))


def mask_of(p):
    """voxels_delete.ply 또는 복셀 목록 JSON 을 받는다 (W42 는 점유 셀 JSON 이다)."""
    if p.endswith(".json"):
        return set(map(tuple, json.load(open(p))))
    pos = utils3d.io.read_ply(p)[0]
    return set(map(tuple, np.floor((np.asarray(pos) + 0.5) * 64).astype(int).tolist()))


STEPS = [("STEP 1", "렌더"), ("STEP 2", "feature"), ("STEP 3", "마스크복셀화"), ("STEP 4", "편집")]


def parse_steps(log):
    """경과초가 각인된 로그에서 STEP 경계를 뽑아 구간 길이로 바꾼다."""
    marks = []
    for line in open(log, errors="ignore"):
        m = re.match(r"\[\s*([\d.]+)\]\s*(.*)", line)
        if not m:
            continue
        el, body = float(m.group(1)), m.group(2)
        for key, name in STEPS:
            if body.startswith(key + ":"):
                marks.append((name, el))
        if "Pipeline completed" in body or body.startswith("[단계] edit_total"):
            marks.append(("__end__", el))
    out = {}
    for i, (name, el) in enumerate(marks):
        if name == "__end__":
            continue
        nxt = marks[i + 1][1] if i + 1 < len(marks) else None
        if nxt is not None:
            out[name] = round(nxt - el, 2)
    return out


def rss_of(log):
    for line in open(log, errors="ignore"):
        if "Maximum resident set size" in line:
            return round(int(line.split()[-1]) / 1048576, 2)
    return None


def collect(tag, asset, arm, prefix="w41", maskfile=None):
    parent = coords_of(f"{TRACK}/{asset}/slat.safetensors")
    mask = mask_of(maskfile or f"{TRACK}/{tag}-mask/voxels_delete.ply")
    p_in, p_out = parent & mask, parent - mask
    rows = []
    for d in sorted(glob.glob(f"{TRACK}/{prefix}-{tag}-{arm}-r*")):
        tj = os.path.join(d, "timing.json")
        if not os.path.exists(tj):
            continue
        T = json.load(open(tj))
        c = coords_of(os.path.join(d, "slat.safetensors"))
        c_in, c_out = c & mask, c - mask
        r = {
            "run": os.path.basename(d),
            "wall": T["wall"], "edit": T.get("edit_total"), "load": T.get("pipeline_load"),
            "밖_IoU": len(p_out & c_out) / len(p_out | c_out),
            "밖_사라짐": len(p_out - c_out), "밖_생김": len(c_out - p_out),
            "안_신규": len(c_in - p_in), "안_제거": len(p_in - c_in),
            "점유셀": len(c), "정점": T["vertices"],
            "GPU_GiB": T["gpu_max_alloc_GiB"], "RSS_GiB": rss_of(os.path.join(d, "run.log")),
            "sha": T["slat_sha256"][:12],
        }
        r.update(parse_steps(os.path.join(d, "run.log")))
        rows.append(r)
    return rows


def summarize(rows, keys):
    print(f"{'항목':14s}{'n':>3s}{'평균':>12s}{'표준편차':>12s}{'최소':>12s}{'최대':>12s}{'변동계수':>10s}")
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None]
        if not v:
            continue
        m = st.mean(v)
        s = st.stdev(v) if len(v) > 1 else 0.0
        cv = (s / m * 100) if m else 0.0
        fmt = "{:12.4f}" if k.endswith("IoU") else "{:12.2f}"
        print(f"{k:14s}{len(v):3d}" + fmt.format(m) + fmt.format(s) + fmt.format(min(v)) + fmt.format(max(v)) + f"{cv:9.2f}%")


if __name__ == "__main__":
    tag, asset = sys.argv[1], sys.argv[2]
    prefix = sys.argv[3] if len(sys.argv) > 3 else "w41"
    maskfile = sys.argv[4] if len(sys.argv) > 4 else None
    KEYS = ["wall", "load", "렌더", "feature", "마스크복셀화", "편집", "edit",
            "밖_IoU", "밖_사라짐", "밖_생김", "안_신규", "안_제거", "점유셀", "정점", "GPU_GiB", "RSS_GiB"]
    for arm in ["baseline", "ours"]:
        rows = collect(tag, asset, arm, prefix, maskfile)
        print(f"\n{'='*92}\n### {prefix} · {tag} · {arm} · {len(rows)}런\n{'='*92}")
        for r in rows:
            print(f"  {r['run']:26s} wall {r['wall']:7.1f}s  밖IoU {r['밖_IoU']:.4f}  "
                  f"사라짐 {r['밖_사라짐']:5d}  생김 {r['밖_생김']:5d}  안신규 {r['안_신규']:5d}  "
                  f"안제거 {r['안_제거']:5d}  셀 {r['점유셀']:6d}  정점 {r['정점']:7d}  sha {r['sha']}")
        if rows:
            print()
            summarize(rows, KEYS)
            shas = {r["sha"] for r in rows}
            if len(rows) < 2:
                verdict = "런이 1개뿐 — 판정 불가"
            elif len(shas) == 1:
                verdict = "전부 동일 → 비트 재현됨"
            elif len(shas) == len(rows):
                verdict = "전부 다름 → 비트 재현 안 됨"
            else:
                verdict = "일부만 같음"
            print(f"\n  🔴 SLat 해시: {len(shas)}종 / {len(rows)}런  → {verdict}")
