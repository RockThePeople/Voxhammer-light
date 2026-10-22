"""계측 JSON(vhl-run/1) 묶음 → 구간별 평균±표준편차 표, 메모리 표, 회차 간 동등성(Stage 1 coords, Stage 2 feats).

사용: summarize.py <run dir> [<run dir> ...] --out summary.json
프로파일 회차(`profile_run: true`)는 평균에서 뺀다.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

SEG_ORDER = ["registration", "3", "4", "5", "6", "7", "8"]


def load_slat(p):
    from safetensors import safe_open
    with safe_open(str(p), "np") as f:
        return f.get_tensor("coords"), f.get_tensor("feats")


def stage_equivalence(runs):
    """회차 쌍마다 Stage 1 coords 일치 여부, 공통 좌표에서 Stage 2 feats max|Δ|."""
    out = []
    for a, b in itertools.combinations(runs, 2):
        pa, pb = Path(a["outputs"]["slat_tgt"]), Path(b["outputs"]["slat_tgt"])
        if not (pa.exists() and pb.exists()):
            continue
        ca, fa = load_slat(pa)
        cb, fb = load_slat(pb)
        ka = {tuple(r): i for i, r in enumerate(ca.tolist())}
        kb = {tuple(r): i for i, r in enumerate(cb.tolist())}
        common = sorted(set(ka) & set(kb))
        ia = np.array([ka[c] for c in common], int)
        ib = np.array([kb[c] for c in common], int)
        d = np.abs(fa[ia] - fb[ib]) if len(common) else np.zeros((0, 8))
        out.append({"pair": f"{a['run']}-{b['run']}", "coords_equal": set(ka) == set(kb),
                    "n_a": len(ka), "n_b": len(kb), "common": len(common),
                    "only_a": len(set(ka) - set(kb)), "only_b": len(set(kb) - set(ka)),
                    "feats_max_abs_diff_common": float(d.max()) if d.size else None,
                    "feats_mean_abs_diff_common": float(d.mean()) if d.size else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    summary = {}
    for d in a.dirs:
        files = sorted(Path(d).glob("*.json"))
        runs = [json.loads(f.read_text()) for f in files]
        timed = [r for r in runs if not r.get("profile_run")]
        key = Path(d).name
        segs = {}
        for s in SEG_ORDER:
            vals = [r["segments"].get(s, {}).get("seconds") for r in timed if s in r["segments"]]
            if not vals:
                continue
            mem = [r["segments"][s] for r in timed if s in r["segments"]]
            segs[s] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                       "n": len(vals), "values": vals,
                       "max_memory_allocated_GiB": max(m["max_memory_allocated"] for m in mem) / 2 ** 30,
                       "nvidia_smi_MiB": max((m["nvidia_smi_mib_max"] or 0) for m in mem),
                       "rss_GiB": max((m["rss_bytes_max"] or 0) for m in mem) / 2 ** 30,
                       "parts_mean": {n: float(np.mean([r["segments"][s]["parts"].get(n, 0.0) for r in timed]))
                                      for n in mem[0]["parts"]}}
        tot = [r["edit_total_seconds"] for r in timed]
        s47 = [sum(r["segments"].get(k, {}).get("seconds", 0.0) for k in ("4", "5", "6", "7")) for r in timed]
        summary[key] = {
            "mask": timed[0]["mask"] if timed else None, "runs": [r["run"] for r in timed],
            "segments": segs,
            "edit_total": {"mean": float(np.mean(tot)), "std": float(np.std(tot, ddof=1)) if len(tot) > 1 else 0.0},
            "seg4_7": {"mean": float(np.mean(s47)), "std": float(np.std(s47, ddof=1)) if len(s47) > 1 else 0.0},
            "cache_bytes": timed[0]["cache_bytes"] if timed else None,
            "counts": [r["counts"] for r in timed],
            "checks": [r["checks"] for r in timed],
            "errors": [r["error"] for r in runs if r["error"]],
            "equivalence": stage_equivalence(timed),
            "profile": next((r["profile"] for r in runs if r.get("profile_run")), None),
            "json_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        }
    Path(a.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for k, v in summary.items():
        print(f"== {k}  mask {v['mask']['file'] if v['mask'] else None}  runs {v['runs']}")
        for s, x in v["segments"].items():
            print(f"  {s:13s} {x['mean']:7.2f} ± {x['std']:5.2f} s  alloc {x['max_memory_allocated_GiB']:5.2f} GiB  "
                  f"smi {x['nvidia_smi_MiB']} MiB  rss {x['rss_GiB']:.1f} GiB")
        print(f"  total {v['edit_total']['mean']:.2f} ± {v['edit_total']['std']:.2f}   seg4-7 {v['seg4_7']['mean']:.2f} ± {v['seg4_7']['std']:.2f}")
        for e in v["equivalence"]:
            print(f"  eq {e}")


if __name__ == "__main__":
    main()
