"""0-1 P-4 판정 (정오표 E7): 이식–기준 3쌍(port r_i vs ref r_i)이 기준–기준 3쌍 분포 안인가.

- Stage 1 coords 완전 일치, 보존 셀 비트 동일 (모든 쌍)
- 편집 셀 max|Δ|·mean|Δ|: 이식–기준 각 쌍의 값 ≤ 기준–기준 3쌍 최대값이면 "분포 안", 4.7e-2 상한
- GLB 정점·면 수, 시간(구간 3–8), 메모리
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reference.compare_slat_runs import compare  # noqa: E402


def glb_counts(p):
    import trimesh
    m = trimesh.load(p, force="mesh")
    return {"vertices": int(len(m.vertices)), "faces": int(len(m.faces))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port_dir", required=True)
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--slat", required=True)
    ap.add_argument("--mask_json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P, R = Path(a.port_dir), Path(a.ref_dir)
    ref = {r: R / f"{r}.work" / "slat_tgt.safetensors" for r in ("r1", "r2", "r3")}
    port = {r: P / f"port-{r}" / "slat_tgt.safetensors" for r in ("r1", "r2", "r3")}
    rr = {f"{x}-{y}": compare(ref[x], ref[y], a.slat, a.mask_json) for x, y in itertools.combinations(ref, 2)}
    pr = {f"port{r}-ref{r}": compare(port[r], ref[r], a.slat, a.mask_json) for r in ref}
    pp = {f"port{x}-port{y}": compare(port[x], port[y], a.slat, a.mask_json) for x, y in itertools.combinations(port, 2)}
    lim_max = max(v["edited"]["max_abs_diff"] for v in rr.values())
    lim_mean = max(v["edited"]["mean_abs_diff"] for v in rr.values())
    verdict = {
        "coords_equal_all": all(v["coords_equal"] for v in [*rr.values(), *pr.values(), *pp.values()]),
        "preserved_bit_equal_all": all(v["preserved"].get("bit_equal", True) for v in [*rr.values(), *pr.values()]),
        "ref_ref_edit_max_range": [min(v["edited"]["max_abs_diff"] for v in rr.values()), lim_max],
        "ref_ref_edit_mean_range": [min(v["edited"]["mean_abs_diff"] for v in rr.values()), lim_mean],
        "port_ref_edit_max": [v["edited"]["max_abs_diff"] for v in pr.values()],
        "port_ref_edit_mean": [v["edited"]["mean_abs_diff"] for v in pr.values()],
    }
    verdict["port_ref_within_ref_ref"] = all(v["edited"]["max_abs_diff"] <= lim_max and v["edited"]["mean_abs_diff"] <= lim_mean
                                             for v in pr.values())
    verdict["port_ref_bit_equal"] = all(v["edited"].get("bit_equal") and v["preserved"].get("bit_equal", True) for v in pr.values())
    verdict["under_cap_4.7e-2"] = all(v["edited"]["max_abs_diff"] <= 4.7e-2 for v in pr.values())
    glb = {"ref": {r: glb_counts(R / f"{r}.work" / "output.glb") for r in ref},
           "port": {r: glb_counts(P / f"port-{r}" / "output.glb") for r in ref}}
    times = {"ref": {r: json.loads((R / f"{r}.json").read_text())["segments"] for r in ref},
             "port": {r: json.loads((P / f"port-{r}.json").read_text())["segments"] for r in ref}}
    seg = {}
    for impl in ("ref", "port"):
        for r, s in times[impl].items():
            for k, v in s.items():
                seg.setdefault(impl, {}).setdefault(k, []).append(round(v["seconds"], 2))
    out = {"verdict": verdict, "ref_ref": rr, "port_ref": pr, "port_port": pp, "glb": glb, "segments_seconds": seg,
           "mem": {impl: {k: max(times[impl][r][k]["max_memory_allocated"] for r in ref if k in times[impl][r]) / 2 ** 30
                          for k in times[impl]["r1"]} for impl in ("ref", "port")}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"verdict": verdict, "glb": glb, "segments_seconds": seg}, indent=1))


if __name__ == "__main__":
    main()
