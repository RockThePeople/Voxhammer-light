"""2번 R4 집계 (CPU). 켬 = $RUNS_DIR/02-r4/<이름>/r4-on/r{1,2,3}, 끔 = E2a c15 r1–r3(15 스텝, 깨끗한 측정).

phase time   : 구간 7 세부(decode_mesh·chunk_partition·cbin_encode·bookkeeping·cbin_write) 3회 평균±표준편차, 구간 4–7, GLB 조립(구간 밖),
               메모리, 디코더 호출 수(Gaussian 0 확인). 끔은 c15 의 구간 7(decode_slat·to_glb·glb_export).
phase equiv  : slat_tgt 켬 vs 끔 — coords 완전 일치·보존 셀 비트 동일(필수), 편집 셀 E14·E15(tools/e14.py).
               기준–기준 = c15 r1–r3 3쌍, 변경–기준 = 켬 rK vs 끔 rK (K=1..3).
phase chunks : 청크 수·변경(새)·승계·removed·부기, 승계 청크가 부모 바이트와 같은지(직접 대조), 변경 청크 바이트,
               `.cbin` 계약 디코더 왕복(정점·인덱스 수 = manifest, 단위 법선 비율), 면 수(.cbin vs 기존 GLB), B1 면 수 회차별.
결과: $RUNS_DIR/02-r4/metrics/{time,equiv,chunks}.json
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "third_party" / "ai-ar-v3" / "contract" / "python"))
ASSETS = Path(os.environ["ASSETS_DIR"])
RUNS = Path(os.environ["RUNS_DIR"])
E2A_DAY = os.environ.get("E2A_DAY", "2026-10-25")
OUT = RUNS / "02-r4" / "metrics"
RUN_IDS = ("r1", "r2", "r3")
SEG7_ON = ("decode_mesh", "chunk_partition", "cbin_encode", "bookkeeping", "cbin_write")


def scenarios():
    ta = json.loads((ROOT / "scenarios" / "track_a.json").read_text())["scenarios"]
    tb = json.loads((ROOT / "scenarios" / "track_b.json").read_text())["scenarios"]
    out = []
    for sid, s in ta.items():
        a = s["asset"]
        out.append({"id": sid, "name": f"track_a-{sid}", "slat": ASSETS / "track_a" / a / "gen" / "slat.safetensors",
                    "mask": ASSETS / "track_a" / a / "masks" / f"{sid}.mask.json", "parent": ASSETS / "track_a" / a / "gen" / "chunks"})
    for bid in tb:
        out.append({"id": bid, "name": f"track_b-{bid}", "slat": ASSETS / "track_b" / bid / "slat.safetensors",
                    "mask": ASSETS / "track_b" / bid / "mask.json", "parent": RUNS / "02-r4" / "parents" / bid / "chunks"})
    return out


on_dir = lambda s, r: RUNS / "02-r4" / s["name"] / "r4-on" / r
off_dir = lambda s, r: RUNS / E2A_DAY / s["name"] / "e2a-c15" / r
on_json = lambda s, r: RUNS / "02-r4" / s["name"] / "r4-on" / f"{r}.json"
off_json = lambda s, r: RUNS / E2A_DAY / s["name"] / "e2a-c15" / f"{r}.json"


def ms(v):
    v = [x for x in v if x is not None]
    return [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0] if v else None


def time_phase():
    res = {}
    for s in scenarios():
        J = {k: [json.loads(f(s, r).read_text()) for r in RUN_IDS if f(s, r).is_file()] for k, f in (("on", on_json), ("off", off_json))}
        row = {}
        for k, runs in J.items():
            if not runs:
                continue
            seg = lambda j, n: j["segments"].get(n, {})
            row[k] = {"n": len(runs), "seg7": ms([seg(j, "7").get("seconds") for j in runs]),
                      "seg4_7": ms([sum(seg(j, n).get("seconds", 0) for n in ("4", "5", "6", "7")) for j in runs]),
                      "parts7": {p: ms([seg(j, "7").get("parts", {}).get(p) for j in runs])
                                 for p in sorted({p for j in runs for p in seg(j, "7").get("parts", {})})},
                      "alloc7_GiB": max(seg(j, "7").get("max_memory_allocated", 0) for j in runs) / 2 ** 30,
                      "rss_GiB": max(max((d.get("rss_bytes_max") or 0) for d in j["segments"].values()) for j in runs) / 2 ** 30}
            if k == "on":
                row[k]["glb_build"] = ms([seg(j, "r4_glb").get("seconds") for j in runs])
                row[k]["decoder_calls"] = [j.get("decoder_calls") for j in runs]
        res[s["id"]] = row
        if "on" in row and "off" in row:
            print(s["id"], "seg7 off %.1f → on %.1f" % (row["off"]["seg7"][0], row["on"]["seg7"][0]), row["on"]["decoder_calls"][0], flush=True)
    ok = [r for r in res.values() if "on" in r and "off" in r]
    if ok:
        res["_mean14"] = {k: {"seg7": float(np.mean([r[k]["seg7"][0] for r in ok])), "seg4_7": float(np.mean([r[k]["seg4_7"][0] for r in ok]))}
                          for k in ("on", "off")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "time.json").write_text(json.dumps(res, indent=1))


def equiv_phase():
    from reference.compare_slat_runs import compare
    from e14 import judge
    rows, ch, bs = {}, [], []
    for s in scenarios():
        on = {r: on_dir(s, r) / "slat_tgt.safetensors" for r in RUN_IDS if (on_dir(s, r) / "slat_tgt.safetensors").is_file()}
        off = {r: off_dir(s, r) / "slat_tgt.safetensors" for r in RUN_IDS}
        if not on:
            continue
        pick = lambda v: {"coords_equal": v["coords_equal"], "preserved_bit_equal": v["preserved"].get("bit_equal"),
                          "max": v["edited"].get("max_abs_diff"), "mean": v["edited"].get("mean_abs_diff")}
        c = {r: pick(compare(on[r], off[r], s["slat"], s["mask"])) for r in on}
        b = {f"{x}-{y}": pick(compare(off[x], off[y], s["slat"], s["mask"])) for x, y in itertools.combinations(RUN_IDS, 2)}
        rows[s["id"]] = {"change_base": c, "base_base": b,
                         "required_ok": all(v["coords_equal"] and v["preserved_bit_equal"] for v in c.values())}
        ch += list(c.values()); bs += list(b.values())
        print(s["id"], "coords/preserved", rows[s["id"]]["required_ok"], "mean", ["%.2e" % v["mean"] for v in c.values()], flush=True)
    res = {"rows": rows, "pooled": judge(ch, bs) if ch else None}
    if ch:
        print("E14 max", res["pooled"]["max"]["pass"], "mean", res["pooled"]["mean"]["pass"], "E15", {k: v["pass"] for k, v in res["pooled"]["e15"].items()})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "equiv.json").write_text(json.dumps(res, indent=1))


def chunks_phase():
    import trimesh
    import deltacontract as dc
    res = {}
    for s in scenarios():
        row = {}
        for r in RUN_IDS:
            d = on_dir(s, r)
            if not (d / "cbin" / "r4.json").is_file():
                continue
            j = json.loads((d / "cbin" / "r4.json").read_text())
            man = json.loads((d / "cbin" / "manifest.json").read_text())["chunks"]
            inh_ok = all((d / "cbin" / "chunks" / f"{k}.cbin").read_bytes() == (s["parent"] / f"{k}.cbin").read_bytes() for k in j["reused"])
            vc = ic = 0
            unit = []
            for k, e in man.items():
                m = dc.decode((d / "cbin" / "chunks" / f"{k}.cbin").read_bytes())
                vc += int(m.vertex_count == e["vertex_count"]); ic += int(m.index_count == e["index_count"])
                if m.normals is not None:
                    n = np.linalg.norm(np.asarray(m.normals, np.float32), axis=1)
                    unit.append(np.abs(n - 1) < 1e-3)
            faces_cbin = int(sum(e["index_count"] for e in man.values()) // 3)
            glb = trimesh.load(off_dir(s, r) / "output.glb", force="mesh", process=False)
            row[r] = {"chunks": j["chunks"], "sent": len(j["sent"]), "reused": len(j["reused"]), "removed": len(j["removed"]),
                      "book": len(j["book"]), "parent_chunks": len(list(s["parent"].glob("*.cbin"))),
                      "inherited_bytes_equal_parent": inh_ok, "sent_bytes": j["sent_bytes"], "total_bytes": j["total_bytes"],
                      "manifest_vertex_match": f"{vc}/{len(man)}", "manifest_index_match": f"{ic}/{len(man)}",
                      "unit_normal_ratio": float(np.concatenate(unit).mean()) if unit else None,
                      "faces_cbin": faces_cbin, "faces_mesh_decoder": j["mesh"]["faces"], "faces_glb_off": int(len(glb.faces)),
                      "face_ratio": faces_cbin / max(len(glb.faces), 1)}
        res[s["id"]] = row
        if row:
            r1 = row.get("r1") or next(iter(row.values()))
            print(s["id"], {k: r1[k] for k in ("chunks", "sent", "reused", "removed", "book", "inherited_bytes_equal_parent", "faces_cbin", "faces_glb_off")}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chunks.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"time": time_phase, "equiv": equiv_phase, "chunks": chunks_phase}[sys.argv[1]]()
