"""보조 DS 집계 (CPU). A = $RUNS_DIR/01a-ds/e2e/<이름>.json(비트 비교), B = $RUNS_DIR/01a-ds/on/<이름>/r{1,2,3}(토글 켬 이식 CLI, 25 스텝).

phase bits : 16건 × 5쌍(포크–포크, 이식–이식, 이식–포크 3쌍) 비트 동일 요약, 결정적 연산 호출 수.
phase dist : 분포 비교(E14·E15, tools/e14.py) — 켬–끔 = 켬(이식) rK vs F-5 rK(포크 끔, 00s2 동결 기준 3회),
             끔–끔 = F-5 3쌍. 14건. coords·보존 셀(필수)도 같이 본다.
phase time : 구간 6(Stage 2) 켬(B, 3회) vs 끔(0-1 이식 port r1–r3, 25 스텝, 깨끗한 측정) — 같은 장비.
결과: $RUNS_DIR/01a-ds/metrics/{bits,dist,time}.json
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
ASSETS = Path(os.environ["ASSETS_DIR"])
RUNS = Path(os.environ["RUNS_DIR"])
D = RUNS / "01a-ds"
OUT = D / "metrics"
RUN_IDS = ("r1", "r2", "r3")


def scenarios():
    ta = json.loads((ROOT / "scenarios" / "track_a.json").read_text())["scenarios"]
    tb = json.loads((ROOT / "scenarios" / "track_b.json").read_text())["scenarios"]
    out = []
    for sid, s in ta.items():
        a = s["asset"]
        out.append({"id": sid, "name": f"track_a-{sid}", "fp": s["mask"]["fingerprint"][:12],
                    "slat": ASSETS / "track_a" / a / "gen" / "slat.safetensors", "mask": ASSETS / "track_a" / a / "masks" / f"{sid}.mask.json"})
    for bid, s in tb.items():
        out.append({"id": bid, "name": f"track_b-{bid}", "fp": s["mask"]["fingerprint"][:12],
                    "slat": ASSETS / "track_b" / bid / "slat.safetensors", "mask": ASSETS / "track_b" / bid / "mask.json"})
    for f in sorted(RUNS.glob("2026-10-23/01ref-moto-b-*/w34-*")):   # moto-b: 끔 = 0-1 포크 기준 3회
        m = f.parent.name.replace("01ref-", "")
        out.append({"id": m, "name": m, "off_dir": f, "slat": Path.home() / "ai-ar-data/assets/v3-moto-b/v1/slat.safetensors",
                    "mask": ROOT / "third_party" / "ai-ar-v3" / "handoff" / "lasso" / f"moto-b.{m.replace('moto-b-', '')}.mask.json"})
    return out


def on_slats(s):
    """켬(이식) slat_tgt: A단계 저장본(on/<이름>/a) + B단계 CLI(on/<이름>/r1..r3) 중 있는 것. 켬은 결정적이라 서로 비트 동일이어야 한다."""
    d = D / "on" / s["name"]
    return {k: d / k / "slat_tgt.safetensors" for k in ("a", "r1", "r2", "r3") if (d / k / "slat_tgt.safetensors").is_file()}


def bits():
    res = {}
    for p in sorted((D / "e2e").glob("*.json")):
        j = json.loads(p.read_text())
        res[p.stem] = {"all_pairs_bit_equal": j["all_pairs_bit_equal"], "ds_calls": j["ds_calls"],
                       "pairs": {k: v["summary"] for k, v in j["pairs"].items()},
                       "s2_kv": {k: f"{v['detail']['s2_kv']['bit_equal']}/{v['detail']['s2_kv']['n']}" for k, v in j["pairs"].items()},
                       "s2_latent": {k: f"{v['detail']['s2_latent']['bit_equal']}/{v['detail']['s2_latent']['n']}" for k, v in j["pairs"].items()},
                       "edited_cells": j["pairs"]["port1-fork1"]["detail"].get("edited", {}).get("cells"),
                       "preserved_cells": j["pairs"]["port1-fork1"]["detail"].get("preserved", {}).get("cells")}
    res["_summary"] = {"scenarios": len(res), "all_bit_equal": all(v["all_pairs_bit_equal"] for v in res.values())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bits.json").write_text(json.dumps(res, indent=1))
    print(res["_summary"])


def dist():
    from reference.compare_slat_runs import compare
    from e14 import judge
    rows, ch, bs = {}, [], []
    pick = lambda v: {"coords_equal": v["coords_equal"], "preserved_bit_equal": v["preserved"].get("bit_equal"),
                      "max": v["edited"].get("max_abs_diff"), "mean": v["edited"].get("mean_abs_diff")}
    for s in scenarios():
        on = on_slats(s)
        offd = s.get("off_dir") or RUNS / "2026-10-22" / s["name"] / f"w34-{s['fp']}"
        f5 = {r: offd / f"{r}.work" / "slat_tgt.safetensors" for r in RUN_IDS}
        if not on:
            continue
        first = next(iter(on.values()))
        c = {r: pick(compare(first, f5[r], s["slat"], s["mask"])) for r in RUN_IDS}   # 켬 1개 × 끔 3회 = 켬–끔 3쌍
        b = {f"{x}-{y}": pick(compare(f5[x], f5[y], s["slat"], s["mask"])) for x, y in itertools.combinations(RUN_IDS, 2)}
        oo = {f"{x}-{y}": pick(compare(on[x], on[y], s["slat"], s["mask"])) for x, y in itertools.combinations(on, 2)}
        rows[s["id"]] = {"on_off": c, "off_off": b, "on_on": oo, "on_sources": list(on), "off_dir": str(offd),
                         "required_ok": all(v["coords_equal"] and v["preserved_bit_equal"] for v in c.values())}
        ch += list(c.values()); bs += list(b.values())
        print(s["id"], rows[s["id"]]["required_ok"], "on-on max", [v["max"] for v in oo.values()], flush=True)
    res = {"rows": rows, "pooled": judge(ch, bs) if ch else None}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dist.json").write_text(json.dumps(res, indent=1))
    if ch:
        print("E14 max", res["pooled"]["max"], "\nE14 mean", res["pooled"]["mean"], "\nE15", res["pooled"]["e15"])


def time():
    res = {}
    seg = lambda j, n: j["segments"].get(n, {}).get("seconds")
    for s in scenarios():
        on = [json.loads((D / "on" / s["name"] / f"{r}.json").read_text()) for r in RUN_IDS if (D / "on" / s["name"] / f"{r}.json").is_file()]
        if len(on) < 3:   # 시간은 장비가 빈 상태에서 3회 잰 건만(DS_TIME)
            continue
        off = [json.loads((RUNS / "01-port" / s["name"] / f"port-{r}.json").read_text()) for r in RUN_IDS]
        if not on:
            continue
        f = lambda v: [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0]
        res[s["id"]] = {"seg6_on": f([seg(j, "6") for j in on]), "seg6_off": f([seg(j, "6") for j in off]),
                        "seg4_on": f([seg(j, "4") for j in on]), "seg4_off": f([seg(j, "4") for j in off]),
                        "ds_calls": [j.get("ds", {}).get("deterministic_downsample_calls") for j in on],
                        "steps": [j.get("steps") for j in on]}
        print(s["id"], "seg6 off %.2f on %.2f" % (res[s["id"]]["seg6_off"][0], res[s["id"]]["seg6_on"][0]), res[s["id"]]["ds_calls"], flush=True)
    if res:
        res["_mean14"] = {k: float(np.mean([v[k][0] for v in res.values()])) for k in ("seg6_on", "seg6_off", "seg4_on", "seg4_off")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "time.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"bits": bits, "dist": dist, "time": time}[sys.argv[1]]()
