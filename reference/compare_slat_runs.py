"""동일 코드 2회 동등성 바닥값 — 두 회차의 `slat_tgt.safetensors`(R2) 비교.

- Stage 1 coords: 두 회차 좌표 집합이 같은가(완전 일치 기준).
- Stage 2 feats: 공통 좌표에서 max|Δ|·mean|Δ| 를 보존 셀/편집 셀로 나눠서.
  보존 셀 = 입력 SLat 좌표 − 마스크 셀(`run_edit` 의 coords_preserve 와 같은 정의, `edit_pipeline.py:612`),
  편집 셀 = 나머지 공통 좌표.
판정 기준(CLAUDE.md): 입력 SLat 고정 + 보존 셀 비트 동일 + 편집 셀은 같은 시드에서 max|Δ| ≤ 4.7e-2.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from safetensors import safe_open


def load(p):
    with safe_open(str(p), "np") as f:
        return f.get_tensor("coords"), f.get_tensor("feats")


def compare(a, b, src_slat, mask_json):
    ca, fa = load(a)
    cb, fb = load(b)
    cs, _ = load(src_slat)
    mask = {tuple(x) for x in json.load(open(mask_json))["mask"]["voxels"]}
    pres = {tuple(x) for x in cs[:, 1:].tolist()} - mask
    ka = {tuple(r[1:]): i for i, r in enumerate(ca.tolist())}
    kb = {tuple(r[1:]): i for i, r in enumerate(cb.tolist())}
    common = sorted(set(ka) & set(kb))
    out = {"coords_equal": set(ka) == set(kb), "n_a": len(ka), "n_b": len(kb), "common": len(common)}
    for name, keys in (("preserved", [k for k in common if k in pres]), ("edited", [k for k in common if k not in pres])):
        if keys:
            d = np.abs(fa[[ka[k] for k in keys]] - fb[[kb[k] for k in keys]])
            out[name] = {"cells": len(keys), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                         "bit_equal": bool(d.max() == 0)}
        else:
            out[name] = {"cells": 0}
    out["pass_4_7e-2"] = bool(out["coords_equal"] and out["preserved"].get("bit_equal", True)
                              and out["edited"].get("max_abs_diff", 0) <= 4.7e-2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--src_slat", required=True)
    ap.add_argument("--mask_json", required=True)
    a = ap.parse_args()
    print(json.dumps(compare(a.a, a.b, a.src_slat, a.mask_json), indent=1))
