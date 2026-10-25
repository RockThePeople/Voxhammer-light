"""2번 R4 R-3a: DS(E16)로 Stage 2 비트 비교 — 한 프로세스 안에서 시나리오마다 R4 끔 1회 → R4 켬 1회.

- `VHL_DS_DETERMINISTIC=1` 과 같은 경로(`editcore.models.apply_ds`)로 Stage 2 SparseDownsample 을 결정적으로 바꾼다.
  운영 기본값(DS 꺼짐)은 바꾸지 않는다 — 이 스크립트 안에서만 켠다.
- 프로세스가 다르면 spconv `SparseConv3d` 때문에 갈릴 수 있으므로(DS 보고서 §3-4) 끔·켬을 반드시 같은 프로세스에서 돈다.
- 편집은 editcore 기준(15/15, E20). 끔 = `decode_to_glb`(gaussian+mesh+to_glb), 켬 = `decode_to_cbin`(mesh 만 → .cbin).
- 비교: slat_tgt(coords·feats 전체), Stage 1 noise·잠재·K/V, voxel_tgt, coords_tgt, Stage 2 잠재·K/V(다이제스트), 출발 잡음.
사용: r4_ds_bits.py <out.json>   (결과 $RUNS_DIR/02-r4/metrics/ds_bits.json 권장)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests" / "gpu"))
import editcore.models as em  # noqa: E402

import torch  # noqa: E402
from run_stage_equivalence import digest, tsha  # noqa: E402

ASSETS = Path(os.environ["ASSETS_DIR"])
RUNS = Path(os.environ["RUNS_DIR"])


def scenarios():
    ta = json.loads((REPO / "scenarios" / "track_a.json").read_text())["scenarios"]
    tb = json.loads((REPO / "scenarios" / "track_b.json").read_text())["scenarios"]
    out = []
    for sid, s in ta.items():
        a = s["asset"]
        out.append({"id": sid, "slat": ASSETS / "track_a" / a / "gen" / "slat.safetensors", "mask": ASSETS / "track_a" / a / "masks" / f"{sid}.mask.json",
                    "images": ASSETS / "track_a" / a / "cond" / sid, "parent": ASSETS / "track_a" / a / "gen" / "chunks"})
    for bid in tb:
        out.append({"id": bid, "slat": ASSETS / "track_b" / bid / "slat.safetensors", "mask": ASSETS / "track_b" / bid / "mask.json",
                    "images": ASSETS / "track_b" / bid / "cond", "parent": RUNS / "02-r4" / "parents" / bid / "chunks"})
    return out


def run_once(pipe, C, s, r4: bool, tmp: Path):
    """editcore 편집 1회(기준 15/15) + 토글에 맞는 디코딩. 단계 출력을 다이제스트로 잡는다."""
    from editcore.export import decode_to_cbin, decode_to_glb
    from editcore.pipeline import Editor
    from editcore.service import load_request
    ed = Editor(pipe, C)
    cap = {"steps": dict(ed.steps)}
    o1i, o1d, o2i, o2d = ed.stage1_inverse, ed.stage1_denoise, ed.stage2_inverse, ed.stage2_denoise

    def s1i(*a, **k):
        noise, lat, kv = o1i(*a, **k)
        cap["s1_noise"] = tsha(noise); cap["s1_latent"] = digest(lat); cap["s1_kv"] = digest(kv)
        return noise, lat, kv

    def s1d(*a, **k):
        v = o1d(*a, **k); cap["voxel_tgt"] = tsha(v); return v

    def s2i(*a, **k):
        lat, kv = o2i(*a, **k); cap["s2_latent"] = digest(lat); cap["s2_kv"] = digest(kv); return lat, kv

    def s2d(cond_tgt, coords_tgt, *a, **k):
        gen = a[-1] if a and isinstance(a[-1], torch.Generator) else k["generator"]
        cap["start_noise"] = tsha(torch.randn(coords_tgt.shape[0], 8, generator=torch.Generator().set_state(gen.get_state())))
        cap["coords_tgt"] = tsha(coords_tgt)
        return o2d(cond_tgt, coords_tgt, *a, **k)

    ed.stage1_inverse, ed.stage1_denoise, ed.stage2_inverse, ed.stage2_denoise = s1i, s1d, s2i, s2d
    res = ed.edit(load_request(str(s["slat"]), str(s["mask"]), str(s["images"]), seed=0))
    st = res.slat_tgt
    cap["slat_tgt_coords"] = tsha(st.coords)
    cap["slat_tgt_feats"] = tsha(st.feats)
    cap["slat_tgt_n"] = int(st.coords.shape[0])
    if r4:
        from safetensors import safe_open
        with safe_open(str(s["slat"]), "np") as f:
            pc = f.get_tensor("coords")[:, 1:]
        mask = json.loads(Path(s["mask"]).read_text())["mask"]["voxels"]
        r = decode_to_cbin(pipe, st, tmp / "cbin", mask_cells=mask, parent_dir=s["parent"], parent_cells=pc)
        cap["decode"] = {"path": "decode_to_cbin", "chunks": r["chunks"], "sent": len(r["sent"])}
    else:
        decode_to_glb(pipe, st, tmp / "off.glb")
        cap["decode"] = {"path": "decode_to_glb"}
    del res
    return cap


def compare(a, b):
    out = {}
    for k in ("s1_noise", "voxel_tgt", "start_noise", "coords_tgt", "slat_tgt_coords", "slat_tgt_feats"):
        out[k] = a[k] == b[k]
    for k in ("s1_latent", "s1_kv", "s2_latent", "s2_kv"):
        same = sum(1 for kk in a[k] if b[k].get(kk) == a[k][kk])
        out[k] = f"{same}/{len(a[k])}" + ("" if a[k].keys() == b[k].keys() else " (키 다름)")
    out["all_bit_equal"] = all(v is True or (isinstance(v, str) and v.split("/")[0] == v.split("/")[1].split(" ")[0] and "키" not in v)
                               for v in out.values())
    return out


def main():
    out_path = Path(sys.argv[1])
    assert not em.ds_enabled() and not em.r4_enabled(), "이 스크립트가 토글을 직접 다룬다 — 환경변수를 설정하지 않는다"
    pipe = em.load_pipeline()
    names, restore = em.apply_ds(pipe)   # 이 프로세스 안에서만 DS 켬
    print(f"[ds] slat_flow_model SparseDownsample {len(names)}개 교체 {names}", flush=True)
    C = em.derive_constants(pipe)
    from editcore.blocks import DeterministicSparseDownsample as D
    res = json.loads(out_path.read_text()) if out_path.is_file() else {}
    for s in scenarios():
        if s["id"] in res:
            continue
        with tempfile.TemporaryDirectory() as td:
            c0 = D.calls
            off = run_once(pipe, C, s, False, Path(td))
            c1 = D.calls
            on = run_once(pipe, C, s, True, Path(td))
            c2 = D.calls
        r = compare(off, on)
        r.update({"steps": off["steps"], "slat_tgt_cells": off["slat_tgt_n"], "ds_calls_off": c1 - c0, "ds_calls_on": c2 - c1,
                  "decode_off": off["decode"], "decode_on": on["decode"]})
        res[s["id"]] = r
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=1))
        print(s["id"], r, flush=True)
        torch.cuda.empty_cache()   # 계측 구간 밖(검증 스크립트)
    restore()
    print("ALL", all(v["all_bit_equal"] for v in res.values()), len(res))


if __name__ == "__main__":
    main()
