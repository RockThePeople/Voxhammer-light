"""00 A-6 moto-b feats SLat 확인 절차 1–5 (정오표 E1). 판정은 chief 가 한다 — 여기서는 수치만 낸다.

1. 파일 sha256·크기·메타 전체·coords/feats shape·dtype
2. coords(batch 열 제외)가 ai-ar-v3 `handoff/lasso/moto-b.slat_coords.npy` 와 정렬 후 완전 일치 (필수)
3. TRELLIS `slat_decoder_mesh` 로 2회 디코딩 (to_glb 후처리 없음, 디코더 native = 복셀 프레임 D9-b)
4. ai-ar-v3 `assets/moto-b/parent/*.cbin` 376청크와 면 수, 중복 제거 정점 수(계약 양자화
   `np.rint(pos * 2**20)`, chunkbin.py:115), Chamfer(양방향 평균·최대, 정규화 공간) 비교
5. chief 기준값과 나란히 적는다
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

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "third_party" / "ai-ar-v3"
sys.path.insert(0, str(V3 / "contract" / "python"))

from deltacontract import decode  # noqa: E402

QUANT = 2 ** 20
CHIEF = {"chunks": 376, "faces": 359220, "vertices_sum": 207537, "vertices_dedup": 179596,
         "voxel_count_sum": 9149}


def dedup_count(v):
    return int(len(np.unique(np.rint(np.asarray(v, np.float64) * QUANT).astype(np.int64), axis=0)))


def chamfer(a, b):
    from scipy.spatial import cKDTree
    da, _ = cKDTree(b).query(a, workers=-1)
    db, _ = cKDTree(a).query(b, workers=-1)
    return {"a_to_b_mean": float(da.mean()), "a_to_b_max": float(da.max()),
            "b_to_a_mean": float(db.mean()), "b_to_a_max": float(db.max()),
            "symmetric_mean": float((da.mean() + db.mean()) / 2), "n_a": int(len(a)), "n_b": int(len(b))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default=os.environ.get("REFERENCE_DIR", "build/reference-v2"))
    ap.add_argument("--no_decode", action="store_true")
    a = ap.parse_args()
    from safetensors import safe_open

    rec: dict = {"chief_reference": CHIEF}
    p = Path(a.slat)
    rec["1_file"] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
    with safe_open(str(p), "np") as f:
        meta = dict(f.metadata() or {})
        coords, feats = f.get_tensor("coords"), f.get_tensor("feats")
    rec["1_file"].update(metadata=meta, coords={"shape": list(coords.shape), "dtype": str(coords.dtype)},
                         feats={"shape": list(feats.shape), "dtype": str(feats.dtype)})

    ref = np.load(V3 / "handoff" / "lasso" / "moto-b.slat_coords.npy").astype(np.int64)
    c3 = coords[:, 1:].astype(np.int64) if coords.shape[1] == 4 else coords.astype(np.int64)
    so = lambda x: x[np.lexsort((x[:, 2], x[:, 1], x[:, 0]))]  # noqa: E731
    exact = bool(c3.shape == ref.shape and np.array_equal(so(c3), so(ref)))
    rec["2_coords_match"] = {"exact_after_sort": exact, "slat_n": int(len(c3)), "ref_n": int(len(ref)),
                             "batch_col_unique": np.unique(coords[:, 0]).tolist() if coords.shape[1] == 4 else None,
                             "ref_spec_job": json.loads((V3 / "handoff/lasso/moto-b.slat_coords_spec.json").read_text())["a5000_job_id"]}
    if not exact:
        Path(a.out).write_text(json.dumps(rec, indent=2))
        print(json.dumps(rec, indent=2))
        raise SystemExit("2 불일치 — 멈춘다")

    files = sorted((V3 / "assets" / "moto-b" / "parent").glob("*.cbin"))
    vs, n_idx, vox = [], 0, 0
    for fp in files:
        m = decode(fp.read_bytes())
        vs.append(np.asarray(m.positions, np.float64))
        n_idx += int(np.asarray(m.indices).size)
        vox += int(getattr(m, "voxel_count", 0) or 0)
    cb = np.concatenate(vs)
    rec["4_cbin"] = {"chunks": len(files), "faces": n_idx // 3, "vertices_sum": int(len(cb)),
                     "vertices_dedup": dedup_count(cb), "voxel_count_sum": vox}

    if not a.no_decode:
        REF = (ROOT / a.reference).resolve()
        sys.path.insert(0, str(REF))
        os.chdir(REF)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("SPCONV_ALGO", "native")
        import torch
        import trellis.modules.sparse as sp
        from trellis.pipelines import TrellisImageTo3DPipeline

        pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        dec = pipe.models["slat_decoder_mesh"].cuda().eval()
        ct = torch.as_tensor(coords).int()
        if ct.shape[1] == 3:
            ct = torch.cat([torch.zeros(len(ct), 1, dtype=ct.dtype), ct], 1)
        slat = sp.SparseTensor(feats=torch.as_tensor(feats).float().cuda(), coords=ct.cuda())
        runs = []
        for i in range(2):
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.no_grad():
                mesh = dec(slat)[0]
            torch.cuda.synchronize()
            v = mesh.vertices.detach().cpu().numpy().astype(np.float64)
            f = mesh.faces.detach().cpu().numpy()
            runs.append({"seconds": time.perf_counter() - t, "faces": int(len(f)), "vertices": int(len(v)),
                         "vertices_dedup": dedup_count(v), "_v": v})
        rec["3_decode"] = {k: [{kk: vv for kk, vv in r.items() if kk != "_v"} for r in runs] for k in ["runs"]}
        rec["3_decode"]["two_runs_same_counts"] = (runs[0]["faces"] == runs[1]["faces"]
                                                  and runs[0]["vertices"] == runs[1]["vertices"])
        rec["3_decode"]["two_runs_vertices_bit_equal"] = bool(
            runs[0]["_v"].shape == runs[1]["_v"].shape and np.array_equal(runs[0]["_v"], runs[1]["_v"]))
        dv = runs[0]["_v"]
        rec["3_decode"]["bbox"] = [dv.min(0).tolist(), dv.max(0).tolist()]
        rec["4_compare"] = {
            "faces": {"decoder": runs[0]["faces"], "cbin": rec["4_cbin"]["faces"]},
            "vertices_dedup": {"decoder": runs[0]["vertices_dedup"], "cbin": rec["4_cbin"]["vertices_dedup"]},
            "chamfer_decoder_vs_cbin": chamfer(np.unique(np.rint(dv * QUANT), axis=0) / QUANT,
                                               np.unique(np.rint(cb * QUANT), axis=0) / QUANT),
            "chamfer_points": "중복 제거(계약 양자화) 정점 집합끼리, scipy cKDTree 최근접, 단위=정규화 공간",
        }
    Path(a.out).write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
