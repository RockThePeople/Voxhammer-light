"""보조 DS D-3 3: 토글 켬 종단 비트 비교 — 포크 2회, 이식 2회, 이식–포크 3쌍.

- 포크: 파일은 고치지 않고, 이 스크립트 안에서만 TRELLIS `SparseDownsample.forward` 를 결정적 구현으로 바꿔 끼운다
  (정오표 E16 검증 한정 예외). 시드·입력·디코딩 전 정지는 0-1 `run_stage_equivalence.run_fork` 그대로.
- 이식: `editcore.models.apply_ds` 로 Stage 2 flow 모델의 모듈을 교체(VHL_DS_DETERMINISTIC=1 과 같은 경로).
- 비교: Stage 1 noise·잠재·K/V, voxel_tgt, coords, Stage 2 잠재·K/V(다이제스트), 출발 잡음, slat_tgt(보존·편집 셀).

사용: run_ds.py --slat S --mask_json M --image_dir I --out OUT.json
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests" / "gpu"))
import editcore.models as em  # noqa: E402

import torch  # noqa: E402
from run_stage_equivalence import compare, run_fork, run_port  # noqa: E402


def patch_fork():
    """TRELLIS 클래스의 forward 를 결정적 구현으로 바꿔 끼우고, 되돌리는 함수를 돌려준다(검증 스크립트 한정)."""
    import trellis.modules.sparse.spatial as S
    from editcore.blocks import DeterministicSparseDownsample
    orig = S.SparseDownsample.forward
    S.SparseDownsample.forward = DeterministicSparseDownsample.forward
    return lambda: setattr(S.SparseDownsample, "forward", orig)


def counted(fn, *a):
    from editcore.blocks import DeterministicSparseDownsample as D
    before = D.calls
    cap = fn(*a)
    cap["ds_calls"] = D.calls - before
    cap.setdefault("rng_state_at_denoise_equal_seed_state", None)
    gc.collect()
    return cap


def bit_summary(r):
    s2 = all(r[k]["keys_equal"] and r[k]["bit_equal"] == r[k]["n"] for k in ("s2_latent", "s2_kv"))
    s1 = all(r[k]["keys_equal"] and r[k]["bit_equal"] == r[k]["n"] for k in ("s1_latent", "s1_kv"))
    return {"stage1_cache_bit_equal": s1, "stage2_cache_bit_equal": s2, "coords_equal": r["slat_coords_equal"],
            "preserved_bit_equal": r.get("preserved", {}).get("bit_equal"),
            "edited_bit_equal": r.get("edited", {}).get("bit_equal"),
            "edited_max": r.get("edited", {}).get("max_abs"), "edited_mean": r.get("edited", {}).get("mean_abs"),
            "all_bit_equal": bool(s1 and s2 and r["slat_coords_equal"] and r.get("edited", {}).get("bit_equal")
                                  and r.get("preserved", {}).get("bit_equal") is not False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slat", required=True)
    ap.add_argument("--mask_json", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    paths = {"slat": str(Path(a.slat).resolve()), "mask": str(Path(a.mask_json).resolve()),
             "images": str(Path(a.image_dir).resolve())}
    out_path = Path(a.out).resolve()
    assert not em.ds_enabled(), "이 스크립트는 토글을 직접 다룬다 — VHL_DS_DETERMINISTIC 을 설정하지 않는다"
    pipe = em.load_pipeline()

    unpatch = patch_fork()
    print("[ds] fork: trellis SparseDownsample.forward 교체", flush=True)
    f1 = counted(run_fork, pipe, paths, a.seed)
    f2 = counted(run_fork, pipe, paths, a.seed)
    unpatch()

    names, restore = em.apply_ds(pipe)
    print(f"[ds] port: slat_flow_model 모듈 {len(names)}개 교체 {names}", flush=True)
    p1 = counted(run_port, pipe, paths, a.seed)
    p2 = counted(run_port, pipe, paths, a.seed)
    restore()

    from safetensors import safe_open
    with safe_open(paths["slat"], "np") as sf:
        src = {tuple(x) for x in sf.get_tensor("coords")[:, 1:].tolist()}
    mask = {tuple(x) for x in json.loads(Path(paths["mask"]).read_text())["mask"]["voxels"]}
    pres = src - mask
    pairs = {"fork1-fork2": (f1, f2), "port1-port2": (p1, p2),
             "port1-fork1": (p1, f1), "port2-fork2": (p2, f2), "port1-fork2": (p1, f2)}
    res = {"replaced_modules": names, "ds_calls": {k: c["ds_calls"] for k, c in
                                                    (("fork1", f1), ("fork2", f2), ("port1", p1), ("port2", p2))},
           "pairs": {}}
    for k, (x, y) in pairs.items():
        r = compare(x, y, pres)
        res["pairs"][k] = {"summary": bit_summary(r), "detail": r}
    res["all_pairs_bit_equal"] = all(v["summary"]["all_bit_equal"] for v in res["pairs"].values())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"ds_calls": res["ds_calls"], "all_pairs_bit_equal": res["all_pairs_bit_equal"],
                      "pairs": {k: v["summary"] for k, v in res["pairs"].items()}}, indent=1), flush=True)


if __name__ == "__main__":
    main()
