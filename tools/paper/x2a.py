"""X2-A P0 (ops/tasks/X-paper.md X2-A): 동결 14건 × 2회(사용자 23:18 축소, 원래 3회), 원저자 경로 `reference/run_reference.py --mode original --seed 0`
(150시점 재등록 + 25 스텝 + to_glb, 포크 무수정, `--reuse_render` 없음). 조건 이미지 = 동결본(cond/), 마스크 = 동결 셀.

포크는 입력 GLB 를 bbox 로 정규화(normalize_scene)해 인코딩하고 마스크 GLB 는 정규화 없이 복셀화한다(X3 P0 와 같음).
그래서 v0 모델에 그 정규화 T 를 미리 건 in.glb 와, 같은 T 를 건 마스크 GLB 를 넘긴다(포크 정규화가 항등이 된다).
마스크 GLB(v0 좌표): 트랙 A = masks/<ID>.mask.glb(s5_masks), 트랙 B = mask.json 셀의 노출면(raw = v0, 00s2 S-8).
T 는 시나리오 폴더 prep.json 에 남긴다. P0 출력 = T(v0) 좌표 = paper_metrics 의 N 좌표(같은 v0 bbox 변환).
P1 은 새로 돌리지 않는다(E2a c15 r1–r3 slat_tgt 재사용, X0 과 같은 출처).

    python tools/paper/x2a.py [시나리오 …]     # 결과 $RUNS_DIR/paper/x2a/p0/<ID>/r{1,2,3}/
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scenarios"), str(ROOT / "third_party" / "ai-ar-v3"),
                str(ROOT / "third_party" / "ai-ar-v3" / "contract" / "python")]
from tools.paper.x0 import scenarios  # noqa: E402
from tools.paper.x3 import PY, REF, affine, glb_bbox, sh, sha, transform_glb  # noqa: E402

RUNS = Path(os.environ["RUNS_DIR"])
OUT = RUNS / "paper" / "x2a" / "p0"
RUN_IDS = ("r1", "r2")          # 사용자 23:18 축소: P0 14건 × 2회(r3 없음)


def mask_glb_v0(sid, s):
    """시나리오 마스크 GLB(v0 좌표)."""
    if sid.startswith("A-"):
        return Path(str(s["mask"]).replace(".mask.json", ".mask.glb"))
    d = OUT / sid
    p = d / "mask_v0.glb"
    if not p.exists():
        import trimesh
        from s5_masks import boundary_mesh
        from server.pipeline.frames import VOXEL_TO_GLB
        cells = np.asarray(json.loads(Path(s["mask"]).read_text())["mask"]["voxels"], np.int64)
        v, f = boundary_mesh(cells)
        d.mkdir(parents=True, exist_ok=True)
        trimesh.Trimesh(VOXEL_TO_GLB.apply(v), f, process=True).export(p)
    return p


def prep(sid, s):
    d = OUT / sid
    if (d / "prep.json").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    lo, hi = glb_bbox(s["v0"])
    T = affine(1.0 / float((hi - lo).max()), (lo + hi) / 2)
    transform_glb(s["v0"], T, d / "in.glb")
    mg = mask_glb_v0(sid, s)
    transform_glb(mg, T, d / "mask.glb")
    (d / "prep.json").write_text(json.dumps({"T": T.tolist(), "v0": str(s["v0"]), "mask_glb_v0": str(mg), "mask_json": str(s["mask"]),
                                             "cond": str(s["cond"]), "note": "T = 포크 normalize_scene 과 같은 v0 bbox 정규화"}, indent=1))
    return d


def run(sid, s, r):
    d = OUT / sid
    rd = d / r
    if (rd / "round.json").exists():
        return
    if rd.exists():
        shutil.rmtree(rd)
    rd.mkdir(parents=True)
    t0 = time.time()
    sh([PY, ROOT / "reference" / "run_reference.py", "--mode", "original", "--model", d / "in.glb", "--mask_glb", d / "mask.glb",
        "--mask_json", s["mask"], "--image_dir", s["cond"], "--asset", f"x2a-{sid}", "--run", r, "--seed", 0,
        "--runs_dir", rd / "ref", "--reference", REF], rd / "reference.log")
    work = next((rd / "ref").glob(f"*/x2a-{sid}/*/{r}.work"))
    rj = next((rd / "ref").glob(f"*/x2a-{sid}/*/{r}.json"), None)
    shutil.copy(work / "output.glb", rd / "output.glb")
    shutil.copy(work / "slat_tgt.safetensors", rd / "slat_tgt.safetensors")
    shutil.rmtree(work / "render", ignore_errors=True)   # 150시점 렌더(크다)
    (rd / "view.glb").symlink_to("output.glb")
    checks = json.loads(rj.read_text()).get("checks") if rj else None
    (rd / "round.json").write_text(json.dumps({"path": "P0", "scenario": sid, "run": r, "seed": 0, "checks": checks,
                                               "files_sha256": {"output.glb": sha(rd / "output.glb")}, "wall_s": time.time() - t0}, indent=1))
    print(f"x2a p0 {sid} {r} done {time.time() - t0:.0f}s", flush=True)


def main(ids):
    S = scenarios()
    ids = ids or list(S)
    for sid in ids:
        prep(sid, S[sid])
    for r in RUN_IDS:              # 회차 우선(r1 14건 → r2 → r3): 중간에도 3회 중 1회분이 다 모인다
        for sid in ids:
            run(sid, S[sid], r)
    print("X2A_P0_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
