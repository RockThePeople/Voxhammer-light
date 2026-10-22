"""00s S-4 트랙 B 후보표 — Edit3D-Bench 300건(100 객체 × prompt 1–3).

건마다:
- 편집 영역 셀: 포크 `voxhammer/delete_region_voxel.process_delete_ply(3d_edit_region.glb, dir)`
  (원저자 경로와 같은 bpy 임포트 + volume 필터, 64³) → `voxels_delete.ply` → 셀.
- 원본 점유 셀: 같은 bpy 임포트(`glb_to_ply`)로 `mesh.ply` 를 만든 뒤 포크
  `voxhammer/extract_feature.voxelize_mesh` 로 64³ 표면 복셀화(원저자 150뷰 렌더가 저장하는 mesh.ply
  와 같은 좌표계: Bench 모델은 이미 최대 변 1·원점 중심으로 정규화돼 있다).
- 마스크 비율 = 영역 셀 / 원본 점유 셀, 겹침 = |영역 ∩ 원본| / 원본.
- 편집 유형(추정): 영역 안 원본 점유가 영역의 5% 미만이면 "추가", 프롬프트에 without/remove/no 가 있으면
  "삭제", 나머지 "교체". 추정이므로 선정 때 육안으로 다시 본다.

병렬: `--jobs` 개 프로세스(각 프로세스가 bpy 를 따로 연다). 결과는 건별 JSON 캐시 후 CSV 로 모은다.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def one(job):
    bench, ref, ds, obj, k, cache = job
    out = Path(cache) / f"{ds}__{obj}__p{k}.json"
    if out.exists():
        return json.loads(out.read_text())
    code = r"""
import json, os, sys
import numpy as np
ref, bench, ds, obj, k, tmp = sys.argv[1:7]
sys.path.insert(0, ref); os.chdir(ref)
from voxhammer.delete_region_voxel import process_delete_ply, glb_to_ply
from voxhammer.extract_feature import voxelize_mesh
import utils3d
base = os.path.join(bench, "data", ds, obj)
region = os.path.join(base, f"prompt_{k}", "3d_edit_region.glb")
process_delete_ply(region, tmp)
pos = utils3d.io.read_ply(os.path.join(tmp, "voxels_delete.ply"))[0]
rc = np.unique(np.floor((np.asarray(pos) + 0.5) * 64).astype(np.int64), axis=0) if pos is not None and len(pos) else np.zeros((0, 3), np.int64)
glb_to_ply(os.path.join(base, "source_model", "model.glb"), os.path.join(tmp, "mesh.ply"))
voxelize_mesh(tmp)
spos = utils3d.io.read_ply(os.path.join(tmp, "voxels.ply"))[0]
sc = np.unique(np.floor((np.asarray(spos) + 0.5) * 64).astype(np.int64), axis=0)
np.save(os.path.join(tmp, "region_cells.npy"), rc)
print("RESULT " + json.dumps({"region_cells": int(len(rc)), "source_cells": int(len(sc)),
      "overlap": int(len({tuple(x) for x in rc.tolist()} & {tuple(x) for x in sc.tolist()}))}))
"""
    with tempfile.TemporaryDirectory() as tmp:
        p = subprocess.run([sys.executable, "-c", code, ref, bench, ds, obj, str(k), tmp],
                           capture_output=True, text=True, timeout=1800)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            return {"dataset": ds, "object": obj, "prompt_id": k, "error": (p.stderr or p.stdout)[-400:]}
        r = json.loads(line[0][7:])
        cells = Path(cache) / f"{ds}__{obj}__p{k}.cells.npy"
        import shutil
        shutil.copy(Path(tmp) / "region_cells.npy", cells)
    r.update(dataset=ds, object=obj, prompt_id=k)
    out.write_text(json.dumps(r))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=os.path.join(os.environ.get("ASSETS_DIR", ""), "Edit3D-Bench"))
    ap.add_argument("--reference", default=str(ROOT.parent / "Voxhammer-light" / "build" / "reference-v2"))
    ap.add_argument("--cache", default=os.path.join(os.environ.get("RUNS_DIR", "runs"), "track_b_cells"))
    ap.add_argument("--out", default=str(ROOT / "scenarios" / "track_b_candidates.csv"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    meta = json.loads((Path(a.bench) / "data" / "metadata.json").read_text())
    jobs, prompts = [], {}
    for m in meta:
        for k in (1, 2, 3):
            prompts[(m["dataset"], m["source_model"], k)] = (m["source_prompt"], m[f"prompt_{k}"])
            jobs.append((a.bench, str(Path(a.reference).resolve()), m["dataset"], m["source_model"], k, a.cache))
    if a.limit:
        jobs = jobs[: a.limit]
    rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, f in enumerate(as_completed([ex.submit(one, j) for j in jobs]), 1):
            r = f.result()
            rows.append(r)
            print(f"[{i}/{len(jobs)}] {r.get('dataset')} {r.get('object')} p{r.get('prompt_id')} "
                  f"{'ERR' if 'error' in r else (r['region_cells'], r['source_cells'])}", flush=True)
    out_rows = []
    for r in sorted(rows, key=lambda r: (r["dataset"], r["object"], r["prompt_id"])):
        sp, tp = prompts[(r["dataset"], r["object"], r["prompt_id"])]
        if "error" in r:
            out_rows.append({"dataset": r["dataset"], "object": r["object"], "prompt_id": r["prompt_id"],
                             "source_prompt": sp, "prompt": tp, "error": r["error"].replace("\n", " ")[:200]})
            continue
        ratio = r["region_cells"] / r["source_cells"] if r["source_cells"] else None
        inside = r["overlap"] / r["region_cells"] if r["region_cells"] else 0.0
        low = tp.lower()
        kind = ("삭제" if any(w in low.split() for w in ("without", "remove", "no")) else
                "추가" if inside < 0.05 else "교체")
        out_rows.append({"dataset": r["dataset"], "object": r["object"], "prompt_id": r["prompt_id"],
                         "source_prompt": sp, "prompt": tp, "region_cells": r["region_cells"],
                         "source_cells": r["source_cells"], "overlap_cells": r["overlap"],
                         "mask_ratio": round(ratio, 4) if ratio is not None else "",
                         "region_occupied_fraction": round(inside, 4), "edit_type_guess": kind, "error": ""})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "object", "prompt_id", "source_prompt", "prompt",
                                          "region_cells", "source_cells", "overlap_cells", "mask_ratio",
                                          "region_occupied_fraction", "edit_type_guess", "error"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {a.out} ({len(out_rows)} rows, errors {sum(1 for r in out_rows if r['error'])})")


if __name__ == "__main__":
    main()
