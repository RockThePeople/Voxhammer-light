"""X2-B Edit3D-Bench 3건 (사용자 23:10 축소, 지시서 main 82da407).

선정: 데이터셋 목록(`Edit3D-Bench/data/metadata.json`) 순서대로 GSO 앞 2건 + PartObjaverse-Tiny 앞 1건, 모델마다 prompt_1.
트랙 B 8건과 (데이터셋, 모델)이 겹치면 다음 건으로 넘어간다. 목록 = scenarios/x2b_list.json(커밋).
준비는 트랙 B(00s2 F-3, scenarios/f3_track_b.py)와 같은 절차를 그 함수 그대로 쓴다 — 출력만 $ASSETS_DIR/x2b/<ID>/:
  encode : 원 VoxHammer 인코딩 1회(포크 150시점 렌더 → DINOv2 특징 → feats_to_slat) = P1 최초 SLat
  cond   : 마스크 = 3d_edit_region.glb → 포크 process_delete_ply(64³, S-4 캐시) / 조건 = 데이터셋 2D(편집 시점 IoU ≥ 0.9) 또는
           포크 렌더 + bf16 FLUX(seed 0)
  parent : 최초 SLat → R4 .cbin(부모 없음) = P1 부모 청크
P0 1회: X2-A 와 같은 원저자 경로(`run_reference --mode original --seed 0`, 입력·마스크 GLB 에 같은 정규화 T).
P1 1회: editcore CLI `VHL_R4=1 VHL_R4_STITCH=1`(봉합 채택, chief 22:4x), seed 0, 부모 = parent.
FID·FVD·TOST 는 하지 않는다(건별 지표만).

    python tools/paper/x2b.py list | encode | cond | parent | p0 | p1
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scenarios"), str(ROOT / "third_party" / "ai-ar-v3"),
                str(ROOT / "third_party" / "ai-ar-v3" / "contract" / "python")]
ASSETS = Path(os.environ["ASSETS_DIR"])
RUNS = Path(os.environ["RUNS_DIR"])
BENCH = ASSETS / "Edit3D-Bench" / "data"
LIST = ROOT / "scenarios" / "x2b_list.json"
OUT_A = ASSETS / "x2b"
OUT = RUNS / "paper" / "x2b"


def make_list():
    meta = json.loads((BENCH / "metadata.json").read_text())
    tb = json.loads((ROOT / "scenarios" / "track_b.json").read_text())["scenarios"]
    used = {(s["dataset"], s["object"]) for s in tb.values()}
    out, skipped = [], []
    for ds, n in (("GSO", 2), ("PartObjaverse-Tiny", 1)):
        got = 0
        for m in meta:
            if m["dataset"] != ds or got == n:
                continue
            if (ds, m["source_model"]) in used:
                skipped.append({"dataset": ds, "object": m["source_model"], "reason": "트랙 B 와 겹침"})
                continue
            got += 1
            out.append({"id": f"X2B-{len(out) + 1}", "dataset": ds, "object": m["source_model"], "prompt_id": 1,
                        "prompt": m["prompt_1"], "source_prompt": m["source_prompt"]})
    LIST.write_text(json.dumps({"note": "사용자 23:10 — 목록 순서, GSO 2 + PartObjaverse-Tiny 1, prompt_1, 트랙 B 와 겹치면 다음 건",
                                "source": "Edit3D-Bench/data/metadata.json", "skipped": skipped, "scenarios": out},
                               ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=1))


def items():
    return json.loads(LIST.read_text())["scenarios"]


def f3():
    """트랙 B 함수를 X2-B 선정·출력으로 쓴다."""
    import f3_track_b as F
    F.OUT = OUT_A
    F.selection = lambda: [(s["id"], {**s, "prompt_id": str(s["prompt_id"])}) for s in items()]
    return F


def parent():
    import torch
    from safetensors import safe_open
    import editcore.models as em
    import trellis.modules.sparse as sp
    from editcore.export import decode_to_cbin
    pipe = em.load_pipeline()
    for s in items():
        d = OUT_A / s["id"]
        if (d / "parent" / "r4.json").exists():
            continue
        with safe_open(str(d / "slat.safetensors"), "pt") as f:
            slat = sp.SparseTensor(feats=f.get_tensor("feats").float().cuda(), coords=f.get_tensor("coords").int().cuda())
        with torch.inference_mode():
            r = decode_to_cbin(pipe, slat, d / "parent", glb_path=d / "parent" / "output.glb")
        print("x2b parent", s["id"], "chunks", r["chunks"], flush=True)
        del slat
        torch.cuda.empty_cache()   # 준비 스크립트(계측 구간 밖)


def p0():
    import numpy as np
    import trimesh
    from s5_masks import boundary_mesh
    from server.pipeline.frames import VOXEL_TO_GLB
    from tools.paper.x3 import PY, REF, affine, glb_bbox, sh, sha, transform_glb
    for s in items():
        a = OUT_A / s["id"]
        d = OUT / "p0" / s["id"]
        rd = d / "r1"
        if (rd / "round.json").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        src = BENCH / s["dataset"] / s["object"] / "source_model" / "model.glb"
        lo, hi = glb_bbox(src)
        T = affine(1.0 / float((hi - lo).max()), (lo + hi) / 2)
        transform_glb(src, T, d / "in.glb")
        cells = np.asarray(json.loads((a / "mask.json").read_text())["mask"]["voxels"], np.int64)
        v, f = boundary_mesh(cells)
        trimesh.Trimesh(VOXEL_TO_GLB.apply(v), f, process=True).export(d / "mask_v0.glb")   # 트랙 B 와 같이 raw = v0
        transform_glb(d / "mask_v0.glb", T, d / "mask.glb")
        (d / "prep.json").write_text(json.dumps({"T": T.tolist(), "v0": str(src)}, indent=1))
        if rd.exists():
            shutil.rmtree(rd)
        rd.mkdir(parents=True)
        t0 = time.time()
        sh([PY, ROOT / "reference" / "run_reference.py", "--mode", "original", "--model", d / "in.glb", "--mask_glb", d / "mask.glb",
            "--mask_json", a / "mask.json", "--image_dir", a / "cond", "--asset", f"x2b-{s['id']}", "--run", "r1", "--seed", 0,
            "--runs_dir", rd / "ref", "--reference", REF], rd / "reference.log")
        work = next((rd / "ref").glob(f"*/x2b-{s['id']}/*/r1.work"))
        shutil.copy(work / "output.glb", rd / "output.glb")
        shutil.copy(work / "slat_tgt.safetensors", rd / "slat_tgt.safetensors")
        shutil.rmtree(work / "render", ignore_errors=True)
        (rd / "view.glb").symlink_to("output.glb")
        (rd / "round.json").write_text(json.dumps({"path": "P0", "scenario": s["id"], "seed": 0, "wall_s": time.time() - t0,
                                                   "files_sha256": {"output.glb": sha(rd / "output.glb")}}, indent=1))
        print(f"x2b p0 {s['id']} done {time.time() - t0:.0f}s", flush=True)


def p1():
    from tools.paper.x3 import ALLOC, PY, sh, sha
    from editcore.render import vertex_color_render_glb
    for s in items():
        a = OUT_A / s["id"]
        rd = OUT / "p1" / s["id"] / "r1"
        if (rd / "round.json").exists():
            continue
        rd.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        sh([PY, "-m", "editcore.cli", "--slat", a / "slat.safetensors", "--mask_json", a / "mask.json", "--image_dir", a / "cond",
            "--out", rd / "output.glb", "--slat_out", rd / "slat_tgt.safetensors", "--seed", 0,
            "--parent_cbin", a / "parent" / "chunks", "--cbin_out", rd / "cbin"], rd / "edit.log",
           env={"VHL_R4": "1", "VHL_R4_STITCH": "1", "HF_HUB_OFFLINE": "1", **ALLOC})
        (rd / "view.glb").write_bytes(vertex_color_render_glb((rd / "output.glb").read_bytes()))
        r4 = json.loads((rd / "cbin" / "r4.json").read_text())
        (rd / "round.json").write_text(json.dumps({"path": "P1", "scenario": s["id"], "seed": 0, "wall_s": time.time() - t0,
                                                   "stitch": (r4.get("seam") or {}).get("stitch"),
                                                   "files_sha256": {"output.glb": sha(rd / "output.glb")}}, indent=1))
        print(f"x2b p1 {s['id']} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":
        make_list()
    elif cmd in ("encode", "cond"):
        getattr(f3(), cmd)()
    else:
        {"parent": parent, "p0": p0, "p1": p1}[cmd]()
