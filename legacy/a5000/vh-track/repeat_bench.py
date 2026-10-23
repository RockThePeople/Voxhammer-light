"""반복 측정 하네스 — Baseline 과 Ours 를 **같은 계측**으로 돌린다 (W41 기준선 재수립).

왜 있나
-------
`run_voxhammer_checked.sh` 는 Baseline 을 `inference.py` 로 돌리는데 두 가지가 안 된다:
  · 시드 고정 — 편집 경로의 유일한 RNG 추첨(`edit_pipeline.py:479` torch.randn)이 안 잠긴다
  · SLat 포획 — Baseline 에는 포획 훅이 없어 산출 SLat 을 못 본다 (PIPELINE §4)
둘 다 없으면 "같은 자산·같은 시드로 돌려도 다른가"(분산의 출처 ②)를 못 잰다.
⇒ 이 하네스가 두 팔을 **파이썬에서 직접** 부르고 시드·포획을 바깥에서 건다.
   **VoxHammer 소스는 안 고친다.**

`run_voxhammer_checked.sh` 의 D10 검사 중 산출에 영향 있는 둘은 여기서 재현한다:
  · 삼킨 예외 마커("Error happened during processing.") 검출 — 호출자가 로그에서 grep
  · features.npz 재생성 확인 — mtime 을 전후로 찍어 track.json 에 남긴다
"""
import argparse
import hashlib
import json
import os
import sys
import time

VOXHAMMER = os.environ.get("VOXHAMMER_DIR", "<HOME>/ai-ar-v2/VoxHammer")
os.chdir(VOXHAMMER)
sys.path.insert(0, VOXHAMMER)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ATTN_BACKEND", "flash-attn")
os.environ.setdefault("SPCONV_ALGO", "native")

import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import trimesh                                                 # noqa: E402

T = {}


def mark(name, t0):
    T[name] = round(time.time() - t0, 2)
    print(f"[단계] {name:22s} {T[name]:8.2f}s", flush=True)
    return time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["baseline", "ours"])
    ap.add_argument("--model", required=True, help="원본 model.glb (baseline 입력)")
    ap.add_argument("--mask_glb", help="baseline 용 mask.glb")
    ap.add_argument("--slat", help="ours 용 생성 SLat")
    ap.add_argument("--mask_ply", help="ours 용 voxels_delete.ply")
    # W42: 점유 셀로 정의한 마스크. 프리미티브가 돌출부(당근 코·손)를 반경 밖으로 흘리는 것을
    #      고친다. 주면 **두 팔 모두** 이 목록을 쓴다 — Baseline 은 edit_params 로 전달되어
    #      STEP3 산출(voxels_delete.ply)이 무시된다. STEP3 는 그대로 돌아 시간 구조가 유지된다.
    ap.add_argument("--mask_json", help="복셀 목록 JSON (N,3). 주면 mask_ply 보다 우선한다")
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    out_glb = os.path.join(a.out_dir, "output.glb")
    out_slat = os.path.join(a.out_dir, "slat.safetensors")
    render_dir = os.path.join(a.out_dir, "render")     # 🔴 런마다 새로 — STEP1 캐시를 피한다

    t0 = time.time()
    from trellis.pipelines import TrellisImageTo3DPipeline
    from slat_capture import capture_slat
    t = mark("import", t0)

    pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipe.cuda()
    t = mark("pipeline_load", t)

    # 🔴 시드는 래퍼 바깥에서 건다. VoxHammer 의 `# torch.manual_seed(0)` 은 주석 처리돼 있고
    #    편집 경로의 RNG 추첨은 sample_slat_denoise 의 초기 노이즈 하나뿐이다.
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    feat_npz = os.path.join(render_dir, "features.npz")
    pre_mtime = os.path.getmtime(feat_npz) if os.path.exists(feat_npz) else 0

    cells = None
    if a.mask_json:
        cells = np.asarray(json.load(open(a.mask_json)), dtype=np.int32)
        print(f"  마스크 셀(JSON) {len(cells)}", flush=True)

    with capture_slat(pipe, out_slat) as cap:
        if a.arm == "baseline":
            from inference import run_complete_pipeline
            run_complete_pipeline(
                pipeline=pipe, input_model_path=a.model, mask_glb_path=a.mask_glb,
                render_dir=render_dir, output_path=out_glb, image_dir=a.image_dir,
                is_text=False, source_prompt="", target_prompt="",
                edit_params=({"mask_cells": cells} if cells is not None else None))
        else:
            from voxhammer.edit_pipeline import run_edit
            if cells is None:
                import utils3d
                pos = utils3d.io.read_ply(a.mask_ply)[0]
                cells = np.floor((np.asarray(pos) + 0.5) * 64).astype(np.int32)
                print(f"  마스크 복셀(PLY) {len(cells)}", flush=True)
            # 🔴 SLat 재사용 경로는 extract_feature 를 임포트하지 않아 전역 grad 가 켜져 있다
            #    → to_glb 가 detach 없이 죽는다 (PIPELINE §5 함정 3)
            with torch.no_grad():
                run_edit(pipe, render_dir=None, output_path=out_glb, image_dir=a.image_dir,
                         is_text=False, source_prompt="", target_prompt="",
                         slat_path=a.slat, mask_cells=cells)
    assert cap.n_calls == 1, f"decode_slat {cap.n_calls}회"
    t = mark("edit_total", t)

    T["wall"] = round(time.time() - t0, 2)
    T["gpu_max_alloc_GiB"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
    T["gpu_max_reserved_GiB"] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)
    post_mtime = os.path.getmtime(feat_npz) if os.path.exists(feat_npz) else 0
    T["features_regenerated"] = bool(post_mtime > pre_mtime) if a.arm == "baseline" else None
    T["slat_sha256"] = hashlib.sha256(open(out_slat, "rb").read()).hexdigest()
    g = trimesh.load(out_glb)
    g = g.to_geometry() if hasattr(g, "to_geometry") else g
    T["vertices"] = int(len(g.vertices))
    T["faces"] = int(len(g.faces))
    T["slat_cells"] = int(cap.shape["coords"][0])
    json.dump(T, open(os.path.join(a.out_dir, "timing.json"), "w"), indent=2)
    print(f"[단계] {'WALL':22s} {T['wall']:8.2f}s", flush=True)
    print(f"DONE {a.arm} 셀 {T['slat_cells']} 정점 {T['vertices']} "
          f"sha {T['slat_sha256'][:16]} GPU {T['gpu_max_alloc_GiB']}GiB", flush=True)


if __name__ == "__main__":
    main()
