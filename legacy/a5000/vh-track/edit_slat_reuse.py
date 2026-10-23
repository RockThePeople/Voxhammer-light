"""SLat 재사용 편집 경로 — 생성 때 만든 SLat 을 그대로 넣어 STEP1/2/3 을 없앤다.

기존 경로가 하는 일 (실측 297s):
    STEP1  GLB → 150뷰 렌더 (bpy CYCLES)                  83.3s
    STEP2  150뷰 → dinov2 → features.npz                  24.4s
    STEP3  mask.glb → bpy → 262,144 복셀 부피검사          38.0s
    STEP4  역추적 + 디코드 + to_glb                       110.2s
    (+ 모델 적재 36.0s)

`run_edit` 이 render_dir 에서 읽는 것은 정확히 셋뿐이고(코드 확인),
    voxels.ply       → coords_src
    features.npz     → feats_to_slat() → slat_src      ← 유일한 소비처
    voxels_delete.ply → coords_delete
셋 다 우리가 이미 갖고 있는 것으로 대체된다:
    coords_src, slat_src ← 생성 SLat (slat.safetensors)
    coords_delete        ← 마스크 복셀 목록

🔴 두 가지를 코드로 확인하고 들어간다:
  · 공간 — 생성 `sample_slat` 은 `slat*std+mean` 로 끝나고(denormalized),
    `sample_slat_inverse` 는 `(slat-mean)/std` 로 시작한다. 같은 공간이다.
  · 프레임 — slat 좌표가 상위 계약의 정본 격자다(`grid_source="slat_coords"`). 항등이다.
    v3-moto-b v1 에서 `mask_fingerprint` 로 대조해 확인했다 (b8a621ab…).

⚠️ 이 경로는 **더 나은 것**이지 등가가 아니다. 기존 경로는 왕복에서 점유를 깎는다
   (실측 t1s-snowman: 13,460 → 11,467 · 손실 21.2% · IoU 0.741).
   판정은 "같은가"가 아니라 **"어느 쪽이 부모와 정합한가"** 다.
"""
import argparse
import json
import os
import sys
import time

VOXHAMMER = os.environ.get("VOXHAMMER_DIR", "<HOME>/ai-ar-v2/VoxHammer")
os.chdir(VOXHAMMER)
sys.path.insert(0, VOXHAMMER)
sys.path.insert(0, "<HOME>/ai-ar-v3/tools")
os.environ.setdefault("ATTN_BACKEND", "flash-attn")
os.environ.setdefault("SPCONV_ALGO", "native")

import numpy as np                                             # noqa: E402
import trimesh                                                 # noqa: E402

T = {}


def mark(name, t0):
    T[name] = round(time.time() - t0, 2)
    print(f"[단계] {name:26s} {T[name]:8.2f}s", flush=True)
    return time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slat", required=True, help="생성 SLat (slat.safetensors)")
    ap.add_argument("--mask_ply", help="마스크 복셀 (voxels_delete.ply). --mask_json 과 택일")
    ap.add_argument("--mask_json", help="마스크 복셀 목록 JSON (N,3)")
    ap.add_argument("--image_dir", required=True, help="2d_render/2d_mask/2d_edit")
    ap.add_argument("--output", required=True)
    ap.add_argument("--timing", help="단계별 시간 JSON 출력 경로")
    ap.add_argument("--emit_slat", help="편집된 SLat 을 여기에 저장한다 (다음 편집의 소스)")
    a = ap.parse_args()

    t0 = time.time()
    from trellis.pipelines import TrellisImageTo3DPipeline
    from voxhammer.edit_pipeline import run_edit
    t = mark("import", t0)

    pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipe.cuda()
    t = mark("pipeline_load", t)

    if a.mask_json:
        cells = np.asarray(json.load(open(a.mask_json)), dtype=np.int32)
    else:
        import utils3d
        pos = utils3d.io.read_ply(a.mask_ply)[0]
        cells = np.floor((np.asarray(pos) + 0.5) * 64).astype(np.int32)
    print(f"  마스크 복셀 {len(cells)}", flush=True)
    t = mark("mask_load", t)

    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    # 🔴 no_grad 로 감싼다. `to_glb` 는 `mesh.vertices.cpu().numpy()` 를 detach 없이 부르는데,
    #    주변 grad 가 켜진 채로 decode_slat 이 돌면 정점에 grad_fn 이 붙어 죽는다(실측).
    #    run_edit 안의 `torch.set_grad_enabled(True)` 는 전역 플래그라 이 블록 안에서도
    #    그대로 켜지므로, 텍스처 베이킹은 영향받지 않는다.
    # 🔴 편집된 SLat 을 내보낸다. 이게 없으면 **편집을 이어 붙일 수 없다** —
    #    새 경로는 SLat 을 먹는데 SLat 을 안 내놓아서 연쇄가 구조적으로 불가능했다.
    #    (실측: t1s-edit-new 에 slat.safetensors 가 없다. 포획을 안 붙인 탓이다.)
    import torch as _torch
    from slat_capture import capture_slat
    _cm = capture_slat(pipe, a.emit_slat) if a.emit_slat else None
    with _torch.no_grad():
        if _cm is None:
            run_edit(pipe, render_dir=None, output_path=a.output, image_dir=a.image_dir,
                     is_text=False, source_prompt="", target_prompt="",
                     slat_path=a.slat, mask_cells=cells)
        else:
            with _cm as cap:
                run_edit(pipe, render_dir=None, output_path=a.output, image_dir=a.image_dir,
                         is_text=False, source_prompt="", target_prompt="",
                         slat_path=a.slat, mask_cells=cells)
            assert cap.n_calls == 1, f"decode_slat {cap.n_calls}회 — 마지막 것이 저장된다"
            print(f"[slat] {a.emit_slat}  {cap.shape}", flush=True)
    t = mark("edit_total", t)

    T["wall"] = round(time.time() - t0, 2)
    print(f"[단계] {'WALL':26s} {T['wall']:8.2f}s", flush=True)
    if a.timing:
        json.dump(T, open(a.timing, "w"), indent=2)
    g = trimesh.load(a.output)
    g = g.to_geometry() if hasattr(g, "to_geometry") else g
    print(f"DONE {a.output}  정점 {len(g.vertices)}  면 {len(g.faces)}")


if __name__ == "__main__":
    main()
