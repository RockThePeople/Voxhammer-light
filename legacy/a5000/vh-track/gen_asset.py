"""입력 이미지 하나로 TRELLIS 자산(GLB)을 만든다 — VoxHammer 단독 트랙용.

이 트랙은 상위 프로젝트의 청크·델타·계약을 안 쓴다. 그래서 오케스트레이터를 거치지 않고
TRELLIS 파이프라인을 직접 부른다. 만드는 것은 `model.glb` · `input.png` · `gen_stats.json`
셋뿐이고, 편집(`inference.py`)이 요구하는 것도 `model.glb` 하나다.

⚠️ 배경 제거는 `preprocess_image=True` 로 **TRELLIS 내부 u2net** 이 한다.
   3090 이 붙여 준 BiRefNet 알파 수치와 다르게 나올 수 있다 — 실측된 사실이다
   (같은 이미지에서 BiRefNet 0.2% vs u2net 99.8%). 그래서 여기서 **다시 재고** 보고한다.

시드는 42 로 고정한다 (t1·t2·t3 자산이 전부 42 였다 — 자산끼리 비교 가능해야 한다).
"""
import argparse
import json
import os
import sys
import time

VOXHAMMER = os.environ.get("VOXHAMMER_DIR", "<HOME>/ai-ar-v2/VoxHammer")
os.chdir(VOXHAMMER)
sys.path.insert(0, VOXHAMMER)

os.environ.setdefault("ATTN_BACKEND", "flash-attn")
os.environ.setdefault("SPCONV_ALGO", "native")

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
from PIL import Image                                 # noqa: E402
from scipy import ndimage                             # noqa: E402

from trellis.pipelines import TrellisImageTo3DPipeline  # noqa: E402
from trellis.utils import postprocessing_utils          # noqa: E402

sys.path.insert(0, "<HOME>/ai-ar-v3/tools")
from slat_capture import capture_slat                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    t0 = time.time()
    pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipe.cuda()
    print(f"[시간] 파이프라인 적재 {time.time()-t0:.1f}s", flush=True)

    img = Image.open(a.image)
    t1 = time.time()
    # 🔴 생성 SLat 을 붙잡는다. 지금까지는 이걸 버리고 편집 때 GLB→150뷰→dinov2 로
    #    **재구성**했다. 원본 잠재를 그대로 쓰면 그 왕복 손실이 사라진다.
    #    decode_slat 은 run() 안에서 정확히 1회 불린다 — n_calls 로 확인한다.
    slat_path = os.path.join(a.out_dir, "slat.safetensors")
    with capture_slat(pipe, slat_path, meta={"seed": str(a.seed)}) as cap:
        out = pipe.run(img, seed=a.seed, formats=["mesh", "gaussian"])
    gen_s = time.time() - t1
    assert cap.n_calls == 1, f"decode_slat 이 {cap.n_calls}회 불렸다 — 마지막 것이 저장된다"
    print(f"[slat] {slat_path}  {cap.shape}", flush=True)
    print(f"[시간] 생성 {gen_s:.1f}s", flush=True)

    mesh, gs = out["mesh"][0], out["gaussian"][0]
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=0.95, texture_size=1024)
    glb_path = os.path.join(a.out_dir, "model.glb")
    glb.export(glb_path)
    Image.open(a.image).save(os.path.join(a.out_dir, "input.png"))

    # 점유 복셀 = sparse structure 격자. D28 정본 격자다(표면 샘플링이 아니다).
    coords = None
    for k in ("coords", "voxel_coords"):
        if hasattr(mesh, k):
            coords = getattr(mesh, k)
            break
    if coords is None and hasattr(gs, "_xyz"):
        coords = None                                  # gaussian 은 격자가 아니다
    stats = {
        "label": os.path.basename(a.out_dir),
        "input": os.path.basename(a.image),
        "seed": a.seed,
        "gen_s": round(gen_s, 1),
        "mesh_vertices": int(mesh.vertices.shape[0]),
        "glb": "ok",
        "slat": "ok",
        "total_s": round(time.time() - t0, 1),
    }
    if coords is not None:
        c = coords.detach().cpu().numpy()
        c = c[:, 1:] if c.shape[1] == 4 else c
        occ = np.zeros((64, 64, 64), bool)
        ci = np.clip(c.astype(int), 0, 63)
        occ[ci[:, 0], ci[:, 1], ci[:, 2]] = True
        lab, n = ndimage.label(occ, structure=np.ones((3, 3, 3)))
        sizes = sorted(ndimage.sum(occ, lab, range(1, n + 1)).astype(int).tolist(), reverse=True)
        stats.update({
            "occupied_voxels": int(occ.sum()),
            "components": int(n),
            "components_ge_20": int(sum(1 for s in sizes if s >= 20)),
            "component_sizes_top12": sizes[:12],
        })
    json.dump(stats, open(os.path.join(a.out_dir, "gen_stats.json"), "w"),
              ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"DONE {glb_path}")


if __name__ == "__main__":
    main()
