"""X0 P1-cbin (chief 검토 1a): P1-glb 와 같은 저장 E2a c15 r1 slat_tgt 를 R4(정정 규칙 R11 + 경계 스냅)로 다시 디코딩한다.

입력 $RUNS_DIR/2026-10-25/<이름>/e2a-c15/r1/slat_tgt.safetensors, 부모 = R4 와 같은 v0 청크(트랙 A gen/chunks, 트랙 B 02-r4/parents).
출력 $RUNS_DIR/paper/x0/cbin/<시나리오>/{cbin/, output.glb, view.glb, decode.json} (r1), r2·r3 는 그 아래 r2/·r3/.
view.glb = output.glb 에 (iv) 보정(editcore/render.py:vertex_color_render_glb) — 웹·VSCode glTF 미리보기용(chief 15:2x 4a).
한 프로세스, 계측 구간 밖.

    X0_RUNS=r1,r2,r3 python tools/paper/x0_cbin.py [시나리오 …]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tests" / "gpu"), str(REPO / "third_party" / "ai-ar-v3")]
RUNS = Path(os.environ["RUNS_DIR"])
OUT = RUNS / "paper" / "x0" / "cbin"


def main(ids):
    import torch
    from safetensors import safe_open
    import editcore.models as em
    from editcore.export import decode_to_cbin
    from r4_checks import load_slat, scenarios
    from tools.paper.x0 import E2A_DAY, scenarios as x0s
    masks = {k: v["mask"] for k, v in x0s().items()}
    pipe = em.load_pipeline()
    from editcore.render import vertex_color_render_glb
    for s, run in [(s, r) for s in scenarios() for r in os.environ.get("X0_RUNS", "r1").split(",")]:
        if ids and s["id"] not in ids:
            continue
        d = OUT / s["id"] if run == "r1" else OUT / s["id"] / run
        if (d / "output.glb").is_file():
            if not (d / "view.glb").is_file():
                (d / "view.glb").write_bytes(vertex_color_render_glb((d / "output.glb").read_bytes()))
            continue
        src = RUNS / E2A_DAY / s["name"] / "e2a-c15" / run / "slat_tgt.safetensors"
        slat = load_slat(src)
        with safe_open(str(s["slat"]), "np") as f:
            pc = f.get_tensor("coords")[:, 1:]
        mask = json.loads(Path(masks[s["id"]]).read_text())["mask"]["voxels"]
        with torch.inference_mode():
            w = decode_to_cbin(pipe, slat, d / "cbin", mask_cells=mask, parent_dir=s["parent"], parent_cells=pc,
                               glb_path=d / "output.glb")
        (d / "view.glb").write_bytes(vertex_color_render_glb((d / "output.glb").read_bytes()))
        (d / "decode.json").write_text(json.dumps({"slat_tgt": str(src), "parent": str(s["parent"]),
                                                   **{k: w[k] for k in ("chunks", "sent", "reused", "removed", "seam") if k in w}},
                                                  indent=1, default=str))
        print("x0 cbin", s["id"], run, "sent", w.get("sent"), "reused", w.get("reused"), "snapped", w.get("seam", {}).get("snapped"), flush=True)
        del slat
        torch.cuda.empty_cache()   # 검증 스크립트(계측 구간 밖)


if __name__ == "__main__":
    main(sys.argv[1:])
