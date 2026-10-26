"""C4 P1 검은 점 확인(지시서 main 88d2ab0 5절): 같은 회차 slat_tgt 를 승계 없이 통째로 R4 디코딩한다(부모 없음 → 모든 청크 새로, 스냅 없음).

비교 대상: 그 회차 P1 output.glb(승계 + 정정 규칙 + 경계 스냅). 결과 $RUNS_DIR/paper/x3/c4/fulldecode/<자산>/r<k>/{cbin/, output.glb, view.glb}.
한 프로세스, GPU 는 X3 회차 경계에서(감시 스크립트가 X3 를 잠깐 세운다).

    python tools/paper/c4_fulldecode.py jeep 1 2 3 4 5
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party" / "ai-ar-v3"), str(ROOT / "third_party" / "ai-ar-v3" / "contract" / "python")]
from tools.paper.x3 import OUT  # noqa: E402


def main(asset, rounds):
    import torch
    from safetensors import safe_open
    import editcore.models as em
    import trellis.modules.sparse as sp
    from editcore.export import decode_to_cbin
    from editcore.render import vertex_color_render_glb
    pipe = em.load_pipeline()
    for k in rounds:
        src = OUT / "p1" / asset / f"r{k}" / "slat_tgt.safetensors"
        d = OUT / "c4" / "fulldecode" / asset / f"r{k}"
        if (d / "view.glb").exists():
            continue
        with safe_open(str(src), "pt") as f:
            slat = sp.SparseTensor(feats=f.get_tensor("feats").float().cuda(), coords=f.get_tensor("coords").int().cuda())
        with torch.inference_mode():
            r = decode_to_cbin(pipe, slat, d / "cbin", glb_path=d / "output.glb")
        (d / "view.glb").write_bytes(vertex_color_render_glb((d / "output.glb").read_bytes()))
        (d / "decode.json").write_text(json.dumps({"slat_tgt": str(src), "parent": None, "chunks": r["chunks"]}, indent=1))
        print(f"c4 fulldecode {asset} r{k} chunks {r['chunks']}", flush=True)
        del slat
        torch.cuda.empty_cache()   # 검증 스크립트(계측 구간 밖)


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]])
