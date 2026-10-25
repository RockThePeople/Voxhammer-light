"""2번 R4 — 구간 7 정정 규칙(부기 ∪ 점유 변경) 14건 × 3회 실측 (chief 검토 2026-10-26).

측정 방식: 전체 편집 실행이 아니라, 저장된 E2a c15 r1–r3 slat_tgt 를 **한 프로세스에서** `decode_to_cbin`(정정 규칙)으로
다시 디코딩하며 구간 7 span(decode_mesh·chunk_partition·cbin_encode·bookkeeping·cbin_write)을 잰다(reference.meter, 구간 경계 synchronize).
- 부기 선행 계산(`bookkeeping_async`)은 실제 편집에서 구간 4–6(≥ 40 s) 동안 끝나므로, 여기서도 선행 계산이 끝난 뒤 구간 7 을 시작한다.
  선행 계산 자체 시간은 따로 적는다(구간 7 밖).
- 결과 무영향 확인: r1 마다 선행 계산 없이(동기 부기) 한 번 더 디코딩해 새로 쓴/승계/생략/removed 청크 집합이 같은지 본다(시간 미기록).
출력: $RUNS_DIR/02-r4/metrics/seg7_r11.json (회차별 span), .cbin 은 임시 폴더(남기지 않음).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import editcore.models as em  # noqa: E402

import torch  # noqa: E402

ASSETS = Path(os.environ["ASSETS_DIR"])
RUNS = Path(os.environ["RUNS_DIR"])
E2A_DAY = os.environ.get("E2A_DAY", "2026-10-25")


def main():
    import trellis.modules.sparse as sp
    from safetensors import safe_open
    from editcore.export import bookkeeping, bookkeeping_async, decode_to_cbin
    from reference.meter import Meter
    sys.path.insert(0, str(REPO))
    from tools.r4_metrics import scenarios
    pipe = em.load_pipeline()
    out = RUNS / "02-r4" / "metrics" / "seg7_r11.json"
    res = json.loads(out.read_text()) if out.is_file() else {}
    for s in scenarios():
        mask = json.loads(Path(s["mask"]).read_text())["mask"]["voxels"]
        with safe_open(str(s["slat"]), "np") as f:
            pc = f.get_tensor("coords")[:, 1:]
        for r in ("r1", "r2", "r3"):
            key = f"{s['id']}/{r}"
            if key in res:
                continue
            st = RUNS / E2A_DAY / s["name"] / "e2a-c15" / r / "slat_tgt.safetensors"
            with safe_open(str(st), "pt") as f:
                slat = sp.SparseTensor(feats=f.get_tensor("feats").float().cuda(), coords=f.get_tensor("coords").int().cuda())
            t0 = time.perf_counter()
            fut = bookkeeping_async(mask, 2)
            fut.result()                              # 실제 편집에서는 구간 4–6 동안 끝난다
            pre_s = time.perf_counter() - t0
            meter = Meter()
            hooks = {"span": (lambda n, seg: meter.span(n, segment=seg))}
            with tempfile.TemporaryDirectory() as td:
                w = decode_to_cbin(pipe, slat, Path(td) / "cbin", mask_cells=mask, parent_dir=s["parent"], parent_cells=pc,
                                   hooks=hooks, book=fut)
            parts = {}
            for sp_ in meter.spans:
                if sp_["segment"] == 7:
                    parts[sp_["name"]] = parts.get(sp_["name"], 0.0) + sp_["seconds"]
            row = {"seg7": sum(parts.values()), "parts": parts, "precompute_bookkeeping_s": pre_s,
                   "sent": len(w["sent"]), "reused": len(w["reused"]), "removed": len(w["removed"]),
                   "omitted_outside": len(w["omitted_outside"]), "book_existing": w["book_existing"],
                   "occupancy_only_added": w["occupancy_only_added"]}
            if r == "r1":   # 결과 무영향: 동기 부기로 한 번 더(시간 미기록)
                with tempfile.TemporaryDirectory() as td:
                    w2 = decode_to_cbin(pipe, slat, Path(td) / "cbin", mask_cells=mask, parent_dir=s["parent"], parent_cells=pc)
                row["precompute_same_sets"] = all(sorted(w[k]) == sorted(w2[k]) for k in ("sent", "reused", "removed", "omitted_outside"))
                row["book_same"] = set(fut.result()) == bookkeeping(mask, 2)
            res[key] = row
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, indent=1))
            print(key, "seg7 %.2f" % row["seg7"], {k: round(v, 2) for k, v in parts.items()}, "pre %.2f" % pre_s,
                  row.get("precompute_same_sets"), flush=True)
            del slat
    print("DONE", len(res))


if __name__ == "__main__":
    main()
