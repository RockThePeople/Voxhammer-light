"""2번 R4 — R11 규칙 정정(chief 2026-10-26, 요구사항 4.4 0번 R11 "점유 비교") 재계산.

변경 청크 = 부기(마스크 ∪ halo 2 가 닿는 청크) ∪ 점유 변경 청크(그 청크 안 SLat 점유 셀 집합이 부모와 다른 청크).
나머지는 부모 `.cbin` 바이트 승계. 이미 돈 GPU 실행(부기만 쓴 규칙)은 다시 돌리지 않는다.

phase sets    (CPU) : 저장된 slat_tgt coords vs 부모(입력) SLat coords → 시나리오·회차별 청크 수, 부기 청크 수(원래 / 기하가 있는 것),
                      점유 변경 청크 수, **점유 변경으로만 추가된 청크 수**(= 마스크 밖 생성이 화면에 반영되는 청크), 승계 청크 수.
phase rebuild (GPU) : 한 프로세스에서 저장된 slat_tgt 를 mesh 만 다시 디코딩 → 청크 분할 → 정정 규칙으로 조립
                      → `<회차>/cbin_r11/{chunks,manifest.json,r4.json}` + `output_r11.glb`(glbbuild). 계측 구간 밖.
결과: $RUNS_DIR/02-r4/metrics/r11_sets.json, r11_rebuild.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "ai-ar-v3" / "contract" / "python"))
from tools.r4_metrics import RUN_IDS, on_dir, scenarios  # noqa: E402

RUNS = Path(os.environ["RUNS_DIR"])
OUT = RUNS / "02-r4" / "metrics"


def coords(p):
    from safetensors import safe_open
    with safe_open(str(p), "np") as f:
        return f.get_tensor("coords")[:, 1:].astype(np.int64)


def occupancy_changed_chunks(parent_cells, child_cells):
    """청크 안 점유 셀 집합이 부모와 다른 청크 키 = 대칭차 셀이 속한 청크."""
    from deltacontract.coords import chunk_keys_sorted, voxel_to_chunk
    code = lambda c: c[:, 0] * 4096 + c[:, 1] * 64 + c[:, 2]
    pc, cc = code(parent_cells), code(child_cells)
    diff = np.setxor1d(pc, cc)
    if diff.size == 0:
        return set()
    cells = np.stack([diff // 4096, (diff // 64) % 64, diff % 64], 1)
    return set(chunk_keys_sorted(voxel_to_chunk(cells)))


def changed_sets(s, slat_tgt):
    from editcore.export import bookkeeping
    mask = json.loads(Path(s["mask"]).read_text())["mask"]["voxels"]
    book = bookkeeping(mask, 2)
    occ = occupancy_changed_chunks(coords(s["slat"]), coords(slat_tgt))
    return book, occ


def sets():
    res = {}
    for s in scenarios():
        parent_keys = {p.stem for p in s["parent"].glob("*.cbin")}
        row = {}
        for r in RUN_IDS:
            d = on_dir(s, r)
            if not (d / "slat_tgt.safetensors").is_file() or not (d / "cbin" / "manifest.json").is_file():
                continue
            book, occ = changed_sets(s, d / "slat_tgt.safetensors")
            fresh_keys = set(json.loads((d / "cbin" / "manifest.json").read_text())["chunks"])   # 원래 규칙 결과(승계 포함)
            exist = parent_keys | fresh_keys
            changed = book | occ
            occ_only = occ - book
            row[r] = {"chunks_parent": len(parent_keys), "book_raw": len(book), "book": len(book & exist),
                      "occupancy_changed": len(occ & exist), "occupancy_only_added": len(occ_only & exist),
                      "occupancy_only_added_keys": sorted(occ_only & exist),
                      "changed_total": len(changed & exist), "inherited": len(exist - changed)}
        res[s["id"]] = row
        if row:
            x = row.get("r1") or next(iter(row.values()))
            print(s["id"], {k: v for k, v in x.items() if not k.endswith("keys")}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "r11_sets.json").write_text(json.dumps(res, indent=1))


def rebuild():
    import torch
    import editcore.models as em   # trellis 경로·ATTN_BACKEND 를 먼저 맞춘다
    import trellis.modules.sparse as sp
    from safetensors import safe_open
    from editcore.export import mesh_to_chunks, write_cbin
    pipe = em.load_pipeline()
    v3 = str(ROOT / "third_party" / "ai-ar-v3")
    if v3 not in sys.path:
        sys.path.insert(0, v3)
    from server.glbbuild import build_glb
    res = {}
    for s in scenarios():
        for r in RUN_IDS:
            d = on_dir(s, r)
            st = d / "slat_tgt.safetensors"
            if not st.is_file() or (d / "cbin_r11" / "r4.json").is_file():
                continue
            with safe_open(str(st), "pt") as f:
                slat = sp.SparseTensor(feats=f.get_tensor("feats").float().cuda(), coords=f.get_tensor("coords").int().cuda())
            blobs, chunks, info = mesh_to_chunks(pipe, slat)
            book, occ = changed_sets(s, st)
            w = write_cbin(d / "cbin_r11", blobs, chunks, book=book | occ, parent_dir=s["parent"])
            exist = set(w["sent"]) | set(w["reused"]) | set(w["removed"])
            w.update({"mesh": info, "rule": "부기 ∪ 점유 변경 ∪ 신규 (R11, chief 2026-10-26)", "book_raw": len(book),
                      "occupancy_changed_raw": len(occ), "book_existing": len(book & exist),
                      "occupancy_only_added": len((occ - book) & exist)})
            # 원래 실행(다른 프로세스)의 새 청크 바이트와 같은지 — 프로세스 간 비결정(spconv, DS §3-4) 확인용
            old = d / "cbin" / "chunks"
            same = sum((old / f"{k}.cbin").is_file() and (old / f"{k}.cbin").read_bytes() == blobs[k] for k in w["sent"] if k in blobs)
            w["fresh_equal_to_original_run"] = f"{same}/{len(w['sent'])}"
            build_glb(d / "cbin_r11" / "chunks", d / "output_r11.glb")
            (d / "cbin_r11" / "r4.json").write_text(json.dumps(w, indent=1))
            res.setdefault(s["id"], {})[r] = {k: (len(v) if isinstance(v, list) else v) for k, v in w.items() if k != "parent_dir"}
            print(s["id"], r, res[s["id"]][r], flush=True)
            del slat, blobs, chunks
            torch.cuda.empty_cache()   # 계측 구간 밖
    OUT.mkdir(parents=True, exist_ok=True)
    prev = json.loads((OUT / "r11_rebuild.json").read_text()) if (OUT / "r11_rebuild.json").is_file() else {}
    for k, v in res.items():
        prev.setdefault(k, {}).update(v)
    (OUT / "r11_rebuild.json").write_text(json.dumps(prev, indent=1))


if __name__ == "__main__":
    {"sets": sets, "rebuild": rebuild}[sys.argv[1]]()
