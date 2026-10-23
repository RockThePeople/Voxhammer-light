"""GLB 를 계약대로 청킹하고, 부모 대비 changed/removed 를 낸다 — 시간도 같이 잰다.

왜 필요한가:
    단독 트랙은 GLB in / GLB out 이라 **청킹과 델타가 계측에서 빠져 있었다.**
    제품 경로(`/v2/trellis/edit`)에는 그게 들어간다. 그래서 여기서 따로 잰다.

무엇을 지키나 (사용자 제약):
    ① 청킹 규칙 불변 — CHUNK_SIZE·CHUNK_GRID_RES·contract 판본을 계약에서 **읽어** 쓴다
    ② 승계 불변 — 안 바뀐 청크는 **부모 바이트를 그대로** 둔다. 다시 디코드하지 않는다
    ③ 부기는 배치에서 유도 — 마스크+halo → `affected_chunks()`.
       `diff_chunk_sets()` 는 그 부기가 맞는지 **검증**하는 용도로만 쓴다 (계약 docstring)
    ④ 결정적 청킹 — 같은 기하 → 같은 바이트. sha256 으로 확인한다

⚠️ GLB 정점은 GLB 좌표계다. 청킹은 NORMALIZED 공간을 받는다.
   `frames.GLB_TO_VOXEL` 과 같은 축 규약(D9)을 여기서도 명시적으로 적용한다 —
   매직넘버를 흩뿌리지 않는다.
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, "<HOME>/ai-ar-prototype/contract/python")

import numpy as np                                                    # noqa: E402
import trimesh                                                        # noqa: E402

from deltacontract import (CHUNK_GRID_RES, CHUNK_SIZE, CONTRACT_VERSION,  # noqa: E402
                           VOXEL_RES, affected_chunks, diff_chunk_sets,
                           dilate_cells, encode, partition_mesh)


def load_normalized(path):
    """GLB → (vertices in NORMALIZED, faces). D9: voxel = (x, -z, y)."""
    g = trimesh.load(path)
    g = g.to_geometry() if hasattr(g, "to_geometry") else g
    V = np.asarray(g.vertices, dtype=np.float64)
    V = np.stack([V[:, 0], -V[:, 2], V[:, 1]], axis=1)   # GLB(y-up) → 격자(z-up)
    return V, np.asarray(g.faces, dtype=np.int64)


def chunk_glb(path, label):
    t0 = time.time()
    V, F = load_normalized(path)
    t_load = time.time() - t0
    t1 = time.time()
    chunks = partition_mesh(V, F)
    t_part = time.time() - t1
    t2 = time.time()
    blobs = {k: encode(m) for k, m in chunks.items()}
    t_enc = time.time() - t2
    hashes = {k: hashlib.sha256(b).hexdigest() for k, b in blobs.items()}
    total = sum(len(b) for b in blobs.values())
    print(f"[{label}] 정점 {len(V):,} 면 {len(F):,} → 청크 {len(chunks)}개 · {total/1e6:.2f} MB")
    print(f"    load {t_load:.2f}s  partition {t_part:.2f}s  encode {t_enc:.2f}s  "
          f"합계 {t_load+t_part+t_enc:.2f}s")
    return blobs, hashes, {"load_s": round(t_load, 2), "partition_s": round(t_part, 2),
                           "encode_s": round(t_enc, 2), "chunks": len(chunks),
                           "bytes": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--children", nargs="+", required=True, help="label=path 형태")
    ap.add_argument("--mask_ply", help="편집 볼륨 복셀 — 부기(배치 유도)에 쓴다")
    ap.add_argument("--halo", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    print(f"계약 v{CONTRACT_VERSION} · CHUNK_SIZE={CHUNK_SIZE} · GRID={CHUNK_GRID_RES} "
          f"· VOXEL_RES={VOXEL_RES}\n")
    pb, ph, pt = chunk_glb(a.parent, "부모")
    rep = {"contract": {"version": CONTRACT_VERSION, "chunk_size": CHUNK_SIZE,
                        "chunk_grid_res": CHUNK_GRID_RES, "voxel_res": VOXEL_RES},
           "parent": pt, "children": {}}

    bookkeeping = None
    if a.mask_ply:
        import utils3d
        pos = utils3d.io.read_ply(a.mask_ply)[0]
        cells = np.floor((np.asarray(pos) + 0.5) * VOXEL_RES).astype(np.int64)
        if a.halo:
            cells = dilate_cells(cells, a.halo)
        # ③ 부기는 **배치에서 유도**한다 — 해시 비교로 정하지 않는다
        bookkeeping = set(affected_chunks(None, None, cells))
        print(f"\n부기(배치 유도): 마스크 {len(cells)}셀 → 영향 청크 {len(bookkeeping)}개")
        rep["bookkeeping_chunks"] = len(bookkeeping)

    for spec in a.children:
        label, path = spec.split("=", 1)
        print()
        cb, ch, ct = chunk_glb(path, label)
        changed, removed = diff_chunk_sets(ph, ch)
        added = [k for k in ch if k not in ph]
        same = [k for k in ch if k in ph and ch[k] == ph[k]]
        d = {**ct, "changed": len(changed), "removed": len(removed),
             "added": len(added), "identical_to_parent": len(same)}
        print(f"    부모 대비  changed {len(changed)}  removed {len(removed)}  "
              f"added {len(added)}  바이트 동일 {len(same)}")
        if bookkeeping is not None:
            touched = set(changed) | set(removed) | set(added)
            outside = touched - bookkeeping
            inside_untouched = bookkeeping - touched
            d["touched_outside_bookkeeping"] = len(outside)
            d["bookkeeping_untouched"] = len(inside_untouched)
            print(f"    ★ 부기 밖인데 바뀐 청크 {len(outside)}   부기 안인데 안 바뀐 청크 "
                  f"{len(inside_untouched)}")
            if outside:
                print(f"       예: {sorted(outside)[:8]}")
        rep["children"][label] = d
    json.dump(rep, open(a.out, "w"), ensure_ascii=False, indent=2)
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
