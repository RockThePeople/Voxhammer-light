"""04L 시드 판본: 트랙 A 생성 자산(moto·snowman·chair)을 워커 저장소의 v1(커밋됨)으로 넣는다. 입력은 읽기만 한다.

v1 = gen/slat.safetensors(R2, denormalized) · gen/chunks/*.cbin · manifest.json(계약 ChunkManifest) · gen/model.glb(텍스처, 정규화본,
조건 렌더 입력) · asset.json(정규화 xf = gen.json glb_normalize). asset_id 는 `vhl-ta-<asset>`.
종단(E2E) 반복용 v0 복제 `vhl-ta-<asset>-e2e-r1..r3` 도 같은 v1 로 넣는다(회차마다 깨끗한 판본에서 출발).
`assets_track_a.json`(커밋) = 시나리오마다 asset_id · base_version · 동결 편집 시점 · 복제 id.

    python services/a5000_edit/seed.py            # 이미 있으면 건너뛴다
    python services/a5000_edit/seed.py --manifest # assets_track_a.json 만 다시 쓴다
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "third_party" / "ai-ar-v3" / "contract" / "python"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from services.a5000_edit import store  # noqa: E402

ASSETS = Path(os.environ["ASSETS_DIR"])
TRACK_A = ("moto", "snowman", "chair")
E2E_RUNS = ("r1", "r2", "r3")
MANIFEST = Path(__file__).resolve().parent / "assets_track_a.json"


def asset_id(name, run=None):
    return f"vhl-ta-{name}" + (f"-e2e-{run}" if run else "")


def write_manifest():
    sc = json.loads((ROOT / "scenarios" / "track_a.json").read_text())["scenarios"]
    m = {"note": "04L 트랙 A 종단 자산. base_version 은 시드 v1. edit_view 는 동결 시나리오(scenarios/track_a.json) 값. "
                 "e2e_clones 는 같은 v1 복제로, 종단 반복 회차마다 하나씩 쓴다.",
         "scenarios": {sid: {"asset": s["asset"], "asset_id": asset_id(s["asset"]), "base_version": 1, "edit_view": s["edit_view"],
                             "prompt": s["inpaint_prompt"], "mask": s["mask"]["file"], "mask_fingerprint": s["mask"]["fingerprint"],
                             "e2e_clones": [asset_id(s["asset"], r) for r in E2E_RUNS]} for sid, s in sc.items()}}
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=1) + "\n")
    return MANIFEST


def seed(name, run=None):
    import deltacontract as dc
    from deltacontract.schemas import ChunkEntry, ChunkManifest
    aid = asset_id(name, run)
    vd = Path(store.version_dir(aid, 1))
    if (vd / "manifest.json").is_file() and store.committed(aid) >= 1:
        return aid, "exists"
    gen = ASSETS / "track_a" / name / "gen"
    (vd / "chunks").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(gen / "slat.safetensors", vd / "slat.safetensors")
    shutil.copyfile(gen / "model.glb", vd / "model.glb")
    entries = {}
    for p in sorted((gen / "chunks").glob("*.cbin")):
        b = p.read_bytes()
        k = p.stem
        (vd / "chunks" / p.name).write_bytes(b)
        m = dc.decode(b)
        entries[k] = ChunkEntry(uri=dc.chunk_uri(aid, k, 1), hash=dc.blob_hash(b), byte_length=len(b), vertex_count=int(m.vertex_count),
                                index_count=int(m.index_count), voxel_count=int(m.voxel_count), version=1)
    (vd / "manifest.json").write_text(ChunkManifest(asset_id=aid, version=1, chunks=entries).model_dump_json(indent=2))
    g = json.loads((gen / "gen.json").read_text())["glb_normalize"]
    (vd / "asset.json").write_text(json.dumps({"xf": {"center": g["center_glb_frame"], "scale": g["scale"]},
                                                "render_mesh": "model.glb (생성 판본 텍스처 GLB)", "source": str(gen)}))
    (Path(store.asset_dir(aid)) / "committed.json").write_text(json.dumps({"latest": max(1, store.committed(aid))}))
    return aid, f"seeded v1 ({len(entries)} chunks)"


if __name__ == "__main__":
    if "--manifest" not in sys.argv:
        for n in TRACK_A:
            for r in (None, *E2E_RUNS):
                print(*seed(n, r))
    print("manifest", write_manifest())
