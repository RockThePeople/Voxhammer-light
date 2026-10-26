"""논문 채점 도구 CPU 테스트(ops/tasks/X-paper.md 1절): 같은 메시 CD 0·NC 1·F1 1, 알려진 평행이동, 실루엣 투영 vs 동결 2D 마스크."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
trimesh = pytest.importorskip("trimesh")


def _box():
    return trimesh.creation.box(extents=(0.6, 0.4, 0.8))


def test_same_mesh_scores_perfect():
    from tools.paper_metrics import sample, surface_metrics
    p, n = sample(_box())
    q, m = sample(_box())          # 같은 seed → 같은 점
    assert np.array_equal(p, q)
    r = surface_metrics(p, n, q, m)
    assert r["cd"] == 0.0 and abs(r["nc"] - 1.0) < 1e-9 and r["f1"] == 1.0


def test_known_translation_cd():
    """넓은 평면을 법선 방향으로 t 만큼 옮기면 내부 점의 최근접 거리는 t → CD ≈ t, F1@0.01 은 t < 0.01 이면 1."""
    from tools.paper_metrics import sample, surface_metrics
    plane = trimesh.creation.box(extents=(1.0, 1.0, 1e-6))
    t = 0.004
    moved = plane.copy(); moved.apply_translation([0, 0, t])
    p, n = sample(plane, n=20000); q, m = sample(moved, n=20000)
    r = surface_metrics(q, m, p, n)
    assert abs(r["cd"] - t) < 2e-4 and r["f1"] == 1.0 and r["nc"] > 0.999
    far = plane.copy(); far.apply_translation([0, 0, 0.02])
    q2, m2 = sample(far, n=20000)
    assert surface_metrics(q2, m2, p, n)["f1"] == 0.0


def test_frame_roundtrip_and_preserved_mask():
    from tools.paper_metrics import AssetFrame, excluded_set, preserved_mask
    fr = AssetFrame.from_v0(_box(), raw_c=(0.01, -0.02, 0.0), raw_s=0.98)
    cells = np.array([[5, 7, 9], [40, 30, 20], [63, 0, 1]])
    assert np.array_equal(fr.points_to_cells(fr.cells_to_n(cells)), cells)
    excl = excluded_set(np.array([[40, 30, 20]]))
    keep = preserved_mask(fr.cells_to_n(cells), fr, excl)
    assert keep.tolist() == [True, False, True]


def test_silhouette_matches_frozen_2d_mask():
    """A-S1 마스크 셀을 포크 편집 시점(v1)으로 투영한 실루엣 vs 동결 2d_mask(포크 depth 기반) IoU ≥ 0.9 (T1 V-2 실측 0.9955)."""
    assets = os.environ.get("ASSETS_DIR")
    if not assets or not Path(assets, "track_a/snowman/cond/A-S1/2d_mask.png").is_file():
        pytest.skip("ASSETS_DIR 없음")
    from PIL import Image
    from tools.paper_metrics import FORK_VIEWS, AssetFrame, silhouette
    g = json.loads(Path(assets, "track_a/snowman/gen/gen.json").read_text())["glb_normalize"]
    fr = AssetFrame.from_v0(Path(assets, "track_a/snowman/gen/model.glb"), g["center_glb_frame"], g["scale"])
    cells = json.loads(Path(assets, "track_a/snowman/masks/A-S1.mask.json").read_text())["mask"]["voxels"]
    s = silhouette(cells, fr, FORK_VIEWS[1], kind_frame="v0")
    m = np.asarray(Image.open(Path(assets, "track_a/snowman/cond/A-S1/2d_mask.png")).convert("L")) > 127
    iou = (s & m).sum() / (s | m).sum()
    assert iou >= 0.9, iou
