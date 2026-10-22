"""eval/rkeep.py CPU 테스트 (00s S-7). (1)·(3) 은 보존만 재는 지표가 무편집을 통과시키는지 보는 대조다."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from eval.rkeep import rkeep_metrics  # noqa: E402


def _block(lo, hi):
    r = [np.arange(lo[i], hi[i]) for i in range(3)]
    return np.stack(np.meshgrid(*r, indexing="ij"), -1).reshape(-1, 3)


PARENT = _block((10, 10, 10), (30, 30, 30))          # 8000 셀
MASK = _block((12, 12, 12), (16, 16, 16))            # 64 셀, 부모 안


def _without(a, b):
    sb = {tuple(x) for x in b}
    return np.array([x for x in a if tuple(x) not in sb])


def test_1_no_edit():
    r = rkeep_metrics(PARENT, PARENT.copy(), MASK)
    assert r["rkeep_iou"] == 1.0 and r["rkeep_lost"] == 0 and r["rkeep_new"] == 0
    assert r["edit_new"] == 0 and r["edit_lost"] == 0


def test_2_delete_one_outside_mask():
    child = _without(PARENT, [(29, 29, 29)])
    r = rkeep_metrics(PARENT, child, MASK)
    assert r["rkeep_lost"] == 1 and r["rkeep_new"] == 0 and r["rkeep_iou"] < 1.0


def test_3_change_only_inside_mask():
    child = _without(PARENT, MASK[:10])                # 마스크 안 10셀 삭제
    r = rkeep_metrics(PARENT, child, MASK)
    assert r["rkeep_iou"] == 1.0 and r["rkeep_lost"] == 0 and r["rkeep_new"] == 0
    assert r["edit_lost"] == 10 and r["edit_new"] == 0


def test_4_new_cell_in_empty_space():
    child = np.vstack([PARENT, [[50, 50, 50]]])
    r = rkeep_metrics(PARENT, child, MASK)
    assert r["rkeep_new"] == 1 and r["rkeep_lost"] == 0


def test_halo_uses_dilate_cells_and_accepts_n4():
    one = np.array([[32, 32, 32]])
    r = rkeep_metrics(np.zeros((0, 3), int), np.zeros((0, 3), int), one)
    assert r["rkeep_cells"] == 64 ** 3 - 125           # 5³ 팽창
    corner = np.array([[0, 0, 0]])
    r = rkeep_metrics(np.zeros((0, 3), int), np.zeros((0, 3), int), corner)
    assert r["rkeep_cells"] == 64 ** 3 - 27
    p4 = np.hstack([np.zeros((len(PARENT), 1), int), PARENT])
    assert rkeep_metrics(p4, p4, MASK)["rkeep_iou"] == 1.0
