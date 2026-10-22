"""보존 지표 R_keep (요구사항서 6장 「품질 지표」, 정오표 E2).

요구사항서 6장 원문:
    R_keep = 64³ 격자 전체에서 (편집 마스크 ⊕ halo 2)를 뺀 셀(빈 칸 포함).
    R_keep 안에서 부모 대비 점유 IoU, 사라진 셀 수, 새로 생긴 셀 수를 보고.
    마스크 안의 새 셀·사라진 셀 수는 편집량으로 함께. 입력은 coords 뿐.

"⊕ halo 2" 는 `deltacontract.dilate_cells(cells, 2)` 를 그대로 호출한다
(ai-ar-v3 19ed91f `contract/python/deltacontract/coords.py:406`, 고객사 회신 A6(b), 정오표 E2).
그 함수의 이웃 정의: 26-이웃(체비셰프 거리 ≤ halo)으로 팽창, 64³ 밖 셀은 버림.
팽창을 여기서 다시 구현하지 않는다.

입력 coords 는 (N,3) 또는 SLat 형식 (N,4)[batch,x,y,z] 둘 다 받는다((N,4) 는 0번 열을 뗀다).
부모 = 그 편집의 입력 판본(트랙 B 첫 편집은 판본 0).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_DC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "third_party", "ai-ar-v3", "contract", "python")
if _DC not in sys.path:
    sys.path.insert(0, _DC)
from deltacontract.coords import VOXEL_RES, dilate_cells  # noqa: E402

HALO = 2
__all__ = ["rkeep_metrics", "HALO"]


def _cells(coords) -> np.ndarray:
    a = np.asarray(coords)
    if a.ndim != 2 or a.shape[1] not in (3, 4):
        raise ValueError(f"coords shape {a.shape} — (N,3) 또는 (N,4) 여야 한다")
    if a.shape[1] == 4:
        a = a[:, 1:]
    a = a.astype(np.int64)
    if a.size and (a.min() < 0 or a.max() >= VOXEL_RES):
        raise ValueError(f"coords 가 [0,{VOXEL_RES}) 밖에 있다")
    return a


def _flat(cells: np.ndarray) -> np.ndarray:
    return np.unique((cells[:, 0] * VOXEL_RES + cells[:, 1]) * VOXEL_RES + cells[:, 2])


def rkeep_metrics(parent_coords, child_coords, mask_cells, halo: int = HALO) -> dict:
    """부모·자식 점유와 편집 마스크로 R_keep 지표와 편집량을 계산한다."""
    p = _flat(_cells(parent_coords))
    c = _flat(_cells(child_coords))
    m_cells = _cells(mask_cells)
    m = _flat(m_cells)
    excl = _flat(dilate_cells(m_cells, halo)) if m_cells.size else m

    p_keep = np.setdiff1d(p, excl, assume_unique=True)
    c_keep = np.setdiff1d(c, excl, assume_unique=True)
    inter = np.intersect1d(p_keep, c_keep, assume_unique=True).size
    union = np.union1d(p_keep, c_keep).size

    p_in = np.intersect1d(p, m, assume_unique=True)
    c_in = np.intersect1d(c, m, assume_unique=True)
    ring = np.setdiff1d(excl, m, assume_unique=True)
    p_ring = np.intersect1d(p, ring, assume_unique=True)
    c_ring = np.intersect1d(c, ring, assume_unique=True)

    return {
        "halo": int(halo),
        "dilation": "deltacontract.dilate_cells (26-neighbour, clipped to 64^3)",
        "rkeep_cells": int(VOXEL_RES ** 3 - excl.size),
        "rkeep_parent_occupied": int(p_keep.size),
        "rkeep_child_occupied": int(c_keep.size),
        "rkeep_iou": float(inter / union) if union else 1.0,
        "rkeep_lost": int(p_keep.size - inter),
        "rkeep_new": int(c_keep.size - inter),
        "mask_cells": int(m.size),
        "edit_new": int(np.setdiff1d(c_in, p_in, assume_unique=True).size),
        "edit_lost": int(np.setdiff1d(p_in, c_in, assume_unique=True).size),
        # 참고: 마스크도 R_keep 도 아닌 halo 고리 안의 변화
        "halo_ring_new": int(np.setdiff1d(c_ring, p_ring, assume_unique=True).size),
        "halo_ring_lost": int(np.setdiff1d(p_ring, c_ring, assume_unique=True).size),
    }
