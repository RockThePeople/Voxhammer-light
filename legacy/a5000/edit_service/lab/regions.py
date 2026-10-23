"""--region 문자열 -> 활성 복셀 부분집합. 사람이 좌표를 손으로 못 짜니까.

축은 NORMALIZED [-0.5, 0.5] 기준이고 z 가 상단축이다.
"""
import re

import numpy as np

AXIS = {"x": 0, "y": 1, "z": 2}

# 이름 -> (축, 큰 쪽인가)
NAMED = {
    "top": (2, True), "bottom": (2, False),
    "right": (0, True), "left": (0, False),
    "back": (1, True), "front": (1, False),
}

INEQ = re.compile(r"^\s*([xyz])\s*(>=|<=|>|<)\s*(-?\d*\.?\d+)\s*$")


def describe(region, fraction):
    if region == "center":
        return f"중심 {fraction:.0%}"
    if region in NAMED:
        return f"{region} {fraction:.0%}"
    if region.endswith("-half"):
        return f"{region} (50%)"
    return region


def fraction_ignored(region):
    """--fraction 이 무시되는 region 인지. 조용히 무시하면 사람이 속는다.

    부등식은 임계값이 이미 명시적이고, *-half 는 0.5 로 고정이다.
    """
    if INEQ.match(region):
        return "부등식은 임계값이 이미 명시적이라 --fraction 을 쓰지 않는다"
    if region.lower().endswith("-half"):
        return "*-half 는 0.5 로 고정이라 --fraction 을 쓰지 않는다"
    return None


def select(xyz_voxel, region, fraction=0.25):
    """(N,3) VOXEL 좌표에서 region 에 해당하는 행 마스크 (N,) bool 을 돌려준다.

    fraction 은 **활성 복셀 개수 기준 분위수**로 자른다. 바운딩박스 기준으로 자르면
    오브젝트가 한쪽으로 치우쳤을 때 아무것도 안 잡히는 경우가 생긴다.
    """
    xyz = np.asarray(xyz_voxel, np.int64).reshape(-1, 3)

    m = INEQ.match(region)
    if m:
        ax, op, val = AXIS[m.group(1)], m.group(2), float(m.group(3))
        # NORMALIZED -> VOXEL. 셀 인덱스 기준으로 비교한다.
        thr = (val + 0.5) * 64.0
        col = xyz[:, ax].astype(np.float64)
        return {">": col > thr, ">=": col >= thr,
                "<": col < thr, "<=": col <= thr}[op]

    r = region.lower()
    if r.endswith("-half"):
        r, fraction = r[:-5], 0.5

    if r == "center":
        keep = np.ones(len(xyz), bool)
        for ax in range(3):
            lo = np.quantile(xyz[:, ax], (1 - fraction) / 2)
            hi = np.quantile(xyz[:, ax], 1 - (1 - fraction) / 2)
            keep &= (xyz[:, ax] >= lo) & (xyz[:, ax] <= hi)
        return keep

    if r not in NAMED:
        raise ValueError(
            f"모르는 region: {region!r}. "
            f"쓸 수 있는 것: {', '.join(sorted(NAMED))}, center, "
            f"*-half, 또는 부등식(예: 'z>0.2')"
        )
    ax, upper = NAMED[r]
    col = xyz[:, ax]
    if upper:
        return col >= np.quantile(col, 1 - fraction)
    return col <= np.quantile(col, fraction)
