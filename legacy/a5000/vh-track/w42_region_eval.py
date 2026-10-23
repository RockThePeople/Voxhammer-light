"""W42 — **고정 영역** 기준 채점.  마스크 셀 집합을 채점에 쓰면 항진명제가 된다.

왜 필요한가
-----------
W42 의 마스크는 `mask ⊆ 부모 점유 셀` 이다 (점유 셀로만 정의했으므로).
⇒ `마스크 안 신규 = |자식∩mask − 부모∩mask|` 은 **정의상 0** 이다. 모델이 무엇을 하든 0 이다.
  (MEASUREMENTS §5.1 의 "마스크 476셀이 전부 점유라 신규가 정의상 불가능" 과 같은 자리)
⇒ 그리고 새 기하가 자란 자리가 "안" 이 아니라 "밖" 으로 재분류되어
  `밖_생김` 이 부풀고 `밖_IoU` 가 떨어진다. **보존이 나빠진 것이 아니다.**

그래서 채점은 **빈 칸을 포함하는 고정 영역**으로 한다. 이 영역은 두 웨이브(W41·W42)에
동일하게 적용되므로 비교가 성립한다.

    t1s  편집 영역 = {z >= 45}                     (목 위 = 머리.  z=44 가 코어 최소반경)
    t2s  편집 영역 = 기존 타원체 볼륨 ∪ {z >= 36}    (라이더 + 손잡이 높이)
"""
import json
import os
import sys

import numpy as np
import utils3d
from safetensors import safe_open

T = os.environ.get("TRACK_ROOT", "<HOME>/ai-ar-data/vh-track")


def coords(p):
    with safe_open(p, "pt") as f:
        return f.get_tensor("coords").numpy()[:, 1:]


def volume(p):
    pos = utils3d.io.read_ply(p)[0]
    return np.floor((np.asarray(pos) + 0.5) * 64).astype(int)


def region_mask(tag):
    """빈 칸을 포함하는 고정 편집 영역 (64³ bool)."""
    r = np.zeros((64, 64, 64), bool)
    if tag == "t1s":
        r[:, :, 45:] = True
    else:
        v = volume(f"{T}/{tag}-mask/voxels_delete.ply")
        r[v[:, 0], v[:, 1], v[:, 2]] = True
        r[:, :, 36:] = True
    return r


def stats(parent, child, reg):
    P = {tuple(t) for t in parent.tolist()}
    C = {tuple(t) for t in child.tolist()}
    inr = lambda S: {t for t in S if reg[t]}
    out = lambda S: {t for t in S if not reg[t]}
    pi, ci, po, co = inr(P), inr(C), out(P), out(C)
    return {
        "보존_IoU": len(po & co) / len(po | co) if po | co else 1.0,
        "보존_사라짐": len(po - co), "보존_생김": len(co - po), "보존_부모셀": len(po),
        "편집_신규": len(ci - pi), "편집_제거": len(pi - ci), "편집_부모셀": len(pi),
        "총셀": len(C),
    }


if __name__ == "__main__":
    ASSET = {"t1s": "t1s-snowman", "t2s": "t2s-motorcycle"}
    for tag in sys.argv[1:] or ["t1s", "t2s"]:
        par = coords(f"{T}/{ASSET[tag]}/slat.safetensors")
        reg = region_mask(tag)
        print(f"\n{'='*104}\n### {tag} — 고정 편집 영역 (빈 칸 포함) · 부모 {len(par)}셀\n{'='*104}")
        print(f"{'런':30s}{'보존_IoU':>10s}{'사라짐':>8s}{'생김':>8s}{'보존부모':>9s}"
              f"{'편집_신규':>10s}{'편집_제거':>10s}{'편집부모':>9s}{'총셀':>8s}")
        for wave in ["w41", "w42"]:
            for arm in ["baseline", "ours"]:
                p = f"{T}/{wave}-{tag}-{arm}-r1/slat.safetensors"
                if not os.path.exists(p):
                    continue
                s = stats(par, coords(p), reg)
                print(f"{wave+' '+arm:30s}{s['보존_IoU']:10.4f}{s['보존_사라짐']:8d}{s['보존_생김']:8d}"
                      f"{s['보존_부모셀']:9d}{s['편집_신규']:10d}{s['편집_제거']:10d}"
                      f"{s['편집_부모셀']:9d}{s['총셀']:8d}")
