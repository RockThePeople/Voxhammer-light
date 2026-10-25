"""정오표 E14·E15 판정(편집 셀 feats, 14건 합산 분포 비교) — 항목 보고서들이 같이 쓴다.

E14: 변경–기준 쌍 vs 기준–기준 쌍(시나리오당 3쌍씩), max|Δ|·mean|Δ| 각각에
     단측 순위 검정(Mann-Whitney U, 대립가설 변경–기준 > 기준–기준) p ≥ 0.05,
     중앙값 비(변경–기준 / 기준–기준) ≤ 1.1, P(변경–기준 > 기준–기준) = U / (n₁n₂) ≤ 0.6.
E15: 쌍별 mean|Δ| ≤ 상한. 원문 3.2e-3, 잠정 4.06e-3(c15 바닥값 최대 3.38e-3 × 1.2, 2026-10-25) — 둘 다 적는다.
"""
from __future__ import annotations

import numpy as np

E15_LIMITS = {"원문 3.2e-3": 3.2e-3, "잠정 4.06e-3": 4.06e-3}


def e14(change: list[float], base: list[float]) -> dict:
    from scipy.stats import mannwhitneyu
    c, b = np.asarray(change, float), np.asarray(base, float)
    u, p = mannwhitneyu(c, b, alternative="greater")
    ratio = float(np.median(c) / np.median(b)) if np.median(b) > 0 else float("inf")
    prob = float(u / (len(c) * len(b)))
    return {"n_change": int(len(c)), "n_base": int(len(b)), "p_one_sided": float(p), "median_ratio": ratio,
            "P_change_gt_base": prob, "pass": bool(p >= 0.05 and ratio <= 1.1 and prob <= 0.6),
            "median_change": float(np.median(c)), "median_base": float(np.median(b))}


def e15(change_means: list[float]) -> dict:
    m = np.asarray(change_means, float)
    return {name: {"limit": lim, "max": float(m.max()), "n_over": int((m > lim).sum()), "pass": bool((m <= lim).all())}
            for name, lim in E15_LIMITS.items()}


def judge(change_pairs: list[dict], base_pairs: list[dict]) -> dict:
    """pairs: [{"max": max|Δ|, "mean": mean|Δ|}, ...] (편집 셀). 필수 항목(coords·보존 셀)은 호출부가 따로 본다."""
    return {"max": e14([x["max"] for x in change_pairs], [x["max"] for x in base_pairs]),
            "mean": e14([x["mean"] for x in change_pairs], [x["mean"] for x in base_pairs]),
            "e15": e15([x["mean"] for x in change_pairs])}
