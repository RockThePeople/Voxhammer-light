"""보고서 맨 앞 "결과 요약" 절(논문용 표) 생성. 채점이 끝난 실험부터 채우고, 나머지는 "채점 중" 자리를 둔다.

    X3_STITCHED=1 python tools/paper/summary.py figure     # 논문 그림 후보: 3자산 × 1·3·5회차 × P0|P1|P2(포크 시점 1 고정, P1·P2 봉합본)
    python tools/paper/summary.py md > 요약.md
보고서에는 <!-- summary:start --> … <!-- summary:end --> 사이에 넣는다(다시 만들면 그 사이만 바꾼다).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
RUNS = Path(os.environ["RUNS_DIR"])
X3S = RUNS / "paper" / "x3" / "score"
X3SS = RUNS / "paper" / "x3" / "score_stitched"
C4 = RUNS / "paper" / "x3" / "c4"
ASSETS3 = ("jeep", "cottage", "adventurer")
NAMES = {"jeep": "지프", "cottage": "오두막", "adventurer": "모험가"}


def j(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}


def _f(v, fmt):
    return "채점 중" if v is None else format(v, fmt)


def x3_tables():
    pre, st, surf = j(X3S / "x3_score.json"), j(X3SS / "x3_score.json"), j(X3S / "surface_iou.json")
    vf = j(RUNS / "paper" / "visible" / "x3.json")
    out = []
    for a in ASSETS3:
        P, S, U = pre.get(a, {}), st.get(a, {}), surf.get(a, {})
        out += [f"**{NAMES[a]}({a})**", "",
                "| 경로 | 회차 | CD 보이는 표면 ↓ | F1 보이는 표면 ↑ | PSNR(M) ↑ | LPIPS(M) ↓ | 표면 셀 IoU(원본 표면 2M 점이 닿는 셀) ↑ | DINO-I ↑ | CLIP-T ↑ | "
                "고정 영역 PSNR(M) / LPIPS(M) / 렌더 L* | 참고: 전체 표면 CD / F1 | 참고: 봉합 전 CD / PSNR(M) |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for p in ("p0", "p1", "p2"):
            for k in range(1, 6):
                key = f"{p}_r{k}_own"
                o = (S if p != "p0" and key in S else P).get(key)
                if not o:
                    continue
                im, pr = o["image"], o["preserved"]
                v = vf.get(f"{a}/{p}_r{k}", {})
                vc, fx, fl = v.get("visible_cum") or {}, v.get("fixed") or {}, v.get("fixed_render_L") or {}
                fixed = (f"{fx['image']['psnr']:.2f} / {fx['image']['lpips']:.4f} / {fl['L']:.1f}" if fx and fl else "채점 중")
                ref = f"{P[key]['preserved']['cd']:.4f} / {P[key]['image']['psnr']:.2f}" if p != "p0" and key in P and key in S else "–"
                out.append(f"| {p.upper()} | {k} | {_f(vc.get('cd'), '.4f')} | {_f(vc.get('f1'), '.3f')} | {im['psnr']:.2f} | {im['lpips']:.4f} | "
                           f"{U.get(f'{p}_r{k}', {}).get('iou_surf', float('nan')):.4f} | {(im.get('dino_i') or float('nan')):.3f} | "
                           f"{(im.get('clip_t') or float('nan')):.3f} | {fixed} | {pr['cd']:.4f} / {pr['f1']:.3f} | {ref} |")
        out += ["", "설정: P0 = 원저자 경로(매 회차 150시점 재등록 + 25 스텝 + `to_glb`), 인페인팅 A5000 bf16. "
                    "P1 = W34 SLat 재사용 + editcore 15/15 + R4 `.cbin`(R11 승계, 경계 스냅 + 봉합 띠), 인페인팅 3090 INT4. "
                    "P2 = P1 + 매 회차 150시점 재등록, 인페인팅 3090 INT4. 각 경로 자기 직전 결과에서 다음 회차, 반복 없음(표본 1). "
                    "비교 기준 = 자기 출발점(P0 생성 텍스처 GLB, P1·P2 A_origin R4 `.cbin`). 주 열의 영역 = 그 회차 누적 마스크 ∪ halo 2 밖. "
                    "**고정 영역** = 5회차 누적 마스크 ∪ halo 2 밖을 모든 회차에 같게(회차–지표 곡선용). "
                    "**보이는 표면** = Bench 16시점 레이캐스팅 첫 교차점 100k 점(`.cbin` 의 가려진 내부 면 제외). "
                    "DINO-I·CLIP-T = 편집 시점 1장(자기 2d_edit·프롬프트). 표면 셀 IoU(원본 표면 2M 점이 닿는 셀) = A_origin 표면 2M 점이 닿는 셀만(R_keep IoU 는 4절 참고 열).", ""]
    return out


def c4_table():
    vf = j(RUNS / "paper" / "visible" / "x3.json")
    out = ["| 자산 | 경로 | ΔL* r1 | r2 | r3 | r4 | r5 | ΔE r5 | 렌더 L* r0 → r5 |", "|---|---|---|---|---|---|---|---|---|"]
    fixed_all = True
    for a in ASSETS3:
        r = j(C4 / f"{a}_color.json")
        for p in ("p0", "p1", "p2"):
            if f"{p}_r5" not in r:
                continue
            dl = " | ".join(f"{r[f'{p}_r{k}']['albedo']['dL']:+.1f}" for k in range(1, 6))
            f0, f5 = (vf.get(f"{a}/{p}_r{k}", {}).get("fixed_render_L") for k in (0, 5))
            if f0 and f5:
                rl = f"{f0['L']:.1f} → {f5['L']:.1f}"
            else:
                fixed_all = False
                rl = f"{r[f'{p}_r0']['render']['L']:.1f} → {r[f'{p}_r5']['render']['L']:.1f} (참고)"
            out.append(f"| {NAMES[a]} | {p.upper()} | {dl} | {r[f'{p}_r5']['albedo']['dE']:.1f} | {rl} |")
    out += ["", "렌더 L*: " + ("고정 영역(5회차 누적 마스크 ∪ halo 2 밖, 모든 회차 같은 화소 규칙)." if fixed_all else
                              "**참고(영역이 회차마다 다름)** — 누적 마스크가 커지며 비교 화소가 바뀐다. 고정 영역 값은 채점 중이다. 색 표류의 주 지표는 같은 점 비교인 ΔL*.")]
    return out


def human_table():
    c = j(RUNS / "paper" / "x3" / "eval" / "choices.json")
    key = j(RUNS / "paper" / "x3" / "blind_key.json").get("key", {})
    from collections import Counter
    t = {q: Counter() for q in ("preserve", "prompt")}
    for item, v in c.items():
        for q in t:
            if q in v and item in key:
                t[q][key[item][v[q]["slot"]]] += 1
    lab = {"preserve": "보존 상태가 가장 좋은 것", "prompt": "편집이 프롬프트대로 된 것"}
    return (["| 질문 | P0 | P1 | P2 | 응답 |", "|---|---|---|---|---|"] +
            [f"| {lab[q]} | {t[q]['p0']} | **{t[q]['p1']}** | {t[q]['p2']} | {sum(t[q].values())} |" for q in t])


def _mean(v):
    v = [x for x in v if x is not None]
    return float(sum(v) / len(v)) if v else None


def _m(r, m):
    return r["preserved"][m] if m in ("cd", "f1", "nc") else r["image"].get(m)


def _x0_runs():
    """r1 은 14건 모두 x0.json(먼저 채점), r2·r3 는 x0_runs.json(채점된 만큼)."""
    runs = j(RUNS / "paper" / "x0" / "x0_runs.json")
    for sid, cfgs in j(RUNS / "paper" / "x0" / "x0.json").items():
        runs.setdefault(sid, {}).setdefault("r1", {})
        for c, v in cfgs.items():
            runs[sid]["r1"].setdefault(c, v)
    return runs


def _keep(J):
    from tools.paper.x0 import case_filter
    k = case_filter()
    return J if k is None else {sid: v for sid, v in J.items() if sid in k}


def x0_table():
    runs = _keep(_x0_runs())
    st = _keep(j(RUNS / "paper" / "x0" / "x0_runs_st.json"))
    rows = ["| 설정 | 시나리오 | 회차(합) | CD ↓ | F1 ↑ | PSNR(M) ↑ | SSIM(M) ↑ | LPIPS(M) ↓ | DINO-I(편집 시점) ↑ | CLIP-T(편집 시점) ↑ | CLIP-T 6시점 ↑ | DINO-I 16시점 ↑ |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cfg, J, lab in (("base", runs, "base(25 스텝, `to_glb`)"), ("c15-glb", runs, "P1-glb(15 스텝, `to_glb`)"),
                        ("c15-cbin-st", st, "P1-cbin(15 스텝, R4 + 봉합)"), ("c15-cbin", runs, "참고: P1-cbin 봉합 전")):
        per = {}
        for sid, rr in J.items():
            got = [rr[r][cfg] for r in ("r1", "r2", "r3") if r in rr and cfg in rr[r]]
            if got:
                per[sid] = got
        if not per:
            continue
        nr = sum(len(v) for v in per.values())
        vals = {m: _mean([_mean([_m(x, m) for x in v]) for v in per.values()])
                for m in ("cd", "f1", "psnr", "ssim", "lpips", "dino_i", "clip_t", "clip_t_front6", "dino_i_ref16")}
        rows.append(f"| {lab} | {len(per)} | {nr} | {vals['cd']:.4f} | {vals['f1']:.3f} | {vals['psnr']:.2f} | {vals['ssim']:.4f} | "
                    f"{vals['lpips']:.4f} | {vals['dino_i']:.3f} | {vals['clip_t']:.3f} | {vals['clip_t_front6']:.4f} | {vals['dino_i_ref16']:.4f} |")
    return rows + ["", f"값 = 시나리오별 회차 평균의 {len(runs)}건 평균(회차 수는 표의 '회차(합)', 채점된 회차만). 기준 = 원본 v0, 마스크 ∪ halo 2 밖. "
                       "인페인팅 = 동결 조건 이미지(A5000 bf16, 00s2). 전체 표면 형상 지표(보이는 표면 변형은 하지 않았다)."]


def x4_table():
    J = j(RUNS / "paper" / "x4" / "score" / "x4_score.json")
    assets = list(J)
    rows = ["| 경로 | 회차 | CD ↓ | F1 ↑ | PSNR(M) ↑ | LPIPS(M) ↓ | R_keep IoU(원본 SLat 대비) ↑ |", "|---|---|---|---|---|---|---|"]
    for p, tag in (("p0", "own=common"), ("p1", "own"), ("p1", "common")):
        for k in range(1, 6):
            rs = [J[a].get(f"{p}_r{k}_{tag}") for a in assets]
            rs = [r for r in rs if r]
            if not rs:
                continue
            iou = _mean([(r.get("occupancy_vs_origin") or {}).get("rkeep_iou") for r in rs])
            lab = {"own=common": "P0", "own": "P1(자기 출발점 대비)", "common": "P1(공통 기준 대비)"}[tag]
            if tag == "own" and _mean([_m(r, "psnr") for r in rs]) > 60:      # 같은 SLat 재디코딩(디코더 1e-7 차이만)
                rows.append(f"| {lab} | {k} | ≈0 | 1.000 | 78–83(같은 SLat 재디코딩) | ≈0 | " + ("–" if iou is None else f"{iou:.4f}") + " |")
                continue
            rows.append(f"| {lab} | {k} | {_mean([_m(r, 'cd') for r in rs]):.4f} | {_mean([_m(r, 'f1') for r in rs]):.3f} | "
                        f"{_mean([_m(r, 'psnr') for r in rs]):.2f} | {_mean([_m(r, 'lpips') for r in rs]):.4f} | "
                        + ("–" if iou is None else f"{iou:.4f}") + " |")
    return rows + ["", f"채점된 자산 {len(assets)}/11 평균. "
                       "기준 = 자기 출발점(P0 = 생성·원본 GLB = 공통 기준, P1 = 원본 SLat 의 R4 `.cbin`) 또는 공통 기준(원본 GLB), 편집 없음, 전체 영역. "
                       "P1 은 같은 SLat 을 다시 디코딩할 뿐이라 디코더 비결정성(1e-7)으로 바이트는 다르지만 지표는 소수 넷째 자리까지 같고(자기 출발점 대비 PSNR 78–83 dB), 공통 기준 대비 값은 회차와 무관하게 R4 메시 표현 차이만 남는다. "
                       "P1 공통 기준 행의 PSNR(M) 차이는 R4 정점색 메시 대 텍스처 GLB 렌더 차이(전체 표면 형상 지표 — 가려진 내부 면 포함)."]


def x2a_table():
    from tools.paper.x0 import scenarios
    p0 = j(RUNS / "paper" / "x2a" / "score" / "x2a_p0.json")
    runs, st = _x0_runs(), j(RUNS / "paper" / "x0" / "x0_runs_st.json")
    rows = ["| 시나리오 | P0 회차 | P0 PSNR(M) | P1-glb PSNR(M) | P1-cbin(봉합) PSNR(M) | P0 CD | P1-glb CD | P1-cbin CD | "
            "P0 DINO-I | P1-glb DINO-I | P0 CLIP-T | P1-glb CLIP-T |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    agg = {k: [] for k in ("p0p", "gp", "cp", "p0c", "gc", "cc", "p0d", "gd", "p0t", "gt")}
    for sid in scenarios():
        a = [p0[f"{sid}/{r}"] for r in ("r1", "r2") if f"{sid}/{r}" in p0]
        if not a:
            continue
        g = [runs[sid][r]["c15-glb"] for r in ("r1", "r2", "r3") if sid in runs and r in runs[sid] and "c15-glb" in runs[sid][r]]
        c = [st[sid][r]["c15-cbin-st"] for r in ("r1", "r2", "r3") if sid in st and r in st[sid] and "c15-cbin-st" in st[sid][r]]
        v = {"p0p": _mean([_m(x, "psnr") for x in a]), "gp": _mean([_m(x, "psnr") for x in g]), "cp": _mean([_m(x, "psnr") for x in c]),
             "p0c": _mean([_m(x, "cd") for x in a]), "gc": _mean([_m(x, "cd") for x in g]), "cc": _mean([_m(x, "cd") for x in c]),
             "p0d": _mean([_m(x, "dino_i") for x in a]), "gd": _mean([_m(x, "dino_i") for x in g]),
             "p0t": _mean([_m(x, "clip_t") for x in a]), "gt": _mean([_m(x, "clip_t") for x in g])}
        for k in agg:
            agg[k].append(v[k])
        f = lambda x, n: "–" if x is None else format(x, n)   # noqa: E731
        rows.append(f"| {sid} | {len(a)} | {f(v['p0p'], '.2f')} | {f(v['gp'], '.2f')} | {f(v['cp'], '.2f')} | {f(v['p0c'], '.4f')} | "
                    f"{f(v['gc'], '.4f')} | {f(v['cc'], '.4f')} | {f(v['p0d'], '.3f')} | {f(v['gd'], '.3f')} | {f(v['p0t'], '.3f')} | {f(v['gt'], '.3f')} |")
    n = len(agg["p0p"])
    m = {k: _mean(v) for k, v in agg.items()}
    f = lambda x, n_: "–" if x is None else format(x, n_)   # noqa: E731
    rows.append(f"| **평균({n}건)** | | {f(m['p0p'], '.2f')} | {f(m['gp'], '.2f')} | {f(m['cp'], '.2f')} | {f(m['p0c'], '.4f')} | "
                f"{f(m['gc'], '.4f')} | {f(m['cc'], '.4f')} | {f(m['p0d'], '.3f')} | {f(m['gd'], '.3f')} | {f(m['p0t'], '.3f')} | {f(m['gt'], '.3f')} |")
    return rows + ["", f"P0 = 원저자 경로(재등록 + 25 스텝 + `to_glb`), 채점된 시나리오 {n}건 · 시나리오당 r1·r2 중 채점된 회차 평균. "
                       "P1 = E2a c15(W34 재사용 + 15/15) r1–r3 중 채점된 회차 평균 — P1-glb(`to_glb`)·P1-cbin(R4 + 봉합). 조건 이미지 = 동결본(A5000 bf16). "
                       "기준 = 원본 v0, 마스크 ∪ halo 2 밖."] + x2a_test()


def x2a_test(tag=None):
    tag = os.environ.get("VHL_PAPER_TAG", "") if tag is None else tag
    t = j(RUNS / "paper" / "x2a" / "score" / (f"tost_{tag}.json" if tag else "tost.json"))
    if not t:
        return ["", "비열등성 검정: 채점 중."]
    lab = {"psnr": "PSNR(M)", "ssim": "SSIM(M)", "lpips": "LPIPS(M)", "dino_i": "DINO-I", "clip_t": "CLIP-T", "cd": "CD"}
    rows = ["", f"**단측 비열등성 검정(chief 보정 7):** 시나리오 평균 쌍 차이 d = P1 − P0(P0 2회, P1 3회), H0: 좋은 방향 기준 d ≤ −한계(E19), n = {t.get('n')}, α = 0.05. "
                "우위 p = H0: 좋은 방향 기준 d ≤ 0.", "",
            "| P1 | 지표 | 한계 | P0 평균 | P1 평균 | d | 비열등 p | 비열등 | 우위 p | P1 이 한계 넘게 좋은/나쁜 시나리오 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for cfg in ("P1-glb", "P1-cbin(봉합)"):
        for m, v in (t.get(cfg) or {}).items():
            fmt = ".4f" if m in ("cd", "ssim", "lpips") else (".2f" if m == "psnr" else ".3f")
            rows.append(f"| {cfg} | {lab[m]} | {v['margin']} | {format(v['p0_mean'], fmt)} | {format(v['p1_mean'], fmt)} | {v['d_mean']:+{fmt}} | "
                        f"{v['p_noninferiority']:.2g} | {'**예**' if v['noninferior'] else '아니오'} | {v['p_superiority']:.2g} | "
                        f"{v['p1_better_beyond_margin']} / {v['p1_worse_beyond_margin']} |")
    return rows


def x2b_table():
    J = j(RUNS / "paper" / "x2b" / "score" / "x2b.json")
    items = json.loads((ROOT / "scenarios" / "x2b_list.json").read_text())["scenarios"]
    rows = ["| 건 | 모델 · 프롬프트 | 경로 | CD ↓ | F1 ↑ | PSNR(M) ↑ | SSIM(M) ↑ | LPIPS(M) ↓ | DINO-I ↑ | CLIP-T ↑ |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in items:
        for p in ("p0", "p1"):
            r = J.get(f"{s['id']}/{p}")
            if not r:
                rows.append(f"| {s['id']} | {s['object'][:24]} · {s['prompt']} | {p.upper()} | 채점 중 | | | | | | |")
                continue
            rows.append(f"| {s['id']} | {s['object'][:24]} · {s['prompt']} | {p.upper()} | {_m(r, 'cd'):.4f} | {_m(r, 'f1'):.3f} | "
                        f"{_m(r, 'psnr'):.2f} | {_m(r, 'ssim'):.4f} | {_m(r, 'lpips'):.4f} | {_m(r, 'dino_i'):.3f} | {_m(r, 'clip_t'):.3f} |")
    return rows + ["", "P0 = 원저자 경로 1회, P1 = 원 VoxHammer 인코딩 1회로 만든 SLat → editcore 15/15 + R4 + 봉합 1회. "
                       "조건 이미지 = 데이터셋 2D(X2B-1·3, 편집 시점 마스크 IoU 1.000) / 포크 렌더 + A5000 bf16 FLUX(X2B-2, 데이터셋 2D IoU 0.859). "
                       "기준 = 데이터셋 원본 GLB, 마스크 ∪ halo 2 밖. FID·FVD·검정 없음."]


def md():
    L = ["## 결과 요약 (논문용 표)", "",
         "모든 수치는 실측이고, 원자료·정의는 아래 각 절에 있다. 생성 `tools/paper/summary.py`. **사용자 지시(10-27 01:4x)로 추가 계산(보이는 표면·고정 영역)은 하지 않았고, 예정된 채점(X0 r1–r3, X4 11자산, X2-A·X2-B, X2-A 비열등성 검정)은 모두 마쳤다(03:38).** 각 표에 건수·회차를 적는다.", "",
         "### S1. X3 연속 5회 편집 (P1·P2 = 봉합본)", ""]
    L += x3_tables()
    L += ["### S2. C4 색 표류 (보존 영역 표면 색, A_origin 대비)", ""]
    L += c4_table()
    L += ["", "설정: 표면 색 = A_origin(생성 텍스처 GLB) 보존 영역 표면 100k 점마다 결과의 최근접 점(2셀 이내) 색 — P0 텍스처 UV, P1·P2 정점색(sRGB). "
               "CIELAB(D65). 렌더 L* = Bench 16시점 보존 영역 화소 평균. 인페인팅·회차 설정은 S1 과 같다.", "",
          "![회차–평균 L*](img/X-paper-c4-c4_L_curves.png)", "",
          "**논문 그림 후보 — 1·3·5회차, 같은 시점(포크 시점 1), P0 | P1 | P2(P1·P2 봉합본):**", "",
          "![X3 1·3·5회차](img/X-paper-summary-x3-figure.png)", "",
          "### S3. 사람 평가 (X3, 15 항목 × 2 질문)", ""]
    L += human_table()
    L += ["", "한계:", "1. 평가자 1명, 반복 없음 — 통계 검정은 쓰지 않는다.",
          "2. 경로 이름은 가렸지만 P0·P2 는 회차가 지날수록 어두워져(C4) 그림만으로 P1 과 구별될 수 있다 — 블라인드가 완전하지 않을 수 있다.",
          "3. 편집 반영 판단은 사람 선택 하나로 정하지 않고 S1 의 CLIP-T·DINO-I 와 함께 본다.", "",
          "### S4. X0 기존 14건 재채점(25 vs 15, P1-glb·P1-cbin)", ""]
    L += x0_table()
    L += ["", "### S5. X4 무편집 재등록 체인(P0 vs P1, 5회)", ""]
    L += x4_table()
    L += ["", "### S6. X2-A 3D 만 비교(동결 14건, P0 vs P1)", ""]
    L += x2a_table()
    L += ["", "### S7. X2-B Edit3D-Bench 3건(P0·P1 각 1회)", ""]
    L += x2b_table()
    L += [""]
    return "\n".join(L)


def figure():
    from PIL import Image, ImageDraw
    from tools.paper.x3 import round_info
    from tools.paper.x3_score import result_views
    from tools.paper_metrics import FORK_TO_BENCH
    VIEW = 1                       # 회차마다 편집 시점이 달라도 그림은 자산마다 같은 시점(포크 시점 1 = 앞쪽 3/4)
    S = 240
    rounds = (1, 3, 5)
    W, H = S * 9 + 90, (S + 6) * 3 + 50
    sh = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(sh)
    for c, k in enumerate(rounds):
        dr.text((90 + c * 3 * S + 6, 6), f"round {k}", fill="black")
        for i, p in enumerate(("P0", "P1", "P2")):
            dr.text((90 + c * 3 * S + i * S + 6, 26), p, fill="black")
    for r, a in enumerate(ASSETS3):
        y = 50 + r * (S + 6)
        dr.text((6, y + S // 2), a, fill="black")
        for c, k in enumerate(rounds):
            _, info = round_info(a, k)
            for i, p in enumerate(("p0", "p1", "p2")):
                d = result_views(p, a, k, info["edit_view"])[0].parent       # Bench 16시점 렌더 폴더(캐시)
                im = Image.open(d / f"render_{FORK_TO_BENCH[VIEW]:04d}.png").convert("RGBA")
                bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                bg.alpha_composite(im)
                sh.paste(bg.convert("RGB").crop((112, 112, 912, 912)).resize((S, S)), (90 + c * 3 * S + i * S, y))
    p = ROOT / "docs" / "reports" / "img" / "X-paper-summary-x3-figure.png"
    sh.save(p, optimize=True)
    print("figure", p)


def _sub(cases, tag, fn):
    """VHL_PAPER_CASES·VHL_PAPER_TAG 를 잠시 바꿔 fn() 을 부른다(트랙 소계)."""
    old = {k: os.environ.get(k) for k in ("VHL_PAPER_CASES", "VHL_PAPER_TAG")}
    os.environ.update(VHL_PAPER_CASES=cases, VHL_PAPER_TAG=tag)
    try:
        return fn()
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


E18_CFG = {"P1-glb": "c15-glb", "P1-cbin(봉합)": "c15-cbin-st"}


def worse_lists():
    """11건 안에서 P1 이 E19 한계보다 나쁜 시나리오·지표(X0 base→P1, X2-A P0→P1) + R_keep 신규 E18 상한 초과."""
    from tools.paper.x0 import scenarios
    from tools.paper.x2_score import MARGINS
    runs, st = _x0_runs(), j(RUNS / "paper" / "x0" / "x0_runs_st.json")
    p0 = j(RUNS / "paper" / "x2a" / "score" / "x2a_p0.json")
    lab = {"psnr": "PSNR(M)", "ssim": "SSIM(M)", "lpips": "LPIPS(M)", "dino_i": "DINO-I", "clip_t": "CLIP-T", "cd": "CD"}
    def grab(sid, cfg):
        J = st if cfg == "c15-cbin-st" else runs
        return [J[sid][r][cfg] for r in ("r1", "r2", "r3") if sid in J and r in J[sid] and cfg in J[sid][r]]
    rows = ["| 비교 | P1 | 시나리오 | 지표 | 기준 평균 | P1 평균 | d(P1 − 기준) | 한계 |", "|---|---|---|---|---|---|---|---|"]
    rk = ["| 비교 | P1 | 시나리오 | 기준 R_keep 신규 | 상한 max(기준×1.2, 기준+20) | P1 R_keep 신규 | P1 R_keep 소실 |", "|---|---|---|---|---|---|---|"]
    tot = {}
    for cmp_, base_of in (("X0 25→15", lambda sid: grab(sid, "base")),
                          ("X2-A P0→P1", lambda sid: [p0[f"{sid}/{r}"] for r in ("r1", "r2") if f"{sid}/{r}" in p0])):
        for p1, cfg in E18_CFG.items():
            nb = n1 = 0
            for sid in scenarios():
                a, b = base_of(sid), grab(sid, cfg)
                if not a or not b:
                    continue
                for m, (mg, good) in MARGINS.items():
                    ma, mb = _mean([_m(x, m) for x in a]), _mean([_m(x, m) for x in b])
                    if ma is not None and mb is not None and good * (mb - ma) < -mg:
                        fmt = ".4f" if m in ("cd", "ssim", "lpips") else (".2f" if m == "psnr" else ".3f")
                        rows.append(f"| {cmp_} | {p1} | {sid} | {lab[m]} | {ma:{fmt}} | {mb:{fmt}} | {mb - ma:+{fmt}} | {mg} |")
                oa = [x.get("occupancy") for x in a if x.get("occupancy")]
                ob = [x.get("occupancy") for x in b if x.get("occupancy")]
                if not oa or not ob:
                    continue
                na, nb_ = _mean([o["rkeep_new"] for o in oa]), _mean([o["rkeep_new"] for o in ob])
                lost = max(o["rkeep_lost"] for o in ob)
                nb += na; n1 += nb_
                cap = max(na * 1.2, na + 20)
                if nb_ > cap or lost > 0:
                    rk.append(f"| {cmp_} | {p1} | {sid} | {na:.1f} | {cap:.1f} | {nb_:.1f} | {lost} |")
            tot[(cmp_, p1)] = (nb, n1)
    rk += ["", "**합계(11건, E18 합계 ≤ 기준 × 1.1)**", "", "| 비교 | P1 | 기준 신규 합계 | 상한(×1.1) | P1 신규 합계 | 합계 판정 |", "|---|---|---|---|---|---|"]
    for (c, p1), (a, b) in tot.items():
        rk.append(f"| {c} | {p1} | {a:.1f} | {a * 1.1:.1f} | {b:.1f} | {'초과' if b > a * 1.1 else '안'} |")
    return rows, rk


def cases11():
    from tools.paper.x0 import CASES11
    A3, B8 = ",".join(CASES11[:3]), ",".join(CASES11[3:])
    L = ["# 논문용 11건 재집계 (A-M1·A-M2·A-C1 제외)", "",
         "사용자 결정 10-27 17:27. 남는 11건 = " + ", ".join(CASES11) + ". 새 채점·새 실행 없이 14건과 같은 원값 JSON 을 같은 코드"
         "(`tools/paper/summary.py`, `tools/paper/x2_score.py tost`)로 다시 집계했다. 정의·한계값(E19)·검정(단측 t, α 0.05)은 14건판과 같다. "
         "14건 원본 표는 ../ 에 그대로 있다.", "",
         "## S4. X0 25 vs 15 (11건)", ""]
    L += _sub("11", "11cases", x0_table)
    L += ["", "### S4 트랙 소계", "", "**트랙 A (A-S1·A-S2·A-C2, 3건)**", ""] + _sub(A3, "11cases_A3", x0_table)[:-2]
    L += ["", "**트랙 B (B1–B8, 8건)**", ""] + _sub(B8, "11cases_B8", x0_table)[:-2]
    L += ["", "## S6. X2-A P0 vs P1 (11건)", ""]
    L += _sub("11", "11cases", x2a_table)
    for name, cs, tag in (("트랙 A (A-S1·A-S2·A-C2, 3건)", A3, "11cases_A3"), ("트랙 B (B1–B8, 8건)", B8, "11cases_B8")):
        t = _sub(cs, tag, x2a_table)
        L += ["", f"### S6 소계 — {name}", ""] + [r for r in t if r.startswith("| **평균") or r.startswith("| 시나리오") or r.startswith("|---|---|---|---|---|---|---|---|---|---|---|---|")][:3]
        L += _sub(cs, tag, x2a_test)
    L += ["", "트랙 A 소계는 n = 3 이라 검정력이 매우 낮다(참고). 트랙 B 는 n = 8.", "",
          "## 1c. 11건 안에서 P1 이 한계(E19)보다 나쁜 시나리오·지표", "",
          "기준 평균·P1 평균 = 시나리오별 회차 평균(X0 base r1–r3, X2-A P0 r1·r2, P1 E2a c15 r1–r3). d 를 좋은 방향으로 맞춰 −한계보다 작으면 넣었다.", ""]
    rows, rk = _sub("11", "11cases", worse_lists)
    L += rows
    L += ["", "### R_keep 신규 상한(E18) 초과·소실", "",
          "R_keep 신규·소실 = 부모 SLat 대비 셀 점유(occupancy). P1-glb·P1-cbin 은 같은 SLat 이라 같은 값이다. 표에는 상한 초과 또는 소실 > 0 인 건만 적는다. "
          "X2-A 는 E18 의 기준 자리에 P0 를 넣었다(E18 원래 기준은 직전 승인 상태 — 참고). 초과 표가 비어 있으면 해당 건이 없다는 뜻이다.", "",
          "**시나리오별 초과**", ""] + rk
    return "\n".join(L) + "\n"


def cases11_out(out_dir):
    """11건판을 OUT/README.md + 표마다 CSV 로(제목 = 바로 앞 '#' 머리글 + 굵은 줄)."""
    import csv
    import re
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = cases11()
    (out / "README.md").write_text(md)
    tabs, head, bold, rows = [], "", "", []
    for line in md.splitlines() + [""]:
        if line.startswith("|"):
            cells = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
            if not set("".join(cells)) <= set("-: "):
                rows.append(cells)
            continue
        if rows:
            tabs.append((head + (" " + bold if bold else ""), rows)); rows = []
        if line.startswith("#"):
            head, bold = line.lstrip("# ").strip(), ""
        elif line.startswith("**") and "**" in line[2:]:
            bold = line[2:line.index("**", 2)].split("(")[0].strip()
    names = []
    for i, (title, rows) in enumerate(tabs):
        name = f"{i:02d}_" + re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", title)[:60].strip("_") + ".csv"
        with open(out / name, "w", newline="") as f:
            csv.writer(f).writerows(rows)
        names.append(name)
    print("\n".join(names))


if __name__ == "__main__":
    {"md": lambda: print(md()), "figure": figure, "cases11": lambda: print(cases11(), end=""),
     "cases11_out": lambda: cases11_out(sys.argv[2])}[sys.argv[1]]()
