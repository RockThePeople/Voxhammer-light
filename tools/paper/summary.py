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
                "| 경로 | 회차 | CD 보이는 표면 ↓ | F1 보이는 표면 ↑ | PSNR(M) ↑ | LPIPS(M) ↓ | IoU 표면 셀 ↑ | DINO-I ↑ | CLIP-T ↑ | "
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
                    "DINO-I·CLIP-T = 편집 시점 1장(자기 2d_edit·프롬프트). IoU 표면 셀 = A_origin 표면이 닿는 셀만(R_keep IoU 는 4절 참고 열).", ""]
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


def x0_table():
    runs = _x0_runs()
    st = j(RUNS / "paper" / "x0" / "x0_runs_st.json")
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
    return rows + ["", "값 = 시나리오별 회차 평균의 14건 평균(회차 수는 표의 '회차(합)', 채점된 회차만). 기준 = 원본 v0, 마스크 ∪ halo 2 밖. "
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
                rows.append(f"| {lab} | {k} | ≈0 | 1.000 | > 60(같은 메시) | ≈0 | " + ("–" if iou is None else f"{iou:.4f}") + " |")
                continue
            rows.append(f"| {lab} | {k} | {_mean([_m(r, 'cd') for r in rs]):.4f} | {_mean([_m(r, 'f1') for r in rs]):.3f} | "
                        f"{_mean([_m(r, 'psnr') for r in rs]):.2f} | {_mean([_m(r, 'lpips') for r in rs]):.4f} | "
                        + ("–" if iou is None else f"{iou:.4f}") + " |")
    return rows + ["", f"채점된 자산 {len(assets)}/11 평균. "
                       "기준 = 자기 출발점(P0 = 생성·원본 GLB = 공통 기준, P1 = 원본 SLat 의 R4 `.cbin`) 또는 공통 기준(원본 GLB), 편집 없음, 전체 영역. "
                       "P1 은 같은 SLat 을 다시 디코딩할 뿐이라 자기 출발점과 같은 메시이고(∞), 공통 기준 대비 값은 회차와 무관하게 R4 메시 표현 차이만 남는다. "
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
    return rows + ["", f"P0 = 원저자 경로(재등록 + 25 스텝 + `to_glb`), 채점된 시나리오 {n}/14 · 시나리오당 r1·r2 중 채점된 회차 평균. "
                       "P1 = E2a c15(W34 재사용 + 15/15) r1–r3 중 채점된 회차 평균 — P1-glb(`to_glb`)·P1-cbin(R4 + 봉합). 조건 이미지 = 동결본(A5000 bf16). "
                       "기준 = 원본 v0, 마스크 ∪ halo 2 밖."] + x2a_test()


def x2a_test():
    t = j(RUNS / "paper" / "x2a" / "score" / "tost.json")
    if not t:
        return ["", "비열등성 검정: 채점 중."]
    lab = {"psnr": "PSNR(M)", "ssim": "SSIM(M)", "lpips": "LPIPS(M)", "dino_i": "DINO-I", "clip_t": "CLIP-T", "cd": "CD"}
    rows = ["", "**단측 비열등성 검정(chief 보정 7):** 시나리오 평균 쌍 차이 d = P1 − P0(P0 2회, P1 3회), H0: 좋은 방향 기준 d ≤ −한계(E19), n = 14, α = 0.05. "
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


if __name__ == "__main__":
    {"md": lambda: print(md()), "figure": figure}[sys.argv[1]]()
