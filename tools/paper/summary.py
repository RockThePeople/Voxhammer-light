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


def md():
    L = ["## 결과 요약 (논문용 표)", "",
         "채점이 끝난 실험부터 채운다. 모든 수치는 실측이고, 원자료·정의는 아래 각 절에 있다. 생성 `tools/paper/summary.py`.", "",
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
          "### S4. X0 기존 14건 재채점(25 vs 15, P1-glb·P1-cbin) — 채점 중", "",
          "r1–r3 채점·봉합본 재채점이 끝나면 3회 평균 ± 범위로 채운다(r1 단발 결과는 3절).", "",
          "### S5. X4 무편집 재등록 체인 — 채점 중", "",
          "11자산 중 5개 채점, 나머지는 X2-B 뒤 재개.", "",
          "### S6. X2-A 3D 만 비교(14건, P0 2회 · P1 3회, 단측 비열등성) — 채점 중", "",
          "P0 실행 r2 진행 중. 판정 = 시나리오 평균 쌍 차이, H0: P1 − P0 ≤ −E19 한계(좋은 방향 기준), n = 14, α = 0.05, 우위 p 함께(chief 보정 7).", "",
          "### S7. X2-B Edit3D-Bench 3건(P0·P1 각 1회) — 실행·채점 전", "",
          "건별 지표만(FID·FVD·검정 없음).", ""]
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
