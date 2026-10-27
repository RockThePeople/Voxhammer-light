"""paper-results 마지막 커밋: 사람 평가 화면·보고서 그림·표(CSV + 원값 JSON)·MANIFEST·SHA256SUMS.

    python tools/paper/capture/finish.py <out>
CSV 는 보고서(docs/reports/X-paper-a5000.md) 맨 앞 "결과 요약" 표를 그대로 옮기고(S1 의 빈 자리 열은 뺀다), 같은 원값 JSON 으로
요약 생성기(tools/paper/summary.py)를 다시 돌린 결과와 칸마다 대조한다(불일치 수를 _build/finish.json 에).
JSON 의 내부 절대경로는 $RUNS_DIR·$ASSETS_DIR·$REPO 로 바꾼다.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tools" / "paper" / "capture")]
RUNS = Path(os.environ["RUNS_DIR"])
ASSETS = Path(os.environ["ASSETS_DIR"])
P = RUNS / "paper"
REPORT = ROOT / "docs" / "reports" / "X-paper-a5000.md"


def scrub(text):
    return (text.replace(str(RUNS), "$RUNS_DIR").replace(str(ASSETS), "$ASSETS_DIR").replace(str(ROOT.parent / "Voxhammer-light"), "$MAIN")
            .replace(str(ROOT), "$REPO").replace(str(Path.home()), "~"))


def md_tables(md):
    """'### S…' 절마다 markdown 표들을 [(제목, [행…])] 로."""
    out, cur, sec, rows = [], None, None, []
    for line in md.splitlines():
        if line.startswith("### S"):
            if rows:
                out.append((cur, rows)); rows = []
            cur = sec = line[4:].strip()
        elif line.startswith("**") and line.endswith("**") and sec and sec.startswith("S1"):
            if rows:
                out.append((cur, rows)); rows = []
            cur = sec.split(" (")[0] + " " + line.strip("*")
        elif line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            rows.append(cells)
        elif rows and not line.startswith("|"):
            out.append((cur, rows)); rows = []
    if rows:
        out.append((cur, rows))
    return out


def tables(out):
    t = out / "tables"
    (t / "raw").mkdir(parents=True, exist_ok=True)
    md = REPORT.read_text()
    s = md[md.index("<!-- summary:start -->"): md.index("<!-- summary:end -->")]
    files = []
    tabs = md_tables(s)
    for i, (title, rows) in enumerate(tabs):
        hdr, body = rows[0], rows[1:]
        drop = [j for j, h in enumerate(hdr) if any(r[j] == "채점 중" for r in body if j < len(r)) and all(r[j] == "채점 중" for r in body if j < len(r))]
        hdr2 = [h for j, h in enumerate(hdr) if j not in drop]
        body2 = [[c.replace("**", "") for j, c in enumerate(r) if j not in drop] for r in body]
        name = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", title)[:60].strip("_")
        p = t / f"{i:02d}_{name}.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([h.replace("**", "") for h in hdr2])
            w.writerows(body2)
        files.append({"csv": p.name, "title": title, "rows": len(body2), "dropped_empty_columns": [hdr[j] for j in drop]})
    # 대조: 같은 원값 JSON 으로 요약 생성기를 다시 돌린 표 vs 보고서 표(칸마다)
    regen = subprocess.run([sys.executable, str(ROOT / "tools" / "paper" / "summary.py"), "md"], capture_output=True, text=True,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, cwd=ROOT).stdout
    t2 = md_tables(regen)
    mism, cells = [], 0
    for (ta, ra), (tb, rb) in zip(tabs, t2):
        for x, y in zip(ra, rb):
            for a, b in zip(x, y):
                cells += 1
                if a != b:
                    mism.append({"table": ta, "report": a, "regenerated": b})
    if len(tabs) != len(t2):
        mism.append({"table": "*", "report": len(tabs), "regenerated": len(t2)})
    raws = {"x3_score.json": P / "x3" / "score" / "x3_score.json", "x3_score_stitched.json": P / "x3" / "score_stitched" / "x3_score.json",
            "x3_surface_iou.json": P / "x3" / "score" / "surface_iou.json", "c4_jeep_color.json": P / "x3" / "c4" / "jeep_color.json",
            "c4_cottage_color.json": P / "x3" / "c4" / "cottage_color.json", "c4_adventurer_color.json": P / "x3" / "c4" / "adventurer_color.json",
            "c4_jeep_seam_variants.json": P / "x3" / "c4" / "jeep_seam_variants.json", "c4_jeep_stitch_darkpx.json": P / "x3" / "c4" / "jeep_stitch_darkpx.json",
            "x0_r1.json": P / "x0" / "x0.json", "x0_runs.json": P / "x0" / "x0_runs.json", "x0_runs_stitched.json": P / "x0" / "x0_runs_st.json",
            "x4_score.json": P / "x4" / "score" / "x4_score.json", "x2a_p0.json": P / "x2a" / "score" / "x2a_p0.json",
            "x2a_noninferiority.json": P / "x2a" / "score" / "tost.json", "x2b.json": P / "x2b" / "score" / "x2b.json",
            "x3_eval_choices.json": P / "x3" / "eval" / "choices.json", "x3_eval_blind_key.json": P / "x3" / "blind_key.json"}
    raw_list = []
    for n, src in raws.items():
        if src.exists():
            (t / "raw" / n).write_text(scrub(src.read_text()))
            raw_list.append({"file": f"raw/{n}", "source": scrub(str(src)), "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest()})
    return {"csv": files, "raw": raw_list, "cells_compared": cells, "mismatches": mism}


def evalpage(out):
    e = out / "eval"
    src = P / "x3" / "eval"
    for d in sorted(src.iterdir()):
        if d.is_dir():
            shutil.copytree(d, e / "items" / d.name, dirs_exist_ok=True)
    shutil.copyfile(src / "items.json", e / "items.json")
    shutil.copyfile(src / "choices.json", e / "choices.json")
    shutil.copyfile(P / "x3" / "blind_key.json", e / "blind_key.json")
    shot = e / "x3_eval_page.png"
    r = subprocess.run(["google-chrome", "--headless=new", "--no-sandbox", "--hide-scrollbars", f"--screenshot={shot}", "--window-size=1400,16000",
                        "--virtual-time-budget=30000", "http://localhost:8083/x3-eval"], capture_output=True, text=True, timeout=180)
    return {"items": len([d for d in (e / "items").iterdir() if d.is_dir()]), "page_screenshot": shot.exists(), "chrome_rc": r.returncode}


def report_imgs(out):
    d = out / "img"
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted((ROOT / "docs" / "reports" / "img").glob("X-paper-*")):
        shutil.copyfile(p, d / p.name)
        n += 1
    return n


def silhouette_excl(out):
    """알파 IoU < 0.90 쌍: 편집 마스크(∪ halo 2) 투영 화소를 뺀 영역의 IoU 로 시점 어긋남과 편집 몫을 가른다."""
    import numpy as np
    from PIL import Image
    import capture as C
    from build import _asset_of, mask_cells
    from deltacontract.coords import dilate_cells
    from tools.paper_metrics import silhouette
    from tools.paper.x0 import scenarios
    from tools.paper.x3 import round_info
    reg, cam, S = C.registry(), C.cams(), scenarios()
    alpha = lambda p: np.asarray(Image.open(p).convert("RGB")).min(2) < 250   # noqa: E731
    res = []
    for e in ("x3", "x0", "x2a", "x2b"):
        bp = out / "_build" / f"{e}.json"
        if not bp.exists():
            continue
        for sl in json.loads(bp.read_text())["silhouette"]:
            if sl["alpha_iou"] >= 0.9:
                continue
            sid = sl["asset"]
            if e == "x3":
                a, mp, A, B = sid, round_info(sid, 1)[0] / "mask.json", out / e / sid / "original" / "r0", out / e / sid / sl["path"] / "r1"
            elif e == "x2b":
                a, mp, A, B = sid, ASSETS / "x2b" / sid / "mask.json", out / e / sid / "original", out / e / sid / sl["path"]
            else:
                a, mp, A, B = _asset_of(sid), S[sid]["mask"], out / e / sid / "original", out / e / sid / sl["path"] / "r1"
            ex = silhouette(np.asarray(dilate_cells(mask_cells(mp), 2)), C.frame(reg, a), (cam[a]["elev"], cam[a]["az"]), res=1024)
            x, y = alpha(A / "shaded.png"), alpha(B / "shaded.png")
            k = ~ex
            res.append({**sl, "iou_outside_edit": float((x & y & k).sum() / max(((x | y) & k).sum(), 1))})
    return res


def manifest(out, info):
    sys.path.insert(0, str(ROOT / "tools" / "paper" / "capture"))
    import capture as C
    cam = json.loads(C.CAMS.read_text())
    reg = C.registry()
    b = {e: json.loads((out / "_build" / f"{e}.json").read_text()) for e in ("x3", "stitch", "x4", "x0", "x2a", "x2b") if (out / "_build" / f"{e}.json").exists()}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    L = ["# paper-results — 논문 캡처·표 (X-paper, a5000)", "",
         f"생성: `tools/paper/capture/{{capture,bpy_render,build,finish}}.py`, 저장소 `item/X-paper-a5000` 커밋 `{commit}`. 새 편집·새 채점 없음 — 보고서 `0531459` 기준 기존 출력만 다시 찍었다.", "",
         "## 렌더 설정", "",
         "- 렌더러: Blender Cycles(Edit3D-Bench `render.py` 와 같은 bpyrenderer 설정 — 64 샘플, 디노이즈, 확산·광택 반사 1회), GPU(OptiX). 1024×1024.",
         "- 조명: HDRI `brown_photostudio_02_1k.exr`(Bench 와 같음), 바닥 없음(그림자 없음).",
         "- shaded = 고정 조명, glTF 재질 그대로(Blender 기본 색 변환). albedo = 재질의 Base Color(텍스처·정점색 연결)를 Emission(세기 1)으로, 월드 조명 0, 색 변환 Standard.",
         "- 배경: 투명 RGBA 로 찍고 알파로 흰색(255,255,255) 위에 합성한 불투명 RGB PNG(가장자리 검은 테두리 없음). 이미지 안에 글자 없음(라벨은 sheets/ 의 시트에만).",
         "- crop = 3D 마스크 셀 bbox 8 꼭짓점을 같은 카메라로 투영한 상자 + 양쪽 15% 여유, 정사각형, 1024² 로 확대(X4 제외). X3 원본 r0 는 회차 마스크마다 crop_r<k>.png.",
         "- P1·P2(정점색 R4 메시)는 output.glb 를 N 좌표로 옮긴 뒤 (iv) 보정(`editcore/render.py:vertex_color_render_glb`) — view.glb 와 같은 보정이다. view.glb 를 trimesh 로 옮기면 보정 재질·선형 float 정점색이 빠져 검게 나와서 이렇게 했다. P0 는 to_glb 출력.", "",
         "## 카메라·좌표", "",
         "- 카메라: Bench/포크 정의 — 반지름 1.8, 원점 주시, 렌즈 35 mm·센서 32 mm. az = Bench 방위(Blender 방위 = az − 90°). 정면은 원본 r0 의 고도 10° 8방향 렌더에서 정했다(cameras.json).",
         "- 캐릭터 = 정면(az = 정면, 고도 10°), 사물 = 3/4 정면(az = 정면 + 45°, 고도 25°). 모든 경로·회차·실행에 같은 카메라.",
         "- 좌표: 자산마다 v0 메시 bbox 로 정한 N 좌표(중심 원점, 최대 변 1; `tools/paper_metrics.py` AssetFrame) 하나에 모든 결과를 옮긴다. raw(R4·editcore) → v0: (p − raw_c)·raw_s(트랙 A·X3 gen.json, 트랙 B·X2-B 항등) → N: (p − t0_c)·t0_s.",
         "- P0: X3 = round.json 의 누적 변환 C_k(v0 → 그 회차 입력 좌표)의 역행렬로 v0 에 되돌림(output_v0.glb). X2-A·X2-B = 입력에 건 정규화 T(prep.json) 좌표 = N(같은 v0 bbox 변환). X4 = 채점과 같이 N 으로 본다(회차마다의 재정규화 표류는 되돌리지 않음 — 실루엣 확인 참조).", "",
         "| 자산 | 분류 | 정면 az | 카메라 az | 고도 | 근거 |", "|---|---|---|---|---|---|"]
    for a, c in cam["assets"].items():
        L.append(f"| {a} | {'캐릭터' if c['class'] == 'character' else '사물'} | {c['front_az']} | {c['az']} | {c['elev']} | {c['front_reason']} |")
    L += ["", "**좌표 변환(자산 N 좌표):**", "", "| 자산 | v0 원본 | raw_c | raw_s | t0_c | t0_s |", "|---|---|---|---|---|---|"]
    for a in reg:
        f = C.frame(reg, a).as_dict()
        L.append(f"| {a} | `{scrub(str(reg[a]['v0']))}` | {[round(x, 6) for x in f['raw_c']]} | {f['raw_s']:.6f} | {[round(x, 6) for x in f['t0_c']]} | {f['t0_s']:.6f} |")
    L += ["", "**X3 P0 누적 변환 C_k(v0 → 그 회차 좌표, 렌더는 C_k⁻¹ 적용):** `x3/<자산>/p0/r<k>/` 의 원본 round.json 값.", ""]
    for a in ("jeep", "cottage", "adventurer"):
        for k in range(1, 6):
            rj = P / "x3" / "p0" / a / f"r{k}" / "round.json"
            if rj.exists():
                Cm = json.loads(rj.read_text())["C"]
                L.append(f"- {a} r{k}: scale {Cm[0][0]:.6f}, translate {[round(Cm[i][3], 6) for i in range(3)]}")
    L += ["", "**X2-A·X2-B P0 정규화 T(prep.json):**", ""]
    for d in sorted((P / "x2a" / "p0").glob("*/prep.json")) + sorted((P / "x2b" / "p0").glob("*/prep.json")):
        T = json.loads(d.read_text())["T"]
        L.append(f"- {d.parent.parent.name}/{d.parent.name}: scale {T[0][0]:.6f}, translate {[round(T[i][3], 6) for i in range(3)]}")
    L += ["", "## 폴더 → 실험·표", "",
          "| 폴더 | 내용 | 보고서 |", "|---|---|---|",
          "| `x3/<자산>/{original/r0, p0, p1, p2}/r<k>/` | X3 연속 5회 편집(P1·P2 = 봉합본), `cond/r<k>/<경로>/` 조건 이미지·prompt.txt | 4절, 요약 S1·S2 |",
          "| `stitch/<자산>/{p1,p2}_prestitch/`, `stitch/jeep/{p1_skirt, p1_stitch_skirt, p1_fulldecode}/` | 봉합 전·치마 벽 시험·통째 디코딩 | 10·10-1절 |",
          "| `x4/<자산>/{p0/r0–r5, p1/r1-5}/` | 무편집 재등록(P1 r1–r5 는 같은 메시라 1회) | 2절, 요약 S5 |",
          "| `x0/<시나리오>/{original, base25/r1, c15_glb/r1, c15_cbin/r1}/` | X0 r1(c15_cbin = R4 + 봉합, 채택본) | 3절, 요약 S4 |",
          "| `x2a/<시나리오>/{original, p0/r1–r2, p1_glb/r1–r3, p1_cbin/r1–r3}/`, `cond/` | X2-A, 동결 조건 이미지 | 5절, 요약 S6 |",
          "| `x2b/<건>/{original, p0, p1}/`, `cond/` | X2-B, 사용한 조건 이미지(데이터셋 2D 또는 FLUX) | 요약 S7 |",
          "| `eval/` | 사람 평가 x3-eval: 평가자가 본 익명 칸 이미지(items/), 페이지 캡처, items.json, choices.json, blind_key.json(칸 ↔ 경로) | 요약 S3 |",
          "| `sheets/` | 경로 × 회차 시트(라벨), 실루엣 확인(빨강 = 원본만, 파랑 = 결과만, 회색 = 겹침), 시점 확인 | — |",
          "| `img/` | 보고서 그림 원본 해상도 | 각 절 |",
          "| `tables/` | 요약 S1–S7 CSV(S1 빈 자리 열 제거), `raw/` 원값 JSON(경로는 $RUNS_DIR 등으로 바꿈) | 요약 |", ""]
    # 실루엣
    sil = [s for e in b.values() for s in e.get("silhouette", [])]
    L += ["## 실루엣 확인(원본 대 각 경로 r1, 알파 IoU)", "",
          f"- {len(sil)} 쌍, 최소 {min(s['alpha_iou'] for s in sil):.3f}, 0.90 미만 {sum(s['alpha_iou'] < 0.9 for s in sil)} 쌍(시점 어긋남 판정 기준 0.90).", ""]
    low = sorted(sil, key=lambda s: s["alpha_iou"])[:8]
    ex = info.get("silhouette_excl", [])
    L += [f"- 0.90 미만 쌍은 모두 편집량이 큰 시나리오다. 편집 마스크(∪ halo 2) 투영 화소를 빼고 다시 재면 최소 "
          f"{min([x['iou_outside_edit'] for x in ex] or [1]):.3f} — **시점 어긋남 0 건**(낮은 값은 편집으로 실루엣이 바뀐 몫).", "",
          "| 실험 | 자산·시나리오 | 경로 | 알파 IoU | 편집 영역 밖 알파 IoU |", "|---|---|---|---|---|"]
    L += [f"| {x['exp']} | {x['asset']} | {x['path']} | {x['alpha_iou']:.3f} | {x['iou_outside_edit']:.3f} |" for x in sorted(ex, key=lambda x: x["alpha_iou"])] + [""]
    if "x4" in b:
        L += ["**X4 P1 r1–r5 output.glb sha256 동일 여부:** " + ", ".join(f"{a} {'같음' if v['identical'] else '다름'}" for a, v in b["x4"].get("x4_p1_sha", {}).items()), ""]
    miss = [m for e in b.values() for m in e.get("missing", [])]
    occ = [o for e in b.values() for o in e.get("occluded", [])]
    L += ["## 없음(기존 출력이 없어 만들지 않은 것)", ""] + ([f"- {m['exp']}: `{scrub(m['dest'])}` ← `{scrub(m['mesh'])}`" for m in miss] or ["- 없음"]) + [""]
    L += ["## 편집 영역 가려짐(그 시점에서 편집 영역이 보이지 않음, 시점은 늘리지 않음)", "",
          f"판정: 원본 메시를 그 카메라로 256² 레이캐스팅해 마스크(∪1셀) 셀에 닿는 화소가 50 미만.", ""]
    L += [f"- {o['item']}: 보이는 화소 {o['visible_px_256']}" for o in occ] or ["- 없음"]
    L += ["", "## 모델 원본(메시 sha256)", "", "| 실험 | 결과 폴더 | 원본 메시 | 좌표 | sha256 | 비고 |", "|---|---|---|---|---|---|"]
    for e, d in b.items():
        for m in d.get("models", []):
            L.append(f"| {e} | `{scrub(m['dest']).split('vhl-paper-results/')[-1]}` | `{scrub(m['mesh'])}` | {m['kind']}{' + (iv)' if m['vertex_color'] else ''} | `{m['sha256'][:16]}…` | {m['note']} |")
    L += ["", "## 표·대조", "", f"- CSV {len(info['tables']['csv'])}개, 원값 JSON {len(info['tables']['raw'])}개(tables/raw/, 원본 sha256 은 tables/raw_sources.json).",
          f"- 보고서 요약 표 대 같은 원값으로 다시 만든 표: {info['tables']['cells_compared']} 칸 대조, 불일치 {len(info['tables']['mismatches'])} 칸.",
          "- 모든 파일의 sha256: `SHA256SUMS`.", ""]
    (out / "MANIFEST.md").write_text("\n".join(L) + "\n")
    (out / "tables" / "raw_sources.json").write_text(json.dumps(info["tables"]["raw"], ensure_ascii=False, indent=1))


def sums(out):
    lines = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and ".git" not in p.parts and "_build" not in p.parts and p.name != "SHA256SUMS":
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return len(lines)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    info = {"tables": tables(out), "eval": evalpage(out), "img": report_imgs(out), "silhouette_excl": silhouette_excl(out)}
    manifest(out, info)
    info["files"] = sums(out)
    (out / "_build" / "finish.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print("finish", {k: (v if k != "tables" else {"csv": len(v["csv"]), "raw": len(v["raw"]), "cells": v["cells_compared"], "mismatch": len(v["mismatches"])}) for k, v in info.items()})
