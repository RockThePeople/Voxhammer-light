"""X3 회차별 소요 시간(초) 표 — 최초 생성 + 3자산 × 5회차 × P0·P1·P2.

출처(새 편집 없음, 재계측만):
- 회차 단계 경계 = X3 실행(tools/paper/x3.py)의 단계별 산출 파일 mtime. 회차 시작 t0 = round.json mtime − wall_s.
  단계 합 = wall_s(1초 해상도).
- 단계 안 세부 = 각 단계 프로세스가 남긴 구간 기록:
  조건 렌더 cond_render/timing.json, FLUX bf16 inpaint/timing.json, INT4 inpaint_int4.json(워커 spans·created/finished),
  원저자 3D 편집 ref/**/r<k>.json(spans), P2 재등록 round.json rereg.seconds, 최초 생성 gen.json seconds.
- editcore(P1·P2 3D 편집) 세부: X3 실행은 --timing 없이 돌았다. 같은 입력(slat_in·image·mask·parent_cbin·seed 0)으로
  editcore.cli --timing 을 한 번 더 돌린 값(remeasure, RUNS/paper/x3_timing). 출력은 그 폴더에만 쓴다.

    python tools/paper/x3_timing.py remeasure      # GPU, 30회 순차
    python tools/paper/x3_timing.py tables OUT.md  # 표
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(os.environ["RUNS_DIR"])
ASSETS = Path(os.environ["ASSETS_DIR"]) / "x3" / "track_a"
X3 = RUNS / "paper" / "x3"
MASKS = RUNS / "x3"
TM = RUNS / "paper" / "x3_timing"
PY = os.environ.get("VH_PY", str(Path.home() / "ai-ar-v2/VoxHammer/.venv/bin/python"))
NAMES = {"jeep": "지프", "cottage": "오두막", "adventurer": "모험가"}


# X3 실행 중 다른 GPU 작업(채점·X0 .cbin·X4·C4)과 겹친 회차(회차 구간 안에 다른 작업의 파일 쓰기가 있었던 것).
# .viewer_cache(뷰어 미리보기) 쓰기만 있던 회차는 겹침으로 치지 않는다.
OVERLAP = [("p0", a, k) for a in NAMES for k in range(1, 6) if (a, k) != ("cottage", 5)] + \
          [("p1", "jeep", k) for k in (1, 2, 3, 4)] + [("p1", "adventurer", k) for k in (1, 2, 3)]
# 재계측은 겹친 P1 회차만(사용자 10-27). P0 는 지프 r1·r2 만 재계측이 끝났고(그 뒤 중단) 나머지는 X3 실행 값에 "겹침" 표시.
CONTAMINATED = [x for x in OVERLAP if x[0] == "p1"]


def remeasure():
    """겹친 회차만 같은 단계·같은 입력으로 다시 돌려 시간만 잰다(다른 GPU 작업 없이 순차). 출력은 TM 아래에만.
    3D 편집 입력 이미지는 원래 회차의 image/ 를 쓴다(인페인팅은 시간만 재고 결과는 3D 에 넣지 않는다 — 3D 입력을 원 회차와 같게)."""
    sys.path[:0] = [str(ROOT)]
    import numpy as np
    from tools.paper import x3
    for path, a, k in CONTAMINATED:
        d = X3 / path / a / f"r{k}"
        o = TM / path / a / f"r{k}"
        if (o / "steps.json").exists():
            continue
        if o.exists():
            subprocess.run(["rm", "-rf", str(o)])
        o.mkdir(parents=True)
        gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        if gpu:
            raise SystemExit(f"GPU 에 다른 프로세스: {gpu}")
        rec = json.loads((d / "round.json").read_text())
        md, info = x3.round_info(a, k)
        prev = X3 / path / a / f"r{k - 1}"
        st = {}
        t0 = st["t0"] = time.time()
        if path == "p0":
            C, T = np.asarray(rec["C"]), np.asarray(rec["T"])
            x3.transform_glb(prev / "output.glb", T, o / "in.glb")
            st["in.glb"] = time.time()
            x3.transform_glb(md / f"mask_r{k}.glb", C, o / "mask.glb")
            st["mask.glb"] = time.time()
            x3.condition(o, o / "in.glb", o / "mask.glb", info["edit_view"])
            st["cond_render.log"] = time.time()
            x3.inpaint_bf16(o, info["prompt"])
            st["inpaint.log"] = time.time()
            x3.sh([PY, ROOT / "reference" / "run_reference.py", "--mode", "original", "--model", o / "in.glb", "--mask_glb", o / "mask.glb",
                   "--mask_json", md / "mask.json", "--image_dir", d / "image", "--asset", f"x3-{a}-p0", "--run", f"r{k}", "--seed", 0,
                   "--runs_dir", o / "ref", "--reference", x3.REF], o / "reference.log")
            st["reference.log"] = time.time()
            work = next((o / "ref").glob(f"*/x3-{a}-p0/*/r{k}.work"))
            shutil.copy(work / "output.glb", o / "output.glb")
            if (work / "slat_tgt.safetensors").exists():
                shutil.copy(work / "slat_tgt.safetensors", o / "slat_tgt.safetensors")
            shutil.rmtree(work / "render", ignore_errors=True)
            {n: x3.sha(o / n) for n in ("2d_render.png", "2d_mask.png", "2d_edit.png", "output.glb")}   # 원 회차 마무리와 같은 일(시간만)
            x3.write_view(o, "p0")
        else:
            nz = x3.gen(a)[1]
            x3.transform_glb(prev / "output.glb", x3.affine(nz["scale"], nz["center_glb_frame"]), o / "cond_in.glb", vertex_color=True)
            st["cond_in.glb"] = time.time()
            x3.condition(o, o / "cond_in.glb", md / f"mask_r{k}.glb", info["edit_view"])
            st["cond_render.log"] = time.time()
            x3.inpaint_int4(o, info["prompt"])
            st["inpaint_int4.json"] = time.time()
            x3.sh([PY, "-m", "editcore.cli", "--slat", prev / "slat_tgt.safetensors", "--mask_json", md / "mask.json", "--image_dir", d / "image",
                   "--out", o / "output.glb", "--slat_out", o / "slat_tgt.safetensors", "--seed", 0, "--timing", o / "timing.json",
                   "--parent_cbin", prev / "cbin" / "chunks", "--cbin_out", o / "cbin"], o / "edit.log",
                  env={"VHL_R4": "1", "HF_HUB_OFFLINE": "1", **x3.ALLOC})
            st["edit.log"] = time.time()
            json.loads((o / "cbin" / "r4.json").read_text())
            {n: x3.sha(o / n) for n in ("2d_render.png", "2d_mask.png", "2d_edit.png", "output.glb")}
            x3.write_view(o, path)
        st["round.json"] = time.time()
        st["wall_s"] = st["round.json"] - t0
        st["gpu_before"] = gpu or "없음"
        (o / "steps.json").write_text(json.dumps(st, indent=1))
        for n in ("output.glb", "view.glb", "in.glb", "cond_in.glb", "slat_tgt.safetensors"):
            (o / n).unlink(missing_ok=True)
        shutil.rmtree(o / "cbin", ignore_errors=True)
        for w in (o / "ref").glob("*/*/*/*.work"):   # 큰 작업 폴더만 지우고 구간 기록 r<k>.json 은 남긴다
            shutil.rmtree(w, ignore_errors=True)
        print(f"remeasure {path} {a} r{k} {st['wall_s']:.0f}s (X3 실행 {rec['wall_s']:.0f}s)", flush=True)


# ── 표 ────────────────────────────────────────────────────────────────────────
def mt(p):
    return Path(p).stat().st_mtime if Path(p).exists() else None


def spans(j):
    out = {}
    for s in j.get("spans", []):
        out[s["name"]] = out.get(s["name"], 0.0) + s["seconds"]
    return out


def cond_parts(d, total):
    t = json.loads((d / "cond_render" / "timing.json").read_text())
    sp = spans(t)
    a, b = sp.get("render_source_5views", 0), sp.get("render_mask_5views", 0)
    return [("　└ 원본 5시점 렌더(Blender)", a), ("　└ 마스크 5시점 렌더", b), ("　└ 프로세스 기동·불러오기·후처리", total - a - b)]


def tqdm_secs(log):
    """editcore 로그의 tqdm 완료 줄 [mm:ss<…] → [(이름, 초)] (같은 줄이 두 번씩 찍힌다)."""
    txt = Path(log).read_text(errors="replace").replace("\r", "\n")
    seen, out = set(), []
    for m in re.finditer(r"(Inversing|Sampling): 100%\|[^|]*\| (\d+)/\d+ \[(\d+):(\d+)<", txt):
        key = m.group(0)
        if key in seen:
            continue
        seen.add(key)
        out.append((m.group(1), int(m.group(3)) * 60 + int(m.group(4))))
    return out


def round_rows(path, a, k):
    d = X3 / path / a / f"r{k}"
    r = json.loads((d / "round.json").read_text())
    rows, notes = [], []
    o = TM / path / a / f"r{k}"
    if (o / "steps.json").exists():                       # 재계측 회차: 단계 경계 = steps.json, 세부 = 그 폴더의 구간 기록
        st = json.loads((o / "steps.json").read_text())
        x3_wall = r["wall_s"]
        r = {**r, "wall_s": st["wall_s"], "remeasured": True, "x3_wall_s": x3_wall}
        d = o
        mt = lambda p: st.get(Path(p).name)  # noqa: E731
        end, t0 = st["round.json"], st["t0"]
        notes.append(f"X3 실행이 다른 GPU 작업과 겹쳐 같은 입력으로 다시 잰 값(X3 실행 기록 {x3_wall:.0f}초).")
        return _rows(path, a, k, d, r, mt, end, t0, rows, notes)
    if (path, a, k) in OVERLAP:
        notes.append("X3 실행 중 다른 GPU 작업(채점·X0·X4·C4)과 겹친 회차다. 값이 실제보다 크게 나왔을 수 있다(재계측 안 함).")
    mt = globals()["mt"]
    end = mt(d / "round.json")
    t0 = end - r["wall_s"]
    return _rows(path, a, k, d, r, mt, end, t0, rows, notes)


def _rows(path, a, k, d, r, mt, end, t0, rows, notes):
    edit_end_name = "decode.log" if (d / "decode.log").exists() else ("reference.log" if path == "p0" else "edit.log")
    if path == "p1" and a == "cottage" and k == 4:       # 첫 실행이 디코딩에서 죽고, 디코딩만 나중에 이어 함 → 두 구간을 이어 붙인다
        t0 = mt(d / "cond_in.glb") - 0.5
        gap0, gap1 = mt(d / "edit.log"), end - r["wall_s"]
        notes.append(f"첫 실행이 메시 디코딩 중 GPU 메모리 부족으로 끝나고, {(gap1 - gap0) / 60:.0f}분 뒤 디코딩만 새 프로세스로 이어 했다. "
                     "그 사이 대기는 합계에서 뺐다. 회차 시작은 cond_in.glb 저장 0.5초 전으로 잡았다(입력 변환 실측 0–2초).")
    cur = t0

    def step(name, until, parts=()):
        nonlocal cur
        s = until - cur
        rows.append((name, s, True))
        for pn, pv in parts:
            rows.append((pn, pv, False))
        cur = until

    if path == "p2":
        rr = r["rereg"]["seconds"]
        e = mt(d / "rereg.log")
        tot = e - cur
        step("0. 재등록(150시점 렌더 → 특징 → SLat 인코딩)", e,
             [("　└ 150시점 렌더(Blender)", rr["render_150views"]), ("　└ DINOv2 특징 추출", rr["feature_extraction"]),
              ("　└ 복셀화·모델 로드·SLat 인코딩·프로세스 기동(나눠 재지 않음)", tot - rr["render_150views"] - rr["feature_extraction"])])
    inp = "in.glb" if path == "p0" else "cond_in.glb"
    prep_end = max(x for x in (mt(d / inp), mt(d / "mask.glb")) if x)
    step("1. 입력 준비(좌표 변환·GLB 저장)", prep_end)
    ce = mt(d / "cond_render.log")
    cp = cond_parts(d, ce - cur)
    if cp[2][1] > 60:
        notes.append(f"조건 렌더 단계에 렌더 구간 밖 시간이 {cp[2][1]:.0f}초 있다. Blender 렌더가 시작되기 전의 대기로, 원인은 기록에 없다(다른 GPU 작업과 겹친 때).")
    step("2. 조건 렌더(편집 시점 2D 렌더·마스크, 5시점)", ce, cp)
    if path == "p0":
        ie = mt(d / "inpaint.log")
        sp = spans(json.loads((d / "inpaint" / "timing.json").read_text()))
        tot = ie - cur
        step("3. 2D 인페인팅(FLUX.1-Fill bf16, A5000, CPU 오프로드)", ie,
             [("　└ 모델 불러오기", sp["flux_load"]), ("　└ 인페인팅 50 스텝", sp["flux_inpaint"]),
              ("　└ 프로세스 기동·저장", tot - sp["flux_load"] - sp["flux_inpaint"])])
        re_ = mt(d / "reference.log")
        j = json.loads(next((d / "ref").glob(f"*/x3-{a}-p0/*/r{k}.json")).read_text())
        sp = spans(j)
        tot = re_ - cur
        g = lambda *ns: sum(sp.get(n, 0.0) for n in ns)  # noqa: E731
        parts = [("　└ 모듈 import", g("import_modules")), ("　└ 모델 로드", g("model_load")),
                 ("　└ 재등록: 150시점 렌더", g("render_150views")), ("　└ 재등록: DINOv2 특징 추출", g("feature_extraction")),
                 ("　└ 재등록: SLat 인코딩(복셀 좌표 포함)", g("ply_to_coords", "slat_encode")),
                 ("　└ 3D 마스크 복셀화(mask.glb)", g("mask_glb_voxelize")),
                 ("　└ 조건 이미지 전처리·인코딩", g("preprocess_image", "get_cond")),
                 ("　└ Stage 1 inversion(25)", g("stage1_inverse")), ("　└ Stage 1 denoising(25)", g("stage1_denoise")),
                 ("　└ Stage 2 inversion(25)", g("stage2_inverse")), ("　└ Stage 2 denoising(25)", g("stage2_denoise")),
                 ("　└ 디코딩(메시·가우시안)", g("decode_mesh", "decode_gaussian")),
                 ("　└ to_glb(텍스처 굽기)", g("to_glb")), ("　└ GLB 저장", g("glb_export"))]
        named = sum(v for _, v in parts)
        parts.append(("　└ 기타(마스크 변환·프로세스 기동/종료 등 구간 밖)", tot - named))
        step("4. 3D 편집(원저자 VoxHammer, 재등록 포함)", re_, parts)
    else:
        ij = json.loads((d / "inpaint_int4.json").read_text())
        ie = mt(d / "inpaint_int4.json")
        sp = spans(ij["result"])
        tot = ie - cur
        w = ij["finished"] - ij["started"]
        step("3. 2D 인페인팅(FLUX.1-Fill INT4, RTX 3090 워커, HTTP)", ie,
             [("　└ 워커 인페인팅", sp.get("inpaint", 0.0)), ("　└ 워커 합성(마스크 밖 원본 복원)", sp.get("composite", 0.0)),
              ("　└ 워커 기타(대기·저장)", w - sp.get("inpaint", 0.0) - sp.get("composite", 0.0)),
              ("　└ 업로드·다운로드·요청 처리", tot - w)])
        ee = mt(d / edit_end_name)
        tot = ee - cur
        tq = tqdm_secs(d / "edit.log")
        parts = []
        if len(tq) >= 4 and not (d / "timing.json").exists():
            parts = [("　└ (실행 로그) Stage 1 inversion(15)", tq[0][1]), ("　└ (실행 로그) Stage 1 denoising(15)", tq[1][1]),
                     ("　└ (실행 로그) Stage 2 inversion(15)", tq[2][1]), ("　└ (실행 로그) Stage 2 denoising(15)", tq[3][1])]
            parts.append(("　└ (실행 로그) 그 밖(모델 로드·전처리·디코딩·R4·GLB) 합", tot - sum(v for _, v in parts)))
        if path == "p1" and a == "cottage" and k == 4:
            el = mt(d / "edit.log")
            first = el - cur
            dec = ee - (end - r["wall_s"])
            parts = [("　└ (실행 로그) 첫 프로세스(편집 완료 후 디코딩 중 메모리 부족 종료)", first),
                     ("　└ (실행 로그) 디코딩만 새 프로세스(R4)", dec)]
            tot = first + dec
            rows.append(("4. 3D 편집(editcore 15/15 + R4 + 봉합)", tot, True))
            rows.extend((n, v, False) for n, v in parts)
            cur = ee
        else:
            step("4. 3D 편집(editcore 15/15 + R4 + 봉합)", ee, parts)
        if (d / "decode.log").exists() and not (path == "p1" and a == "cottage" and k == 4):
            notes.append("editcore 프로세스가 편집을 마친 뒤 메시 디코딩 중 GPU 메모리 부족으로 끝나서, 같은 회차 안에서 디코딩만 새 프로세스로 이어 했다(4단계에 포함).")
        tm = d / "timing.json"
        if tm.exists():
            jt = json.loads(tm.read_text())
            s2 = spans(jt)
            rm = [("model_load", "모델 로드"), ("load_slat", "입력 SLat 불러오기"), ("preprocess_image", "조건 이미지 전처리"),
                  ("get_cond", "조건 인코딩"), ("masks", "마스크 준비"), ("stage1_inverse", "Stage 1 inversion(15)"),
                  ("stage1_denoise", "Stage 1 denoising(15)"), ("slat_mask", "SLat 마스크"), ("stage2_inverse", "Stage 2 inversion(15)"),
                  ("stage2_denoise", "Stage 2 denoising(15)"), ("decode_mesh", "메시 디코딩"), ("bookkeeping", "R4 부기"),
                  ("chunk_partition", "청크 분할"), ("cbin_encode", "새 청크 부호화"), ("seam_snap", "경계 스냅"),
                  ("seam_stitch", "경계 봉합 띠"), ("cbin_write", ".cbin 쓰기"), ("to_glb", "GLB 조립(glbbuild)"), ("glb_export", "GLB 저장")]
            used = 0.0
            for key, nm in rm:
                if key in s2:
                    rows.append((f"　└ (재계측) {nm}", s2[key], False))
                    used += s2[key]
            rest = [n for n in s2 if n not in dict(rm)]
            if rest:
                v = sum(s2[n] for n in rest)
                rows.append((f"　└ (재계측) 기타 구간({', '.join(rest)})", v, False))
                used += v
            rows.append(("　└ (재계측) 구간 밖(import·프로세스 기동 등)", jt["process_wall_seconds"] - used, False))
            rows.append(("　└ (재계측) editcore 프로세스 합계", jt["process_wall_seconds"], False))
    fe = end
    rows.append(("5. 마무리(출력 복사·sha256·보기용 GLB·기록)", fe - cur, True))
    total = sum(v for _, v, top in rows if top)
    return rows, total, r, notes, datetime.fromtimestamp(t0)


def gen_rows(a):
    g = json.loads((ASSETS / a / "gen" / "gen.json").read_text())
    s = g["seconds"]
    rows = [("파이프라인(TRELLIS) 로드", g["pipeline_load_seconds"]), ("샘플링(Stage 1 + Stage 2, 각 25 스텝)", s["sample"]),
            ("디코딩(메시·가우시안)", s["decode"]), ("R4 .cbin 부호화(P1·P2 출발점)", s["cbin"]), ("to_glb(텍스처 GLB, P0 출발점)", s["to_glb"])]
    return rows, g


def fmt(v):
    return f"{v:.1f}"


def tables(out_md):
    L = []
    L.append("## 최초 생성(TRELLIS image-large, A5000) — 자산별 초\n")
    L.append("| 단계 | " + " | ".join(NAMES.values()) + " |\n|---|" + "---:|" * 3)
    gr = {a: gen_rows(a)[0] for a in NAMES}
    for i, (n, _) in enumerate(gr["jeep"]):
        L.append(f"| {n} | " + " | ".join(fmt(gr[a][i][1]) for a in NAMES) + " |")
    L.append("| **합계** | " + " | ".join(f"**{fmt(sum(v for _, v in gr[a]))}**" for a in NAMES) + " |")
    L.append("| 합계(로드 제외) | " + " | ".join(fmt(sum(v for _, v in gr[a][1:])) for a in NAMES) + " |\n")
    for a, nm in NAMES.items():
        for k in range(1, 6):
            res = {p: round_rows(p, a, k) for p in ("p0", "p1", "p2")}
            prompt = res["p0"][2]["prompt"]
            L.append(f"### {nm}({a}) {k}회차 — \"{prompt}\"\n")
            L.append("| 단계 | P0(원저자) | P1(제안) | P2(P1 + 재등록) |\n|---|---:|---:|---:|")
            keyed, names = {}, []
            for p in ("p0", "p1", "p2"):
                st = 0
                for n, v, top in res[p][0]:
                    if top:
                        st = _rank(n)
                    keyed[(p, st, n)] = (v, top)
                    if (st, n) not in names:
                        names.append((st, n))
            for st, n in sorted(names, key=lambda x: (x[0], names.index(x))):
                cells = []
                top = False
                for p in ("p0", "p1", "p2"):
                    hit = [keyed[(p, st, n)]] if (p, st, n) in keyed else []
                    if hit:
                        cells.append(f"**{fmt(hit[0][0])}**" if hit[0][1] else fmt(hit[0][0]))
                        top = top or hit[0][1]
                    else:
                        cells.append("–")
                L.append(f"| {'**' + n + '**' if top else n} | " + " | ".join(cells) + " |")
            L.append("| **회차 합계** | " + " | ".join(f"**{fmt(res[p][1])}**" for p in ("p0", "p1", "p2")) + " |")
            L.append("| 출처 | " + " | ".join("재계측" if res[p][2].get("remeasured") else "X3 실행" for p in ("p0", "p1", "p2")) + " |")
            L.append("| 참고: X3 실행 기록 wall_s | " + " | ".join(fmt(res[p][2].get("x3_wall_s", res[p][2]["wall_s"])) for p in ("p0", "p1", "p2")) + " |")
            L.append("| 시작 시각 | " + " | ".join(f"{res[p][4]:%m-%d %H:%M:%S}" for p in ("p0", "p1", "p2")) + " |")
            for p in ("p0", "p1", "p2"):
                for t in res[p][3]:
                    L.append(f"\n- {p.upper()}: {t}")
            L.append("")
    Path(out_md).write_text("\n".join(L) + "\n")
    print("tables", out_md)


def per_path(out_dir):
    """자산 × 회차 × 경로마다 표 하나(45개) + 자산별 최초 생성 표(3개): CSV 각각 + 모두 담은 README.md."""
    import csv
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    PN = {"p0": "P0(원저자)", "p1": "P1(제안)", "p2": "P2(P1 + 재등록)"}
    L = ["# X3 단계별 소요 시간 — 자산 × 회차 × 경로", "",
         "단위 초. 굵은 줄 = 단계(합 = 회차 합계), `└` 줄 = 그 단계 안 세부. 출처·방법·한계는 [../10_X3_회차별_소요시간.md](../10_X3_회차별_소요시간.md) 앞부분.",
         "CSV 는 같은 폴더 `<자산>_r<회차>_<경로>.csv`(열: 단계, 세부, 초). 최초 생성은 `<자산>_r0_gen.csv`.", ""]
    for a, nm in NAMES.items():
        L += [f"## {nm}({a})", ""]
        rows, g = gen_rows(a)
        with open(out_dir / f"{a}_r0_gen.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["단계", "세부", "초"])
            for n, v in rows:
                w.writerow([n, "", f"{v:.1f}"])
            w.writerow(["합계", "", f"{sum(v for _, v in rows):.1f}"])
        L += [f"### {nm} 최초 생성(TRELLIS image-large, A5000, seed {g['seed']})", "", "| 단계 | 초 |", "|---|---:|"]
        L += [f"| {n} | {fmt(v)} |" for n, v in rows] + [f"| **합계** | **{fmt(sum(v for _, v in rows))}** |", ""]
        for k in range(1, 6):
            for p in ("p0", "p1", "p2"):
                rows, total, r, notes, t0 = round_rows(p, a, k)
                with open(out_dir / f"{a}_r{k}_{p}.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["단계", "세부", "초"])
                    st = ""
                    for n, v, top in rows:
                        if top:
                            st = n
                            w.writerow([n, "", f"{v:.1f}"])
                        else:
                            w.writerow([st, n.replace("　└ ", ""), f"{v:.1f}"])
                    w.writerow(["회차 합계", "", f"{total:.1f}"])
                src = "재계측(10-27)" if r.get("remeasured") else "X3 실행(10-26)"
                L += [f"### {nm} {k}회차 · {PN[p]}", "", f"프롬프트: \"{r['prompt']}\" · 출처: {src} · 시작 {t0:%m-%d %H:%M:%S}", "",
                      "| 단계 | 초 |", "|---|---:|"]
                for n, v, top in rows:
                    L.append(f"| **{n}** | **{fmt(v)}** |" if top else f"| {n} | {fmt(v)} |")
                L.append(f"| **회차 합계** | **{fmt(total)}** |")
                L += [f"\n- {t}" for t in notes] + [""]
    (out_dir / "README.md").write_text("\n".join(L) + "\n")
    print("per_path", len(list(out_dir.glob("*.csv"))), "csv")


def _rank(n):
    m = re.match(r"\s*(\d)\.", n)
    if m:
        return int(m.group(1)) * 10
    return 99


if __name__ == "__main__":
    {"remeasure": remeasure, "tables": lambda: tables(sys.argv[2]), "per_path": lambda: per_path(sys.argv[2])}[sys.argv[1]]()
