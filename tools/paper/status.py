"""논문 실험 진행 상황 페이지 (T1 뷰어 `/paper`). 읽기만 한다 — 실험 폴더를 훑어 상태를 JSON 으로 낸다.

뷰어(tools/viewer/server.py)가 VHL_PAPER_STATUS=<이 파일> 이면 불러서 `/paper`·`/api/paper`·`/pimg` 를 이 모듈에 넘긴다.
표준 라이브러리만 쓴다(뷰어와 같음). 이미지는 $RUNS_DIR/paper·$RUNS_DIR/x3 아래 .png 만 내보낸다.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = Path(os.environ.get("RUNS_DIR", "runs")).resolve()
P = RUNS / "paper"
ALLOWED = (P, RUNS / "x3")
X3_ASSETS = ("jeep", "cottage", "adventurer")
X3_PATHS = ("p0", "p1", "p2")
X4_ASSETS = ("moto", "snowman", "chair", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8")
X0_IDS = ("A-M1", "A-M2", "A-S1", "A-S2", "A-C1", "A-C2", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8")
STAGES_P0 = (("cond_render.log", "2d_render.png", "조건 렌더"), ("2d_render.png", "2d_edit.png", "인페인팅 A5000 bf16"),
             ("2d_edit.png", "output.glb", "원저자 편집(150시점 재등록 + 25스텝 + to_glb)"))
STAGES_P12 = (("rereg.log", "slat_in.safetensors", "150시점 재등록"), ("cond_render.log", "2d_render.png", "조건 렌더"),
              ("2d_render.png", "2d_edit.png", "인페인팅 3090 INT4"), ("2d_edit.png", "output.glb", "editcore 편집 + R4"))


def _j(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return None


def _rel(p):
    return str(Path(p).resolve().relative_to(RUNS))


def gpu():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5).stdout.strip().split(", ")
        return {"name": o[0], "mem_used_mib": int(o[1]), "mem_total_mib": int(o[2]), "util": int(o[3])}
    except Exception:  # noqa: BLE001
        return None


def procs():
    pat = re.compile(r"tools/paper/(x\d\w*)\.py|run_reference\.py|run_inpaint\.py|editcore\.cli|run_condition_render\.py")
    out = []
    try:
        for line in subprocess.run(["ps", "-eo", "pid,etimes,args"], capture_output=True, text=True, timeout=5).stdout.splitlines()[1:]:
            pid, et, args = line.strip().split(None, 2)
            if pat.search(args) and "bash -c" not in args:
                m = pat.search(args)
                out.append({"pid": int(pid), "elapsed_s": int(et), "what": m.group(0), "args": args[:200]})
    except Exception:  # noqa: BLE001
        pass
    return out


def tail(p, n=14):
    try:
        lines = Path(p).read_text(errors="replace").splitlines()
    except Exception:  # noqa: BLE001
        return []
    keep = [l for l in lines if l.strip() and "%|" not in l and "it/s" not in l and "Warning" not in l]
    return keep[-n:]


def x3():
    grid, walls = {}, {p: [] for p in X3_PATHS}
    for a in X3_ASSETS:                    # 실행 순서(자산별 P0 → P1 → P2)대로
        for path in X3_PATHS:
            row = []
            for k in range(0, 6):
                d = P / "x3" / path / a / f"r{k}"
                c = {"round": k, "state": "pending"}
                rj = _j(d / "round.json")
                if k == 0:
                    if (d / "output.glb").exists():
                        c["state"] = "start"
                elif rj:
                    c.update(state="done", wall_s=rj.get("wall_s"), r4=rj.get("r4"), rereg=bool(rj.get("rereg")))
                    if rj.get("wall_s"):
                        walls[path].append(rj["wall_s"])
                elif d.exists():
                    c["state"] = "running"
                    for start, end, name in (STAGES_P0 if path == "p0" else STAGES_P12):
                        if (d / start).exists() and not (d / end).exists():
                            c["stage"] = name
                    c["since_s"] = time.time() - d.stat().st_mtime
                for n in ("2d_render.png", "2d_edit.png"):
                    if (d / n).exists():
                        c[n[:-4]] = _rel(d / n)
                mi = _j(RUNS / "x3" / a / f"r{k}" / "info.json") if k else None
                if mi:
                    c["prompt"], c["part"] = mi["prompt"], mi["part"]
                    c["mask_view"] = _rel(RUNS / "x3" / a / f"r{k}" / "view.png")
                row.append(c)
            grid[f"{path}/{a}"] = row
    done = {p: sum(1 for a in X3_ASSETS for c in grid[f"{p}/{a}"][1:] if c["state"] == "done") for p in X3_PATHS}
    total_done = sum(done.values())
    eta = 0.0
    for p in X3_PATHS:
        mean = (sum(walls[p]) / len(walls[p])) if walls[p] else {"p0": 420, "p1": 150, "p2": 270}[p]
        eta += (15 - done[p]) * mean
    sheet = P / "x3" / "p2" / "rereg_sheet.png"
    running = any(c["state"] == "running" for r in grid.values() for c in r)
    return {"grid": grid, "done": done, "total": 45, "total_done": total_done, "eta_s": eta, "running": running,
            "mean_wall_s": {p: (sum(w) / len(w) if w else None) for p, w in walls.items()},
            "rereg_sheet": _rel(sheet) if sheet.exists() else None, "log": tail(P / "x3.log"),
            "scored": sorted(_j(P / "x3" / "score" / "x3_score.json") or {})}


def x4():
    r = {}
    for path in ("p0", "p1"):
        n = sum(1 for a in X4_ASSETS for k in range(1, 6) if (P / "x4" / path / a / f"r{k}" / ("round.json")).exists())
        r[path] = n
    note = None
    for a in ("moto",):
        j = _j(P / "x4" / "p1" / a / "r1" / "round.json")
        if j:
            note = j.get("note")
    scored = sorted(_j(P / "x4" / "score" / "x4_score.json") or {})   # 다른 세션의 tools/paper/x4_score.py 출력(자산별 키)
    return {"p0": r["p0"], "p1": r["p1"], "total": 55, "chain_done": (P / "x4" / "chain.done").exists(), "empty_mask_edit": note, "scored": scored}


def x0():
    n = sum(1 for s in X0_IDS for c in ("base", "c15-glb", "c15-cbin") if (P / "x0" / s / f"{c}.json").exists())
    s = _j(P / "x0" / "x0.json") or {}
    means = {}
    for cfg in ("base", "c15-glb", "c15-cbin"):
        rows = [v[cfg] for v in s.values() if cfg in v]
        if rows:
            m = lambda f: sum(f(r) for r in rows) / len(rows)  # noqa: E731
            means[cfg] = {"n": len(rows), "cd": m(lambda r: r["preserved"]["cd"]), "f1": m(lambda r: r["preserved"]["f1"]),
                          "psnr": m(lambda r: r["image"]["psnr"]), "lpips": m(lambda r: r["image"]["lpips"]),
                          "clip_t": m(lambda r: r["image"]["clip_t"]), "clip_t_front6": m(lambda r: r["image"].get("clip_t_front6") or 0)}
    return {"done": n, "total": 42, "means": means}


def paused():
    """일시정지한 채점 프로세스($RUNS_DIR/paper/paused.json)와 지금 상태(T = 정지)."""
    j = _j(P / "paused.json")
    if not j:
        return None
    for g in j.get("groups", []):
        st = []
        for pid in g.get("pids", []):
            try:
                st.append(Path(f"/proc/{pid}/stat").read_text().split(") ")[1][0])
            except Exception:  # noqa: BLE001
                st.append("없음")
        g["state"] = st
    return j


def status():
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "gpu": gpu(), "procs": procs(), "paused": paused(),
            "experiments": [
                {"id": "X0", "title": "기존 14건 재채점(25 vs 15, P1-glb·P1-cbin)", **x0()},
                {"id": "X4", "title": "무편집 재등록 체인 11자산 × 5회", **x4()},
                {"id": "X3", "title": "연속 5회 편집 3자산 × P0·P1·P2", **x3()},
                {"id": "X2-A", "title": "기존 14건 P0 × 3 (재등록 포함)", "state": "pending", "note": "X3 뒤"},
                {"id": "X2-B", "title": "Edit3D-Bench GSO 30 · PartObjaverse-Tiny 30", "state": "pending", "note": "데이터 확인 완료(내려받기 불요)"},
                {"id": "X2-C", "title": "종단(인페인팅 → 편집)", "state": "pending", "note": "P1 = 04L 1차 종단 18회(9cf1fda) 재사용"},
            ]}


def handle(h, path, q):
    """뷰어 핸들러 h 에서 부른다. 처리하면 True."""
    if path in ("/paper", "/paper/"):
        h._file(HERE / "status.html")
        return True
    if path == "/api/paper":
        h._json(status())
        return True
    if path == "/pimg":
        p = (RUNS / q.get("p", [""])[0]).resolve()
        if p.suffix.lower() == ".png" and p.is_file() and any(str(p).startswith(str(a.resolve()) + os.sep) for a in ALLOWED):
            h._file(p)
        else:
            h._send(404, b"not found", "text/plain")
        return True
    return False
