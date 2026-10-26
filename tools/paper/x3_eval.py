"""X3 사람 평가 페이지 `x3-eval` (ops/tasks/X-paper.md X3 "사람 평가"). 표준 라이브러리만(뷰어와 같다).

prep (X3 가 끝난 뒤, a5000 venv):
    python tools/paper/x3_eval.py prep
  자산 × 회차(15 항목)마다 P0·P1·P2 결과를 v0 → N 좌표로 옮겨 Bench render.py 로 렌더하고(정점색은 (iv) 보정), 편집 시점·3/4 시점
  두 장을 항목 폴더에 **익명 칸 이름(A·B·C)** 으로 복사한다. 칸 ↔ 경로 순서는 항목마다 무작위(secrets)이고, 키는
  $RUNS_DIR/paper/x3/blind_key.json 에만 둔다(이 페이지·보고서는 키를 읽지 않는다). 항목 목록 $RUNS_DIR/paper/x3/eval/items.json.
뷰어 연결: 서버가 이 모듈의 handle(h, path, q)·handle_post(h, path, q) 를 부른다.
  GET  /x3-eval              평가 페이지(x3_eval.html)
  GET  /api/x3eval           항목 목록 + 저장된 선택
  GET  /x3img?p=<상대경로>    eval 폴더 안 이미지
  POST /api/x3eval/choice    {"item", "question": "preserve"|"prompt", "slot": "A"|"B"|"C"} → eval/choices.json
"""
from __future__ import annotations

import json
import os
import secrets
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = Path(os.environ.get("RUNS_DIR", "runs")).resolve()
EVAL = RUNS / "paper" / "x3" / "eval"
KEY = RUNS / "paper" / "x3" / "blind_key.json"
CHOICES = EVAL / "choices.json"
QUESTIONS = ("preserve", "prompt")


# ── 서버 쪽 ─────────────────────────────────────────────────────────────────────
def _items():
    p = EVAL / "items.json"
    return json.loads(p.read_text()) if p.is_file() else []


def _choices():
    return json.loads(CHOICES.read_text()) if CHOICES.is_file() else {}


def handle(h, path, q):
    if path in ("/x3-eval", "/x3-eval/"):
        h._file(HERE / "x3_eval.html")
        return True
    if path == "/api/x3eval":
        h._json({"items": _items(), "choices": _choices()})
        return True
    if path == "/x3img":
        rel = q.get("p", [""])[0]
        p = (EVAL / rel).resolve()
        if EVAL in p.parents and p.suffix == ".png" and p.is_file():
            h._file(p)
        else:
            h._send(404, b"not found", "text/plain")
        return True
    return False


def handle_post(h, path, q):
    if path != "/api/x3eval/choice":
        return False
    n = int(h.headers.get("Content-Length", "0") or 0)
    try:
        d = json.loads(h.rfile.read(min(n, 4096)) or b"{}")
    except ValueError:
        h._json({"error": "bad json"}, 400)
        return True
    ids = {it["id"] for it in _items()}
    if d.get("item") not in ids or d.get("question") not in QUESTIONS or d.get("slot") not in ("A", "B", "C"):
        h._json({"error": "bad request"}, 400)
        return True
    c = _choices()
    c.setdefault(d["item"], {})[d["question"]] = {"slot": d["slot"], "t": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = CHOICES.with_suffix(".tmp")
    tmp.write_text(json.dumps(c, indent=1))
    tmp.replace(CHOICES)
    h._json({"ok": True, "choices": c})
    return True


# ── prep ────────────────────────────────────────────────────────────────────────
def prep():
    import shutil
    sys.path[:0] = [str(HERE.parents[1])]
    from tools.paper.x3 import ASSET_NAMES, OUT, ROUNDS, round_info
    from tools.paper.x3_score import result_views
    if KEY.exists():
        raise SystemExit(f"{KEY} 가 이미 있다 — 키를 다시 만들지 않는다(평가 중 순서가 바뀌지 않게)")
    EVAL.mkdir(parents=True, exist_ok=True)
    items, key = [], {}
    for a in ASSET_NAMES:
        for k in range(1, ROUNDS + 1):
            _, info = round_info(a, k)
            iid = f"{a}-r{k}"
            paths = ["p0", "p1", "p2"]
            order = [paths.pop(secrets.randbelow(len(paths))) for _ in range(3)]
            key[iid] = dict(zip("ABC", order))
            d = EVAL / iid
            d.mkdir(exist_ok=True)
            slots = []
            for slot, path in zip("ABC", order):
                ev, v34 = result_views(path, a, k, info["edit_view"])
                shutil.copy(ev, d / f"{slot}_edit.png")
                shutil.copy(v34, d / f"{slot}_34.png")
                slots.append({"slot": slot, "edit": f"{iid}/{slot}_edit.png", "v34": f"{iid}/{slot}_34.png"})
            items.append({"id": iid, "asset": a, "round": k, "prompt": info["prompt"], "slots": slots})
    (EVAL / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1))
    KEY.write_text(json.dumps({"note": "x3-eval 칸 ↔ 경로. 평가가 끝날 때까지 보고서·뷰어에 노출하지 않는다.", "key": key}, indent=1))
    print("items", len(items), "key", KEY)


if __name__ == "__main__":
    if sys.argv[1:] == ["prep"]:
        prep()
