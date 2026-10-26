"""04L / 05 R9 R-7 종단: 트랙 A 6건 × N회 — 3090 오케스트레이터 → A5000 렌더 → 3090 INT4 인페인팅 → A5000 편집.

ai-ar-v3 오케스트레이터(`server/voxhammer.run_voxhammer_edit`, 계약 3.28.0 조건 이미지 경로)를 그대로
부른다. LLM 계획(plan_edit)은 건너뛰고 시나리오의 편집 지시를 그대로 spec 으로 준다(op 는 유형에서).
`VHL_R9=1` 로 켠다(종단 시험 규칙, 05-R9 R-7).

구간(3090 에서 본 값, `R9_TIMING_LOG` 를 회차마다 따로 받는다):
  ① render_s + render_fetch_s      렌더 요청 ~ 응답, 렌더 파일 2장 수신 (render_bytes)
  ② inpaint_s (워커 내부 inpaint_worker_s)
  ③ submit_s + job_wait_s + chunk_fetch_s   편집 제출(조건 3장 업로드) ~ 잡 완료 ~ 청크 수신
     전송: condition_upload_bytes(이미지 3장), chunk_bytes·n_chunks
  wall_s   run_voxhammer_edit 1회 벽시계(검증·저장 포함)

산출(회차마다 RUNS_DIR/e2e-track-a-<시각>/<ID>/r<k>/):
  2d_render.png · 2d_mask.png · 2d_edit.png, store/<asset>/v<n>/*.cbin, manifest.json, patch.json, timing.jsonl
  요약 e2e.json (+ sha256 출력)

    # 모의 A5000 (기본) — 경로 점검용. 인페인팅은 상주 R9 워커(127.0.0.1:INPAINT_PORT)
    AIAR_DIR=<ai-ar-v3 작업본> <aiar-tests python> reference/rtx3090/e2e_track_a.py --repeat 1 --only A-M1
    # 실주소 — A5000 R8a 워커
    AIAR_DIR=... A5000_URL=http://<A5000>:8082 <python> reference/rtx3090/e2e_track_a.py --target real \\
        --asset-map <자산 id·base_version JSON> --repeat 3

--asset-map 형식: {"moto": {"asset_id": "...", "base_version": 1}, "snowman": {...}, "chair": {...}}
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import socket
import statistics as st
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
OPS = {"교체": "replace_region", "형태 변경": "replace_region", "추가": "add", "제거": "remove"}

ap = argparse.ArgumentParser()
ap.add_argument("--target", choices=["mock", "real"], default="mock")
ap.add_argument("--repeat", type=int, default=3)
ap.add_argument("--only", nargs="*", help="시나리오 ID 일부만 (예: A-M1 A-S2)")
ap.add_argument("--asset-map", help="실주소일 때 필수. 자산 키 → {asset_id, base_version}")
ap.add_argument("--view", choices=["frozen", "auto"], default="frozen",
                help="frozen: 동결 편집 시점을 보낸다. auto: view 를 생략(A5000 이 고른다)")
ap.add_argument("--poll", default="0.25", help="잡 폴링 간격 s (EDIT_POLL_INTERVAL)")
a = ap.parse_args()

AIAR = os.environ.get("AIAR_DIR")
if not AIAR or not (Path(AIAR) / "server" / "condrender.py").is_file():
    sys.exit("AIAR_DIR 가 계약 3.28.0 오케스트레이터가 있는 ai-ar-v3 작업본을 가리켜야 한다")
sys.path.insert(0, AIAR)
sys.path.insert(0, str(Path(AIAR) / "contract" / "python"))

import numpy as np  # noqa: E402
from deltacontract import mask_fingerprint  # noqa: E402


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def cells_from_boxes(boxes):
    out = set()
    for lo, hi in boxes:                      # 양끝 포함 상자 (00s S-5)
        for x in range(lo[0], hi[0] + 1):
            for y in range(lo[1], hi[1] + 1):
                for z in range(lo[2], hi[2] + 1):
                    out.add((x, y, z))
    return sorted(out)


def tcp_ok(url: str) -> tuple[bool, str]:
    u = urlparse(url)
    try:
        with socket.create_connection((u.hostname, u.port or 80), timeout=5):
            return True, "connected"
    except OSError as e:
        return False, f"{type(e).__name__}: {e}"


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"mean": round(st.mean(xs), 3), "sd": round(st.stdev(xs), 3) if len(xs) > 1 else 0.0, "n": len(xs)}


def main():
    track = json.loads((ROOT / "scenarios/track_a.json").read_text())["scenarios"]
    masks = json.loads((ROOT / "scenarios/track_a_masks.json").read_text())
    ids = [s for s in track if not a.only or s in a.only]

    os.environ["VHL_R9"] = "1"
    os.environ["EDIT_POLL_INTERVAL"] = a.poll
    mock = None
    if a.target == "real":
        url = os.environ.get("A5000_URL")
        if not url:
            sys.exit("A5000_URL 이 없다 (.env)")
        if not a.asset_map:
            sys.exit("--asset-map 이 필요하다 (A5000 의 트랙 A 자산 id·base_version)")
        amap = json.loads(Path(a.asset_map).read_text())
        ok, why = tcp_ok(url)
        if not ok:
            u = urlparse(url)
            sys.exit(f"A5000 에 연결되지 않는다: {why}\n시험 명령: curl -v http://<A5000>:{u.port}/")
        os.environ.pop("BLOCKEDIT_B_URL", None)
    else:
        from server.tests.mock_condition_upstream import KEY, MockA5000, moto_b_chunks

        mock = MockA5000(moto_b_chunks(3), latest_version=1).__enter__()
        os.environ["A5000_URL"], os.environ["A5000_KEY"] = mock.url, KEY
        amap = {k: {"asset_id": f"mock-{k}", "base_version": 1} for k in ("moto", "snowman", "chair")}

    from server import voxhammer
    from server.assetstore import AssetStore
    from server.condrender import inpaint_url
    from deltacontract.schemas import EditMask, EditRequest

    import httpx
    h = httpx.get(inpaint_url() + "/health", timeout=5).json()
    if not h.get("ok"):
        sys.exit(f"인페인팅 워커가 준비되지 않았다: {h}")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(os.environ.get("RUNS_DIR") or ROOT / "runs") / f"e2e-track-a-{stamp}"
    out.mkdir(parents=True)
    res = {"item": "04L / R9 R-7 종단", "target": a.target, "repeat": a.repeat, "view_mode": a.view,
           "poll_interval_s": float(a.poll), "started": stamp,
           "inpaint_worker": {k: h["load"].get(k) for k in ("base", "base_snapshot", "transformer",
                                                             "transformer_snapshot", "precision")},
           "aiar_commit": os.popen(f"git -C {AIAR} rev-parse --short HEAD").read().strip(),
           "runs": []}

    for sid in ids:
        sc, mk = track[sid], masks[sid]
        cells = cells_from_boxes(mk["mask_boxes"])
        fp = mask_fingerprint(np.asarray(cells, dtype=np.int64))
        assert fp == mk["mask_fingerprint"], f"{sid} 마스크 복원 지문이 다르다"
        asset = amap[sc["asset"]]
        mask = EditMask(mode="voxels", voxels=cells, grid_source="slat_coords",
                        halo_margin_voxels=int(sc["mask"].get("halo_margin_voxels", 2)))
        req = EditRequest(session_id="e2e", base_version=int(asset["base_version"]),
                          raw_prompt=sc["inpaint_prompt"], mask=mask, seed=int(sc.get("edit_seed", 0)),
                          view=int(sc["edit_view"]) if a.view == "frozen" else None)
        spec = SimpleNamespace(op=OPS[sc["type"]], target_prompt=sc["inpaint_prompt"])
        for k in range(a.repeat):
            d = out / sid / f"r{k}"
            d.mkdir(parents=True)
            os.environ["R9_KEEP_DIR"], os.environ["R9_TIMING_LOG"] = str(d), str(d / "timing.jsonl")
            voxhammer.STORE = AssetStore(root=d / "store")
            rec = {"scenario": sid, "run": k, "asset_id": asset["asset_id"], "base_version": req.base_version,
                   "view_requested": req.view, "mask_fingerprint": fp, "n_mask_cells": len(cells)}
            t0 = time.perf_counter()
            try:
                o = voxhammer.run_voxhammer_edit(asset["asset_id"], req, spec, lambda *x: None)
                rec["wall_s"] = round(time.perf_counter() - t0, 3)
                (d / "manifest.json").write_text(o["manifest"].model_dump_json(indent=1))
                (d / "patch.json").write_text(o["patch"].model_dump_json(indent=1))
                rec.update(ok=True, to_version=o["patch"].to_version, stage_detail=o["stage_detail"],
                           n_changed=len(o["patch"].changed_chunks), n_removed=len(o["patch"].removed_chunk_ids))
            except Exception as e:                                   # noqa: BLE001
                rec.update(ok=False, wall_s=round(time.perf_counter() - t0, 3),
                           error_code=getattr(e, "error_code", type(e).__name__), error=str(e)[:500])
                (d / "error.txt").write_text(traceback.format_exc())
            lines = (d / "timing.jsonl").read_text().splitlines() if (d / "timing.jsonl").is_file() else []
            if lines:
                t = json.loads(lines[-1])
                rec.update(timing=t.get("timings"), view=t.get("view"), render_attempts=t.get("render_attempts"),
                           edit_sha256=t.get("edit_sha256"), inpaint_reused=t.get("inpaint_reused"),
                           job_id=t.get("job_id"), render_a5000=t.get("render_a5000"),
                           edit_stats_a5000=t.get("edit_stats_a5000"))
            res["runs"].append(rec)
            tm = rec.get("timing") or {}
            print(f"{sid} r{k} {'ok' if rec['ok'] else 'FAIL ' + str(rec.get('error_code'))} wall {rec['wall_s']:.1f}s "
                  f"①{(tm.get('render_s') or 0) + (tm.get('render_fetch_s') or 0):.1f} ②{tm.get('inpaint_s', 0):.1f} "
                  f"③{(tm.get('submit_s') or 0) + (tm.get('job_wait_s') or 0) + (tm.get('chunk_fetch_s') or 0):.1f} "
                  f"chunks {tm.get('n_chunks')} {tm.get('chunk_bytes')}B view {rec.get('view')}", flush=True)

    keys = ["render_s", "render_fetch_s", "inpaint_s", "inpaint_worker_s", "submit_s", "job_wait_s",
            "chunk_fetch_s", "render_bytes", "condition_upload_bytes", "chunk_bytes", "n_chunks"]
    res["summary"] = {}
    for sid in ids:
        rs = [r for r in res["runs"] if r["scenario"] == sid]
        okr = [r for r in rs if r["ok"]]
        res["summary"][sid] = {"ok": f"{len(okr)}/{len(rs)}", "wall_s": mean_sd([r["wall_s"] for r in okr]),
                               **{k: mean_sd([(r.get("timing") or {}).get(k) for r in okr]) for k in keys}}
    if mock is not None:
        mock.__exit__(None, None, None)
    p = out / "e2e.json"
    p.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("RESULT", p, sha(p.read_bytes()))


if __name__ == "__main__":
    main()
