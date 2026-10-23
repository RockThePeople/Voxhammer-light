"""R9 상주 인페인팅 워커 (rtx3090). `VHL_R9=1` 일 때만 뜬다.

- 프로세스 1개가 INT4 FLUX.1-Fill-dev 를 한 번 올리고 HTTP 로 요청을 받는다. 호출마다 subprocess 를 띄우지 않는다.
- localhost 바인딩, 포트 `INPAINT_PORT`(기본 8082). 오케스트레이터만 호출한다(정오표 E4).
- 인페인팅은 포크 `utils/inpaint.py:inpaint()` 를 그대로 호출한다(1024², 50스텝, guidance 30,
  max_sequence_length 512, CPU generator, seed 기본 0 = 정오표 E5). A5000 기준 경로(`reference/run_inpaint.py`)와
  다른 것은 파이프라인 구성(INT4 transformer, GPU 상주)뿐이다.
- 그 뒤 마스크 밖을 원본 렌더로 덮는다(composite.py). 합성 전(`2d_edit_raw.png`)·후(`2d_edit.png`) 모두 남긴다.
- 큐: 한 번에 하나. 나머지는 대기.

HTTP (규약 초안: services/common/http-contract.md)
    GET  /health                   상태·큐 길이·적재 정보
    POST /inpaint                  multipart: image(file), mask(file), prompt, seed(선택)  → 202 {job_id}
                                   ?wait=1 이면 끝날 때까지 기다려 200 으로 결과
    GET  /jobs/<id>                상태·결과
    GET  /jobs/<id>/<name>.png     2d_render · 2d_mask · 2d_edit · 2d_edit_raw

환경변수: VHL_R9, INPAINT_PORT, RUNS_DIR, REFERENCE_DIR, FLUX_FILL_BASE, FLUX_FILL_TRANSFORMER, HF_HUB_OFFLINE(기본 1)
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import queue
import re
import sys
import threading
import time
import traceback
import uuid
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if os.environ.get("VHL_R9") != "1":
    sys.exit("VHL_R9=1 이 아니다 — R9 워커는 토글이 켜진 경우에만 뜬다 (기본 꺼짐)")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
REF = (ROOT / os.environ.get("REFERENCE_DIR", "build/reference-v2")).resolve()
sys.path.insert(0, str(REF))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from diffusers import FluxFillPipeline  # noqa: E402
from nunchaku import NunchakuFluxTransformer2DModelV2  # noqa: E402
from nunchaku.utils import get_precision  # noqa: E402
from PIL import Image  # noqa: E402

from reference.meter import Meter, Sampler  # noqa: E402
from services.rtx3090_inpaint.composite import composite, outside_diff  # noqa: E402
from utils.inpaint import inpaint  # noqa: E402  포크 함수 그대로

BASE = os.environ.get("FLUX_FILL_BASE", "black-forest-labs/FLUX.1-Fill-dev")
QUANT = os.environ.get("FLUX_FILL_TRANSFORMER", "nunchaku-ai/nunchaku-flux.1-fill-dev")
PORT = int(os.environ.get("INPAINT_PORT") or 8082)
RUNS = Path(os.environ.get("RUNS_DIR") or ROOT / "runs") / "r9-inpaint"
SIZE = 1024
OUT_NAMES = ("2d_render", "2d_mask", "2d_edit", "2d_edit_raw")


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def snapshot_of(repo: str) -> str | None:
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo, local_files_only=True,
                                      allow_patterns=["model_index.json", "*.json"])).name
    except Exception:
        return None


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.q: "queue.Queue[str]" = queue.Queue()
        self.busy: str | None = None
        self.load: dict = {}
        self.pipe = None
        self.calls = 0


S = State()
sampler = Sampler().start()


def load():
    meter = Meter(sampler)
    prec = get_precision()
    with meter.span("load_transformer_int4"):
        tr = NunchakuFluxTransformer2DModelV2.from_pretrained(
            f"{QUANT}/svdq-{prec}_r32-flux.1-fill-dev.safetensors")
    with meter.span("load_pipeline"):
        pipe = FluxFillPipeline.from_pretrained(BASE, transformer=tr, torch_dtype=torch.bfloat16)
    with meter.span("to_cuda"):
        pipe.to("cuda")
    S.pipe = pipe
    S.load = {"base": BASE, "base_snapshot": snapshot_of(BASE), "transformer": QUANT,
              "transformer_snapshot": snapshot_of(QUANT), "precision": prec,
              "spans": meter.spans, "seconds": round(sum(s["seconds"] for s in meter.spans), 3),
              "memory_allocated_after_load": torch.cuda.memory_allocated(),
              "torch": torch.__version__, "torch_cuda": torch.version.cuda}
    print(f"[load] {S.load['seconds']:.1f}s  precision={prec}  "
          f"alloc={S.load['memory_allocated_after_load'] / 2**30:.2f} GiB", flush=True)


def run_job(jid: str):
    job = S.jobs[jid]
    d = RUNS / jid
    d.mkdir(parents=True, exist_ok=True)
    (d / "2d_render.png").write_bytes(job.pop("_image"))
    (d / "2d_mask.png").write_bytes(job.pop("_mask"))
    meter = Meter(sampler)
    S.calls += 1
    with meter.span("inpaint", call_index=S.calls):
        # 포크 함수: 파일을 읽고 2d_edit 을 저장한다. 여기서는 합성 전 원본을 2d_edit_raw 로 받는다.
        inpaint(pipeline=S.pipe, image_path=str(d / "2d_render.png"), mask_path=str(d / "2d_mask.png"),
                output_path=str(d / "2d_edit_raw.png"), prompt=job["prompt"], seed=job["seed"])
    with meter.span("composite"):
        render, mask = Image.open(d / "2d_render.png"), Image.open(d / "2d_mask.png")
        raw = Image.open(d / "2d_edit_raw.png")
        comp = composite(raw, render, mask)
        comp.save(d / "2d_edit.png")
        diff = {"before": outside_diff(raw, render, mask), "after": outside_diff(comp, render, mask)}
    files = {n: sha256((d / f"{n}.png").read_bytes()) for n in OUT_NAMES}
    job.update(result={"sha256": files, "outside_diff": diff, "spans": meter.spans,
                       "settings": {"steps": 50, "guidance": 30, "resolution": SIZE,
                                    "max_sequence_length": 512, "generator": "cpu",
                                    "seed": job["seed"], "base": BASE, "transformer": QUANT}})
    (d / "result.json").write_text(json.dumps({k: v for k, v in job.items()}, indent=1, default=str))


def loop():
    while True:
        jid = S.q.get()
        with S.lock:
            S.busy = jid
            S.jobs[jid]["state"] = "running"
            S.jobs[jid]["started"] = time.time()
        try:
            run_job(jid)
            state = "done"
        except Exception as e:
            S.jobs[jid]["error"] = {"error_code": "INPAINT_FAILED", "message": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
            state = "failed"
        with S.lock:
            S.jobs[jid]["state"] = state
            S.jobs[jid]["finished"] = time.time()
            S.busy = None
        S.jobs[jid]["done_event"].set()


def parse_multipart(ctype: str, body: bytes) -> dict:
    msg = BytesParser(policy=email_policy).parsebytes(
        b"Content-Type: " + ctype.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
    if not msg.is_multipart():
        raise ValueError("multipart/form-data 가 아니다")
    out = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name:
            out[name] = part.get_payload(decode=True)
    return out


def validate(fields: dict) -> tuple[dict, tuple[int, str, str] | None]:
    for k in ("image", "mask", "prompt"):
        if not fields.get(k):
            return {}, (400, "MISSING_FIELD", f"필드가 없다: {k}")
    try:
        im = Image.open(io.BytesIO(fields["image"]))
        mk = Image.open(io.BytesIO(fields["mask"]))
        im.load(), mk.load()
    except Exception as e:
        return {}, (400, "BAD_IMAGE", f"이미지를 읽을 수 없다: {e}")
    if im.size != (SIZE, SIZE) or mk.size != (SIZE, SIZE):
        return {}, (400, "BAD_SIZE", f"1024² 이어야 한다: image {im.size}, mask {mk.size}")
    try:
        seed = int(fields.get("seed") or b"0")
    except ValueError:
        return {}, (400, "BAD_SEED", "seed 는 정수")
    return {"_image": fields["image"], "_mask": fields["mask"],
            "prompt": fields["prompt"].decode("utf-8"), "seed": seed,
            "input_sha256": {"2d_render": sha256(fields["image"]), "2d_mask": sha256(fields["mask"])}}, None


def public(job: dict) -> dict:
    return {k: v for k, v in job.items() if not k.startswith("_") and k != "done_event"}


class H(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _err(self, code, ecode, msg):
        # ai-ar-v3 deltacontract ErrorBody 와 같은 모양
        self._json(code, {"error_code": ecode, "message": msg})

    def log_message(self, fmt, *args):
        print(f"[http] {self.command} {self.path} {fmt % args}", flush=True)

    def do_GET(self):
        if self.path == "/health":
            with S.lock:
                return self._json(200, {"ok": S.pipe is not None, "vhl_r9": True, "busy": S.busy,
                                        "queue": S.q.qsize(), "calls": S.calls, "load": S.load})
        m = re.fullmatch(r"/jobs/([0-9a-f]{32})(?:/([\w]+)\.png)?", self.path)
        if not m or m.group(1) not in S.jobs:
            return self._err(404, "NOT_FOUND", self.path)
        jid, name = m.groups()
        if name is None:
            return self._json(200, public(S.jobs[jid]))
        p = RUNS / jid / f"{name}.png"
        if name not in OUT_NAMES or not p.is_file():
            return self._err(404, "NOT_FOUND", self.path)
        b = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("X-Sha256", sha256(b))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        path, _, qs = self.path.partition("?")
        if path != "/inpaint":
            return self._err(404, "NOT_FOUND", self.path)
        ctype = self.headers.get("Content-Type", "")
        if not ctype.startswith("multipart/form-data"):
            return self._err(415, "BAD_CONTENT_TYPE", "multipart/form-data 만 받는다")
        try:
            fields = parse_multipart(ctype, self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        except Exception as e:
            return self._err(400, "BAD_MULTIPART", str(e))
        job, err = validate(fields)
        if err:
            return self._err(*err)
        jid = uuid.uuid4().hex
        job.update(job_id=jid, state="queued", created=time.time(), done_event=threading.Event())
        with S.lock:
            S.jobs[jid] = job
            pos = S.q.qsize() + (1 if S.busy else 0)
        S.q.put(jid)
        if "wait=1" in qs.split("&"):
            job["done_event"].wait()
            return self._json(200 if job["state"] == "done" else 500, public(job))
        self._json(202, {"job_id": jid, "state": "queued", "ahead": pos})


def main():
    load()
    threading.Thread(target=loop, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print(f"[ready] 127.0.0.1:{PORT}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
