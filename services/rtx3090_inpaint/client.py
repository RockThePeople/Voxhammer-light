"""R9 인페인팅 워커 클라이언트 (표준 라이브러리만). 오케스트레이터·R-4 비교 스크립트가 쓴다.

    from services.rtx3090_inpaint.client import inpaint
    res = inpaint(render_png_bytes, mask_png_bytes, prompt, seed=0, out_dir=Path(...))
"""
from __future__ import annotations

import json
import os
import urllib.request
import uuid
from pathlib import Path


def base_url() -> str:
    return f"http://127.0.0.1:{os.environ.get('INPAINT_PORT') or 8082}"


def _multipart(fields: dict) -> tuple[bytes, str]:
    b = uuid.uuid4().hex
    out = []
    for k, v in fields.items():
        if isinstance(v, bytes):
            out.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"; filename="{k}.png"\r\n'
                       f"Content-Type: image/png\r\n\r\n".encode() + v + b"\r\n")
        else:
            out.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n'.encode()
                       + str(v).encode() + b"\r\n")
    out.append(f"--{b}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={b}"


def health(timeout=5) -> dict:
    with urllib.request.urlopen(base_url() + "/health", timeout=timeout) as r:
        return json.load(r)


def inpaint(render: bytes, mask: bytes, prompt: str, seed: int = 0, out_dir: Path | None = None,
            timeout: float = 1800) -> dict:
    body, ctype = _multipart({"image": render, "mask": mask, "prompt": prompt, "seed": seed})
    req = urllib.request.Request(base_url() + "/inpaint?wait=1", data=body, method="POST",
                                 headers={"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        res = json.load(r)
    if out_dir is not None and res.get("state") == "done":
        out_dir.mkdir(parents=True, exist_ok=True)
        for n in res["result"]["sha256"]:
            with urllib.request.urlopen(f"{base_url()}/jobs/{res['job_id']}/{n}.png", timeout=60) as r:
                (out_dir / f"{n}.png").write_bytes(r.read())
    return res
