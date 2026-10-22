#!/usr/bin/env python3
"""결과 뷰어 HTTP 서버 (A5000). 표준 라이브러리만 쓴다.

GLB 가 들어 있는 폴더 하나를 "항목" 하나로 본다. 목록 페이지는 항목마다 썸네일을 보여 주고,
항목을 누르면 GLB 를 three.js 로 띄워 마우스로 돌려 볼 수 있다.

썸네일 순서: 폴더 안 이미지(thumb.png → 2d_edit.png → 2d_render.png → input.png → image_rgb.png → 그 밖 png/jpg)
→ 없으면 브라우저가 GLB 를 한 번 렌더해 서버 캐시에 올린다(다음부터는 캐시 사용).

실행:
    python3 tools/viewer/server.py                      # .env 의 RUNS_DIR, ASSETS_DIR + 저장소 scenarios/
    python3 tools/viewer/server.py --root /path/a --root /path/b --port 8083

환경변수: VHL_VIEWER_PORT(기본 8083), VHL_VIEWER_ROOTS(':' 구분, 있으면 기본 루트 대신 사용),
VHL_VIEWER_CACHE(썸네일 캐시, 기본 <RUNS_DIR 또는 저장소>/.viewer_cache).
결과에 영향이 없는 보기 전용 도구다. 파일을 고치거나 지우지 않는다(썸네일 캐시 쓰기만).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

MODEL_EXT = {".glb", ".gltf"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
TEXT_EXT = {".json", ".txt", ".md", ".log", ".csv"}
THUMB_ORDER = ["thumb.png", "2d_edit.png", "2d_render.png", "input.png", "image_rgb.png"]
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".viewer_cache", ".venv", "venv"}
MAX_IMAGES = 40
MAX_FILES = 80
SCAN_TTL = 10.0
MAX_UPLOAD = 8 * 1024 * 1024

mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("model/gltf+json", ".gltf")


def load_dotenv(path: Path) -> dict:
    env = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


class Catalog:
    """루트들을 훑어 GLB 가 있는 폴더 목록을 만든다. SCAN_TTL 초 동안 결과를 재사용한다."""

    def __init__(self, roots: list[Path], cache_dir: Path):
        self.roots = roots
        self.cache_dir = cache_dir
        self._lock = threading.Lock()
        self._items: list[dict] = []
        self._by_id: dict[str, dict] = {}
        self._t = 0.0

    def _scan(self) -> None:
        items = []
        for ri, root in enumerate(self.roots):
            if not root.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
                dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
                models = sorted(f for f in filenames if Path(f).suffix.lower() in MODEL_EXT)
                if not models:
                    continue
                d = Path(dirpath)
                rel = d.relative_to(root).as_posix()
                rel = "" if rel == "." else rel
                images = sorted(f for f in filenames if Path(f).suffix.lower() in IMAGE_EXT)
                texts = sorted(f for f in filenames if Path(f).suffix.lower() in TEXT_EXT)
                try:
                    mtime = max((d / f).stat().st_mtime for f in models)
                except OSError:
                    continue
                item_id = f"{ri}:{rel}"
                thumb = None
                for name in THUMB_ORDER:
                    if name in images:
                        thumb = name
                        break
                if thumb is None and images:
                    thumb = images[0]
                items.append({
                    "id": item_id,
                    "root": ri,
                    "root_name": root.name or str(root),
                    "rel": rel,
                    "title": rel or root.name,
                    "mtime": mtime,
                    "models": models,
                    "images": images[:MAX_IMAGES],
                    "texts": texts[:MAX_FILES],
                    "thumb_file": thumb,
                    "thumb_key": hashlib.sha1(f"{item_id}|{mtime}".encode()).hexdigest()[:20],
                })
        items.sort(key=lambda x: x["mtime"], reverse=True)
        self._items = items
        self._by_id = {it["id"]: it for it in items}
        self._t = time.time()

    def items(self, force: bool = False) -> list[dict]:
        with self._lock:
            if force or time.time() - self._t > SCAN_TTL:
                self._scan()
            out = []
            for it in self._items:
                it = dict(it)
                it["thumb_cached"] = (self.cache_dir / f"{it['thumb_key']}.png").is_file()
                out.append(it)
            return out

    def get(self, item_id: str) -> dict | None:
        self.items()
        with self._lock:
            return self._by_id.get(item_id)

    def file_path(self, item_id: str, name: str) -> Path | None:
        """항목 폴더 안, 목록에 오른 파일만 돌려준다."""
        it = self.get(item_id)
        if it is None or "/" in name or "\\" in name:
            return None
        if name not in it["models"] and name not in it["images"] and name not in it["texts"]:
            return None
        root = self.roots[it["root"]]
        p = (root / it["rel"] / name).resolve()
        if root.resolve() not in p.parents:
            return None
        return p if p.is_file() else None


class Handler(BaseHTTPRequestHandler):
    catalog: Catalog = None  # type: ignore[assignment]
    server_version = "vhl-viewer/1"

    def log_message(self, fmt, *args):  # 접근 로그는 짧게
        sys.stderr.write("%s %s\n" % (self.command, self.path[:160]))

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8",
                   {"Cache-Control": "no-store"})

    def _file(self, p: Path) -> None:
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if p.suffix == ".js":
            ctype = "text/javascript"
        if ctype.startswith("text/") or p.suffix.lower() in TEXT_EXT:
            ctype = (ctype if ctype.startswith("text/") or ctype.endswith("json") else "text/plain") + "; charset=utf-8"
        size = p.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if self.command == "HEAD":
            return
        with p.open("rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        path = u.path
        if path in ("/", "/index.html"):
            return self._file(HERE / "index.html")
        if path in ("/view", "/view.html"):
            return self._file(HERE / "view.html")
        if path == "/common.js":
            return self._file(HERE / "common.js")
        if path == "/api/items":
            return self._json({"roots": [str(r) for r in self.catalog.roots],
                               "items": self.catalog.items(force="refresh" in q)})
        if path == "/api/item":
            it = self.catalog.get(q.get("id", [""])[0])
            return self._json(it) if it else self._json({"error": "not found"}, 404)
        if path == "/file":
            p = self.catalog.file_path(q.get("id", [""])[0], q.get("name", [""])[0])
            return self._file(p) if p else self._send(404, b"not found", "text/plain")
        if path == "/thumb":
            it = self.catalog.get(q.get("id", [""])[0])
            if it is None:
                return self._send(404, b"not found", "text/plain")
            if it["thumb_file"]:
                p = self.catalog.file_path(it["id"], it["thumb_file"])
                if p:
                    return self._file(p)
            c = self.catalog.cache_dir / f"{it['thumb_key']}.png"
            return self._file(c) if c.is_file() else self._send(404, b"no thumb", "text/plain")
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path != "/api/thumb":
            return self._send(404, b"not found", "text/plain")
        it = self.catalog.get(q.get("id", [""])[0])
        n = int(self.headers.get("Content-Length", "0") or 0)
        if it is None or n <= 0 or n > MAX_UPLOAD:
            return self._json({"error": "bad request"}, 400)
        data = self.rfile.read(n)
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            return self._json({"error": "png only"}, 400)
        self.catalog.cache_dir.mkdir(parents=True, exist_ok=True)
        dst = self.catalog.cache_dir / f"{it['thumb_key']}.png"
        tmp = dst.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(dst)
        return self._json({"ok": True})


def main() -> int:
    env = {**load_dotenv(REPO / ".env"), **os.environ}
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", action="append", default=None, help="훑을 폴더(여러 번 지정 가능)")
    ap.add_argument("--port", type=int, default=int(env.get("VHL_VIEWER_PORT") or 8083))
    ap.add_argument("--host", default=env.get("VHL_VIEWER_HOST") or "0.0.0.0")
    ap.add_argument("--cache", default=env.get("VHL_VIEWER_CACHE") or "")
    a = ap.parse_args()

    if a.root:
        roots = [Path(r).expanduser() for r in a.root]
    elif env.get("VHL_VIEWER_ROOTS"):
        roots = [Path(r).expanduser() for r in env["VHL_VIEWER_ROOTS"].split(":") if r]
    else:
        roots = [Path(env[k]).expanduser() for k in ("RUNS_DIR", "ASSETS_DIR") if env.get(k)]
        roots.append(REPO / "scenarios")
    roots = [r for r in roots if r.is_dir()]
    if not roots:
        print("훑을 폴더가 없다. --root 또는 .env 의 RUNS_DIR/ASSETS_DIR 를 지정하라.", file=sys.stderr)
        return 2
    base = Path(env["RUNS_DIR"]).expanduser() if env.get("RUNS_DIR") else REPO / "runs"
    cache = Path(a.cache).expanduser() if a.cache else base / ".viewer_cache"

    Handler.catalog = Catalog(roots, cache)
    n = len(Handler.catalog.items(force=True))
    print(f"viewer: http://{a.host}:{a.port}/  항목 {n}개, 루트 {[str(r) for r in roots]}, 썸네일 캐시 {cache}")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
