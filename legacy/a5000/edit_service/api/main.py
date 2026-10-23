"""8082 한 프로세스에 두 라우터를 올린다.

    A5000 :8082
    ├── /lab/*         개발 도구. 인증 없음 (읽기 전용 + LAN 전용)
    ├── /debug/*       사람이 보는 관측 화면. 인증 없음 · 읽기 전용
    │                  파일은 /lab/runs/{id}/files/ 를 그대로 쓴다 (경로 검증 한 곳)
    └── /v2/trellis/*  제품 경로. X-Blockedit-Key 필수

한 프로세스인 이유는 포트 제약만이 아니다 — **GPU 가 직렬 자원**이라 별도
프로세스면 큐가 둘이 되어 서로 GPU 를 뺏는다. 한 프로세스 · 한 워커 큐 ·
모델 1회 로드가 옳다.

⚠️ 인증은 **라우터 단위**로 건다(routes.py 의 APIRouter dependencies). 경로
프리픽스로 미들웨어를 걸면 면제 규칙이 구멍이 된다.
⚠️ 그래서 lab 에는 쓰기 엔드포인트를 추가하면 안 된다 — 인증 없음의 전제가 깨진다.
"""
import os
import sys

os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("TQDM_DISABLE", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SERVER = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_SERVER), "vendor"))
sys.path.insert(0, _SERVER)

from fastapi import FastAPI                       # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402
from fastapi.exceptions import RequestValidationError   # noqa: E402
from fastapi.responses import JSONResponse          # noqa: E402

from deltacontract.errors import DeltaContractError, InvalidRequest  # noqa: E402

from api import jobs, routes                       # noqa: E402
from lab import debug as lab_debug                 # noqa: E402
from lab.api import app as lab_app                 # noqa: E402

app = FastAPI(title="A5000 (lab + trellis)", version="1.0")
app.include_router(lab_app.router)                 # /lab/*        인증 없음
app.include_router(lab_debug.router)               # /debug/*      인증 없음 · 읽기 전용
app.include_router(routes.router)                  # /v2/trellis/* 인증 필수
app.include_router(routes.v2)                      # /v2/assets/*  인증 필수


# ── 에러 본문은 한 겹도 싸지 않는다 ────────────────────────────────────
#
# HTTPException(detail=...) 을 쓰면 FastAPI 가 {"detail": {...}} 로 감싼다.
# 그러면 소비 측이 자기 쪽에서 벗겨 받아야 하고, "4xx/5xx 는 전부 한 가지 모양"
# 이라는 계약이 무너진다. 아래 핸들러들이 ErrorBody 모양을 그대로 낸다.
@app.exception_handler(DeltaContractError)
async def _contract_error(request, exc: DeltaContractError):
    return JSONResponse(exc.to_dict(), status_code=exc.http_status)


@app.exception_handler(RequestValidationError)
async def _validation_error(request, exc: RequestValidationError):
    # pydantic 검증 실패도 같은 모양으로 내린다.
    e = InvalidRequest(str(exc.errors()[:3]))
    return JSONResponse(e.to_dict(), status_code=e.http_status)


@app.exception_handler(StarletteHTTPException)
async def _http_error(request, exc: StarletteHTTPException):
    """lab 라우터의 HTTPException + **Starlette 의 라우트 미매치 404** 를 잡는다.

    ⚠️ fastapi.HTTPException 에만 핸들러를 걸면 매치되지 않은 경로의 404 가
    Starlette 기본 처리로 빠져 {"detail":"Not Found"} 가 나간다 — 계약이 못박은
    "4xx/5xx 는 전부 한 가지 모양" 이 거기서 깨진다.
    fastapi.HTTPException 은 Starlette 것의 하위라 이 핸들러가 둘 다 잡는다.
    """
    d = exc.detail
    if isinstance(d, dict) and "error_code" in d:
        return JSONResponse(d, status_code=exc.status_code)
    # ⚠️ INTERNAL 로 뭉개지 마라 (계약 3.7.0 이 경고한 그것).
    # Unity 가 "라우트 미등록 vs 파일 부재" 를 이 error_code 로 가르고, 3090 도
    # 같은 상황에 NOT_FOUND "모르는 경로: {path}" 를 준다. 두 서버가 다른 말을
    # 하면 그 판별식이 죽는다. 맨 "Not Found" 는 정보가 0 이므로 경로를 싣는다.
    if exc.status_code == 404:
        return JSONResponse(
            {"error_code": "NOT_FOUND",
             "message": f"모르는 경로: {request.url.path}"},
            status_code=404)
    code = "UNAUTHORIZED" if exc.status_code == 401 else "INTERNAL"
    return JSONResponse({"error_code": code, "message": str(d)},
                        status_code=exc.status_code)


@app.on_event("startup")
def _startup():
    def preload():
        from lab.pipeline import load_pipe

        class _Boot:
            run_id = "boot"
            def begin(self, *a): pass
            def stage(self, *a, **k): pass

        load_pipe(_Boot())
        print("[api] 모델 상주 로드 완료", flush=True)

    jobs.start(preload=preload)


def serve(host="0.0.0.0", port=8082):
    import uvicorn

    print(f"[api] /lab/* + /v2/trellis/*  http://{host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--no-snap", action="store_true",
                    help="경계 스냅 끄기 (기본은 켜짐, 2셀)")
    a = ap.parse_args()
    if a.no_snap:
        routes.SNAP_CELLS = 0.0
    serve(a.host, a.port)
