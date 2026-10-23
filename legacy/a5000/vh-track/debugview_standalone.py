"""DebugView 를 GPU 서비스와 **분리해** 띄운다 — 포트 8081, 인증 없음.

왜 분리하나 (실측된 증상):
    A5000 이 실험할 때마다 VRAM 을 비우려고 :8082 (TRELLIS 편집 서비스, 약 6 GB) 를 내린다.
    FLUX 인페인트가 최대 22.79 GiB 를 쓰기 때문에 안 내리면 OOM 이다.
    그런데 `/debug` 가 그 앱에 얹혀 있어서 **서비스를 내릴 때마다 DebugView 도 같이 죽었다.**

    DebugView 는 GPU 를 한 톨도 안 쓴다 — 파일시스템 읽기와 HTML 생성뿐이다.
    한 프로세스에 묶일 이유가 없다.

⚠️ 읽기 전용이다. `lab` 라우터가 인증 없음을 전제로 하므로(main.py 주석)
   여기에 쓰기·삭제·실행 엔드포인트를 붙이면 그 전제가 깨진다.

기동:
    tools/start_debugview.sh            # setsid 로 PPID=1 — 세션이 끝나도 산다
검증:
    :8082 를 **내린 상태에서** /debug/vh 가 200 이어야 한다. 코드상 독립은 증거가 아니다.
"""
import os
import sys

_SERVER = "<HOME>/ai-ar-prototype/server"
sys.path.insert(0, os.path.join(os.path.dirname(_SERVER), "vendor"))
sys.path.insert(0, _SERVER)

from fastapi import FastAPI  # noqa: E402

from lab import debug  # noqa: E402
from lab.api import app as lab_app  # noqa: E402

app = FastAPI(title="DebugView (standalone, no GPU)")
app.include_router(lab_app.router)
app.include_router(debug.router)

if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("DEBUGVIEW_PORT", "8081"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
