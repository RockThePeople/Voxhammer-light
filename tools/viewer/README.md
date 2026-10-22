# 결과 뷰어 (A5000)

GLB 가 있는 폴더 하나가 목록의 항목 하나다. 목록은 썸네일 격자, 항목을 누르면 three.js 로 GLB 를 띄워 마우스로 돌려 본다(왼쪽 회전, 오른쪽 이동, 휠 확대). 같은 폴더의 이미지(2d_render·2d_mask·2d_edit 등)와 json·로그도 옆에 보여 준다. 마스크 GLB 는 "겹치기"로 반투명 빨강 표시.

```
python3 tools/viewer/server.py            # .env 의 RUNS_DIR, ASSETS_DIR + 저장소 scenarios/ 를 훑는다
python3 tools/viewer/server.py --root <폴더> --root <폴더> --port 8083
```

- 표준 라이브러리만 쓴다(어느 python3 든 됨). 브라우저가 three.js 를 jsdelivr CDN 에서 받는다.
- 포트 기본 8083(`VHL_VIEWER_PORT`). 8082 는 편집·렌더 워커용으로 남겨 둔다.
- 썸네일: 폴더 안 `thumb.png` → `2d_edit.png` → `2d_render.png` → `input.png` → `image_rgb.png` → 그 밖 이미지. 없으면 브라우저가 GLB 를 한 번 렌더해 `RUNS_DIR/.viewer_cache/` 에 저장한다.
- 폴더 목록은 10초마다 다시 훑는다("새로고침" 버튼은 즉시). 파일을 고치거나 지우지 않는다.
- 이후 편집 워커(R8a)의 요청 결과도 GLB 를 `RUNS_DIR` 아래 요청별 폴더에 두면 그대로 목록에 뜬다.
