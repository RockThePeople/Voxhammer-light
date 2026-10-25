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

## 시나리오 비교 페이지 (T1)

`/scenario/<ID>` — 동결 16건(트랙 A 6, 트랙 B 8, moto-b 2): 조건 이미지, 카메라가 연동되는 3D 칸(원본·base·c15·c12·s1-12·s2-12, 회차 r1–r3), 마스크·halo·신규·소실 복셀 겹치기, R_keep 신규 성분 표(거리·9a 대상), 설정별 수치 표(E19 판정). 목록 페이지의 "시나리오" 탭(`/?tab=scenarios`)에서 16건을 고른다.

```
# 1) 미리 계산 (CPU 만, 입력은 읽기만, 16건 약 80 s) → $RUNS_DIR/.viewer_cache/scenario/
~/ai-ar-v2/VoxHammer/.venv/bin/python tools/viewer/scenario_prep.py            # 새 결과가 생기면 다시 돌린다
~/ai-ar-v2/VoxHammer/.venv/bin/python tools/viewer/scenario_prep.py --only B7  # 일부만
# 2) 서버 (위와 같다)
python3 tools/viewer/server.py --port 8083
```

- 접속: 서버는 `0.0.0.0:8083` 에 바인딩한다. 사용자 PC 에서 `http://117.16.244.35:8083/scenario/A-S1`(A5000 공인 주소, firewalld 8083 개방). 인증 없음(기존과 같음).
- 좌표 변환은 `common.js:voxelToWorld`(셀 → ((i,j,k)+0.5)/64 − 0.5 → glTF (x, z, −y) → (p − c) × s) 한 곳, 카메라는 `common.js:forkCamera`(포크 조건 렌더와 같은 정의) 한 곳. `scenario_prep.py:cells_to_world` 는 같은 식을 정렬 검사에만 쓴다.
- 새 설정(이후 항목 결과)은 `scenario_prep.py` 의 `CONFIGS`·`run_dirs` 에 한 줄 더하고 다시 돌린다.
- 캡처(headless Chrome, 의존성 없음): `node tools/viewer/capture.mjs out.png "http://localhost:8083/scenario/B7?only3d=1&panels=base:r1,c12:r1&view=34" 1500 850`. URL 인자: `panels=src,c15:r1,...` `view=edit|34|0..4` `ov=mask,halo,new_in,new_out,new_out_9a,lost` `mode=both|mesh|voxel` `comp=<성분 번호>` `capture=edit`(1024² 한 칸, 초록 배경 — 실루엣 검사용).
- GLB 가 큰 경우(R4 이후) 칸 수를 줄인다(칸마다 GLB 를 따로 읽는다).
