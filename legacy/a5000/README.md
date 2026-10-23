# legacy/a5000 — A5000 현행 코드 보관 (요구사항서 2장, 00 A-2)

2026-10-23 a5000 세션이 원본을 **수정 없이 복사**했다. 바꾼 것은 홈 경로(→ `<HOME>`)뿐이고 위치는 [REDACTIONS.md](REDACTIONS.md).
가중치·`.safetensors`·자산·렌더·로그·`__pycache__`·venv 는 넣지 않았다. 기준선(0번)에는 쓰지 않고 보관만 한다.

- 현행 구조와 실행 방식은 00a 보고 [docs/reports/00a-env-a5000.md](../../docs/reports/00a-env-a5000.md) A4–A8 참조: A4 코드 위치·실행 방식(HTTP 서비스 vs VoxHammer 트랙 스크립트), A5 모델 상주(서비스는 1회 로드, 트랙 스크립트는 실행마다), A6 인페인팅(A5000 bf16 offload, `inpaint_offload.py`), A7 `run_edit` 인자·W39·`slat_tgt` 저장(`slat_capture.py`), A8 조건 시점(자산별 수동 선택).
- `vh-track/`: ai-ar-v3 `tools/` 아래 A5000 전용 미추적 스크립트 19개 + VoxHammer 작업 트리의 `env.sh`. (00a 보고에 "15개"로 적은 것은 목록 출력이 잘린 것이었다.)
- `edit_service/`: ai-ar-prototype `server/api/`(+ `api/tests/test_api.py`)와 그것이 import 하는 `server/lab/*.py` 11개·`server/repaint/masked_sampler.py`. import 폐포는 AST 로 계산했다. 폐포 밖 `lab/cli.py`, `lab/static/`(HTML·JS), `repaint/` 의 다른 실험 스크립트는 넣지 않았다. `vendor/trellis1`(TRELLIS 동봉본)도 넣지 않았다.

## 파일별 역할

| 파일 | 하는 일 |
|---|---|
| `edit_service/api/__init__.py` | 패키지 표시(빈 파일) |
| `edit_service/api/build.py` | 실행 중인 코드의 커밋·빌드 정보(계약 3.16.0) |
| `edit_service/api/jobs.py` | 단일 워커 큐(GPU 직렬), 시작 시 모델 상주 로드(preload) |
| `edit_service/api/main.py` | :8082 한 프로세스에 /lab·/debug·/v2/trellis·/v2/assets 라우터를 올리는 진입점 |
| `edit_service/api/ops.py` | 제품 경로 계산 코어 — op_generate·op_edit(MaskedSlatSampler)·op_assemble, 부기 합성·승계 |
| `edit_service/api/routes.py` | /v2/trellis/* HTTP 라우트(X-Blockedit-Key 인증, generate·edit·assemble·잡 조회·청크 제공) |
| `edit_service/api/store.py` | 자산/버전 저장소 + 2단계 커밋(staging → committed) |
| `edit_service/api/tests/test_api.py` | GPU 없이 도는 계약·저장소 불변식 테스트 |
| `edit_service/lab/__init__.py` | 패키지 표시(빈 파일) |
| `edit_service/lab/api.py` | lab 읽기 전용 API(LAB_API.md) |
| `edit_service/lab/debug.py` | /debug 읽기 전용 관측 화면(lab run·VoxHammer 트랙) |
| `edit_service/lab/pipeline.py` | gen/edit 실행 코어 — 모델 로드(load_pipe 전역 캐시)·디코딩·청크 분할·렌더 |
| `edit_service/lab/regions.py` | --region 문자열 → 활성 복셀 부분집합 |
| `edit_service/lab/render.py` | 청크 메시 직접 렌더(to_glb 미사용) |
| `edit_service/lab/representation.py` | 렌더용 최소 MeshExtractResult 생성 |
| `edit_service/lab/seam.py` | 부기 기반 청크 합성 + 이음새 스냅·측정(계약 3.4.0) |
| `edit_service/lab/store.py` | LAB_ROOT 아래 run 디렉터리 관리·진행 상태 기록 |
| `edit_service/lab/vhmask.py` | VoxHammer 트랙용 마스크 생성(CPU 전용) |
| `edit_service/lab/view.py` | 터미널 ASCII 청크 맵·델타 표 |
| `edit_service/repaint/masked_sampler.py` | SLat RePaint 샘플러 — 마스크 안만 재생성, 밖은 원본 강제 보존(op_edit 이 사용) |
| `vh-track/chunk_compare.py` | GLB 를 계약대로 청킹해 부모 대비 changed/removed 와 시간을 잰다. 부기는 마스크+halo → affected_chunks, diff_chunk_sets 는 검증용 |
| `vh-track/debugview_standalone.py` | DebugView(:8081) 를 GPU 서비스와 분리해 띄우는 단독 서버(인증 없음, 읽기 전용) |
| `vh-track/edit_slat_reuse.py` | W34 SLat 재사용 편집 — 생성 SLat·마스크 셀로 run_edit(slat_path, mask_cells), --emit_slat 로 편집 SLat 저장 |
| `vh-track/env.sh` | VoxHammer 실행 환경변수(CUDA 11.8 경로, eigen include, ATTN_BACKEND=flash_attn, SPCONV_ALGO=native) — 원래 <VH_CUR>/env.sh |
| `vh-track/exr_mask_postprocess.py` | opencv 에 OpenEXR 가 없어 포크 조건 렌더 후처리가 죽을 때, EXR 을 bpy 로 읽어 같은 식으로 2D 마스크를 만드는 후처리 |
| `vh-track/gen_asset.py` | 입력 이미지 하나로 TRELLIS 자산(GLB·SLat) 생성 — 단독 트랙용 |
| `vh-track/inpaint_offload.py` | 포크 utils/inpaint.inpaint 를 bf16 + enable_model_cpu_offload 로 구동(파라미터 원저자 기본값 그대로) |
| `vh-track/make_mask_glb.py` | 편집 영역을 프리미티브 mask.glb(구·상자)로 만들어 원저자 경로 입력으로 쓴다 |
| `vh-track/make_thumb.py` | GLB 정면 텍스처 캡처를 PNG 로 굽는다(DebugView 썸네일) |
| `vh-track/repeat_bench.py` | Baseline(원저자 경로)·Ours(W34) 를 같은 계측으로 반복 실행하는 하네스(W41) |
| `vh-track/repeat_bench_analyze.py` | W41 반복 측정 집계(평균·표준편차·최소/최대, 팔별) |
| `vh-track/repeat_bench_run.sh` | W41 반복 측정 실행(자산당 Baseline 3회 + Ours 3회, 로그에 경과초) |
| `vh-track/restart_edit_service.sh` | A5000 편집 서비스(ai-ar-prototype server/api/main.py, :8082) 재기동 — 키는 환경변수 TRELLIS_API_KEY |
| `vh-track/run_voxhammer_checked.sh` | inference.py 를 감싸 VRAM 사전 확인·features.npz 갱신·삼킨 예외·산출 GLB 를 검사하는 실행기 |
| `vh-track/slat_capture.py` | decode_slat 호출을 가로채 편집 SLat 을 safetensors(R2 형식)로 저장 |
| `vh-track/start_debugview.sh` | debugview_standalone.py 를 setsid 로 띄우는 기동 스크립트 |
| `vh-track/w42_region_eval.py` | W42 고정 영역 기준 채점(마스크 셀 집합 대신 고정 영역으로) |
| `vh-track/w42_run.sh` | W42 재측정 실행(점유 셀 기반 새 마스크, 나머지 고정) |
| `vh-track/w43_panels.py` | W43 DebugView 상세 패널(조건 3종 나란히 + 해시) 생성 |
| `vh-track/w43_run.sh` | W43 재실행(조건까지 갱신) — t1s·t2s × ours/baseline |

## 변경 청크 결정 방식 (R11): 점유 비교인가, 해시인가

**HTTP 편집 서비스(제품 경로)는 해시가 아니라 마스크 배치(부기)로 정한다.** 해시는 결정에 쓰지 않는다(코드확인).
- `edit_service/api/ops.py:96-117` `mask_rows`: 요청 마스크 셀을 `dilate_cells(core, halo)` 로 팽창하고 → `book = affected_chunks(None, None, cells)`(110행). 주석: "부기가 '무엇이 바뀌었는가'의 정의다"(97행), "부기는 마스크+halo **전체**에서 계산한다. 그 청크에 실제로 삼각형이 생기는지는 모르고, 안 생기면 removed_chunk_ids 로 간다"(108–109행). 재생성 행은 마스크 셀 ∩ 활성(점유) 복셀(103–107행).
- `ops.py:294-312` `op_edit` 부기 합성: 새로 디코딩한 청크 중 `k in bookset or k not in parent_man` 이면 새 청크(299–301행), 아니면 **부모 바이트를 그대로 복사**해 승계(309–312행). 승계 전에 `_chunk_drift` 로 부모 대비 차이를 재지만 기록용이다(305–308행).
- `ops.py:341-346`: 부기 청크 중 새 결과에 없는 것은 `removed`(계약 3.4.3 전칭 규칙).
- 해시(`blob_hash`)는 이음새 스냅으로 바뀐 청크의 manifest `hash`·`byte_length` 를 갱신할 때만 쓴다(`ops.py:316-326`). `op_assemble`(`ops.py:427-476`)도 같은 구조.
- 따라서 R11 의 "변경 청크" = **마스크(+halo) 셀이 걸치는 청크 = 부기**이고, 점유 비교도 해시 비교도 아니다. 부기 밖은 디코딩 결과와 무관하게 부모 바이트 승계.

**VoxHammer 단독 트랙(`vh-track/chunk_compare.py`)은 측정용이다:** GLB 를 계약대로 청킹해 부모·자식 청크 sha256 을 만들고(54행) `diff_chunk_sets` 로 changed/removed 를 센다(96행). 같은 파일 docstring(10–11행)이 "부기는 마스크+halo → `affected_chunks()`, `diff_chunk_sets()` 는 그 부기가 맞는지 **검증**하는 용도로만"이라고 적고, 88–108행에서 해시 diff 결과가 부기 밖으로 새는 청크 수(`touched_outside_bookkeeping`)를 잰다. 즉 해시는 검증·계측에만 쓰인다.

(고객사 검증 스크립트 `tests/customer/v_cbin_direct.py` 가 합성 메시에서 변경 청크를 해시로 정하는 것과 다르다 — `tests/customer/README.md` 주의 문구와 같은 결론.)

## sha256 대조 (원본 / 치환 전 복사본 / 커밋된 복사본, 앞 16자)

치환 전 복사본은 40개 모두 원본과 같다(스크립트로 확인). 홈 경로를 바꾼 19개만 커밋본이 다르다.

| 복사본 | 원본 위치 | 원본 | 치환 전 복사본 | 커밋된 복사본 | 비고 |
|---|---|---|---|---|---|
| `edit_service/api/__init__.py` | `<HOME>/ai-ar-prototype/server/api/__init__.py` | `e3b0c44298fc1c14` | = | `e3b0c44298fc1c14` | 동일(원본 = 복사본) |
| `edit_service/api/build.py` | `<HOME>/ai-ar-prototype/server/api/build.py` | `de15f9e33ec8082a` | = | `de15f9e33ec8082a` | 동일(원본 = 복사본) |
| `edit_service/api/jobs.py` | `<HOME>/ai-ar-prototype/server/api/jobs.py` | `c74bcfd91868972c` | = | `c74bcfd91868972c` | 동일(원본 = 복사본) |
| `edit_service/api/main.py` | `<HOME>/ai-ar-prototype/server/api/main.py` | `dc109c2ea96dc3da` | = | `dc109c2ea96dc3da` | 동일(원본 = 복사본) |
| `edit_service/api/ops.py` | `<HOME>/ai-ar-prototype/server/api/ops.py` | `d11cadf8fdd47d58` | = | `d11cadf8fdd47d58` | 동일(원본 = 복사본) |
| `edit_service/api/routes.py` | `<HOME>/ai-ar-prototype/server/api/routes.py` | `368d7ae824cbae71` | = | `368d7ae824cbae71` | 동일(원본 = 복사본) |
| `edit_service/api/store.py` | `<HOME>/ai-ar-prototype/server/api/store.py` | `5b48298928b26314` | `5b48298928b26314` | `b44a6941e0d8e24b` | 홈 경로 치환 |
| `edit_service/api/tests/test_api.py` | `<HOME>/ai-ar-prototype/server/api/tests/test_api.py` | `b6606eb9fe7fd99e` | = | `b6606eb9fe7fd99e` | 동일(원본 = 복사본) |
| `edit_service/lab/__init__.py` | `<HOME>/ai-ar-prototype/server/lab/__init__.py` | `e3b0c44298fc1c14` | = | `e3b0c44298fc1c14` | 동일(원본 = 복사본) |
| `edit_service/lab/api.py` | `<HOME>/ai-ar-prototype/server/lab/api.py` | `f6d696751d6fa8cf` | = | `f6d696751d6fa8cf` | 동일(원본 = 복사본) |
| `edit_service/lab/debug.py` | `<HOME>/ai-ar-prototype/server/lab/debug.py` | `de79fd1d18c6745f` | = | `de79fd1d18c6745f` | 동일(원본 = 복사본) |
| `edit_service/lab/pipeline.py` | `<HOME>/ai-ar-prototype/server/lab/pipeline.py` | `b5060369eebd7cac` | = | `b5060369eebd7cac` | 동일(원본 = 복사본) |
| `edit_service/lab/regions.py` | `<HOME>/ai-ar-prototype/server/lab/regions.py` | `2b128e218bcb3567` | = | `2b128e218bcb3567` | 동일(원본 = 복사본) |
| `edit_service/lab/render.py` | `<HOME>/ai-ar-prototype/server/lab/render.py` | `418e1b0ac2528aa3` | = | `418e1b0ac2528aa3` | 동일(원본 = 복사본) |
| `edit_service/lab/representation.py` | `<HOME>/ai-ar-prototype/server/lab/representation.py` | `d26997d158064d52` | = | `d26997d158064d52` | 동일(원본 = 복사본) |
| `edit_service/lab/seam.py` | `<HOME>/ai-ar-prototype/server/lab/seam.py` | `b59907c65f4848ff` | = | `b59907c65f4848ff` | 동일(원본 = 복사본) |
| `edit_service/lab/store.py` | `<HOME>/ai-ar-prototype/server/lab/store.py` | `2dd0ffe6f9e315bf` | `2dd0ffe6f9e315bf` | `2d4bd61c9293d6fc` | 홈 경로 치환 |
| `edit_service/lab/vhmask.py` | `<HOME>/ai-ar-prototype/server/lab/vhmask.py` | `0fd7b92ebaaf8c20` | = | `0fd7b92ebaaf8c20` | 동일(원본 = 복사본) |
| `edit_service/lab/view.py` | `<HOME>/ai-ar-prototype/server/lab/view.py` | `54dacbfcf4427d06` | = | `54dacbfcf4427d06` | 동일(원본 = 복사본) |
| `edit_service/repaint/masked_sampler.py` | `<HOME>/ai-ar-prototype/server/repaint/masked_sampler.py` | `7fcf1ecb01a2a31e` | = | `7fcf1ecb01a2a31e` | 동일(원본 = 복사본) |
| `vh-track/chunk_compare.py` | `<HOME>/ai-ar-v3/tools/chunk_compare.py` | `806953837743fc3e` | `806953837743fc3e` | `61c09e1485b4ab2e` | 홈 경로 치환 |
| `vh-track/debugview_standalone.py` | `<HOME>/ai-ar-v3/tools/debugview_standalone.py` | `7782537f86b1273c` | `7782537f86b1273c` | `68b0fcef7efd4a88` | 홈 경로 치환 |
| `vh-track/edit_slat_reuse.py` | `<HOME>/ai-ar-v3/tools/edit_slat_reuse.py` | `19b38d6537af14a5` | `19b38d6537af14a5` | `d20a0635c15ea7f4` | 홈 경로 치환 |
| `vh-track/env.sh` | `<HOME>/ai-ar-v2/VoxHammer/env.sh` | `50b91cdc427d7419` | `50b91cdc427d7419` | `cd3693edfdf19280` | 홈 경로 치환 |
| `vh-track/exr_mask_postprocess.py` | `<HOME>/ai-ar-v3/tools/exr_mask_postprocess.py` | `793f6cb9c148dfe6` | = | `793f6cb9c148dfe6` | 동일(원본 = 복사본) |
| `vh-track/gen_asset.py` | `<HOME>/ai-ar-v3/tools/gen_asset.py` | `6bef3527dcdde568` | `6bef3527dcdde568` | `e2785250f47adff5` | 홈 경로 치환 |
| `vh-track/inpaint_offload.py` | `<HOME>/ai-ar-v3/tools/inpaint_offload.py` | `13e2e3cb639349cc` | `13e2e3cb639349cc` | `76092c6934deb78b` | 홈 경로 치환 |
| `vh-track/make_mask_glb.py` | `<HOME>/ai-ar-v3/tools/make_mask_glb.py` | `e797371e83f378bd` | `e797371e83f378bd` | `649b685edefc29ad` | 홈 경로 치환 |
| `vh-track/make_thumb.py` | `<HOME>/ai-ar-v3/tools/make_thumb.py` | `00e5c843f1c00cd2` | = | `00e5c843f1c00cd2` | 동일(원본 = 복사본) |
| `vh-track/repeat_bench.py` | `<HOME>/ai-ar-v3/tools/repeat_bench.py` | `8bee76eaab7a9d38` | `8bee76eaab7a9d38` | `f46302a3d3636400` | 홈 경로 치환 |
| `vh-track/repeat_bench_analyze.py` | `<HOME>/ai-ar-v3/tools/repeat_bench_analyze.py` | `d48f58a8e2429a49` | `d48f58a8e2429a49` | `c67f9a160d1d6ef7` | 홈 경로 치환 |
| `vh-track/repeat_bench_run.sh` | `<HOME>/ai-ar-v3/tools/repeat_bench_run.sh` | `903d88de93ab8118` | `903d88de93ab8118` | `ccf1121a47408ddf` | 홈 경로 치환 |
| `vh-track/restart_edit_service.sh` | `<HOME>/ai-ar-v3/tools/restart_edit_service.sh` | `17a14c02ee79fa50` | `17a14c02ee79fa50` | `47d44cff49f7d691` | 홈 경로 치환 |
| `vh-track/run_voxhammer_checked.sh` | `<HOME>/ai-ar-v3/tools/run_voxhammer_checked.sh` | `1426d932ec445e52` | `1426d932ec445e52` | `1d33db2cf6c12a0b` | 홈 경로 치환 |
| `vh-track/slat_capture.py` | `<HOME>/ai-ar-v3/tools/slat_capture.py` | `9e64e313a583fb45` | = | `9e64e313a583fb45` | 동일(원본 = 복사본) |
| `vh-track/start_debugview.sh` | `<HOME>/ai-ar-v3/tools/start_debugview.sh` | `43321aa6b8727e64` | `43321aa6b8727e64` | `6afd69a2da32f1b5` | 홈 경로 치환 |
| `vh-track/w42_region_eval.py` | `<HOME>/ai-ar-v3/tools/w42_region_eval.py` | `e3854c2385991ad1` | `e3854c2385991ad1` | `103bf19cf598e77e` | 홈 경로 치환 |
| `vh-track/w42_run.sh` | `<HOME>/ai-ar-v3/tools/w42_run.sh` | `89af94808adf2506` | `89af94808adf2506` | `89adc893647d7e1d` | 홈 경로 치환 |
| `vh-track/w43_panels.py` | `<HOME>/ai-ar-v3/tools/w43_panels.py` | `3bd96878da22228c` | `3bd96878da22228c` | `9b12bc609565248e` | 홈 경로 치환 |
| `vh-track/w43_run.sh` | `<HOME>/ai-ar-v3/tools/w43_run.sh` | `d77a53b865c04b3e` | `d77a53b865c04b3e` | `f2637fb387ebb007` | 홈 경로 치환 |
