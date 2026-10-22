# 00 착수 — 보고서 (a5000)

- 장비: a5000 (RTX A5000 24 GB, 드라이버 575.57.08, Xeon Gold 6226R 32T, RAM 124 GiB)
- 브랜치·커밋: `item/00-kickoff-a5000` (main `4a2e729` 기반)
- 기준 상태: 00a 승인 main `361127c` → 이 브랜치는 `4a2e729` 기반
- 토글: 없음 (0번은 계측만, 코드 동작 변경 없음)
- 실행 env: 현행 `<VH_CUR>/.venv` (Python 3.10.12, torch 2.4.0+cu118) — 설치 없음. 기준 사본 `build/reference-v2` (`PYTHONPATH`·cwd 로 import, 로그로 확인)
- 계측 JSON: `$RUNS_DIR/2026-10-22/moto-b/` (git 제외), sha256 은 7절

## 1. 예상 (계측 전에 적은 것)

- 구간 4–7 합: 116–129 s(기존 W41·W42 실측)와 같은 자릿수(10² s). 원저자 경로는 150뷰 렌더·특징 추출이 붙어 편집 1회 전체는 300 s 이상.
- 구간 2(인페인팅): 00a 기록과 같은 자릿수(10² s). VRAM 피크는 22 GiB 대.
- 구간 8: 30–40 s(모델 적재).
- 동일 코드 2회: Stage 1 coords 일치, Stage 2 feats max|Δ| 는 4.7e-2 근처.

## 2. 변경 파일

| 파일 | 내용 |
|---|---|
| `reference/slat_capture.py` | **A-3** R2 SLat 포획(신규 작성, 복사 아님). `decode_slat` 인스턴스 속성만 잠시 바꿔 입력 SLat 저장 → coords int32 (N,4), feats float32 (N,8), `slat_space=denormalized`, norm 메타 |
| `reference/meter.py` | 구간 계측(동기화·중첩 VRAM 피크·nvidia-smi/RSS 0.1 s 샘플·텐서 바이트) |
| `reference/run_reference.py` | **A-4** 기준선 래퍼. 포크 `inference.run_complete_pipeline` 을 `inference.py __main__` 과 같은 기본 인자로 호출, 모듈 속성 교체로 구간 계측, 호출 횟수 검사, `--profile`, `--reuse_render`(동등성 대조 전용) |
| `reference/run_condition_render.py` | **A-5** 구간 1 (포크 `utils/render_rgb_and_mask.py` 그대로, EXR 읽기만 우회 — 4절 참고) |
| `reference/run_inpaint.py` | **A-5** 구간 2 (포크 `utils/inpaint.inpaint` 기본값 그대로, bf16 + `enable_model_cpu_offload`) |
| `reference/compare_renders.py` | **A-5b** 실행 간 픽셀 차이표, 5시점 썸네일 |
| `reference/prepare_moto_b.py` | moto-b 원저자 경로 입력(정규화 model.glb, mask.glb) 생성 (CPU) |
| `reference/check_moto_b_slat.py` | **A-6** moto-b feats SLat 확인 1–5 |
| `reference/compare_slat_runs.py` | 동일 코드 2회 바닥값(Stage 1 coords, 보존·편집 셀 max\|Δ\|). `run_reference.py --seed` 는 `run_edit` 직전 `torch.manual_seed` (대조 전용) |
| `reference/summarize.py`, `reference/SCHEMA.md` | 계측 JSON 요약, 스키마 `vhl-run/1` |
| `tests/cpu/test_slat_capture.py` | A-3 CPU 테스트 2건 (통과) |
| `tests/customer/gpu/*.py` | **A-7** GPU fp16 사본 (원본 대비 diff 는 부록 A) |
| `env/a5000-flux-fill-dev-sha256.txt` | FLUX.1-Fill-dev 파일별 sha256 |
| `docs/reports/img/00-*.png` | 5시점 썸네일, 조건 이미지 |

새 의존성: 현행 `.venv` 에는 없음. A-8 용 별도 `venv-tests`(시스템 python3 3.10.12): `server/requirements.txt` + pytest 9.1.1, fastapi 0.159.1, python-multipart 0.0.32, httpx 0.28.1, anthropic 2.1.0 (그 밖 pydantic 2.14.1, numpy 2.2.6, trimesh 5.1.1).

## 3. 구간 시간 (moto-b, 원저자 경로, 각 마스크 3회 평균±표준편차, 단위 s, 실측)

조건: A5000 · moto-b(ai-ar-v3 `19ed91f` parent 376청크 → 정규화 GLB, sha `989bb6e8…`) · 커밋 `809a1ad`+작업 트리(이 PR 의 계측 스크립트, 커밋 전 상태) · 실행 3회 + 프로파일 1회(평균 제외).

| 구간 | rider-head (476셀, `9e237b53adf1`) | rear-wheel (1,386셀, `28b22e0004e6`) | 근거 |
|---|---|---|---|
| 1 조건 렌더 (원본 5시점 / 마스크 5시점 / 2D 후처리) | 26.02 ± 1.01 (14.45 / 8.56 / 3.02) | 26.15 ± 0.03 (14.26 / 8.87 / 3.01) | 실측 |
| 2 인페인팅 (적재 / 추론), 장비 간 전송 | 87.97 (0.98 / 86.97 ± 0.36), 전송 0 | 88.40 (0.97 / 87.43 ± 0.11), 전송 0 | 실측 |
| 자산 등록 (150뷰 렌더 / 특징 / SLat 인코딩) | 119.70 ± 0.15 (96.0 / 23.5 / 0.2) | 118.85 ± 0.14 | 실측 |
| 3 전처리 (mask.glb 복셀화 / 이미지 전처리 / 기타) | 43.54 ± 3.33 (≈41.8 / 1.4 / 0.3) | 43.74 ± 2.55 | 실측 |
| 4 Stage 1 inv / den | 42.46 ± 0.21 (27.1 / 15.3) | 42.39 ± 0.11 | 실측 |
| 5 Stage 1 디코딩 | 0.03 | 0.03 | 실측 |
| 6 Stage 2 inv / den | 19.83 ± 0.10 (12.1 / 7.7) | 19.20 ± 0.15 | 실측 |
| 7 디코딩·출력 (mesh / gs / to_glb / export / 재청킹) | 16.62 ± 1.07 (0.4 / 0.0 / 15.6 / 0.4 / 없음) | 18.19 ± 2.79 | 실측 |
| 8 모델 로드 (import / from_pretrained+cuda), 큐 | 31.88 ± 0.62 (5.5 / 26.2), 큐 없음 | 30.57 ± 0.22 | 실측 |
| **4–7 합** | **78.94 ± 1.33** | **79.81 ± 2.94** | 실측 |
| `run_complete_pipeline` 전체(등록+3–7) | 242.25 ± 1.99 | 242.46 ± 3.33 | 실측 |
| 편집 1회 합계 1–8 + 등록 | ≈ 388 | ≈ 390 | 실측 합산 |
| 감싸지 않은 시간(전체 − 구간 합) | 0.06 | 0.06 | 실측 |

- 구간 2 의 "적재" 1 s 는 `from_pretrained` 가 safetensors 를 지연 매핑해서다. 실제 가중치 읽기·CPU→GPU 이동은 오프로딩 훅이 추론 중에 하므로 "추론" 87 s 안에 들어 있다(추론). 두 몫을 span 으로는 나눌 수 없다.
- 구간 5(SS 디코더)는 30 ms 로 nvidia-smi 0.1 s 샘플 창보다 짧아 smi 값이 비는 회차가 있다.
- 재청킹은 현행 VoxHammer 경로(GLB 출력)에 없다. `.cbin` 직행은 2번 R4 에서 생긴다.
- **116–129 s(구간 4–7) 자릿수 대조(참고, 판정 아님):** 78.9 / 79.8 s → 같은 자릿수(10² s 아래쪽). 기존 실측보다 짧은 이유는 가르지 않았다(자산·셀 수·마스크가 다르다).
- **경로 실행 증거:** 모든 회차에서 `calls_match_expected = {}`(Stage 1/2 inv·den 각 1, get_cond 2, preprocess 1, 디코더 각 1, to_glb 1, 150뷰 렌더·특징·mask 복셀화·인코딩 각 1), `decode_slat_calls = 1`, `features.npz`·`voxels.ply`·`voxels_delete.ply`·output.glb 생성, 로그 `[D51] coords_src 9423 · coords_delete 860 · coords_preserve 8915`(rider-head).
- **실행 환경 로그:** `ATTN_BACKEND=sdpa`(dense), sparse attention `flash_attn`, `SPCONV_ALGO=native`, `W39_ATTN_MASK` 미설정·실효 False, 포크 파일 = `build/reference-v2/...`.

### 3a. 구간 4·6 내부 분해 (프로파일 1회, rider-head, 평균 제외, 실측)

`torch.profiler`(CPU+CUDA). 장치 시간 합(µs → ms). 프로파일러 오버헤드로 wall 이 늘었다(전체 512 s, 평소 242 s).

| 함수 | wall (s) | 장치 self 합 (ms) | DtoH memcpy | HtoD memcpy | 매칭 eq / all / argmax | 나머지(대부분 forward) |
|---|---|---|---|---|---|---|
| stage1_inverse | 30.1 | 45,824 | **14,505** | 0.2 | – | 31,319 |
| stage1_denoise | 15.9 | 29,441 | 0.0 | **4,816** | – | 24,625 |
| stage2_inverse | 15.7 | 14,178 | **4,322** | 0.1 | 29.5 / 33.1 / – (8,965 / 8,915 회) | 9,793 |
| stage2_denoise | 9.9 | 12,806 | 1.2 | 1,433 | **403 / 1,154 / 245** (4,902 / 4,852 / 4,852 회) | 9,568 |

- 장치 self 합이 wall 보다 큰 것은 복사 엔진과 계산 커널이 겹쳐 돌기 때문이다(추론).
- stage2_inverse 의 eq·all 8,9xx 회는 `sample_slat_inverse` 의 `for coord in coords_mask` 루프(`edit_pipeline.py:492-493`)다. 장치 시간은 작고 대부분 CPU 커널 발행이다.

### 3b. 캐시 크기 (rider-head r1, 실측)

| 캐시 | 장치 | 바이트 | 항목 |
|---|---|---|---|
| Stage 1 `ss_kv` | CPU | **47.32 GB** | 8,448 |
| Stage 1 `ss_latent` | CPU | 3.3 MB | 25 |
| Stage 2 `slat_kv` | CPU | **15.08 GB** | 4,800 |
| Stage 2 `slat_latent` | CPU | 10.7 MB | 25 |

## 4. 메모리 (rider-head 3회 최대, 실측)

| 구간 | max_memory_allocated | nvidia-smi (MiB) | RSS 피크 (GiB) | 캐시 텐서 바이트 |
|---|---|---|---|---|
| 1 조건 렌더 | (torch 미사용) | – | 0.6 | – |
| 2 인페인팅 | 22.79 GiB | 24,093 | 32.7 | – |
| 자산 등록 | 13.83 GiB | 15,249 | 13.4 | – |
| 3 | 5.70 GiB | 15,249 | 7.8 | – |
| 4 | 5.80 GiB | 15,249 | 49.5 | ss_kv 47.3 GB |
| 5 | 5.79 GiB | – | – | – |
| 6 | 6.25 GiB | 15,251 | 63.6 | slat_kv 15.1 GB |
| 7 | 10.36 GiB | 15,493 | 66.0 | – |
| 8 | 5.62 GiB | 5,839 | 7.0 | – |

nvidia-smi 는 프로세스 전체(캐싱 할당자 예약 포함)라 allocated 보다 크다. RSS 66 GiB 는 요구사항서의 "편집 1회 약 74 GiB"와 같은 자릿수.

## 5. 동등성

| 대상 | 기준 | 결과 | 판정 |
|---|---|---|---|
| Stage 1 coords (원저자 경로, 회차 간) | 완전 일치 | rider-head 9,912 셀 r1=r2=r3, rear-wheel 9,374 셀 r1=r2=r3 | 일치 (실측) |
| Stage 2 feats max\|Δ\| (원저자 경로, 회차 간) | 4.7e-2 | 보존 셀 1.50–1.90, 편집 셀 3.38–8.12 | **허용치와 비교 불가** (아래) |
| Stage 2 feats (렌더 고정 대조 2회) | 4.7e-2 | 보존 셀 0.0(비트 동일), 편집 셀 6.28 | 보존 일치 / 편집은 시드 없는 잡음 (5c) |

**원인 분리(실측):** 회차마다 150뷰 렌더를 새로 해서 `voxels.ply` 는 같지만(sha `add0d0fc…` 3회 동일) `features.npz` 는 회차마다 다르다(Cycles 샘플 잡음 → DINOv2 특징). 그래서 `slat_src` 자체가 회차마다 다르고, Stage 2 가 보존 셀에 덮어쓰는 값도 다르다(보존 셀 max|Δ| ≈ 1.5). 편집 셀에는 추가로 `sample_slat_denoise` 의 시드 없는 `torch.randn`(`edit_pipeline.py:512`) 이 들어간다. 4.7e-2 는 입력 SLat 이 고정된 조건의 바닥값이라 원저자 경로 회차 간 비교에는 맞지 않는다.

### 5c. 렌더 고정 대조 (`--reuse_render`, r1 의 150뷰·mesh 재사용, 특징 추출부터 다시)

rider-head, 2회(eq1·eq2), 실측:

| 대상 | 결과 |
|---|---|
| `features.npz` | 두 회 비트 동일, r1 과도 동일(`d7c5c8dc…`) — 특징 추출(DINOv2)은 같은 이미지에서 결정적 |
| Stage 1 coords | 9,912 셀 완전 일치 |
| Stage 2 feats, 보존 셀 8,915 | **max\|Δ\| = 0.0** (비트 동일) |
| Stage 2 feats, 편집 셀 997 | max\|Δ\| = 6.28 — `sample_slat_denoise` 의 시드 없는 `torch.randn`(`edit_pipeline.py:512`) |

→ 기준 구현 그대로는 편집 셀에 4.7e-2 바닥값을 적용할 수 없다(매 실행 다른 잡음에서 출발). 이후 동등성 비교(0-1 이식 등)는 (a) 보존 셀 비트 동일 + Stage 1 coords 일치, (b) 편집 셀은 같은 시드를 양쪽에 주입한 비교(래퍼에서 `torch.manual_seed` 를 `run_edit` 직전에 설정, 포크 수정 없음)로 하는 것을 제안한다. 판정 방식은 chief 결정.

시간(참고, 렌더 생략): 전체 160.1 / 169.5 s, 구간 4 44.5 / 47.7 s, 구간 6 21.5 / 21.9 s — S-4 복셀화(CPU 8프로세스)와 동시에 돌아 기준선보다 2–5 s 길다.

### 5d. 동일 코드 2회 바닥값 — 입력 SLat 고정(W34) + `run_edit` 직전 `torch.manual_seed(0)` (실측, chief 지정 방식)

`run_reference.py --mode w34 --seed 0` 2회(s1·s2) vs 시드 없는 기준 회차(r1·r2·r3) 비교. 도구 `reference/compare_slat_runs.py`(보존 셀 = 입력 SLat 좌표 − 마스크 셀, `edit_pipeline.py:612` 와 같은 정의).

| 마스크 | 비교 | Stage 1 coords | 보존 셀 max\|Δ\| | 편집 셀 max\|Δ\| (mean) | 4.7e-2 기준 |
|---|---|---|---|---|---|
| rider-head (보존 8,674 / 편집 612) | **seed 0: s1–s2** | 일치 | **0.0 (비트 동일)** | **3.30e-2** (2.85e-3) | **통과** |
| | 시드 없음: r1–r2 / r1–r3 | 일치 | 0.0 | 5.91 / 7.03 | – |
| rear-wheel (보존 7,764 / 편집 1,289) | **seed 0: s1–s2** | 일치 | **0.0 (비트 동일)** | **1.43e-2** (2.19e-3) | **통과** |
| | 시드 없음: r1–r2 / r1–r3 | 일치 | 0.0 | 3.01 / 4.65 | – |

→ 시드를 같게 두면 남는 차이는 CUDA 비결정성뿐이고, 그 바닥값이 1.4e-2–3.3e-2 로 기존 4.7e-2 안이다. 시드 없이는 편집 셀이 출발 잡음(`edit_pipeline.py:512`) 때문에 수 단위로 다르다. 이후 항목의 동등성 비교는 이 조건(입력 SLat 고정, seed 0, 보존 셀 비트 동일 + 편집 셀 ≤ 4.7e-2)으로 한다. 결과 `$RUNS_DIR/2026-10-22/moto-b/w34_floor.txt` sha256 `04252cfc…`. 시드 회차 시간(참고): 78.3–85.4 s.

## 6. 예상과의 차이와 원인

- 구간 4–7 합 79 s 는 예상(116–129 s 자릿수) 안이지만 아래쪽이다. 원인은 가르지 않았다.
- Stage 2 feats 회차 간 차이가 예상(≈4.7e-2)보다 두 자릿수 크다 → 5절의 원인 분리.
- **EXR 읽기 — 증상과 원인:** `OPENCV_IO_ENABLE_OPENEXR=1` 을 설정해도 조건 렌더 후처리가 `load_depth`(`utils/render_rgb_and_mask.py:85-86`)에서 `TypeError: 'NoneType' object is not subscriptable` 로 죽었다. 현행 env 의 opencv 5.0.0 이 **OpenEXR 없이 빌드**돼 있어(`cv2.getBuildInformation()`: `OpenEXR: NO`, 실측) `cv2.imread(*.exr)` 가 None 을 돌려준다. 환경변수로는 해결되지 않는다(코덱 자체가 없다). 포크 스크립트는 `import cv2` 뒤에 환경변수를 설정하지도 않는다 — 변수 위치와 무관하게 이 빌드에서는 실패한다.
- **우회 (래퍼 안, 설치·포크 수정 없음):** `reference/run_condition_render.py` 의 `_Cv2ExrViaBpy` 가 포크 모듈의 `cv2` 이름만 프록시로 바꾼다. `.exr` 이 아닌 호출(렌더 PNG 읽기, `imwrite` 등)은 전부 진짜 cv2 로 넘긴다. `.exr` 은 `bpy.data.images.load` → `img.pixels` 를 (H,W,C) float32 로 만들고 행을 뒤집어(Blender 픽셀 버퍼는 아래→위) 채널을 BGR(A) 순으로 바꿔 돌려준다. 포크의 `load_depth` 식(`[..., 0]`, `> 1000.0`, `10000.0` 대입)은 그대로 돈다.
- **반환값이 원 cv2 와 같은가 (근거):**

  | 항목 | 원 cv2 `imread(path, IMREAD_UNCHANGED)` (OpenEXR 있는 빌드) | 프록시 | 결과 영향 |
  |---|---|---|---|
  | 파일 | 깊이 EXR: 채널 B·G·R 3개, 각 **FLOAT(32bit)**, 알파 없음 (헤더 직접 파싱, 실측). bpyrenderer `enable_depth_output` 이 `file_format="OPEN_EXR"`, `color_depth="32"` 로 저장(`bpyrenderer/render_output.py:211-212`, 코드확인) | – | – |
  | dtype | float32 (32bit float 채널 → CV_32F) | float32 (실측) | 없음 |
  | 채널 수·순서 | 3채널 BGR | **4채널 BGRA**(bpy 가 A=1.0 을 붙인다, 실측 `img.channels = 4`) | 없음 — `load_depth` 는 `[..., 0]`(B) 만 쓴다. 프록시 `[..., 0]` 도 B. 세 채널 값이 같다(실측 `R == B`) |
  | 행 방향 | 위→아래 | 위→아래(뒤집음) | 없음 — 실측: 깊이 < 1000 영역이 같은 시점 RGB 렌더 알파와 5시점 모두 **100.00% 일치**(뒤집지 않으면 85.7–96.4%) |
  | 값 | 파일 float 그대로 | `img.pixels` float 그대로(EXR 은 float 버퍼, 색 변환 없음 — 배경 1e10, 물체 1.56… 실측) | 없음 |

  추가 대조(실측): 실패한 회차가 남긴 원본 렌더(rgb/, mask_3d/ EXR)에 위 프록시 읽기 + 포크 `process` 의 마스크 식을 적용한 결과가, 프록시로 끝까지 돈 다른 회차(run1, 같은 입력)의 `mask_000{0..4}.png` 와 **5시점 모두 비트 동일**. → 실질적으로 원 cv2 경로와 같은 마스크를 낸다고 판단한다(원 cv2 + OpenEXR 빌드로 직접 대조한 것은 아니다 — 그 빌드가 장비에 없어 `미검증`).
- **기존 운영은 어떻게 넘겼나 (코드확인·로그 실측):** `<V3>/tools/exr_mask_postprocess.py`(미추적, A-2 보관 대상)가 같은 문제를 같은 방식으로 우회했다 — 포크 스크립트가 렌더 10장을 끝낸 뒤 후처리에서 죽으면, 남은 `rgb/`·`mask_3d/` EXR 을 bpy(`images.load` → `pixels.foreach_get` → `[::-1, :, 0]`)로 읽어 포크와 글자 그대로 같은 식(`~(depth > 1000.0)`, `depth[mask<0.5]=10000.0`, `diff > 1e-4`, `diff[mask_mask <= 0.5]=0`)으로 `images/mask_*.png` 를 따로 만들었다. `<DATA>/vh-track/w43-cond.log` 에 `load_depth` TypeError 직후 마스크 화소 수가 찍힌 것이 그 흔적이다. 이번 래퍼는 같은 읽기를 포크 `process` 안에서 하도록 바꿨을 뿐이다(렌더를 두 번 돌리지 않고, 후처리 시간도 포크 코드로 잰다).

## 7. 실행 정보

- 자산: moto-b. 원저자 경로는 GLB 를 받지만 **moto-b 의 원본 GLB 는 이 장비에 없다**(`t2s-motorcycle` 은 다른 자산, SLat 12,333셀). 그래서 **ai-ar-v3 `19ed91f` `assets/moto-b/parent/*.cbin` 376청크에서 GLB 를 재구성했다** (`reference/prepare_moto_b.py`, CPU):
  - 청크 이어 붙이기: ai-ar-v3 `server/glbbuild.cbin_dir_to_colored_mesh` 그대로 호출(청크별 정점·면을 오프셋만 더해 연결, 정점 병합 없음, 정점 색 있음). **정점 207,537 / 면 359,220**(다시 읽어도 같음, 실측). 중복 제거하면 정점 179,596(A-6 표와 같은 값).
  - 프레임: `.cbin` 정점은 복셀 프레임(Z-up) → `frames.VOXEL_TO_GLB` 로 GLB(Y-up). glbbuild 의 `build_glb` 와 같은 변환이다.
  - 정규화: VoxHammer 의 조건 렌더(`render_rgb_and_mask.py:34`)와 mask.glb 복셀화(`delete_region_voxel.py`)는 정규화하지 않고, 150뷰 렌더(`bpy_render.py:231-250`)만 `scale = 1/max(bbox 변)`, bbox 중심을 원점으로 옮긴다. 세 단계가 같은 좌표계를 쓰도록 GLB 를 미리 같은 식으로 정규화했다: `p' = (p − c) × s`, c = GLB 프레임 bbox 중심 (−0.0010054, 0.0010255, −0.0001635), s = 1/0.99272 = **1.0073348**. 결과 bounds (−0.1981, −0.3755, −0.5000)–(0.1981, 0.3755, 0.5000), sha256 `989bb6e80d669d130939e66a8640a2f9e8c9d94080d3e175807ef8bbc5754f49`. 원래 bbox 가 이미 거의 단위 크기·원점 중심이라 변화는 0.7% 확대뿐이고, 150뷰 렌더의 자동 정규화는 사실상 항등이 된다.
  - 이 GLB 는 디코더 원 메시(텍스처 없음, 정점 색) 이며 TRELLIS `to_glb` 결과(단순화 0.95 + 텍스처 굽기)가 아니다. 원저자 경로의 입력 GLB 성격이 운영 자산과 다를 수 있다 — 시간 기준선 용도로만 쓰고, 렌더·특징 단계 시간은 이 메시(면 35.9만) 기준이다.
  - 마스크: mask.glb = 마스크 셀 정육면체 합집합의 노출면(닫힌 표면, 같은 정규화). 정규화 후 다시 셀화하면 476 / 1,386 / 2,310 셀 그대로. GLB 정점이 SLat 셀 안에 드는 비율 96.1%(셀 경계 위 정점 때문에 100% 아님).
- 마스크 지문(셀, `deltacontract.mask_fingerprint`, 파일 meta 와 일치 확인): rider-head `9e237b53adf1…`, rear-wheel `28b22e0004e6…`, rear-wheel-full `88270814e860…`. 원저자 경로 `voxels_delete.ply` 는 rider-head 860 셀(마스크 476 전부 포함 + 384), volume 필터(임계 0.1)의 팽창.
- 조건 시점: 두 마스크 모두 **view 0**(방위 0°, 측면). 근거 썸네일 `img/00-moto-b-rider-head-views.png`, `img/00-moto-b-rear-wheel-views.png` — view 0 이 편집 부위가 가려지지 않고 마스크 면적이 가장 크다(10,712 / 32,189 px).
- 인페인팅 프롬프트(트랙 A 표와 같은 문구), 시드 0: rider-head "a red and gold Iron Man helmet as the rider's head", rear-wheel "a thick chunky knobby off-road rear tire". 조건 이미지 `img/00-moto-b-conditions.png`. 3회 모두 `2d_edit.png` 비트 동일(rider-head `fd3a0359…`, rear-wheel `3c0d4782…`). rear-wheel 은 육안상 변화가 작다(효능 약함, 미검증).
- FLUX 가중치: HF 캐시 `models--black-forest-labs--FLUX.1-Fill-dev`, snapshot `358293da0354175698b67ec8299acf928313a78a` — **원본 저장소**. 파일별 sha256 `env/a5000-flux-fill-dev-sha256.txt` (transformer 3샤드 `4cef6b16…`/`cd99fdd4…`/`ed878852…`, T5 2샤드 `ec87bffd…`/`a5640855…`, ae `afc8e282…`).
- 실행 횟수: 마스크당 3회 + 프로파일 1회. 조건 렌더 마스크당 3회, 인페인팅 마스크당 3회.
- 계측 JSON sha256: `summary.json d6a7972a…`, `equivalence_split.json 7c15fdd4…`, `a6/check.json 753e7b14…`; 회차 JSON rider-head r1 `3b9fb9c3…` r2 `e7448e72…` r3 `f0dfb90f…` profile `b9aece61…`, rear-wheel r1 `60357a38…` r2 `3b226b74…` r3 `b1a7cdc9…`.

### A-5b 조건 렌더 실행 간 차이 (같은 입력 3회, 실측)

| 시점 | RGB max (0–255) | RGB 다른 픽셀 (쌍별 최대) | 마스크 다른 픽셀 |
|---|---|---|---|
| 0 (사용 시점) | 1 | 12 | 0 |
| 1 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 |
| 4 | 1 | 9 | 0 |

두 마스크(rider-head, rear-wheel) 모두 같은 표. 쌍 1–2 는 전 시점 RGB 비트 동일, 1–3·2–3 에서 시점 0·4 에 ±1 이 9–12 픽셀. **판정 제안(R10a):** 2D 마스크는 비트 비교, RGB 는 채널 차 ≤ 1 허용. 허용치가 작아 시드 사본은 필요 없다고 본다(판정은 chief).

### A-6 moto-b feats SLat 확인 (실측, 판정은 chief)

| 단계 | 결과 |
|---|---|
| 1 파일 | `v3-moto-b/v1/slat.safetensors` sha256 `b8b1f91c546b3ced690523f3ff4e9da4e9a6b80e5d36a69e9434dfa9d0a8f0ee`, 439,848 B. coords int32 (9150,4)(batch 열 0 뿐), feats float32 (9150,8). 메타: contract_version 3, slat_channels 8, voxel_res 64, coord_order canonical, slat_space denormalized, seed 42, norm_mean/norm_std(8개) |
| 2 coords | `moto-b.slat_coords.npy`(spec `j-00ae8c4f3513`) 와 정렬 후 **완전 일치** 9150/9150 |
| 3 디코딩 2회 | 면 359,220 / 정점 179,596 두 회 같음. 정점 좌표는 비트 동일 아님(fp16 디코더). bbox (−0.198, −0.496, −0.372)–(0.196, 0.497, 0.374) |
| 4 `.cbin` 비교 | 376청크, 면 359,220 = 디코더 359,220. 중복 제거 정점 179,596 = 디코더 179,596. Chamfer(중복 제거 정점 집합, cKDTree, 정규화 공간) 양방향 평균 4.5e-9, 최대 1.35e-6 |
| 5 chief 기준값 | 청크 376, 면 359,220, 청크별 정점 합 207,537, 중복 제거 179,596, voxel_count 합 9,149 — **전부 같음** |

**판정: 통과(chief 2026-10-22, PR #5).** 7단계 W34 참고 측정은 아래 「A-6 W34 참고 측정」.

### A-6 W34 참고 측정 (moto-b feats SLat 재사용, 마스크 2건 × 3회, 실측, 참고값·비교 기준 아님)

`run_reference.py --mode w34`: `run_edit(render_dir=None, slat_path=v3-moto-b/v1/slat.safetensors, mask_cells=마스크 json 셀)`, 조건 이미지는 원저자 경로와 같은 것(인페인팅 run1). 시드 없음(기준 구현 그대로). 모든 회차 `calls_match_expected = {}`, `decode_slat_calls = 1`. 커밋 `d68abd1`+`--seed` 옵션(작업 트리).

| 구간 | rider-head (`9e237b53adf1`) | rear-wheel (`28b22e0004e6`) |
|---|---|---|
| 자산 등록 | 없음 (SLat 재사용) | 없음 |
| 3 전처리 (preprocess_image / get_cond) | 1.73 ± 0.02 (1.44 / 0.29) | 1.73 ± 0.05 |
| 4 Stage 1 inv / den | 42.15 ± 0.23 (26.75 / 15.36) | 42.19 ± 0.09 (26.76 / 15.40) |
| 5 Stage 1 디코딩 | 0.03 | 0.03 |
| 6 Stage 2 inv / den | 19.76 ± 0.14 (12.18 / 7.58) | 19.10 ± 0.08 (11.76 / 7.34) |
| 7 디코딩·출력 (to_glb) | 16.36 ± 0.41 (15.55) | 19.80 ± 1.92 (18.99) |
| 8 모델 로드 (import / load) | 31.20 ± 0.59 (5.36 / 25.83) | 30.87 ± 0.48 (5.29 / 25.58) |
| **4–7 합** | **78.30 ± 0.64** | **81.13 ± 1.86** |
| `run_edit` 전체 | 80.14 ± 0.62 | 82.94 ± 1.91 |
| VRAM allocated 피크 / RSS 피크 | 10.32 GiB / 64.2 GiB | 10.30 GiB / 63.6 GiB |

- coords: 입력 9,150 셀 → Stage 1 결과 9,286(rider-head) / 9,053(rear-wheel), 보존 8,674 / 7,764.
- 캐시: Stage 1 `ss_kv` 47,319,613,440 B — 원저자 경로와 같다(토큰 수가 4,096 고정이라 자산·경로와 무관).
- 원저자 경로 대비: 자산 등록 119–120 s 와 mask.glb 복셀화 ≈42 s 가 빠지고, 구간 4–7 은 같은 자릿수(78–81 s).
- 계측 JSON sha256: rider-head r1 `b5898f17…` r2 `0771068d…` r3 `bd325d81…`, rear-wheel r1 `4b46ff63…` r2 `452f542c…` r3 `50a599cc…`, `summary_w34.json 8f6f98f5…`.

### A-7 고객사 검증 스크립트 (실측)

1. **원본 그대로 (CPU, A5000, `sha256sum -c` OK, 인자 `build/reference-v2`·`third_party/ai-ar-v3`)**

| 스크립트 | A5000 | chief 사전 실행 |
|---|---|---|
| v_stage1_cache | 원본 캐시 419.56 MB · cross 비트 동일 True · A 49.76 MB(11.9%) 비트 동일 True · B 24.88 MB(5.9%) 비트 동일 True · 음성대조 5.007e-06 / 4.130e-02 · 23.1 s | 419.56 MB · True · 11.9% · 5.9% · 5.282e-06 / 4.130e-02 |
| v_stage2_misc | 매칭 비트 동일 True · 원본 94 ms → 인덱스 17.1 ms + 적용 7.53 ms · argmax 0번 덮어씀 True · 개선판 오류 중단 · CFG max\|Δ\| 1.61e-06 / 1.67e-06, 비트 동일 False | True · 127 ms → 0.9 ms · True · 중단 · 1.42e-06, False |
| v_cbin_direct | 968 청크 · 9.4 MB · partition+encode 2.15 / 2.06 s · changed 68/972 · removed 12 · 단위 법선 100% | 968 · 9.4 MB · 1.98 / 1.64 s · 68/972 · 12 · 100% |

불리언·크기·청크 수는 모두 같다. 음성대조 1·CFG 의 작은 수치 차이는 torch 버전 차이(2.4.0 vs 2.14.1)로 본다(추론). 시간은 이 실행 동안 다른 CPU 작업(S-4 복셀화)이 같이 돌아 참고값이다.

2. **GPU fp16 사본** (`tests/customer/gpu/`, diff 부록 A) — 기본 규모

| 스크립트 | 결과 |
|---|---|
| v_stage1_cache | 원본 캐시 209.78 MB(fp16) · cross 비트 동일 True · A 24.88 MB 비트 동일 True · B 12.44 MB 비트 동일 True · 음성대조 1.163e-02 / 5.508e-02 · 2.7 / 2.0 / 1.8 s |
| v_stage2_misc | 매칭 비트 동일 True · 원본 29 ms → 인덱스 42.5 ms + 적용 0.13 ms · argmax 0번 덮어씀 True · 개선판 중단 · CFG(fp16) max\|Δ\| 3.34e-03 / 2.62e-03, 비트 동일 False |
| v_cbin_direct | CPU 경로 — 1의 시간이 A5000 CPU 실행 시간 |

3. **실제 규모 1회** (GPU 사본, `V_R=16 V_C=1024 V_NB=24 V_H=16 V_LCTX=1374`, 보존 546/4,096, fp16)

| 가중치 | 원본 캐시 | A 보존행 K/V + cross 1회 | B 보존행 hidden + cross 1회 | 실행 원본 / A / B | VRAM 피크 | 음성대조 1 / 2 | RSS 피크 |
|---|---|---|---|---|---|---|---|
| 무작위 | 47,319.61 MB · 8,448 항목 | 4,988.53 MB (10.5%) · **비트 동일** | 2,494.27 MB (5.3%) · **비트 동일** | 44.3 / 24.6 / 21.2 s | 1.22 / 1.19 / 1.19 GiB | 1.317e-02 / 3.839e-02 | 102 GiB |
| 실제 `ss_flow_img_dit_L_16l8_fp16` | 47,319.61 MB | 4,988.53 MB · **비트 동일** | 2,494.27 MB · **비트 동일** | 44.6 / 24.7 / 21.3 s | 1.22 / 1.19 / 1.19 GiB | 1.625e+00 / 6.887e+00 | 102 GiB |

- cross K/V 가 모든 t·order 에서 비트 동일: 두 경우 모두 True.
- 원본 캐시 47,319.61 MB 는 실제 파이프라인 계측의 `ss_kv` 47,319,613,440 B(3b)와 **바이트까지 같다** — 스크립트가 실제 규모를 맞게 재현한다.
- 실제 가중치에서 음성대조가 크게 벌어진다(1.6, 6.9) → 비교가 민감하다. 비트 동일 결과의 신뢰도가 무작위 가중치보다 높다.
- CPU RAM 은 RSS 102 GiB 로 124 GiB 호스트에서 돌았다(세 캐시를 동시에 들고 있어서). 중단 없음.

### A-8 테스트 (실측, `venv-tests`)

- conformance **53 passed, 0 failed, 0 skipped**
- ai-ar-v3 서버 테스트 **360 passed, 12 skipped** (20.2 s). 실행 후 submodule 작업 트리 변경 없음.

## A-2 현행 코드 보관 — 보류 (chief 2026-10-22)

다른 로컬 저장소 코드를 이 저장소로 복사하는 단계가 권한 검사에 막혀 보류했다. 대상과 비밀값 스캔 결과만 적는다(실측, 읽기 전용 스캔).

| 대상 | 파일 |
|---|---|
| `legacy/a5000/vh-track/` | ai-ar-v3 `tools/` 미추적 19개: chunk_compare.py, debugview_standalone.py, edit_slat_reuse.py, exr_mask_postprocess.py, gen_asset.py, inpaint_offload.py, make_mask_glb.py, make_thumb.py, repeat_bench.py, repeat_bench_analyze.py, repeat_bench_run.sh, restart_edit_service.sh, run_voxhammer_checked.sh, slat_capture.py, start_debugview.sh, w42_region_eval.py, w42_run.sh, w43_panels.py, w43_run.sh + `<VH_CUR>/env.sh` (00a 의 15개는 `head` 로 잘린 목록이었다) |
| `legacy/a5000/edit_service/` | ai-ar-prototype `server/api/*.py`(build, jobs, main, ops, routes, store, `__init__`) + `api/tests/*.py`, `server/lab/*.py`(api, cli, debug, pipeline, regions, render, representation, seam, store, vhmask, view), `server/repaint/masked_sampler.py`. vendor TRELLIS(`vendor/trellis1`)는 대상 아님 |

스캔: 실제 비밀 값 없음. 키는 환경변수에서만 읽는다(`routes.py:29` `TRELLIS_API_KEY` 기본값 `"dev-key"`, `restart_edit_service.sh:9` 미설정 시 중단). IP 리터럴 없음(0.0.0.0·127.0.0.1 제외). 홈 경로 33행, 포트 기본값(8081·8082·8083) 몇 곳.

## A-9 효과 재산정 (제안값, 확정은 고객사) — 분모: W34 경로 실측 (chief 2026-10-22)

**분모 (나) = W34 경로 편집 1회 실측 226.8 s** = 구간 1 조건 렌더 26.1 + 2 인페인팅 88.2 + 3 전처리 1.7 + 4 Stage 1 42.2 + 5 0.03 + 6 Stage 2 19.4 + 7 디코딩·출력 18.1 + 8 모델 로드 31.0 (두 마스크 평균, 구간 1·2 는 A-5 실측, 3–8 은 A-6 W34 실측). 이전 판의 (나) 추정 226.5 s 와 0.3 s 차. 참고로 (가) 원저자 경로 ≈ 388 s 는 자산 등록 119.7 s·mask.glb 복셀화 41.8 s 를 더 포함한다.

| 순서 | ID | 줄이는 구간 | 대상 시간 (s) | (나) W34 대비 | 근거 |
|---|---|---|---|---|---|
| 1 | E2a 스텝 축소 | 4·6 샘플링 | 61.6 (42.2 + 19.4) | 27.2% (상한, 스텝 비율만큼 일부) | 실측 구간, 감소율은 스텝 비율에 비례(추론) |
| 2 | R4 mesh만 → `.cbin` | 7 (to_glb 15.6–19.0, gs 디코딩 0.04) | 18.1 | 8.0% | 실측 |
| 3 | R10a 조건 렌더 1시점·depth | 1 | 26.1 | 11.5% (상한) | 실측, 5시점→1시점이면 대부분 제거(추론) |
| 4 | R8a 워커 상주 | 8 (import 5.3 + load 25.7) | 31.0 | 13.7% | 실측. 현행 VoxHammer 경로는 상주하지 않는다 → 효과 0 아님 |
| 5 | R9 3090 INT4 FLUX 상주 | 2 | 88.2 | 38.9% (상한) | 실측 |
| 6 | E4 인페인팅 해상도·스텝 | 2 (R9 와 겹침) | 88.2 | – | 실측 |
| 7 | R8b rembg 세션 재사용 | 3 preprocess_image 1.44 의 일부 | < 1.44 | < 0.6% | 실측 |
| 8 | R12 cross K/V 중복 제거 | 4·6 DtoH 의 cross 몫 | ≤ 18.8 (장치 시간 DtoH 14.5 + 4.3 중 일부) | ≤ 8.3% | 프로파일 실측(3a, 원저자 경로 회차), cross 몫 미분리(추론) |
| 9 | R7g 매칭 검사 | – | 0 | 0 | 기능 |
| 10 | R5-1+R5-3 보존 토큰만·GPU 상주 | 4 의 DtoH 14.5 + HtoD 4.8 | ≈ 19 | 8.5% | 프로파일 실측(장치 시간), 캐시 47.3 GB |
| 11 | E3 Stage 1 inversion 선계산 | 4 inv | 26.8 | 11.8% | W34 실측 |
| 12 | R7 매칭 인덱스 1회 | 6 den 매칭 eq/all/argmax 1.8 + inv 좌표 루프 | ≈ 2 | ≈ 0.9% | 프로파일 실측 |
| 13 | E1 Stage 1 CFG 배치 | 4 forward 중 CFG 구간 | ≤ 10 | ≤ 4.4% | 추론 |
| 14 | E2b Euler | 4·6 forward 절반 | ≈ 25 | ≈ 11.0% | 추론(2차→1차, NFE ½) |
| 15 | R10 렌더러 교체 | 1 | 26.1 | 11.5% (상한) | 실측 (W34 경로에는 150뷰 등록이 없다) |
| 16 | R5-2 | 메모리 위주 | – | – | – |
| 17 | R6 | 메모리 위주 | – | – | – |

2·3순위 진입 조건(10%): 11(E3) 11.8%·14(E2b) ≈11.0%·15(R10) 11.5% 가 10% 이상, 10(R5) 8.5%·12(R7) 0.9%·13(E1) ≤4.4%·16·17 은 미만(제안). 13·14 는 추론값이라 해당 항목 착수 전 프로파일로 다시 잰다.

## 8. 남은 문제·요청

- A-6 통과(chief) → W34 참고 측정·시드 고정 바닥값 완료(A-6 W34, 5d). A-9 는 W34 분모로 재산정 완료.
- 원저자 경로의 Stage 2 회차 간 차이는 등록 단계(렌더 잡음)가 지배한다. 이후 항목의 동등성 비교는 입력 SLat 고정(W34) 또는 `--reuse_render` 조건으로 하는 것을 제안.
- opencv 5.0.0(OpenEXR 없음) 우회는 래퍼 안에 있다. env 를 바꾸지 않았다.
- 00a 에서 "15개"로 적은 미추적 스크립트는 19개였다(00a 보고의 `head -15` 절단).
- 발견한 개선 기회(구현 안 함): `sample_slat_inverse` 의 좌표별 파이썬 루프(8,915회 eq·all), Stage 1 inversion DtoH 14.5 s, to_glb 15.6 s(구간 7 의 94%).
