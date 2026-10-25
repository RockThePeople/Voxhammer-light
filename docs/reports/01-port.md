# 0-1 모듈 이식 — 보고서 (a5000)

- 장비: a5000 (RTX A5000 24 GB)
- 브랜치: `item/01-port-a5000` (main `999a54f` 기반)
- 기준 상태: 0번 승인 main `4e215c0`
- 결과 유형: 동일(개선 없음). 토글 없음 — 진입점 `python -m editcore.cli`
- 실행 env: 현행 `<VH_CUR>/.venv`(torch 2.4.0+cu118), TRELLIS 는 `build/reference-v2` 동봉본을 수정 없이 import

## 1. 예상

- 함수 단위: 마스크·키 규칙·스케줄·전처리·Stage 1 샘플러는 같은 연산을 같은 순서로 하므로 **비트 동일**.
- 출발 잡음: 포크는 `manual_seed(seed)` 직후 전역 CPU RNG 에서 첫 `randn` → 이식본의 seed CPU Generator 와 비트 동일.
- 종단: 같은 장치·dtype·연산 순서라 이식–기준 차이는 기준–기준 바닥값(seed 0, 편집 셀 max|Δ| 1.4e-2–3.3e-2, 0번 5d) 안. 보존 셀 비트 동일, Stage 1 coords 일치.
- 시간: 포크와 같은 자릿수(구간 4–7 ≈ 80 s).

## 2. 변경 파일

| 파일 | 내용 |
|---|---|
| `editcore/__init__.py` | 출처·라이선스 고지(VoxHammer MIT, TRELLIS MIT) |
| `editcore/models.py` | `ATTN_BACKEND=sdpa`·`SPCONV_ALGO=native`(포크 1–3행), 모델 적재, **상수 유도** `derive_constants` |
| `editcore/masks.py` | `coords_to_voxel`, D51 `preserve_coords`, `ss_masks`(ply_to_ss_mask), `slat_self_kv_mask`(ply_to_slat_mask) |
| `editcore/cache.py` | 캐시 키 규칙 `kv_key`, `EditContext`(kv·self/cross 마스크·t_latent·order·pos·layer·is_text 한 객체) |
| `editcore/attention.py` | `SSAttention`, `SlatAttention` — `MethodType` 대신 명시적 클래스(원본 모듈 가중치 사용, forward 교체 없음) |
| `editcore/blocks.py` | `SSBlock`, `SSFlow`, `SlatBlock`, `SlatFlow` |
| `editcore/sampler.py` | `InversionSampler`(2차 solver, CFG [0.5,1.0], rescale 3.0, 25 스텝, W38 게이트) |
| `editcore/preprocess.py` | 공동 bbox crop·518 리사이즈·rembg(세션은 호출자가 넘김) |
| `editcore/pipeline.py` | `Editor.edit`(run_edit W34 경로), `EditRequest`, seed 출발 잡음 |
| `editcore/export.py` | 디코딩·`to_glb`·GLB, R2 저장 (파일 쓰기) |
| `editcore/service.py` | 입력 파일 읽기(SLat, 마스크 json, 조건 이미지). R8a 자리 |
| `editcore/render.py` | 포크 조건 렌더러 호출 래퍼(0번 EXR 프록시 스크립트를 별도 프로세스로) |
| `editcore/cli.py` | `python -m editcore.cli` 진입점, `--timing` 구간 3–8·메모리 |
| `tests/cpu/test_editcore_cpu.py` | 함수 단위 CPU 동등성 5건 |
| `tests/gpu/run_stage_equivalence.py` | 실제 모델 단계 동등성·출발 잡음·RNG 확인 |
| `tests/gpu/run_e2e_port.sh`, `tests/gpu/compare_port.py` | 종단 동등성(P-4) |

## 2a. attention 계산 경로 (코드확인, CLAUDE.md main `1fb17a2` 정정의 근거)

- 편집 경로의 Stage 1·2 attention 은 포크가 `MethodType` 으로 바꿔 끼운 `ss_attn_forward`(`edit_pipeline.py:174–212`, 호출 208행)·`slat_attn_forward`(`:256–327`, 호출 322행)가 **`F.scaled_dot_product_attention` 을 직접** 부른다. Stage 2 self-attention 도 sparse 텐서의 feats 를 (1, heads, N, dim) 로 펴서 전 토큰 SDPA 를 한다(298–300·322행).
- 따라서 TRELLIS 의 sparse attention 백엔드(`trellis/modules/sparse/__init__.py:16-25` 의 `ATTN`, 이 env 에서 flash_attn)는 편집 경로(Stage 1·2 flow)에서 쓰이지 않는다. 쓰이는 곳은 교체되지 않은 모듈 — SLat 디코더(swin attention) 등이다.
- `edit_pipeline.py:2` 의 `ATTN_BACKEND=sdpa` 는 dense attention 모듈(`trellis/modules/attention`) 의 `BACKEND` 를 정할 뿐이고, 편집 경로의 dense attention 도 위 직접 호출을 쓴다.
- 00 보고의 로그 `[SPARSE] Backend: spconv, Attention: flash_attn` 은 sparse 모듈 import 시 출력이며 편집 경로의 attention 계산을 뜻하지 않는다. 0번 보고와 이전 CLAUDE.md 의 "Stage 2 는 flash_attn" 서술은 이 점에서 틀렸다.
- 이식본(`editcore/attention.py`)도 같은 직접 호출을 쓴다. SDPA 내부 커널(flash/mem-efficient/math)은 torch 가 dtype·모양으로 고르므로 같은 env·같은 입력이면 같은 커널이 선택된다(추론, 종단 비교로 확인).

## 3. 보존 동작 체크리스트 (CLAUDE.md 「반드시 보존할 동작」)

| 보존 동작 | 이식 위치 | 지키는 테스트 |
|---|---|---|
| 캐시 키: inversion `t_prev`, denoising `t_curr` → 같은 `t_latent` | `sampler.sample`, `cache.kv_key` | CPU `test_cache_keys_and_schedule`(키 집합 동일), `test_stage1_sampler_bit_equal`(캐시 키·값 비트 동일), GPU 단계 동등성(Stage 1·2 캐시 다이제스트) |
| 2차 solver order 1·2 분리 캐시 | `sampler.sample_once` | CPU `test_stage1_sampler_bit_equal`(order 1·2 키 존재·값 동일) |
| CFG 구간 [0.5, 1.0], rescale_t 3.0, 스텝 25 | `sampler` 상수 | CPU `test_cache_keys_and_schedule` |
| W38 cfg=0 게이트 | `sampler.inference_model` | CPU `test_stage1_sampler_bit_equal`(cfg 0 → pos 0 키 없음) |
| Stage 1 voxel 병합 | `pipeline.stage1_denoise` | GPU 단계 동등성(voxel_tgt 해시), 종단 coords |
| Stage 2 보존 feats 최종 덮어쓰기 | `pipeline.stage2_denoise` | 종단(보존 셀 비트 동일) |
| D51 정수 부호화 보존 판정 | `masks.preserve_coords` | CPU `test_masks_bit_equal` |
| Stage 2 inversion 은 보존 좌표로 자른 SLat | `pipeline.stage2_inverse` | GPU 단계 동등성(Stage 2 잠재·캐시 다이제스트) |
| 공동 bbox crop | `preprocess.preprocess_images` | CPU `test_preprocess_bit_equal` |
| 이미지 패치 마스크 CLS·register 5토큰 처리 | `masks.ss_masks`(cond_extra_tokens=5) | CPU `test_masks_bit_equal`, `test_constants_derivation` |
| W34 SLat 재사용 입력과 `slat_space=denormalized` 검사 | `pipeline.edit`(load_slat) | GPU 종단(W34 입력) |
| attention 계산: Stage 1·2 모두 `F.scaled_dot_product_attention` 직접 호출(포크 `ss_attn_forward`:208, `slat_attn_forward`:322) | `attention.SSAttention`/`SlatAttention` 이 같은 호출을 그대로 쓴다. `models.py` 는 포크와 같이 `ATTN_BACKEND=sdpa` 설정 | CPU `test_stage1_sampler_bit_equal`, GPU 단계 동등성 |
| 상수 4096·16·64·37·14·1374 모델 설정에서 유도 | `models.derive_constants` | CPU `test_constants_derivation`(설정값), GPU(실제 파이프라인) |
| 출발 잡음 = 요청 seed (정오표 E7) | `pipeline.stage2_denoise`(CPU Generator) | GPU 단계 동등성(`start_noise_bit_equal`, `rng_untouched_before_512`) |

## 4. 함수 단위 테스트 (CPU, 실측)

`tests/cpu` **12 passed**(이 항목 5건 + 기존 R_keep 5·slat_capture 2). 이식 5건은 포크와 **비트 동일**:
- `test_constants_derivation` — 설정값에서 4096·16·64·37·14·1374·5 유도
- `test_masks_bit_equal` — D51 보존 판정, `coords_to_voxel`, `ss_masks` 4종(dtype 포함), `slat_self_kv_mask`
- `test_cache_keys_and_schedule` — inversion 키 집합 = denoising 키 집합, 25 스텝·rescale 3.0·CFG [0.5,1.0]
- `test_stage1_sampler_bit_equal` — 축소 SS flow(무작위 가중치)로 inversion(cfg 5) → denoising(cfg 6, 마스크·K/V 주입)의 최종 z·잠재 캐시·K/V 캐시 전부 비트 동일, cfg 0 이면 pos 0 키 없음(W38), order 1·2 키 존재
- `test_preprocess_bit_equal` — 공동 bbox crop·518 리사이즈·rembg 출력 3장 비트 동일

기타 테스트: conformance **53/53**, ai-ar-v3 서버 **360 passed**·12 skipped(`venv-tests`), 커밋 `e340360` 기준.

## 5. GPU 단계 동등성 — 실제 모델 (실측)

자산 **moto-b**(v3-moto-b/v1 feats SLat, 9,150 셀), 마스크 **rider-head**(476 셀, 지문 `9e237b53adf1`), 조건 0번 인페인팅 run1, seed 0. 스크립트 `tests/gpu/run_stage_equivalence.py`: 포크 `run_edit` 와 `editcore.Editor.edit` 를 같은 입력으로 각 1회 돌리고, 단계 출력을 잡아 비교한다. 대조로 **포크를 두 번** 돌린 것도 같은 방식으로 비교했다(`--fork_twice`). 결과 JSON sha256: 이식–포크 `59146356d683…`, 포크–포크 `1e0c16a0c9ce…`.

| 항목 | 이식본 – 포크 | 포크 – 포크 |
|---|---|---|
| 상수(4096·16·64·37·14·1374) | 실제 파이프라인에서 유도값 = 포크 숫자 | – |
| Stage 1 inversion 출력(noise) | 비트 동일 | 비트 동일 |
| Stage 1 잠재 캐시 25개 | 25/25 비트 동일 | 25/25 |
| Stage 1 K/V 캐시 8,448개 | 8,448/8,448 비트 동일 | 8,448/8,448 |
| Stage 1 voxel_tgt, coords_tgt (9,286 셀) | 비트 동일 | 비트 동일 |
| 출발 잡음(512행 randn, 9,286×8) | **비트 동일** | 비트 동일 |
| `manual_seed` 직후 ~ 512행 사이 CPU 전역 RNG 소비 | **없음**(진입 시 RNG 상태 = seed 직후 상태) | – |
| Stage 2 잠재 캐시 25개 | 0/25 비트 동일 | **0/25** |
| Stage 2 K/V 캐시 4,800개 | 2,400/4,800 비트 동일 | **2,400/4,800** |
| slat_tgt coords | 동일 | 동일 |
| **Stage 2 보존 셀 8,674개 (필수 합격 조건)** | **비트 동일** | 비트 동일 |
| Stage 2 편집 셀 612개 max\|Δ\| | **1.76e-2** | **1.94e-2** |
| Stage 2 편집 셀 612개 mean\|Δ\| | **2.63e-3** | **2.46e-3** |

- **"패턴이 같다"가 비교한 것:** 두 실행의 Stage 2 캐시를 **항목(키)마다 텐서 바이트의 sha256** 으로 줄여, 같은 키끼리 해시가 같은 개수를 센 것이다(셀 집합이 아니라 **값**). 이식–포크와 포크–포크가 둘 다 잠재 0/25, K/V 2,400/4,800 으로 같은 수가 나왔다. 일치하는 2,400개는 4,800 = 25 스텝 × order 2 × 24 층 × (self, cross) × (k, v) 의 정확히 절반이며, 조건(cond)만으로 정해지는 cross K/V 로 본다(추론 — 키별 일치 목록은 남기지 않았다).
- 이식본–포크 편집 셀 차이(1.76e-2 / 2.63e-3)는 포크–포크(1.94e-2 / 2.46e-3)와 같은 크기다.

### 5a. Stage 2 비결정성의 원인 연산 (실측 위치, 메커니즘은 추론)

`tests/gpu/find_stage2_nondeterminism.py`: 원본(교체 없는) `slat_flow_model` 에 같은 입력(moto-b SLat 정규화, t=500, 고정 cond)을 **두 번** 넣고 하위 모듈 558개 출력의 해시를 순서대로 비교했다(결과 sha256 `aaf67f38548d…`).
- 앞의 20개 모듈(`input_layer`, `input_blocks.0` 의 sparse conv·norm 포함)은 비트 동일. **처음 갈리는 모듈은 `input_blocks.1.updown` = `SparseDownsample`** 이고, 그 뒤 429/558 모듈이 다르다.
- `SparseDownsample.forward` 는 `torch.scatter_reduce(..., reduce='mean')`(`trellis/modules/sparse/spatial.py:36`)로 2배 다운샘플한다. CUDA 에서 이 scatter 는 원자적 덧셈이라 실행마다 누적 순서가 달라질 수 있다(추론).
- sparse conv 는 원인이 아니다: 첫 `SparseConv3d` 하나를 같은 입력으로 두 번 → 비트 동일(`SPCONV_ALGO=native`). (00s2·이전 보고의 "spconv 로 본다"는 추론은 이 측정으로 정정한다.)
- 대조: Stage 1 `sparse_structure_flow_model` 511개 모듈은 두 번 실행이 전부 비트 동일(dense, scatter 없음).
- 결론: Stage 2 편집 셀의 실행 간 차이는 포크 자체의 성질이며, 이식본은 같은 원본 모듈을 그대로 호출하므로 같은 크기의 차이를 갖는다. 이식본에서 고치면 결과가 포크와 달라지므로 0-1 에서는 손대지 않는다(개선 기회로만 기록).

## 6. 종단 동등성 — 16건 × 3회 (P-4, 실측)

시나리오: moto-b 2건(rider-head·rear-wheel, 기준 3회를 새로 돌림) + 00s2 동결 14건(기준 = F-5 r1–r3). 이식본 3회 = `python -m editcore.cli --seed 0`. 입력 SLat 고정, seed 0. 판정(정오표 E7): coords 완전 일치, 보존 셀 비트 동일, 편집 셀은 이식–기준 3쌍(r_i–r_i)이 기준–기준 3쌍 분포 안(max·mean 모두), 4.7e-2 상한.

| 시나리오 | 보존 셀 | 편집 셀 | coords 일치 | 보존 비트 동일 | 기준–기준 max (3쌍) | 이식–기준 max (r_i–r_i) | 이식–이식 max | 기준–기준 mean ×1e-3 | 이식–기준 mean ×1e-3 | 엄격 판정* |
|---|---|---|---|---|---|---|---|---|---|---|
| moto-b-rider-head | 8674 | 612 | 3/3 | 3/3 | 1.35e-02 / 1.25e-02 / 1.30e-02 | 1.97e-02 / 1.54e-02 / 1.58e-02 | 1.79e-02 / 1.25e-02 / 1.25e-02 | 2.42 / 2.45 / 2.42 | 2.44 / 2.44 / 2.48 | 초과 |
| moto-b-rear-wheel | 7764 | 1289 | 3/3 | 3/3 | 1.26e-02 / 1.12e-02 / 1.26e-02 | 1.23e-02 / 1.38e-02 / 1.23e-02 | 1.39e-02 / 1.27e-02 / 1.13e-02 | 2.20 / 2.20 / 2.15 | 2.20 / 2.17 / 2.23 | 초과 |
| track_a-A-M1 | 10590 | 1066 | 3/3 | 3/3 | 1.24e-02 / 1.47e-02 / 1.29e-02 | 1.46e-02 / 1.28e-02 / 1.25e-02 | 1.37e-02 / 1.37e-02 / 1.35e-02 | 2.34 / 2.33 / 2.37 | 2.39 / 2.37 / 2.32 | 초과 |
| track_a-A-M2 | 8845 | 3230 | 3/3 | 3/3 | 1.27e-02 / 1.41e-02 / 1.26e-02 | 1.53e-02 / 1.33e-02 / 1.31e-02 | 1.27e-02 / 1.39e-02 / 1.43e-02 | 2.35 / 2.37 / 2.35 | 2.38 / 2.31 / 2.36 | 초과 |
| track_a-A-S1 | 10394 | 2286 | 3/3 | 3/3 | 1.81e-02 / 4.42e-02 / 5.65e-02 | 1.74e-02 / 1.84e-02 / 3.73e-02 | 1.64e-02 / 1.49e-02 / 1.52e-02 | 2.55 / 2.63 / 2.63 | 2.54 / 2.55 / 2.71 | 초과 |
| track_a-A-S2 | 10517 | 243 | 3/3 | 3/3 | 1.45e-02 / 1.07e-02 / 1.10e-02 | 9.96e-03 / 9.56e-03 / 9.91e-03 | 1.08e-02 / 8.97e-03 / 1.14e-02 | 2.20 / 2.21 / 2.20 | 2.20 / 2.16 / 2.11 | 통과 |
| track_a-A-C1 | 7783 | 2504 | 3/3 | 3/3 | 1.23e-02 / 1.29e-02 / 1.44e-02 | 1.31e-02 / 1.34e-02 / 1.32e-02 | 1.40e-02 / 1.26e-02 / 1.38e-02 | 2.30 / 2.37 / 2.36 | 2.32 / 2.33 / 2.32 | 통과 |
| track_a-A-C2 | 7540 | 1567 | 3/3 | 3/3 | 1.42e-02 / 1.26e-02 / 1.38e-02 | 1.23e-02 / 1.07e-02 / 1.33e-02 | 1.39e-02 / 1.27e-02 / 1.53e-02 | 2.20 / 2.18 / 2.19 | 2.21 / 2.21 / 2.17 | 초과 |
| track_b-B1 | 12615 | 5751 | 3/3 | 3/3 | 1.33e-02 / 1.50e-02 / 1.06e-02 | 1.41e-02 / 1.28e-02 / 1.20e-02 | 1.35e-02 / 1.34e-02 / 1.26e-02 | 1.98 / 2.01 / 1.99 | 2.00 / 1.99 / 2.00 | 통과 |
| track_b-B2 | 21179 | 6344 | 3/3 | 3/3 | 2.53e-02 / 1.23e-02 / 2.44e-02 | 2.04e-02 / 3.11e-02 / 1.27e-02 | 3.08e-02 / 1.72e-02 / 1.46e-02 | 2.12 / 2.08 / 2.11 | 2.10 / 2.15 / 2.11 | 초과 |
| track_b-B3 | 3477 | 507 | 3/3 | 3/3 | 1.17e-02 / 1.11e-02 / 1.04e-02 | 1.08e-02 / 1.09e-02 / 1.08e-02 | 1.04e-02 / 1.04e-02 / 1.15e-02 | 2.13 / 2.17 / 2.20 | 2.18 / 2.18 / 2.13 | 통과 |
| track_b-B4 | 8073 | 4262 | 3/3 | 3/3 | 1.20e-02 / 1.60e-02 / 1.32e-02 | 2.23e-02 / 1.43e-02 / 4.82e-02 | 2.18e-02 / 4.75e-02 / 3.34e-02 | 2.16 / 2.17 / 2.17 | 2.19 / 2.22 / 2.67 | 초과 |
| track_b-B5 | 5105 | 3202 | 3/3 | 3/3 | 1.49e-02 / 1.24e-02 / 1.11e-02 | 1.32e-02 / 1.42e-02 / 1.77e-02 | 1.16e-02 / 1.62e-02 / 1.68e-02 | 2.19 / 2.19 / 2.14 | 2.20 / 2.29 / 2.17 | 초과 |
| track_b-B6 | 6100 | 3032 | 3/3 | 3/3 | 1.47e-02 / 2.02e-02 / 1.26e-02 | 1.43e-02 / 1.66e-02 / 1.43e-02 | 1.22e-02 / 1.65e-02 / 1.49e-02 | 2.26 / 2.30 / 2.14 | 2.18 / 2.25 / 2.19 | 통과 |
| track_b-B7 | 5020 | 14389 | 3/3 | 3/3 | 1.96e-02 / 1.66e-02 / 1.62e-02 | 2.31e-02 / 1.63e-02 / 1.77e-02 | 1.95e-02 / 1.80e-02 / 1.74e-02 | 2.49 / 2.50 / 2.48 | 2.57 / 2.47 / 2.57 | 초과 |
| track_b-B8 | 4043 | 5307 | 3/3 | 3/3 | 1.33e-02 / 1.48e-02 / 1.59e-02 | 1.25e-02 / 1.28e-02 / 1.79e-02 | 1.26e-02 / 1.51e-02 / 1.33e-02 | 2.17 / 2.22 / 2.22 | 2.17 / 2.17 / 2.23 | 초과 |

\* 엄격 판정 = 이식–기준 3쌍의 max 와 mean 이 **모두** 기준–기준 3쌍의 최대값 이하. 엄격 판정 통과 5/16

| 합산(16건 × 3쌍 = 48) | 최소 / 중앙값 / 최대 |
|---|---|
| 편집 max\|Δ\| 기준–기준 | 1.04e-02 / 1.32e-02 / 5.65e-02 |
| 편집 max\|Δ\| 이식–기준 | 9.56e-03 / 1.36e-02 / 4.82e-02 |
| 편집 max\|Δ\| 이식–이식 | 8.97e-03 / 1.38e-02 / 4.75e-02 |
| 편집 mean\|Δ\| 기준–기준 | 1.98e-03 / 2.20e-03 / 2.63e-03 |
| 편집 mean\|Δ\| 이식–기준 | 1.99e-03 / 2.22e-03 / 2.71e-03 |

P(이식–기준 > 기준–기준): max 0.540, mean 0.516

- **필수 조건(coords 완전 일치, 보존 셀 비트 동일)은 16건 × 3쌍 모두 통과.**
- **편집 셀:** 합산 분포로 보면 이식–기준은 기준–기준과 같다(중앙값 max 1.36e-2 vs 1.32e-2, mean 2.22e-3 vs 2.20e-3, 이식–기준이 더 클 확률 0.54 / 0.52 — 0.5 면 같은 분포).
- **엄격 판정은 5/16 통과.** 기준 3개 표본의 최대값과 새 표본 3개를 비교하는 방식이라, 같은 분포에서도 새 표본 셋 중 하나가 그 최대값을 넘을 일이 흔하다(추론). 초과 폭도 대부분 작다(예: A-M2 1.53e-2 vs 기준 최대 1.41e-2, moto-b rear-wheel 1.38e-2 vs 1.26e-2). **판정 방식은 chief 결정이 필요하다** — 합산 분포(또는 시나리오별 기준–기준·이식–이식·이식–기준 9쌍을 한 분포로) 비교를 제안한다.
- **4.7e-2 상한:** 이식–기준 48쌍 중 1쌍(B4 r3, 4.82e-2)이 넘는다. 같은 B4 의 이식–이식도 4.75e-2 로, 이식 r3 한 회차가 크게 흔들린 것이다. 기준 구현도 A-S1 에서 5.65e-2 로 상한을 넘는다(00s2 F-5) — 상한 초과는 이식본만의 성질이 아니다.
- **GLB 면 수**는 기준끼리도 회차마다 다르다(예: B2 52,970 / 52,989 / 52,697). `to_glb` 의 단순화·텍스처 굽기가 편집 셀 값에 따라 달라지기 때문이라 판정에 쓰지 않는다. 다만 **B1 이식 r2·r3 이 33,505·33,446 면으로 기준(29,156–29,452)보다 약 14% 많다** — 원인 미확인(같은 slat 값 차이 범위인데도 면 수가 크게 갈리는 경우가 있는지 확인 필요).

### 6a. 시간·메모리 (3회 평균, s)

| 시나리오 | 구간 3 기준 / 이식 | 4 | 6 | 7 | 8 | GLB 면 기준 r1–r3 | GLB 면 이식 r1–r3 |
|---|---|---|---|---|---|---|---|
| moto-b-rider-head | 1.7 / 1.8 | 42.2 / 42.1 | 19.7 / 19.5 | 16.1 / 17.6 | 31.4 / 33.5 | 13149, 13164, 13129 | 13125, 13142, 13190 |
| moto-b-rear-wheel | 1.7 / 1.7 | 42.1 / 42.0 | 19.0 / 18.8 | 15.9 / 16.4 | 30.7 / 33.4 | 12522, 12274, 12518 | 12282, 12540, 12570 |
| track_a-A-M1 | 1.7 / 1.7 | 41.9 / 42.1 | 22.1 / 22.1 | 18.3 / 17.6 | 30.7 / 33.8 | 15313, 15331, 15347 | 15344, 15349, 15342 |
| track_a-A-M2 | 1.7 / 1.7 | 42.1 / 42.3 | 20.9 / 20.7 | 19.1 / 17.5 | 30.6 / 33.6 | 15727, 15724, 15742 | 15779, 15812, 15778 |
| track_a-A-S1 | 1.7 / 1.8 | 41.9 / 42.1 | 25.8 / 25.9 | 24.5 / 22.9 | 30.5 / 33.3 | 34578, 34582, 34585 | 34587, 34574, 34583 |
| track_a-A-S2 | 1.7 / 1.8 | 42.1 / 42.1 | 25.5 / 25.5 | 20.9 / 21.1 | 30.6 / 33.2 | 31642, 31644, 31646 | 31646, 31644, 31646 |
| track_a-A-C1 | 1.7 / 1.7 | 41.9 / 42.1 | 20.5 / 20.5 | 20.3 / 19.2 | 30.5 / 33.6 | 16194, 16164, 16180 | 16158, 16220, 16190 |
| track_a-A-C2 | 1.7 / 1.8 | 42.0 / 42.1 | 19.9 / 19.9 | 17.9 / 16.6 | 30.5 / 33.4 | 14244, 14232, 14220 | 14240, 14222, 14238 |
| track_b-B1 | 1.7 / 1.7 | 42.1 / 42.1 | 29.3 / 29.4 | 27.2 / 28.7 | 30.7 / 33.5 | 29156, 29452, 29394 | 29394, 33505, 33446 |
| track_b-B2 | 1.7 / 1.8 | 41.8 / 42.1 | 50.0 / 50.9 | 30.1 / 29.1 | 30.4 / 33.3 | 52970, 52989, 52697 | 52699, 52684, 52965 |
| track_b-B3 | 1.8 / 1.7 | 42.0 / 42.0 | 15.6 / 15.4 | 15.6 / 14.7 | 30.7 / 33.3 | 4618, 4608, 4624 | 4604, 4630, 4630 |
| track_b-B4 | 2.0 / 1.8 | 42.4 / 42.1 | 24.8 / 24.6 | 22.2 / 23.4 | 30.6 / 33.5 | 26262, 26242, 26224 | 26266, 26237, 26223 |
| track_b-B5 | 1.7 / 1.7 | 41.9 / 42.1 | 17.4 / 17.1 | 15.8 / 17.6 | 30.6 / 33.4 | 9474, 9530, 9486 | 9520, 9496, 9528 |
| track_b-B6 | 1.8 / 1.8 | 42.1 / 42.2 | 19.1 / 18.9 | 17.7 / 19.6 | 30.7 / 33.7 | 10520, 10624, 10599 | 10578, 10595, 10604 |
| track_b-B7 | 1.8 / 1.7 | 42.1 / 42.3 | 21.9 / 21.9 | 20.8 / 20.6 | 30.8 / 33.4 | 15421, 15419, 15321 | 15313, 15334, 15478 |
| track_b-B8 | 1.7 / 1.8 | 41.9 / 42.1 | 16.7 / 16.6 | 17.5 / 18.4 | 30.7 / 33.3 | 19257, 19386, 19132 | 19439, 19182, 19009 |

| 메모리(최대) | 기준 | 이식 |
|---|---|---|
| max_memory_allocated (GiB, 16건 범위) | 9.97–11.54 | 9.92–11.49 |

- 구간 3–7 은 기준과 이식본이 같은 자릿수이고 거의 같다(Stage 1 42.0–42.3 s, Stage 2 차이 ±0.9 s, 출력 ±1.9 s). 이식본의 구간 4 에는 SS 디코더(구간 5, 0.03 s)가 포함된다.
- **구간 8(모델 로드)은 이식본이 약 3 s 길다(33.2–33.8 vs 30.4–31.4).** 기준의 구간 8 은 import 와 `from_pretrained` 를 나눠 재고, 이식본은 `load_pipeline` 하나의 span 안에서 trellis import·`from_pretrained`·`.cuda()` 를 함께 잰다. 측정 경계 차이로 본다(추론). 결과에는 영향이 없다.
- compare.json sha256: moto-b-rider-head `eee3c02a4a10`, moto-b-rear-wheel `70b9b02cafc2`, track_a-A-M1 `fe4b3dc33662`, track_a-A-M2 `83e215f9c240`, track_a-A-S1 `6493326f9351`, track_a-A-S2 `516e347a8f78`, track_a-A-C1 `8f334b46ed04`, track_a-A-C2 `24d7500f4204`, track_b-B1 `e9108dac67b8`, track_b-B2 `14e8e5bff411`, track_b-B3 `1b93a845b509`, track_b-B4 `270cf0e811bc`, track_b-B5 `9862d7584a6b`, track_b-B6 `894b7afd61cc`, track_b-B7 `054630190287`, track_b-B8 `7e61aedf60c1`

## 7. 예상과의 차이와 원인

- Stage 2 가 비트 동일하지 않은 것은 예상과 다르다 — 원인은 포크 자체의 `SparseDownsample` scatter(5a). 이식본은 포크와 같은 수준의 차이를 낸다.
- 엄격 판정 5/16 — 판정 방식의 표본 수 문제로 본다(6절).
- B1 이식본 GLB 면 수 차이 — 미확인.

## 8. 남은 문제·요청

1. 편집 셀 판정 방식(엄격 vs 합산 분포) 결정(chief).
2. 4.7e-2 상한을 시나리오별 기준–기준 분포가 넘는 경우(A-S1)의 처리(chief).
3. B1 GLB 면 수 차이 확인.
4. 개선 기회(구현 안 함): Stage 2 `SparseDownsample` scatter 를 결정적 연산으로 바꾸면 실행 간 편집 셀 차이가 사라질 수 있다 — 결과가 바뀌는 변경이라 별도 항목·고객사 판단.

