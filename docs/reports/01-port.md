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
| attention 백엔드 dense=sdpa, sparse=flash_attn | `models.py` 환경변수 | GPU 단계 동등성 로그 |
| 상수 4096·16·64·37·14·1374 모델 설정에서 유도 | `models.derive_constants` | CPU `test_constants_derivation`(설정값), GPU(실제 파이프라인) |
| 출발 잡음 = 요청 seed (정오표 E7) | `pipeline.stage2_denoise`(CPU Generator) | GPU 단계 동등성(`start_noise_bit_equal`, `rng_untouched_before_512`) |

(이하 결과는 실행 후 기입)
