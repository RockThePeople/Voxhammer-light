# CLAUDE.md — Voxhammer-light

VoxHammer(f33cd96 + 기준 패치) 기반 3D 편집 1회의 벽시계 시간을 줄이는 구현 저장소다.
**요구사항의 유일한 기준은 [ops/REQUIREMENTS.md](ops/REQUIREMENTS.md) 4장이다.** 고객사 회신으로 정정·확정된 내용은 [ops/REQUIREMENTS-ERRATA.md](ops/REQUIREMENTS-ERRATA.md) 에 있고, 원문과 다르면 정오표가 우선한다.

**우선순위: 빠른 프로토타입.** 결과(편집 출력·계측 수치)에 영향이 없는 절차·형식·보안 세부는 유연하게 처리하고, 막히면 우회하고 보고서에 한 줄 남긴다. 결과에 영향이 있는 규칙(보존 동작, 동등성 허용치, 토글 기본값 꺼짐)은 그대로 지킨다.

## 세션 시작·종료 절차

1. 시작: `ops/STATE.md` → 자기 장비에 배정된 `ops/tasks/NN-ID.md` 순서로 읽는다.
2. 시작: `python3 tools/check_pins.py` 가 OK 인지 확인한다. 실패하면 작업하지 않고 보고한다.
3. 종료: 결과를 `docs/reports/NN-ID.md` 에 쓰고, PR 본문에 그 경로를 적는다.
   `ops/STATE.md` 는 chief 세션만 고친다.

## 세션 구성과 담당

| 세션 | 장비 | 담당 |
|---|---|---|
| chief | (claude.ai) | 지시서 `ops/tasks/` 작성, 보고서 검토, `ops/STATE.md` 갱신, 다음 항목 결정 |
| a5000 | RTX A5000 24 GB | TRELLIS 생성·편집, `editcore/`, `services/a5000_edit/`, 기준 구현 실행, 계측 구간 3–8 |
| rtx3090 | RTX 3090 24 GB | 인페인팅, `services/rtx3090_inpaint/`, 계측 구간 2 |

- 지시서에 적힌 담당 장비가 아닌 작업은 하지 않는다. 필요하면 보고서에 요청으로 적는다.
- 두 장비가 같이 쓰는 인터페이스(HTTP 규약, 파일 형식)는 `services/common/` 에 정의하고,
  정의 PR 을 먼저 main 에 병합한 뒤 각자 구현한다.

## 진행 규칙 (요구사항서 4.1 요약)

- **한 번에 한 항목.** 지시서에 없는 항목을 미리 구현하지 않는다. 여러 항목을 한 PR 에 섞지 않는다.
- **토글.** 모든 항목은 환경변수로 켜고 끈다. 이름은 `VHL_<ID>` 계열. 결과가 바뀌는 항목은 기본값 꺼짐.
- **비교 기준.** 직전 승인 상태(`ops/STATE.md` 승인 이력의 커밋). 해당 항목만 켠 결과와 끈 결과를 비교한다.
- **보고.** 6장 구간 시간·메모리·동등성 또는 품질 지표·변경 파일 목록. 모든 수치에
  `실측 / 코드확인 / 추론 / 미검증` 중 하나를 붙인다. 실측에는 장비·자산·마스크 지문·커밋·실행 횟수.
  **예상을 먼저 쓰고 결과를 쓴다.** 다르면 원인을 가른다.

## 기준 구현(포크) 규칙

- `third_party/VoxHammer` 는 f33cd96 고정 submodule 이다. **수정하지 않는다.**
- 패치 적용본은 `REFERENCE_DIR=build/reference-v2 bash tools/setup_reference.sh` 로 `build/reference-v2/` 에 만든다(00a 의 `build/reference` 는 스탬프가 없어 쓰지 않는다). 여기도 수정하지 않는다.
- 기준 실행에서 `W39_ATTN_MASK` 를 설정하지 않는다 (W39 제외, 요구사항서 5장).
- 포크에 대해 허용되는 것은 포크 밖에서 호출 시간을 재는 계측 래퍼(`reference/`)뿐이다.
- `third_party/ai-ar-v3` 는 계약(`contract/`)과 `server/glbbuild.py` 를 읽기 전용으로 쓴다.
  `.cbin` 포맷은 바꾸지 않는다. conformance 53/53 유지.

## 이식 규칙 (요구사항서 4.2)

- 새 코드는 `editcore/` 에 모듈별로 둔다: models, preprocess, masks, cache, attention,
  blocks, sampler, pipeline, export, render, service.
- 옮긴 코드에는 출처 주석: `# origin: voxhammer/edit_pipeline.py:<함수> @ f33cd96+patch`.
  VoxHammer(MIT)·TRELLIS 라이선스 고지 유지.
- `MethodType` forward 교체 금지 → 명시적 클래스. 캐시·마스크·`t_latent`·`order`·`pos`·`layer` 는
  편집 컨텍스트 객체 하나로 전달.
- 상수 4096·16·64, 37·14, 1374 는 모델 설정에서 유도하고, 원본 값과 같은지 테스트한다.
- 계산 함수 안에 grad 설정·rembg 세션 생성·파일 입출력을 두지 않는다. 파일 입출력은 `service`·`export` 만.
- **반드시 보존할 동작** (어긋나면 결과가 바뀐다):
  - 캐시 키: inversion `t_prev`, denoising `t_curr` → 같은 `t_latent`
  - 2차 solver order 1·2 분리 캐시
  - CFG 구간 [0.5, 1.0], `rescale_t` 3.0, W38 cfg=0 게이트. 스텝은 포크 25, **editcore 기본 15/15**(E2a 채택, 정오표 E20). inversion·denoising 같은 스텝
  - Stage 1 voxel 병합, Stage 2 보존 feats 최종 덮어쓰기
  - D51 정수 부호화 보존 판정, Stage 2 inversion 은 보존 좌표로 자른 SLat
  - 공동 bbox crop, 이미지 패치 마스크 CLS·register 5토큰 처리
  - W34 SLat 재사용 입력과 `slat_space=denormalized` 검사
  - attention 계산: 편집 경로의 Stage 1·2 attention 은 포크의 `ss_attn_forward`(`edit_pipeline.py:183`)·`slat_attn_forward`(`:293`)가 `F.scaled_dot_product_attention` 을 **직접** 호출한다(코드확인). TRELLIS 의 sparse attention 백엔드(flash_attn)는 편집 경로에서 쓰이지 않고 디코더 등 교체되지 않은 모듈에서만 쓰인다. `edit_pipeline.py:2` 의 `ATTN_BACKEND=sdpa` 는 그 밖의 dense 모듈에 적용된다. 이식본도 같은 호출을 그대로 쓴다(00 A5000 로그 `Attention: flash_attn` 은 sparse 모듈 import 시 출력)

## 동등성 허용치 (요구사항서 6장 + 정오표 E14·E15)

| 대상 | 합격 |
|---|---|
| Stage 1 coords | 완전 일치 (필수, 시나리오별) |
| Stage 2 feats (보존 셀) | 비트 동일 (필수, 시나리오별) |
| Stage 2 feats (편집 셀) | **14건 합산 분포 비교**(E14): 변경–기준 쌍 vs 기준–기준 쌍(각 시나리오당 3쌍), max\|Δ\|·mean\|Δ\| 각각에 단측 순위 검정 p ≥ 0.05, 중앙값 비 ≤ 1.1, P(변경–기준 > 기준–기준) ≤ 0.6. **쌍별 mean\|Δ\| ≤ 상한 = 현재 채택 기준의 기준–기준 mean 최댓값 × 1.2**(E21, 현재 15 스텝 4.06e-3). max 상한 4.7e-2 는 폐지 |
| 보존 영역 청크 | 승계 바이트 동일 |
| 테스트 | conformance 53/53, ai-ar-v3 서버 360 passed |

- **비교 기준(E20 이후):** editcore 15 스텝. 기준–기준 바닥값은 E2a c15 r1–r3(14건 × 3쌍). 0-1·F-5 의 25 스텝 값은 포크 대조에만 쓴다.
- **비트 비교(E22):** 판정에는 한 프로세스 안의 끔/켬 비교만 쓴다. 프로세스 사이(spconv Native 커널 선택)는 알려진 한계로 두고 고치지 않는다 — 같은 seed 비트 재현은 한 프로세스 안에서만 보장, R8a 워커는 재기동 전까지만 같은 입력 = 같은 결과. 프로세스가 다른 비교는 E14·E15(E21) 분포 판정.
- Stage 2 실행 간 차이의 원인은 TRELLIS `SparseDownsample` 의 `scatter_reduce(mean)`(`trellis/modules/sparse/spatial.py:36`)이다. 검증 전용 결정화 토글은 E16(기본 꺼짐).

## 결과 변경 항목 판정 (요구사항서 6장 + 정오표 E18·E19)

- R_keep 소실 = 0(위반 시 결함). R_keep 신규: 시나리오별 ≤ max(기준 × 1.2, 기준 + 20), 14건 합계 ≤ 기준 합계 × 1.1.
- 마스크 안 편집량(신규 + 소실) < 기준의 50% 인 시나리오는 효능 저하로 표시하고 육안 확인.
- 렌더 지표·DINO-I·CLIP-T 절대 허용치(정오표 E19): 시나리오별 Δ PSNR ≥ −0.5 dB, SSIM ≥ −0.003, LPIPS ≤ +0.005, DINO-I ≥ −0.015, CLIP-T ≥ −0.015, Chamfer ≤ +0.002. 14건 평균 PSNR ≥ −0.2 dB, DINO-I ≥ −0.005, CLIP-T ≥ −0.005. 넘은 칸은 육안 시트로 확인하고 개선 칸 수도 적는다. 채택은 고객사.
- 시나리오 표시(E12): A-C2 "조건 이미지 약함", A-S1 "마스크 밖 생성(알려진 결함)".

- 비교 조건: 입력 SLat 을 고정한다(W34 SLat 재사용 입력, 또는 원저자 경로는 `--reuse_render`). 원저자 경로는 150뷰 렌더 잡음 때문에 회차마다 입력 SLat 이 달라 비교에 쓸 수 없다. 기준 구현은 래퍼에서 `run_edit` 직전 `torch.manual_seed(seed)`, 새 패키지는 요청의 seed 로 출발 잡음(포크 `edit_pipeline.py:512` 해당)을 만든다. 같은 seed 에서 두 출발 잡음이 비트 동일인지 테스트로 확인한다.

## 계측 규칙 (요구사항서 6장)

- 구간 경계마다 `torch.cuda.synchronize()` 후 시간 측정.
- VRAM: 구간마다 `reset_peak_memory_stats()` → `max_memory_allocated()`. `nvidia-smi` 값은 별도 열.
- CPU: 프로세스 RSS 피크, 캐시 딕셔너리 텐서 바이트 합 따로.
- 자산: 0번 시간 기준선은 moto-b(원저자 경로)에 마스크 크기가 다른 편집 2건 이상. 시나리오 동결 뒤에는 트랙 A 6건 + 트랙 B 8건 = 14건이 이후 모든 항목의 비교 기준이다(`ops/tasks/00s-scenarios.md`). t1s-snowman·W41·W42 수치는 비교 기준으로 쓰지 않는다. 시간은 3회 평균±표준편차.
- 계측 결과는 `runs/` (git 제외)에 JSON 으로 두고, 보고서에는 요약 표와 JSON 의 sha256 을 적는다.

## 브랜치·PR

- 항목 하나 = 브랜치 하나: `item/NN-ID-<장비>` (예: `item/00-kickoff-a5000`). 재사용하지 않는다.
- PR 하나 = 항목 하나. PR 본문에 보고서 경로.
- 고객사 승인 후 main 병합. **승인 기준은 `ops/STATE.md` 승인 이력의 병합 커밋 해시**다(chief 세션은 태그를 만들 수 없다). 태그 `approved/NN-ID` 는 chief 가 요청할 때 작업 세션이 그 해시에 붙여 push 한다.

## 보안

- 토큰·비밀번호·키 값은 커밋하지 않는다(`.env` 에만). 호스트·포트는 `.env.example` 기본값으로 둬도 된다(워커 포트 8082).
- 모델 가중치·자산·렌더·`.safetensors` 는 커밋하지 않는다.
- 새 의존성은 보고서에 이름·버전·이유를 적는다.

## 방법론

- "예외가 안 났다" 는 "그 경로가 돌았다" 의 증거가 아니다. 경로가 실제로 실행됐다는 로그를 남긴다.
- 보존만 재지 않는다. 편집이 실제로 일어났는지(효능)도 같이 잰다.
- 육안 산출물이 없는 주장은 `미검증` 으로 적는다.
