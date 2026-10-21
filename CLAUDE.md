# CLAUDE.md — Voxhammer-light

VoxHammer(f33cd96 + 기준 패치) 기반 3D 편집 1회의 벽시계 시간을 줄이는 구현 저장소다.
**요구사항의 유일한 기준은 [ops/REQUIREMENTS.md](ops/REQUIREMENTS.md) 4장이다.**

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
- **비교 기준.** 직전 승인 태그 `approved/NN-ID` 상태. 해당 항목만 켠 결과와 끈 결과를 비교한다.
- **보고.** 6장 구간 시간·메모리·동등성 또는 품질 지표·변경 파일 목록. 모든 수치에
  `실측 / 코드확인 / 추론 / 미검증` 중 하나를 붙인다. 실측에는 장비·자산·마스크 지문·커밋·실행 횟수.
  **예상을 먼저 쓰고 결과를 쓴다.** 다르면 원인을 가른다.

## 기준 구현(포크) 규칙

- `third_party/VoxHammer` 는 f33cd96 고정 submodule 이다. **수정하지 않는다.**
- 패치 적용본은 `tools/setup_reference.sh` 로 `build/reference/` 에 만든다. 여기도 수정하지 않는다.
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
  - CFG 구간 [0.5, 1.0], `rescale_t` 3.0, 스텝 25, W38 cfg=0 게이트
  - Stage 1 voxel 병합, Stage 2 보존 feats 최종 덮어쓰기
  - D51 정수 부호화 보존 판정, Stage 2 inversion 은 보존 좌표로 자른 SLat
  - 공동 bbox crop, 이미지 패치 마스크 CLS·register 5토큰 처리
  - W34 SLat 재사용 입력과 `slat_space=denormalized` 검사
  - attention 백엔드: 원본 `edit_pipeline.py:2` 가 import 시 `ATTN_BACKEND=sdpa` 로 강제한다. 이식본도 같은 백엔드로 돈다(00a A5000)

## 동등성 허용치 (요구사항서 6장)

| 대상 | 합격 |
|---|---|
| Stage 1 coords | 완전 일치 |
| Stage 2 feats | max\|Δ\| ≤ 4.7e-2, 동일 코드 2회 실행 차이 병기 |
| 보존 영역 청크 | 승계 바이트 동일 |
| 테스트 | conformance 53/53, ai-ar-v3 서버 360 passed |

## 계측 규칙 (요구사항서 6장)

- 구간 경계마다 `torch.cuda.synchronize()` 후 시간 측정.
- VRAM: 구간마다 `reset_peak_memory_stats()` → `max_memory_allocated()`. `nvidia-smi` 값은 별도 열.
- CPU: 프로세스 RSS 피크, 캐시 딕셔너리 텐서 바이트 합 따로.
- 자산 t1s-snowman, t2s-moto. 마스크 크기 다른 편집 각 2건 이상. 시간은 3회 평균±표준편차.
- 계측 결과는 `runs/` (git 제외)에 JSON 으로 두고, 보고서에는 요약 표와 JSON 의 sha256 을 적는다.

## 브랜치·PR

- 항목 하나 = 브랜치 하나: `item/NN-ID-<장비>` (예: `item/00-kickoff-a5000`). 재사용하지 않는다.
- PR 하나 = 항목 하나. PR 본문에 보고서 경로.
- 고객사 승인 후 main 병합. **승인 기준은 `ops/STATE.md` 승인 이력의 병합 커밋 해시**다(chief 세션은 태그를 만들 수 없다). 태그 `approved/NN-ID` 는 chief 가 요청할 때 작업 세션이 그 해시에 붙여 push 한다.

## 보안

- 호스트·IP·포트·키·홈 경로를 코드·문서·보고서에 쓰지 않는다. 환경변수(`.env.example`)로만 받는다.
- 모델 가중치·자산·렌더·`.safetensors` 는 커밋하지 않는다.
- 새 의존성은 보고서에 이름·버전·이유를 적는다.

## 방법론

- "예외가 안 났다" 는 "그 경로가 돌았다" 의 증거가 아니다. 경로가 실제로 실행됐다는 로그를 남긴다.
- 보존만 재지 않는다. 편집이 실제로 일어났는지(효능)도 같이 잰다.
- 육안 산출물이 없는 주장은 `미검증` 으로 적는다.
