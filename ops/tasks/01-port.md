# 0-1 모듈 이식 — editcore (a5000)

- 요구사항: [REQUIREMENTS.md](../REQUIREMENTS.md) 4.2, 4.4 「0-1. 모듈 이식」. 정오표 E7.
- 선행: 0번 승인(main `4e215c0`, 2026-10-22).
- 담당: **a5000**. 브랜치 `item/01-port-a5000`. 보고서 `docs/reports/01-port.md`.
- 결과 유형: **동일**(개선 없음). 토글 없음(이식본 자체가 새 경로). 진입점 `python -m editcore.cli`(또는 동등)로 포크 `run_edit` 와 같은 입력·출력(GLB)을 낸다.
- 하지 않는 것: 스텝 변경(1), `.cbin` 직행(2), 렌더 축소(3), 상주 워커(4). 발견한 개선 기회는 보고서에만.

## P-1 패키지 구조

`editcore/` 아래 모듈: `models`, `preprocess`, `masks`, `cache`, `attention`, `blocks`, `sampler`, `pipeline`, `export`, `render`(0-1 에서는 포크 렌더러 호출 래퍼만), `service`(빈 자리). CLAUDE.md 「이식 규칙」 그대로: 출처 주석, `MethodType` 금지 → 명시적 클래스, 편집 컨텍스트 객체 하나, 상수 4096·16·64·37·14·1374 는 모델 설정에서 유도 + 테스트, 계산 함수 안 grad·rembg 세션·파일 I/O 금지. TRELLIS 모델 코드는 VoxHammer 동봉본을 수정 없이 import.

## P-2 이식 순서와 함수 단위 동등성

권장 순서(4.4): `masks` → `sampler` → `attention`·`blocks`(Stage 1) → `attention`·`blocks`(Stage 2) → `preprocess` → `pipeline` → `export`. 모듈마다:
- 같은 입력에서 포크 함수 vs 이식 함수 출력 비교 테스트를 `tests/cpu/`(축소 모델, GPU 없이)와 `tests/gpu/`(실제 모델) 에 둔다. 고객사 검증 스크립트 방식(작은 무작위 가중치)을 참고.
- 비트 동일이 기대되는 함수(마스크, 키 규칙, 매칭, 상수 유도)는 비트 비교. 부동소수 경로는 같은 장치·dtype 에서 비트 동일을 먼저 확인하고, 아니면 차이와 원인을 적는다.
- 보존할 동작 체크리스트(CLAUDE.md)를 표로 만들어 항목마다 어느 테스트가 지키는지 적는다.

## P-3 출발 잡음과 시드 (정오표 E7)

- 이식본은 요청의 `seed` 로 출발 잡음(포크 `edit_pipeline.py:512` 의 `torch.randn` 해당)을 만든다.
- 기준 구현은 래퍼에서 `run_edit` 직전 `torch.manual_seed(seed)`. 같은 seed 에서 **두 출발 잡음 텐서가 비트 동일**인지 테스트로 확인한다. 포크가 512행 전에 전역 RNG 를 다른 곳에서 소비하면(있는지 코드로 확인) 이식본이 같은 순서·같은 shape·dtype·장치로 소비해 맞춘다. 이 확인이 안 되면 종단 비교 전에 보고한다.

## P-4 단계·종단 동등성

- 입력 SLat 고정(W34). 시나리오는 00s ② 가 동결되면 14건, 그 전에는 moto-b(rider-head, rear-wheel)로 먼저 한다.
- 시나리오마다 이식본 3회, seed 0. 판정(정오표 E7):
  - Stage 1 coords 완전 일치
  - 보존 셀 feats 비트 동일
  - 편집 셀: 이식본–기준 3쌍(이식 r_i vs 기준 r_i)의 max|Δ|·평균|Δ| 가 기준–기준 3쌍 분포(00s2 F-5, moto-b 는 새로 3회) 안. 분포 밖이면 원인 분리 후 보고. 4.7e-2 는 상한.
  - GLB: 정점·면 수, Chamfer(참고)
- 시간: 구간 3–8, 포크와 같은 자릿수인지. 메모리 4종.
- 테스트: conformance 53/53, 서버 360 passed, `tests/cpu` 전부.

## 완료 조건

- [ ] `editcore/` 모듈 전부, 출처 주석·라이선스 고지
- [ ] 함수 단위 CPU·GPU 테스트, 보존 동작 체크리스트 표
- [ ] 출발 잡음 비트 동일 확인
- [ ] 종단 동등성(moto-b 2건 → 14건), 시간·메모리
- [ ] 보고서, PR
