# 보조 DS SparseDownsample 결정화 검증 토글 — 보고서 (a5000)

- 장비: a5000 (RTX A5000 24 GB). 브랜치 `item/01a-DS-a5000` (main `bc4f0ae` 에서 분기)
- 근거: 정오표 E16. 검증 전용, **운영 기본값 꺼짐**. 토글 `VHL_DS_DETERMINISTIC=1` 일 때만 켜진다.
- 결과 유형: 검증 도구. 꺼짐 상태의 서비스·CLI 경로는 바뀌지 않는다(모듈 교체가 일어나지 않음).

## 1. 예상 (실행 전, 2026-10-25)

단위 GPU 결과(§3 의 1)를 본 뒤, 종단 실행 전에 적는다.

- **비트 동일(종단):** 토글을 켜면 다음이 모두 비트 동일하다.
  - 대상: 포크–포크, 이식–이식, 이식–포크 3쌍.
  - 항목: Stage 1 잠재·K/V, coords, Stage 2 잠재 25/25·K/V 4,800/4,800, 출발 잡음, 보존·편집 셀 feats.
  - 근거:
    - 0-1 모듈 추적에서 Stage 2 의 첫 차이 지점이 `input_blocks.1.updown`(SparseDownsample) 하나였다.
    - 같은 추적에서 SparseConv3d 단독 2회는 같았다.
    - Stage 1 은 이미 결정적이다.
- **모듈 추적:** 토글을 켜면 558개 모듈이 2회 모두 같다.
- **결과 분포:** 고정 순서 합과 원자적 덧셈의 차이는 반올림 수준이다(단위 GPU: 원본끼리 차이와 같은 크기). 그래서 켠 결과는 꺼진 결과의 실행 간 분포 안에 든다(E14 통과, 쌍별 mean ≤ 3.2e-3).
- **시간:** 결정적 구현은 호출마다 정렬 1회와 슬롯(≤ 8) 순서 덧셈을 한다. Stage 2 구간 증가는 1 s 미만으로 본다. 운영에는 쓰지 않으므로 판정 대상이 아니다.
- **실패 시:** 비트 동일이 안 되면, 남은 비결정 지점의 모듈 이름과 코드 위치를 적고 멈춘다(다른 연산은 바꾸지 않음).

## 2. 구현 (코드확인)

- **결정적 평균 `editcore/blocks.py:fixed_order_segment_mean`:**
  1. 목표 칸 `idx` 로 안정 정렬(`torch.sort(stable=True)`)한다.
  2. 칸마다 슬롯 번호(원래 순서)를 매겨 `[칸, K ≤ 8, C]` 에 배치한다. 칸·슬롯이 겹치지 않는 index_put 이다.
  3. 0 에서 시작해 슬롯 순서대로 더하고, (개수 + 1) 로 나눈다.
  - 원자적 덧셈을 쓰지 않는다. 개수는 정수 `bincount` 로 구한다.
  - **원본 `scatter_reduce(..., reduce='mean')` 은 `include_self=True` 가 기본이라, 초기값 0 이 평균 개수에 들어간다**(실측: [2, 4] → 2.0). 대체 연산도 같은 정의를 따른다.
  - 누적 dtype 은 입력 dtype 그대로다. 그래서 결과는 원본의 원자적 덧셈이 낼 수 있는 덧셈 순서 중 하나와 같은 값이다.
  - `torch.use_deterministic_algorithms(True)` 는 쓰지 않았다. 전역 설정이라 모듈 안으로 가두기 어렵고, 다른 연산의 동작까지 바꾼다.
- **`editcore/blocks.py:DeterministicSparseDownsample`:**
  - TRELLIS `SparseDownsample.forward` 를 옮긴 명시적 모듈이다(출처 주석). 평균 연산만 다르다.
  - 좌표 부호화와 업샘플용 공간 캐시(coords·layout·idx) 등록은 원본과 같다.
  - 클래스 카운터 `calls` 로 경로가 실제로 돌았음을 기록한다.
- **`editcore/models.py`:**
  - `ds_enabled()` 는 `VHL_DS_DETERMINISTIC == "1"` 일 때만 참이다.
  - `apply_ds(pipe)` 는 `slat_flow_model` 의 `SparseDownsample` 인스턴스를 모듈 교체한다(`MethodType` 아님). 교체 목록과 되돌림 함수를 돌려준다.
  - `load_pipeline()` 이 토글이 켜졌을 때만 적용하고, 로그를 남긴다: `[ds] VHL_DS_DETERMINISTIC=1: slat_flow_model SparseDownsample N개 교체 [...]`.
- **`editcore/cli.py`:** 토글이 켜졌을 때 `[ds] deterministic downsample calls N` 을 출력하고, timing JSON 에 `ds` 를 기록한다.
- **포크:** 파일은 고치지 않는다. `tests/gpu/run_ds.py` 안에서만 `trellis.modules.sparse.spatial.SparseDownsample.forward` 를 같은 구현으로 바꿔 끼우고, 끝나면 되돌린다(E16 검증 한정 예외).

## 3. 결과

### 1) 단위 (실측)

(작성 중)
