# X. 논문 실험 — a5000 보고서

- 장비: a5000 (RTX A5000 24 GB). 브랜치 `item/X-paper-a5000`(`item/02-R4-a5000` `60c951d` = 경계 스냅 포함에서 분기, main 병합).
- 지시: `ops/tasks/X-paper.md`(계획 원문 `ops/paper/PLAN.md`, 지시서 우선). 편집 시간은 재지 않는다(품질만).
- 경로: **P0** 원저자(재등록 + 25 스텝 + `to_glb`, 인페인팅 A5000 bf16). **P1** 우리(W34 + editcore 15/15 + R4 `.cbin`, 인페인팅 3090 INT4).
  **P2** = P1 + 매 편집 전 150시점 재등록(`X_REREG=1`, 인페인팅 3090 INT4).
- 결과 원자료: `$RUNS_DIR/paper/<실험>/`. 보고서에 sha256.

## 0. 준비 상태 (2026-10-26)

- **연결:**
  - 3090 → A5000:8082 는 성공했다(3090 이 04L 워커로 트랙 A 종단 요청 20건을 보냈다, 10:5x–11:18).
  - A5000 → 3090 은 3090 임시 리스너를 기다린다(X3 P1·P2 인페인팅을 A5000 이 직접 부를 수 있는지가 여기에 달렸다).
- **X2-B 데이터(실측):** A5000 `ASSETS_DIR/Edit3D-Bench/data` 에 다음이 있다. **내려받을 것은 없다.**
  - GSO 50 모델 · PartObjaverse-Tiny 50 모델
  - 모델마다 원본 GLB(`source_model/model.glb`)와 프롬프트 3개
  - 프롬프트마다 `3d_edit_region.glb`(3D 마스크)·`2d_render`·`2d_mask`·`2d_edit`. 텍스트는 `metadata.json`(source_prompt, prompt_1–3).
- **04L 워커:** 3090 종단 요청이 11:18 에 끝난 뒤, GPU 를 X4 에 쓰려고 11:29 에 내렸다. 04L 코드(assets_track_a.json·인증 끔·e2e 복제 자산)는 병행한다.

## 1. 채점 도구 `tools/paper_metrics.py` (코드확인, CPU 테스트 통과)

정의는 코드 상단 주석과 같다.

- **좌표:** 자산마다 v0 메시 bbox 로 정한 변환 하나(중심 원점, 최대 변 1) = N 좌표다. 포크 `normalize_scene` 과 같은 식이다.
  - P0 출력은 이미 N 이다(원저자 경로가 입력을 normalize_scene 해서 인코딩).
  - editcore·R4 출력(raw 프레임)은 raw → v0 저장 좌표(트랙 A `glb_normalize`, 트랙 B 항등) → N 으로 옮긴다.
  - 64³ 셀은 raw 프레임 기준이고, T1 뷰어 `voxelToWorld` 와 같은 식이다.
- **표면 샘플:** 면적 비례 100,000점, 면 법선, seed 0.
- **보존 점:** 점의 64³ 셀이 `dilate_cells(제외 셀, 2)` 밖이다.
- **지표 식:**
  - CD = ½(mean_R min_G + mean_G min_R)
  - NC = 최근접 쌍 |n·n| 양방향 평균
  - F1@0.01 = 정밀도(R→G)·재현율(G→R) 조화평균
- **점유:** `eval/rkeep.py`(E18).
- **PSNR·SSIM·LPIPS(M):** Bench `render.py` 16시점(N 좌표 메시, 정점색은 (iv) 보정)으로 렌더한다.
  - 제외 마스크 = 제외 셀 ∪ halo 2 를 같은 카메라로 투영한 실루엣이다.
  - 비교 화소 = (참조 알파 ∪ 결과 알파) − 제외 마스크다.
  - 계산은 Bench `eval_modules/image_metrics.py` 메서드 그대로다(회색 127 합성, 512², LPIPS alex spatial — Bench 와 같게 [0,1] 입력).
- **DINO-I·CLIP-T:** 편집 시점(포크 v → Bench {0:2,1:1,2:0,3:7,4:6}) 결과 렌더 vs 2d_edit·프롬프트. Bench 모델(DINOv2-base, CLIP ViT-B/32)을 쓴다.
- **CPU 테스트 `tests/cpu/test_paper_metrics_cpu.py` 4 passed:**
  - 같은 메시는 CD 0·NC 1·F1 1
  - 평면 평행이동 t = 0.004 는 CD ≈ t·F1 1, 0.02 는 F1 0
  - 셀 ↔ N 왕복, 보존 마스크
  - **A-S1 마스크 셀 실루엣(포크 v1) vs 동결 2d_mask IoU ≥ 0.9**
- **실데이터 1건 확인(A-M1 c15 r1 P1-glb vs v0, 1 m 40 s):**
  - 보존 CD 0.0024, NC 0.941, F1 0.989
  - PSNR(M) 32.2, SSIM 0.965, LPIPS 0.022, DINO-I 0.778, CLIP-T 0.269
  - 정렬 확인(v0 셀 → v0 메시 1셀 이내) 0.843 — T1 V-2 moto 값과 같다(to_glb mincut 이 가려진 면을 지움).

## 2. X4 무편집 재등록 체인

**예상(실행 전):**
- **P0 1회차:** 점유 IoU 가 원문의 0.800–0.959 범위로 떨어진다. 회차마다 150시점 렌더·`to_glb`(단순화 0.95 + mincut)를 다시 거치므로 셀이 조금씩 줄고, CD 가 커지고 PSNR(M) 이 떨어질 것이다.
  - 다만 2회차부터는 입력이 이미 `to_glb` 메시라 손실 폭이 1회차보다 작을 수 있다(포화).
- **P1:** 회차와 무관하게 일정하다. 디코더 1e-7 차이만 있고, 지표는 같은 값이 5번 나온다.
- 빈 마스크 editcore 는 거절될 가능성이 크다(마스크 셀 0 → 편집 컨텍스트 생성 단계). 거절되면 사유를 적는다.

(결과는 실행 뒤 기입)

## 3. X0 재채점

**예상(실행 전):**
- 25 대 15(editcore W34 경로): 보존 영역 CD·NC·F1 은 거의 같다. 보존 셀은 비트 동일이라 차이는 경계·마스크 밖 신규 셀 몫이다.
- PSNR(M)·DINO-I·CLIP-T 차이는 E2a §5·E19 재판정과 같은 방향일 것이다(c15 허용치 안).
- `P1-cbin` 은 `P1-glb` 보다 보존 CD·F1 이 나쁘게 나올 수 있다(가려진 내부 면, R4 3-8 B4 Chamfer).

(결과는 실행 뒤 기입)
