# 00s 평가 시나리오 준비 (0번·0-1번과 병행)

- 근거: [REQUIREMENTS.md](../REQUIREMENTS.md) 6장 「품질 지표」·「평가 자산과 편집 시나리오」, 4.4 0번 "평가 시나리오 준비"
- 결과 유형: 평가 기반 준비. 편집 코드는 바꾸지 않는다
- 담당: **rtx3090** (S-1 트랙 A 이미지), **a5000** (S-2 ~ S-8)
- 브랜치: `item/00s-scenarios-3090`, `item/00s-scenarios-a5000` · 보고서: `docs/reports/00s-scenarios.md` (a5000 이 통합)
- **선행: [00a 환경 점검](00a-env-check.md) 승인.**
- **게이트:** 1번 E2a 는 시나리오가 동결·승인된 뒤 시작한다. 0번 기준선은 moto-b 로 먼저 진행한다(이 작업을 기다리지 않는다).
- 시나리오 정의는 요구사항서 6장 그대로 확정이다(고객사 2026-10-21). 건수·자산·프롬프트를 바꾸지 않는다.

## 확정 사항 요약 (요구사항서 6장)

| 트랙 | 건수 | 자산 | 입력 SLat |
|---|---|---|---|
| A | 6 | 자체 생성 3종(오토바이·눈사람·나무 의자) × 편집 2건 | 생성 시 저장한 `slat.safetensors` (W34 경로) |
| B | 8 | Edit3D-Bench(MIT), 편집 영역 크기 소·중·대 고르게, GSO·PartObjaverse-Tiny 섞음 | 원저자 경로(150뷰 렌더 → 인코딩)로 1회 만들어 동결 → 이후 W34 경로 |
| moto-b | – | 0번 시간 기준선 전용(SLat feats 없음) | 원저자 경로 |

트랙 A 시나리오 (요구사항서 표 그대로):

| ID | 자산 subject | 편집 부위 | 유형 | 인페인팅 프롬프트 | 마스크 규모 |
|---|---|---|---|---|---|
| A-M1 | a man riding a black motorcycle | 라이더 머리 | 교체 | a red and gold Iron Man helmet as the rider's head | 소 |
| A-M2 | (A-M1 자산) | 뒷바퀴 전체 | 형태 변경 | a thick chunky knobby off-road rear tire | 중 |
| A-S1 | a snowman with a carrot nose, stick arms and a red scarf | 머리 윗부분 + 위 빈 공간 | 추가 | a tall black top hat on the snowman's head | 중(빈 공간 위주) |
| A-S2 | (A-S1 자산) | 당근 코 | 교체 | a small round red ball nose | 소 |
| A-C1 | a wooden armchair with armrests | 등받이 | 교체 | a tall backrest with vertical wooden slats | 대 |
| A-C2 | (A-C1 자산) | 양쪽 팔걸이 | 제거 | the chair without armrests, plain seat sides | 중 |

## S-1 트랙 A 입력 이미지 (rtx3090)

- Z-Image-Turbo, 1024², 9 스텝(8 NFE), guidance 0, 시드 고정(기존 3090 설정). 자산 3종.
- 프롬프트 = 템플릿 `{subject}, full body, 3/4 view, product photography, studio lighting, plain white background, centered, entire subject fully visible with margin, no text, no watermark` 의 `{subject}` 만 바꾼다.
- 배경 제거 BiRefNet(기존 생성 경로). 편집 경로의 BiRefNet 제외(5장)와는 별개다.
- 산출: 자산별 `input.png` + 생성 파라미터 JSON(모델, 시드, 스텝, 프롬프트 전문, 환경 버전). 이미지는 git 에 넣지 않고 `ASSETS_DIR` 에 두며 sha256 만 커밋한다.
- 자산 하나가 subject 와 다르게 나오면(부품 누락 등) 시드만 바꿔 다시 만들고, 쓴 시드와 이유를 기록한다.

## S-2 트랙 A 3D 생성과 동결 (a5000)

- TRELLIS-image-large v1, 기본 샘플러 설정, 시드 고정. TRELLIS.2 금지(SLat 공간이 다름).
- 저장: `slat.safetensors`(coords int32 (N,4), feats float32 (N,8), `slat_space=denormalized`), mesh `.cbin`, 조건 렌더용 GLB, 생성 파라미터 JSON.
- 같은 시드에서도 비결정적이므로 한 번 만든 산출물을 동결한다. 다시 만들지 않는다. 각 파일 sha256 을 `scenarios/track_a.json` 에 기록.

## S-3 Edit3D-Bench 확보와 평가 환경 (a5000)

1. `huanngzh/Edit3D-Bench` 를 받는다(LFS 포함, `ASSETS_DIR` 아래, git 제외).
2. `third_party/VoxHammer/Edit3D-Bench/requirements.txt` 를 **별도 venv** 에 설치. 버전 목록 `env/a5000-edit3dbench.txt`.
3. 정상 동작 확인: GT 를 예측 자리에 복사해 평가(PSNR 매우 큼·LPIPS≈0), `source_model/model.glb` 를 `render.py` 로 렌더해 GT `visual_*.png` 와 비교. 결과를 보고서에 적는다.

## S-4 트랙 B 선정안 (a5000)

- `metadata.json` 전체로 후보표 `scenarios/track_b_candidates.csv`: 데이터셋, 객체, prompt 번호, 프롬프트, `3d_edit_region.glb` 를 64³ 로 복셀화한 셀 수, 원본 점유 셀 수, 마스크 비율, 편집 유형(추가·교체·삭제 추정). 복셀화는 포크 `voxhammer/delete_region_voxel.py` 방식(포크 함수 호출).
- 8건 선정안: 편집 영역 크기 소·중·대가 고르고, GSO 와 PartObjaverse-Tiny 가 섞이게. 선정 근거(크기 구간 경계, 데이터셋 비율)를 적는다. 승인 후 동결.
- 승인된 8건만 원저자 경로로 SLat 을 1회 만들어 동결(R2 형식). 이 인코딩 시간은 편집 시간과 분리해 "자산 등록 비용"으로 적는다. 원본 메시 복셀 대비 점유 손실·IoU 를 건마다 적는다.

## S-5 마스크 (a5000)

- 동결 자산의 SLat 격자(64³) 좌표로 만든다. 형식은 `handoff/lasso/*.mask.json` 과 같게: `mode: voxels`, `grid_source: slat_coords`, **`halo_margin_voxels: 2`**.
- 트랙 A: 대상 부위 점유 셀을 감싸는 상자 + 여유 2셀에서 시작. A-S1 은 모자가 들어갈 빈 공간을 상자에 포함.
- 트랙 B: `3d_edit_region.glb` 복셀화 결과.
- 지문은 `deltacontract.mask_fingerprint` 로 계산해 기록.
- **초안 렌더:** 시나리오마다 편집 시점과 3/4 시점에 마스크를 겹친 렌더 2장 → `scenarios/drafts/<ID>_{edit,34}.png` (작은 해상도). 고객사 승인 후 동결.
- 편집 시점: `render_rgb_and_mask.py` 의 5개 시점 중 편집 부위가 가려지지 않는 1개를 시나리오별로 지정해 기록.

## S-6 조건 이미지 (a5000, 마스크 승인 후)

- 편집 시점 렌더 → 기존 bf16 FLUX(CPU offload, 현행 경로) → `2d_render`·`2d_mask`·`2d_edit` 를 시나리오당 한 번, 시드 고정으로 만들어 동결. 트랙 B 프롬프트는 데이터셋 것.
- TRELLIS 쪽 항목(E2a 등)은 이 동결 이미지를 그대로 쓴다. R9·E4 는 새 조건 이미지를 만들어 따로 평가한다.

## S-7 보존 지표 R_keep (a5000, `eval/`)

정의(요구사항서 6장 원문): R_keep = 64³ 격자 전체에서 (편집 마스크 ⊕ halo 2)를 뺀 셀(빈 칸 포함). R_keep 안에서 부모 대비 점유 IoU, 사라진 셀 수, 새로 생긴 셀 수를 보고. 마스크 안의 새 셀·사라진 셀 수는 편집량으로 함께. 입력은 coords 뿐.

- `eval/rkeep.py`. 편집 경로와 분리. 정의를 docstring 에 원문 인용.
- "⊕ halo 2" 는 마스크 셀을 2셀 팽창한 집합으로 구현한다. 팽창 이웃(6/18/26 연결)을 docstring 과 보고서에 명시한다. **기본은 26-이웃(체비셰프 거리 ≤ 2)** 로 하고, 마스크 json 의 `halo_margin_voxels` 정의와 다르면 그 정의를 따른다.
- 부모 = 그 편집의 입력 판본(트랙 B 첫 편집은 판본 0).
- CPU 테스트 `tests/cpu/test_rkeep.py`: (1) 무편집 출력 → IoU 1, 신규·소실 0, 편집량 0 (2) 마스크 밖 셀 1개 삭제 → 소실 1 검출 (3) 마스크 안만 바꾼 출력 → R_keep 불변, 편집량 > 0 (4) 빈 칸에 셀 생성(R_keep 안) → 신규 1 검출. (1)·(3) 은 보존만 재는 지표가 무편집을 통과시키는지 확인하는 대조다.

## S-8 렌더 지표와 출력 정렬 (a5000)

- Edit3D-Bench `eval_main.py` 의 PSNR·SSIM·LPIPS(마스크 밖), DINO-I, CLIP-T, Chamfer. **FID·FVD 제외.** 포크 파일이므로 호출만.
- 예측은 원본과 위치·방향·스케일이 맞아야 한다. 현행 출력(`to_glb` GLB)과 R4 이후 출력(`.cbin` → `glbbuild` GLB, 디코더 native 좌표)이 Bench 원본 프레임과 어떻게 다른지 코드 근거로 적고, Bench 1건에서 원본 → 무편집 재구성 → 렌더의 실루엣 IoU 를 잰다. 변환이 필요하면 `eval/` 의 함수로 둔다.
- 육안: 편집 시점과 3/4 시점의 같은 조명 비교 렌더.

## 결과 변경 항목의 판정 규칙 (이후 항목에서 사용)

기준 설정 3회 실행의 실행 간 편차를 허용 범위로 삼는다. 비교 설정의 평균이 모든 보존 지표(R_keep IoU·소실·신규, 마스크 밖 PSNR·SSIM·LPIPS)와 DINO-I·CLIP-T 에서 기준 평균보다 허용 범위 이상 나빠지지 않으면 통과 후보, 고객사가 육안 비교 후 확정. 실행 수: 14건 × 설정 × 3회.

## 제출 (고객사 동결·승인 대상)

- `scenarios/track_a.json`, `scenarios/track_b.json`: 자산 파일 해시, 마스크(셀 수·지문), 조건 이미지 해시, 프롬프트, 시드, 편집 시점
- `scenarios/drafts/` 마스크 초안 렌더 (편집 시점·3/4 시점)
- `scenarios/track_b_candidates.csv` 와 8건 선정안
- `eval/rkeep.py` + `tests/cpu/test_rkeep.py`
- 보고서: S-1·S-2 생성 기록, S-3 정상 동작 결과, S-4 판본 0 재구성 손실, S-8 정렬 결과

제출 순서: ① 트랙 A 자산 + 마스크 초안, 트랙 B 선정안 → 승인 → ② 조건 이미지 생성 → 동결 → 1번 E2a 시작 가능.
