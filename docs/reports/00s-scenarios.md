# 00s 평가 시나리오 준비 — 보고서 (a5000 통합본, 제출 ①)

- 장비: a5000 (S-2–S-8). S-1(트랙 A 입력 이미지)은 rtx3090 — `item/00s-scenarios-3090` 의 `scenarios/track_a_inputs.json`
- 브랜치: `item/00s-scenarios-a5000` (main `4a2e729` 기반)
- 결과 유형: 평가 기반 준비. 편집 코드 변경 없음
- 자산·렌더·SLat 은 git 밖(`$ASSETS_DIR`, `$RUNS_DIR`)이고 sha256 만 커밋했다
- **제출 ① 범위:** 트랙 A 자산(S-2) + 마스크 초안(S-5), 트랙 B 후보표와 8건 선정안(S-4), R_keep(S-7), 평가 환경(S-3), 정렬(S-8). 조건 이미지(S-6)와 트랙 B 판본 0 인코딩은 승인 뒤.

## 1. 예상 (작업 전)

- 트랙 A 자산: 3종 모두 subject 와 같은 부품을 갖는 메시. SLat 10⁴ 셀 자릿수.
- 평가 환경: GT 복사 → PSNR 무한·LPIPS 0, 재렌더 → 거의 같음.
- 정렬: Bench 모델은 이미 정규화돼 있어 TRELLIS 출력과 같은 프레임(변환 불필요).

## 2. 변경 파일

| 파일 | 내용 |
|---|---|
| `eval/rkeep.py`, `tests/cpu/test_rkeep.py` | **S-7** R_keep (`deltacontract.dilate_cells(cells, 2)` 그대로 호출), CPU 테스트 5건 |
| `scenarios/s2_generate.py` | **S-2** 트랙 A 3D 생성(TRELLIS-image-large v1, seed 42) |
| `scenarios/s5_masks.py`, `scenarios/track_a_masks.json`, `scenarios/track_a_views.json` | **S-5** 마스크 초안, 시점 선택 |
| `scenarios/track_a.json` | 트랙 A 제출 표(자산 해시·마스크·프롬프트·시드·시점, 조건 이미지는 S-6 후) |
| `scenarios/drafts/<ID>_{edit,34}.png` | 마스크 초안 렌더 12장(256²) |
| `scenarios/make_track_b_candidates.py`, `scenarios/track_b_candidates.csv`, `scenarios/track_b_selection_draft.csv` | **S-4** 후보 300건, 8건 선정안 |
| `scenarios/s3_sanity.sh` | **S-3** 평가 환경 정상 동작 |
| `scenarios/s8_align.py` | **S-8** 출력 정렬 |
| `env/a5000-edit3dbench.txt` | 평가 venv 버전 목록 |

새 의존성(별도 venv `edit3dbench`, Python 3.10.12): Edit3D-Bench `requirements.txt` 그대로 + **bpy 4.0.0**(PyPI 에 없어 `download.blender.org/pypi` 에서), **torch 2.7.1 / torchvision 0.22.1 cu126**(requirements 가 끌어온 torch 2.14.1+cu130 은 드라이버 575(CUDA 12.9)에서 못 돈다), **transformers 4.46.3**(5.20 에서 CLIP-T 가 `'BaseModelOutputWithPooling' object has no attribute 'norm'` 로 0건 처리됨 — 포크 평가 코드가 4.x API 를 쓴다). 전체 목록 `env/a5000-edit3dbench.txt`.

## S-1 입력 이미지 수령 (3090 → A5000, git `xfer/track-a-inputs` `2ee08c5`)

`scenarios/inputs/track_a/<asset>/{input.png,image_rgb.png}` 6개를 `$ASSETS_DIR/track_a/<asset>/` 로 복사하고 3090 의 `track_a_inputs.json` sha256 과 대조 — **6/6 일치** (실측). 시드: moto 42, snowman 42, chair 43(3090 기록: seed 42 등받이 세로 살 3개 → chief 지시로 43).

## S-2 트랙 A 3D 생성과 동결 (실측)

TRELLIS-image-large v1(`build/reference-v2/trellis`), 기본 샘플러(pipeline.json: steps 25, cfg 기본값, 인자 안 넘김), `pipeline.run()` 과 같은 순서(preprocess → get_cond → `manual_seed(42)` → SS → SLat → decode)를 풀어 써서 SLat 을 저장. attention 은 dense·sparse 모두 flash_attn(실측 `gen.json`; 이 스크립트는 `edit_pipeline` 을 import 하지 않아 `ATTN_BACKEND=sdpa` 강제가 없다 — TRELLIS 기본 생성 경로와 같은 설정). 산출물은 동결(재생성하지 않음, `gen/` 이 있으면 건너뜀).

| 자산 | SLat 셀 | 메시 정점 / 면 | `.cbin` 청크 | slat sha256 | model.glb sha256 | 정규화 배율 | 샘플링 / to_glb (s) |
|---|---|---|---|---|---|---|---|
| moto | 11,396 | 214,994 / 430,084 | 458 | `dc5f4394c419…` | `ea993b04070b…` | 0.9992 | 8.7 / 21.6 |
| snowman | 10,628 | 312,662 / 625,372 | 610 | `fcd3215e3333…` | `e072dca0f942…` | 1.0008 | 8.5 / 24.6 |
| chair | 9,292 | 210,120 / 420,216 | 467 | `e30ed5fd30f5…` | `67595c621826…` | 0.9994 | 8.4 / 20.9 |

- 저장: `slat.safetensors`(coords int32 (N,4), feats float32 (N,8), slat_space=denormalized, norm 메타), `chunks/*.cbin`(디코더 native → `deltacontract.partition_mesh`·`encode`, voxel_cells=SLat coords), `model_raw.glb`(to_glb 0.95/1024), `model.glb`(최대 변 1·중심 원점 정규화 — 조건 렌더·원저자 경로 입력), `gen.json`. 전체 해시는 `scenarios/track_a.json`.
- 육안(16시점 시트 `$RUNS_DIR/s5_views/*_sheet.png` 확인): moto = 헬멧 쓴 라이더 + 검은 오토바이, snowman = 당근 코·막대 팔·빨간 목도리, chair = 판형 등받이·팔걸이 있는 나무 의자(seed 43, 세로 살 없음 → A-C1 효능을 볼 수 있다).
- 첫 실행이 `to_glb` 에서 `Can't call numpy() on Tensor that requires grad` 로 죽었다(디코딩을 grad 켠 채로 함). `no_grad` 로 샘플링·디코딩 후 `to_glb` 만 grad 켬으로 고쳐 다시 만들었고, 실패 회차의 부분 산출물(moto/gen)은 지운 뒤 생성했다. 동결본은 두 번째 실행 결과다.

## S-3 Edit3D-Bench 확보와 평가 환경 (실측)

- 데이터: `huanngzh/Edit3D-Bench` revision `7ad25a53dd19e6273e9e204862fed64cb11bba33`, 18,504 파일(6.2 GB). git-lfs 가 없어 `huggingface_hub.snapshot_download` 사용. **익명 접근이 HF 속도 제한(429)에 계속 걸려** metadata·GLB 400개(원본 100 + 편집 영역 300)와 확인용 객체 1개를 먼저 받았고, 나머지 PNG 는 재시도하며 받는 중(보고 시점 일부). 장비에 남은 만료 HF 토큰이 요청에 붙어 401 이 났던 것은 `token=False` 로 우회.
- 평가 venv: 위 2절.
- 정상 동작(GSO/3D_Dollhouse_Lamp prompt_1, 16시점):

| 확인 | PSNR (마스크 밖) | SSIM | LPIPS | DINO-I | Chamfer | CLIP-T |
|---|---|---|---|---|---|---|
| GT 복사(예측 = GT render, edit.glb = 원본) | ∞ (MSE 0 → 포크가 None 으로 제외, count 0) | 1.0000000 | 0.0 | 1.0000000 | 0.00332 | 0.2985 |
| 원본 model.glb 를 포크 `render.py render_images` 로 재렌더 | 89.64 | 0.9999996 | 4.1e-08 | 0.9999997 | 0.00328 | 0.2985 |

- Chamfer 는 같은 메시에서도 0.0033 이다: `geometry_metrics.py` 가 마스크 밖 표면에서 점 20,480 개를 **무작위 샘플**한다(코드확인 `geometry_metrics.py:26-95`). 이 값이 Chamfer 의 바닥값이다.
- 포크 `eval_main.py:_print_summary`(284행)는 PSNR 이 전부 None(완전 일치)이면 `TypeError` 로 끝난다(결과 JSON 은 그 전에 저장됨). 포크 파일이라 고치지 않았다.
- 재렌더 PSNR 89.6 → 렌더러(bpy 4.0 Cycles)가 GT 를 거의 비트로 재현한다.

## S-4 트랙 B 후보표와 8건 선정안

- 후보표 `scenarios/track_b_candidates.csv`: 300건(GSO 150, PartObjaverse-Tiny 150), 오류 0. 편집 영역 셀은 포크 `delete_region_voxel.process_delete_ply`(volume 필터), 원본 점유 셀은 포크 `glb_to_ply` + `extract_feature.voxelize_mesh`(64³ 표면). 편집 영역 GLB 는 300건 중 고유 240개(같은 객체의 prompt 들이 영역을 공유하는 경우가 있다).
- 크기 지표: `mask_ratio`(영역 셀/원본 셀)는 영역이 부피·원본이 표면이라 1 을 넘는 경우가 많다(중앙값 1.72). 그래서 선정에는 **`occ_share` = 영역 안 원본 점유 셀 / 원본 점유 셀**(트랙 A 의 occupied_ratio 와 같은 뜻)을 쓴다.
- 편집 유형 추정(영역 안 점유 < 5% → 추가, without/remove/no → 삭제): 296 교체 / 4 추가 / 0 삭제. 추가 4건 중 3건은 영역이 격자 거의 전체(63,180셀)인 같은 객체라 추정이 믿을 만하지 않다. 유형은 선정 근거로 쓰지 않았다(추론).
- **크기 구간 경계(삼분위):** 소 occ_share < 0.1764, 중 0.1764–0.354, 대 ≥ 0.354.
- **선정 규칙:** 소 3·중 3·대 2, 데이터셋 GSO 4·PartObjaverse 4 교대, 구간 중앙값에 가장 가까운 후보, 객체 중복 없음, 영역 셀 ≥ 100.

| # | 데이터셋 | 객체 | prompt | 프롬프트 | 영역 셀 | 원본 셀 | occ_share | 크기 |
|---|---|---|---|---|---|---|---|---|
| B1 | GSO | RJ_Rabbit_Easter_Basket_Blue | 3 | A basket with a watermelon. | 21,884 | 13,590 | 0.085 | 소 |
| B2 | PartObjaverse-Tiny | 0c4e4a0b2a374553ae762bb33e8dd2d0 | 1 | A blue ice cream with knife and fork on it. | 9,964 | 23,222 | 0.083 | 소 |
| B3 | GSO | SpiderMan_Titan_Hero_12Inch_Action_Figure_5Hnn4mtkFsP | 1 | Spider-Man with Iron Man's helmet. | 1,518 | 3,799 | 0.086 | 소 |
| B4 | PartObjaverse-Tiny | dcf35ae14df947a9a700b8bcff8db57f | 1 | A fan with a black metal base. | 16,727 | 10,719 | 0.254 | 중 |
| B5 | GSO | Squirtin_Barnyard_Friends_4pk | 2 | A sheep, a cow, a red cow, and a chicken. | 17,992 | 6,841 | 0.255 | 중 |
| B6 | PartObjaverse-Tiny | 97582e1d16f6471bac014081535c26ad | 1 | A queen with a cuirass. | 7,432 | 8,216 | 0.260 | 중 |
| B7 | GSO | 3D_Dollhouse_TablePurple | 1 | A porcelain dish on wooden supports. | 64,840 | 11,345 | 0.559 | 대 |
| B8 | PartObjaverse-Tiny | 5fb5b1704b464f5c94b667f1b630bb39 | 1 | A merry-go-round. | 20,092 | 8,957 | 0.558 | 대 |

승인 후 이 8건만 원저자 경로로 SLat 을 1회 만들어 동결(판본 0)하고, 등록 비용·점유 손실·IoU 를 건마다 적는다. 데이터셋에는 prompt 별 `2d_render.png`·`2d_mask.png`·`2d_edit.png` 가 들어 있다(실측, Lamp) — S-6 에서 트랙 B 조건으로 그대로 쓸지 chief 판단 필요(지시서는 "트랙 B 프롬프트는 데이터셋 것").

## S-5 트랙 A 마스크 초안

동결 SLat 격자(64³, 복셀 프레임 Z-up)에서 "대상 부위 점유 셀" 을 선택 상자로 고르고(근거: SLat 점유 투영, 실측), 그 bbox + 여유 2셀(격자로 자름)의 **상자 안 모든 셀(빈 칸 포함)** 을 마스크로 했다. 형식은 `handoff/lasso` 와 같음(`mode: voxels`, `grid_source: slat_coords`, **`halo_margin_voxels: 2`**). 지문 = `deltacontract.mask_fingerprint`. 파일 `$ASSETS_DIR/track_a/<asset>/masks/<ID>.mask.{json,glb}`.

| ID | 부위 | 마스크 상자 (x,y,z) | 셀 | 마스크 안 점유 / 자산 점유 | 지문 앞 12자 | 편집 시점 | 3/4 시점 |
|---|---|---|---|---|---|---|---|
| A-M1 | 라이더 머리 | (23,28,47)–(39,43,60) | 3,808 | 806 / 11,396 (7.1%) | `c8d28b2c6837` | 3 | 1 |
| A-M2 | 뒷바퀴 전체 | (23,40,3)–(40,63,29) | 11,664 | 2,551 (22.4%) | `24b2cf6d3d38` | 0 | 1 |
| A-S1 | 머리 윗부분 + 위 빈 공간 | (18,18,52)–(45,47,63) | 10,080 | 1,375 / 10,628 (12.9%) | `fd61aa8cef41` | 1 | 3 |
| A-S2 | 당근 코 | (28,42,49)–(35,49,56) | 512 | 111 (1.0%) | `1996548c340e` | 4 | 1 |
| A-C1 | 등받이 | (7,42,44)–(56,55,63) | 14,000 | 1,509 / 9,292 (16.2%) | `5721aa432ddf` | 2 | 1 |
| A-C2 | 양쪽 팔걸이 | (6,9,34)–(17,45,54) ∪ (46,9,34)–(57,45,54) | 18,648 | 1,752 (18.9%) | `9e6095bd3963` | 1 | 3 |

- 시점: 포크 `render_rgb_and_mask.py` 5시점 중 마스크 화소가 가장 많은(=가려지지 않고 크게 보이는) 시점을 편집 시점으로, 3/4 시점은 view 1(방위 −45°), 편집 시점이 1 이면 view 3. 시점별 마스크 화소는 `scenarios/track_a_views.json`.
- 초안 렌더: `scenarios/drafts/<ID>_edit.png`, `<ID>_34.png`(256², 빨강 = 2D 마스크). 렌더는 포크 조건 렌더러 그대로(EXR 은 00 보고서의 bpy 프록시).
- **확인 요청 1 — A-S1 위 빈 공간 없음:** snowman 머리가 격자 꼭대기 z=63 까지 차 있다(TRELLIS 가 가장 긴 축을 [−0.5, 0.5] 에 맞춤). 머리 **위**에 모자가 들어갈 빈 칸이 64³ 안에 없다. 초안은 머리 윗부분(z≥52) + 옆 빈 칸만 넣었다. 모자는 머리 윗부분을 바꿔서만 생길 수 있다. 시나리오를 유지할지, 프롬프트·자산을 바꿀지 판단 필요.
- **확인 요청 2 — 규모 표기:** 요구사항 표의 규모(A-M2 중, A-C1 대)와 마스크 안 점유 비율(A-M2 22.4% > A-C1 16.2%)의 순서가 다르다. 셀 수(상자 부피)로는 A-C1 14,000 > A-M2 11,664 로 표와 맞다. 어느 기준으로 볼지 확인 필요.
- A-C2 상자 아래쪽(z 34–35)은 여유 2셀 때문에 좌석 옆면 일부에 닿는다(초안 렌더에서 보임).

## S-7 보존 지표 R_keep (실측)

`eval/rkeep.py`: 정의 원문을 docstring 에 인용. halo 는 `deltacontract.dilate_cells(cells, 2)`(26-이웃, 체비셰프 ≤ 2, 64³ 밖 버림) **그대로 호출**. 입력 (N,3)/(N,4) 모두 받음. 출력: R_keep 셀 수, 부모·자식 점유, IoU, 소실·신규, 마스크 안 편집량(신규·소실), 참고로 halo 고리 안 변화.
`tests/cpu/test_rkeep.py` 5건 통과: (1) 무편집 → IoU 1, 신규·소실 0, 편집량 0 (2) 마스크 밖 1셀 삭제 → 소실 1 (3) 마스크 안만 변경 → R_keep 불변, 편집량 10 (4) 빈 칸 생성 → 신규 1 (5) 셀 1개 halo → 125셀, 모서리 → 27셀, (N,4) 입력.

## S-8 출력 정렬 (실측, Bench 1건 GSO/3D_Dollhouse_Lamp)

- 코드 근거: Bench `model.glb` 는 이미 최대 변 1·중심 원점(실측 bounds (−0.145, −0.500, −0.140)–(0.144, 0.500, 0.140)). 원저자 경로 150뷰 렌더의 `normalize_scene`(`bpy_render.py:231-250`)은 이 경우 항등에 가깝고, glTF 임포트(Y-up → Blender Z-up)가 복셀 프레임(Z-up)이 된다. 디코더 native 출력은 복셀 프레임(ai-ar-v3 frames D9-b). (a) 현행 `to_glb` 는 끝에 Z-up → Y-up 회전을 건다. (b) R4 이후 `.cbin` → `glbbuild.build_glb` 는 `frames.VOXEL_TO_GLB` 를 건다 — frames 주석·테스트상 두 회전은 같은 변환이다.
- 무편집 재구성(150뷰 → DINOv2 → slat_encoder → decode; SLat 3,271 셀 = voxels.ply 3,271 셀)을 두 출력으로 내보내 Bench `render_images` 16시점 렌더, GT render 알파와 실루엣 IoU:

| 출력 | 실루엣 IoU 평균 | 최소 | bounds |
|---|---|---|---|
| (a) to_glb | 0.9923 | 0.9902 | (−0.145, −0.499, −0.140)–(0.144, 0.500, 0.140) |
| (b) .cbin → glbbuild | 0.9925 | 0.9904 | (−0.145, −0.499, −0.140)–(0.144, 0.500, 0.140) |

→ 두 출력 모두 Bench 프레임과 맞다. **변환 함수는 필요 없다**(그래서 `eval/` 에 두지 않았다). 1 미만의 몫은 재구성 손실(왕복)이다.
- 육안 같은 조명 비교 렌더(편집 시점·3/4)는 편집 결과가 나오는 S-6 이후에 만든다(미검증).

## 6. 예상과의 차이

- 평가 venv 의존성 3건(bpy 출처, torch CUDA 빌드, transformers 버전)이 requirements 그대로는 맞지 않았다 → 2절.
- A-S1 의 빈 공간이 격자 안에 없다(확인 요청 1).
- Edit3D-Bench PNG 전체 다운로드는 HF 익명 속도 제한으로 진행 중(S-8 의 다른 건·트랙 B 판본 0 에 필요). 토큰(`HF_TOKEN`)이 장비에 설정되면 바로 끝난다.

## 7. 실행 정보

- 후보표 sha256 `44d1d7dc4ac52ccb…`(`track_b_candidates.csv`), 트랙 A 표 `scenarios/track_a.json`.
- 계측·렌더 산출물: `$RUNS_DIR/s3_sanity`, `s5_drafts`, `s5_views`, `s8_align`, `track_b_cells`(건별 셀 캐시).

## 8. 남은 일·요청

1. 확인 요청 1·2 (A-S1 빈 공간, 규모 기준).
2. 트랙 B 8건 선정안 승인 → 판본 0 인코딩·손실 측정.
3. 마스크 승인 → S-6 조건 이미지(A5000 `inpaint_offload` 경로, seed 0) 생성·동결 → 14건 기준선.
4. `HF_TOKEN`(읽기) 설정 요청 — Edit3D-Bench 나머지 PNG.
