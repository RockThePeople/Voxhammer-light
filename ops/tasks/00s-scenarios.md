# 00s 평가 시나리오 준비 (0번·0-1번과 병행)

- 근거: 고객사 답변 2026-10-20, 요구사항서 6장 「평가 자산과 편집 시나리오」(개정판)
- 결과 유형: 평가 기반 준비. 편집 코드는 바꾸지 않는다
- 담당: **a5000** (S-1 ~ S-6). 3090 은 S-3 의 인페인팅 조건 이미지가 필요할 때만 요청받는다
- 브랜치: `item/00s-scenarios-a5000` · 보고서: `docs/reports/00s-scenarios.md`
- **게이트:** 1번 E2a 는 이 시나리오가 동결·승인된 뒤 시작한다. 0번 기준선은 moto-b 로 먼저 진행한다(이 작업을 기다리지 않는다).

> ⚠️ chief 가 가진 요구사항서(2026-10-18판)에는 6장 「평가 자산과 편집 시나리오」와 R_keep 정의가 없다.
> 개정판을 받을 때까지 S-1·S-2·S-6 만 진행한다. S-3~S-5 의 건수·선정 기준·지표 정의는 개정판 6장 원문을
> `ops/REQUIREMENTS.md` 에 반영한 뒤 이 문서에 확정한다. 정의를 추측해서 구현하지 않는다.

## 이미 확인된 사실 (chief, 코드확인)

- Edit3D-Bench 평가 코드는 VoxHammer 포크에 동봉돼 있다: `third_party/VoxHammer/Edit3D-Bench/` (`eval_main.py`, `render.py`, `eval_modules/`). 포크 파일이므로 수정하지 않고 호출만 한다.
- 지표: PSNR·SSIM·LPIPS·DINO-I(image), FID, FVD, Chamfer, CLIP-T. DINO 는 `facebook/dinov2-base`.
- PSNR·SSIM·LPIPS 는 GT 마스크가 0 인 픽셀(비편집 영역)만 비교한다(`image_metrics.py` 59–61행 `include_mask = mask_np <= 0`). 즉 2D 보존 지표다. DINO-I 는 전체 이미지.
- 평가 렌더(`render.py`): Cycles, 1024², 반경 1.8, 고도 [15, 30] × 방위 8개 = 16시점, 환경맵 `brown_photostudio_02_1k.exr`, `normalize_scene` 없음. 예측 결과는 원본 모델과 **위치·방향·스케일이 맞아야** 한다(README).
- 데이터: Hugging Face `huanngzh/Edit3D-Bench` (GSO, PartObjaverse-Tiny). 객체마다 `source_model/model.glb`, `prompt_{1,2,3}/3d_edit_region.glb`, `render/visual_*.png`·`mask_*.png`. chief 환경에서는 huggingface.co 가 차단돼 받지 못했다.

## S-1 Edit3D-Bench 확보와 평가 환경

1. 데이터셋을 받는다(LFS 포함). 경로는 `ASSETS_DIR` 아래, git 에 넣지 않는다. 데이터셋 라이선스를 확인해 보고서에 적는다.
2. `Edit3D-Bench/requirements.txt` 를 **별도 venv** 에 설치한다(bpy-renderer 포함). 편집용 venv 와 섞지 않는다. 버전을 `env/a5000-edit3dbench.txt` 로 커밋.
3. 정상 동작 확인: GT 를 예측으로 복사해 평가(PSNR 무한대·LPIPS 0 근처가 나와야 한다)와, 원본 `source_model/model.glb` 를 `render.py` 로 렌더한 결과를 GT `visual_*.png` 와 비교. 두 결과를 보고서에 적는다.

## S-2 트랙 B 후보 목록 (선정은 하지 않는다)

`metadata.json` 전체를 표로 만든다: 데이터셋, 객체, prompt 번호, 프롬프트 문장, `3d_edit_region.glb` 를 64³ 로 복셀화한 셀 수, 원본 점유 셀 수, 마스크 비율, 편집 유형(추가·교체·삭제 추정). 복셀화는 VoxHammer `voxhammer/delete_region_voxel.py` 와 같은 방식(포크 함수 호출)을 쓴다. 산출: `scenarios/track_b_candidates.csv`.

## S-3 트랙 A (자체 생성 6건) — 개정판 6장 확정 후

자체 생성 자산(TRELLIS 생성 SLat 이 있는 것)으로 6건. moto-b 를 포함할지, 자산·마스크·프롬프트 구성은 개정판 6장 기준으로 이 절을 확정한다.
각 건 산출: 자산 지문, 마스크 셀 목록과 지문(`deltacontract.mask_fingerprint`), 프롬프트, **마스크 초안 렌더**(원본 위에 마스크 셀을 색으로 겹친 고정 시점 이미지 2장 이상).

## S-4 트랙 B (Edit3D-Bench 8건) — 개정판 6장 확정 후

S-2 표에서 8건 선정안을 만든다. 선정 기준은 개정판 6장을 따른다. 선정안에는 건마다 마스크 초안 렌더를 붙인다.
**입력 SLat 경로 (결정 2026-10-20).** Bench 자산에는 TRELLIS 생성 SLat 이 없다.
- **최초 1회 (a):** 자산에 SLat 이 없을 때만 VoxHammer 원 경로(150시점 렌더 → DINOv2 특징 → SLat 인코더)로 인코딩하고, 결과를 R2 형식(safetensors, coords int32 (N,4), feats float32 (N,8), `slat_space=denormalized`)으로 저장한다. 이것이 그 자산의 **판본 0** 이다.
- **이후 모든 편집 (b):** 저장된 SLat 을 W34 `slat_path` 입력으로 쓴다. 편집 결과 `slat_tgt` 도 R2 로 저장해 다음 편집 입력으로 쓴다. 트랙 A(자체 생성 자산)는 생성 시 SLat 이 있으므로 처음부터 (b).
- (a) 인코딩 시간은 편집 1회 시간에 넣지 않고 따로 적는다(자산 등록 비용).
- **재구성 손실 분리.** (a) 인코딩은 원본 메시 대비 점유를 잃는다(기존 실측 t1s-snowman 21.2%). 이 손실이 편집 품질로 잡히지 않도록, Bench 렌더 지표는 두 기준으로 따로 낸다: ① GT 원본 렌더 대비(벤치 표준), ② 판본 0 SLat 을 편집 없이 디코딩한 렌더 대비(편집 자체의 영향). 보존 지표 R_keep 의 기준도 판본 0 이다(개정판 6장 정의와 다르면 정의를 따른다).
- 건마다 판본 0 의 점유 셀 수와 원본 메시 복셀 대비 손실·IoU 를 적는다.

## S-5 보존 지표 R_keep 구현 — 개정판 6장 정의 수령 후

`editcore/` 가 아니라 `eval/` 패키지에 둔다(편집 경로와 분리). 정의를 원문 그대로 docstring 에 인용하고, CPU 테스트(`tests/cpu/test_rkeep.py`)를 붙인다. 테스트에는 음성 대조를 넣는다: 무편집 출력(보존 만점·효능 0)과 전체 재생성 출력(보존 붕괴)이 구분돼야 한다.

## S-6 출력 좌표 정렬 (렌더 지표 전제)

Edit3D-Bench 렌더 지표는 예측이 원본과 같은 위치·방향·스케일이어야 한다. 현행 출력(`to_glb` 결과 GLB)과 2번 R4 이후 출력(`.cbin` → `glbbuild` GLB, 디코더 native 좌표, D9)이 Bench 원본 GLB 프레임과 어떻게 다른지 코드 근거로 적고, Bench 자산 1건에서 원본 → (편집 없이) 재구성 → 렌더의 실루엣 IoU 를 잰다. 변환이 필요하면 `eval/` 에 변환 함수로 두고 편집 경로에는 넣지 않는다.

## 제출 (고객사 동결·승인 대상)

- `scenarios/track_a.json`, `scenarios/track_b.json` (자산·마스크 지문·프롬프트·시드)
- `scenarios/drafts/` 마스크 초안 렌더(PNG, 작은 해상도. 자산 원본은 넣지 않는다)
- `scenarios/track_b_candidates.csv` 와 선정안
- `scenarios/track_b.json` 에 건마다 판본 0 SLat 의 sha256 과 (a) 인코딩 시간
- 보고서: S-1 정상 동작 결과, S-4 판본 0 재구성 손실, S-6 정렬 결과
