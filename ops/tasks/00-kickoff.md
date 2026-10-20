# 00 착수 — 포크 고정, 현행 코드 보관, 기준선 계측

- 요구사항: [REQUIREMENTS.md](../REQUIREMENTS.md) 4.4 "0. 착수", 6장 전체
- 결과 유형: 동일 (개선 없음. 코드 동작을 바꾸지 않는다)
- 담당: **a5000** (A-1 ~ A-9), **rtx3090** (B-1 ~ B-4)
- 브랜치: `item/00-kickoff-a5000`, `item/00-kickoff-3090`
- 보고서: `docs/reports/00-kickoff-a5000.md`, `docs/reports/00-kickoff-3090.md` (TEMPLATE 사용)
- 이 항목에서 하지 않는 것: `editcore/` 이식(0-1), 스텝 변경(1), `.cbin` 직행(2), 렌더 축소(3), 상주화(4), INT4(5). 발견한 개선 기회는 보고서 8절에만 적는다.

## 이미 확인된 사실 (chief, 코드확인)

- `third_party/VoxHammer` HEAD = f33cd96. 기준 패치는 충돌 없이 적용된다. 대상 파일은 `voxhammer/edit_pipeline.py` 하나(적용 후 637행).
- W39 는 `W39_ATTN_MASK=1` 일 때만 켜진다. 기준 실행에서는 이 변수를 설정하지 않는다.
- 패치 적용본 행 위치: K/V 저장 198–199(Stage 1)·315–316(Stage 2), `steps=25` 409, `rescale_t` 410, rembg 세션 생성 64–65, `decode_slat` 624, `set_grad_enabled(True)` 626, `to_glb` 627.
- `argmax(0)` 매칭은 4곳: 273–274, 282–283(attention), 444–445(샘플러 Stage 2 잠재 복원), 526–527(최종 feats 덮어쓰기).
- `utils/render_rgb_and_mask.py`: Cycles, 1024², 고도 15°, 방위 [0, −45, −90, −135, −180](= `[90,45,0,315,270]-90`), 반경 1.8, 환경맵 `brown_photostudio_02_1k.exr`, 원본·마스크 GLB 각 5시점. 2D 마스크 = 원본 depth − 마스크 depth > 1e-4. **시드 설정 코드 없음.**
- 5시점 중 어느 것을 `2d_render.png` 로 쓰는지는 포크가 정하지 않는다. 현행 래퍼가 정한다 → A-2 에서 확인.
- `utils/inpaint.py`: bf16 `FLUX.1-Fill-dev`, 1024², 50스텝, guidance 30, `max_sequence_length` 512, CPU generator seed.
- ai-ar-v3 테스트: conformance 53/53. 서버 테스트는 `fastapi python-multipart httpx` 를 추가 설치해야 360 passed · 12 skipped.

---

## A. a5000 세션

### A-1 환경 고정
1. `git clone --recurse-submodules` 후 `python3 tools/check_pins.py` → OK.
2. `bash tools/setup_reference.sh` → `build/reference/` 생성.
3. 현행 편집에 쓰는 파이썬 환경을 찾아 기록: python, torch, CUDA, xformers/flash-attn, spconv, kaolin, nvdiffrast, rembg, diffusers 버전. `pip freeze` 를 `env/a5000-reference.txt` 로 커밋(경로·호스트 제거).
4. 그 환경에서 `build/reference` 를 import 할 수 있게 한다(PYTHONPATH 등). VoxHammer 동봉 `trellis/` 를 그대로 쓰고, 체크포인트는 `microsoft/TRELLIS-image-large`(현행과 같은 것)인지 확인.

### A-2 현행 코드 보관 (요구사항서 2장 "코드 보관")
현행 A5000 편집 서비스, 편집 래퍼, Blender 조건 렌더 호출, 인페인팅 호출 코드를 찾아 **수정 없이** `legacy/a5000/` 에 복사한다.
- 비밀 정보(호스트·포트·키·홈 경로)만 `<EDIT_HOST>` 등 자리표시자로 바꾸고, 바꾼 위치를 `legacy/a5000/REDACTIONS.md` 에 적는다.
- `legacy/a5000/README.md` 에 다음을 코드 근거(파일:행)와 함께 적는다:
  - (a) 모델을 프로세스에 상주시키는가, 요청마다 로드하는가 → R8a 효과 0 여부
  - (b) 5시점 중 `2d_render.png` 로 쓰는 시점 번호와 그 방위각
  - (c) `run_edit` 호출 인자: `cfg`, `skip_step`, `re_init`, `slat_path`, `mask_cells` 사용 여부
  - (d) `slat_tgt` 를 safetensors 로 저장하는가(R2). 저장한다면 키·dtype·메타
  - (e) 인페인팅이 어디서 어떻게 도는가(A5000 bf16 CPU offload 여부, subprocess 여부, 편집 서비스 언로드·재기동 여부)
  - (f) 변경 청크를 점유 비교로 정하는가, 해시로 정하는가(R11)
  - (g) W39 코드 경로가 현행 실행에서 켜져 있는가

### A-3 R2 SLat 포획 (기본 동작)
현행 래퍼가 `slat_tgt` 를 저장하지 않으면, `reference/` 래퍼에서 저장한다: safetensors, `coords` int32 (N,4), `feats` float32 (N,8), 메타 `slat_space=denormalized`. 다음 편집의 `slat_path` 입력으로 쓴다. 포크 함수는 고치지 않는다(반환값을 래퍼에서 받는다).

### A-4 기준선 계측 래퍼 `reference/`
`reference/run_reference.py` 를 만든다. 포크 코드는 고치지 않는다.
- `build/reference/voxhammer/edit_pipeline` 모듈의 모듈 전역 함수(`sample_sparse_structure_inverse`, `sample_sparse_structure_denoise`, `sample_slat_inverse`, `sample_slat_denoise`, `preprocess_image`, `ply_to_ss_mask`, `ply_to_slat_mask` 등)를 **모듈 속성 교체로 감싸** 구간 시간을 잰다. `run_edit` 는 호출 시점에 모듈 전역 이름을 찾으므로 감싼 함수가 쓰인다. 감쌌을 때 실제로 감싼 함수가 호출됐는지 호출 횟수를 로그로 남긴다.
- 구간 경계마다 `torch.cuda.synchronize()`, VRAM 은 구간마다 `reset_peak_memory_stats()` → `max_memory_allocated()`, `nvidia-smi` 값 별도, 프로세스 RSS 피크, 캐시 딕셔너리(`ss_kv`, `slat_kv`, `latent`) 텐서 바이트 합.
- 구간 4·6 의 내부 분해(forward 연산 / 캐시 저장·적재 / 전송 / 좌표 매칭)는 **1회 실행에 한해** `torch.profiler` 로 잰다: DtoH·HtoD memcpy 합계, `aten::eq`·`aten::all`·`aten::argmax`(매칭) 합계, 나머지. 프로파일 실행은 평균 계산에서 제외한다.
- 구간 7 은 Gaussian 디코딩, mesh 디코딩, `to_glb`, GLB export, 재청킹을 따로 잰다(`decode_slat` 는 형식별로 분리 호출할 수 없으면 `pipeline.models` 의 디코더 호출을 감싼다).
- 구간 8: 모델 로드 시간(최초 1회), 큐 대기(현행 서비스에 큐가 있으면).
- 결과는 `runs/<날짜>/<자산>/<마스크지문>/<회차>.json` (git 제외). 스키마는 `reference/SCHEMA.md` 에 적는다. 이후 모든 항목이 같은 스키마를 쓴다.

### A-5 조건 렌더·인페인팅 계측 (구간 1·2, 현행 경로 그대로)
현행 경로가 A5000 에서 Blender 와 FLUX 를 돌리면 그대로 잰다.
- 구간 1: Blender 프로세스 기동, 원본 렌더(5시점), 마스크 렌더(5시점), 2D 마스크 후처리를 분리. 시점 수·Cycles 샘플 수 기록. 시드는 바꾸지 않는다(기준 그대로).
- 구간 2: FLUX 로드, 인페인팅(해상도·스텝 기록), 편집 서비스 언로드·재기동이 있으면 그 시간.
- Blender·FLUX 실행은 포크 스크립트를 호출하는 래퍼로 잰다. 스크립트는 고치지 않는다.

### A-5b 조건 렌더 실행 간 차이 (R10a 판정 기준 준비, 고객사 지시 2026-10-19)
R10a 기준 렌더러는 포크 `utils/render_rgb_and_mask.py` 를 **수정 없이** 쓴다.
1. 같은 입력(원본 GLB, 마스크 GLB)으로 포크 스크립트를 3회 실행한다.
2. 사용 시점(A-2 (b))의 RGB 이미지와 2D 마스크에 대해 실행 쌍(1–2, 1–3, 2–3)마다 픽셀 차이 최대·평균을 적는다. RGB 는 채널별 0–255 정수 차, 마스크는 다른 픽셀 수와 비율. 나머지 4시점도 같은 표로 적는다.
3. 판정 규칙(3번 R10a 에서 사용):
   - 차이가 0 → R10a 는 비트 비교.
   - 0 이 아님 → 이 범위를 R10a 허용치로 쓴다.
   - 허용치가 판정에 쓰기 어려울 만큼 크면(예: 마스크 경계 픽셀이 실행마다 크게 바뀜) 수치와 함께 chief 에 보고한다. 그때만 포크 밖에 **시드 설정 줄만** 추가한 사본을 기준 렌더러로 허용하며, 포크 대비 diff 를 함께 제출한다. 이 단계에서 사본을 미리 만들지 않는다.
4. 이 시드는 VoxHammer 결과와 무관한 자체 비교용이다. 편집 결과 동등성 판정에는 쓰지 않는다.

### A-6 실행 매트릭스 (고객사 답변 2026-10-20 반영)
- 자산: **moto-b 로 먼저 진행**한다. t1s-snowman 은 제공되지 않는다. 시나리오 자산(00s 트랙 A·B)이 동결되면 그 자산으로 기준선을 다시 잰다(1번 E2a 전).
- moto-b 마스크는 ai-ar-v3 `handoff/lasso/` 에서 크기가 다른 2개 이상 (셀 수·지문은 파일 `meta` 기준, 코드확인):

  | 파일 | 셀 | 자산 대비 | 지문 앞 12자 |
  |---|---|---|---|
  | `moto-b.rider-head.mask.json` | 476 | – | 9e237b53adf1 |
  | `moto-b.rear-wheel.mask.json` | 1,386 | 15.1% | 28b22e0004e6 |
  | `moto-b.rear-wheel-full.mask.json` | 2,310 | – | 88270814e860 |

  최소 rider-head 와 rear-wheel 두 건. 실행 전에 `deltacontract.mask_fingerprint` 로 지문을 다시 계산해 일치를 확인한다.
- **조건 이미지는 현행 경로대로 렌더 → 인페인팅으로 만든다.** `handoff/conditions/*.png` 는 t2i 전신 생성 이미지라 편집 조건으로 쓰지 않는다(요구사항서 5장 제외 항목). 인페인팅 프롬프트와 시드는 보고서에 적는다.
- 각 편집 3회. 시간은 평균±표준편차.
- 같은 편집을 2회 실행해 Stage 1 coords 일치 여부와 Stage 2 feats max|Δ| 를 기록(CUDA 비결정성 바닥값 재확인. 기존값 4.7e-2).

### A-7 고객사 검증 스크립트
스크립트 3개는 `tests/customer/` 에 있다(원본, 수정 금지, `SHA256SUMS`). chief 가 CPU 에서 부록 수치를 재현했다(`tests/customer/README.md`).
1. **원본 그대로 실행.** A5000 에서 `sha256sum -c` 확인 후 원본을 그대로 실행한다(인자: `build/reference`, `third_party/ai-ar-v3`). 결과를 chief 사전 실행 결과와 나란히 적는다.
2. **GPU fp16 사본.** `tests/customer/gpu/` 에 사본을 만들어 다음만 바꾼다: `torch.Tensor.cuda` 항등 치환 제거, 모델·입력을 `cuda` 로, 순전파를 fp16(현행 TRELLIS 실행과 같은 dtype 설정)으로. 결과 비교 항목은 원본과 같게 유지한다. 원본 대비 diff 를 보고서에 붙인다.
3. **실제 규모 1회.** 같은 GPU 사본에서 v_stage1_cache 의 모델 크기만 바꾼다: 해상도 16(=4,096 토큰), 채널 1,024, 블록 24, 헤드 16, 조건 토큰 1,374, cond 채널 1,024. 무작위 가중치 대신 **실제 체크포인트**(`sparse_structure_flow_model`) 가중치로도 1회 돌린다. 보존 토큰 비율은 moto(546/4,096)를 쓴다. 캐시 용량·비트 동일 여부·max|Δ|·실행 시간·VRAM 피크를 적는다. CPU 메모리가 부족하면 그 수치와 함께 중단 사유를 적는다.
4. v_stage2_misc 의 CFG 배치 부분과 v_cbin_direct 는 GPU 사본에서 같은 방식으로 1회씩. v_cbin_direct 는 CPU 경로이므로 A5000 CPU 실행 시간만 추가로 적는다.

### A-8 테스트
- `contract` 와 ai-ar-v3 서버 테스트를 A5000 에서도 1회 실행: conformance 53/53, 서버 360 passed.

### A-9 효과 점수 재산정
기준선 구간 시간으로 순서표 1–17번의 "효과"가 줄이는 구간과 그 구간의 편집 1회 대비 비율을 표로 적는다(점수는 제안값, 확정은 고객사). 2·3순위 진입 조건(10%) 판단에 쓴다.

---

## B. rtx3090 세션

### B-1 환경 고정
- `git clone --recurse-submodules` → `check_pins.py` OK.
- 현재 3090 에 상주하는 프로세스와 VRAM 을 기록(Z-Image-Turbo, BiRefNet, ai-ar-v3 오케스트레이터 등). `nvidia-smi` 와 각 프로세스의 환경(conda env 이름, torch·CUDA 버전).
- `pip freeze` 를 `env/rtx3090-<env이름>.txt` 로 커밋(경로·호스트 제거).

### B-2 현행 코드 보관
- 3090 에서 편집 경로에 관여하는 코드(오케스트레이터가 A5000 으로 편집을 넘기는 부분, 조건 이미지 전달 부분)가 ai-ar-v3 main 과 다르면 차이를 `legacy/rtx3090/` 에 보관하고 `README.md` 에 기록. 같으면 "ai-ar-v3 19ed91f 와 동일" 이라고만 적는다.
- 현재 3090 에 인페인팅 코드가 있는지 확인(없으면 없다고 적는다).

### B-3 VRAM 여유 측정 (R9 준비, 설치하지 않음)
- Z-Image-Turbo 상주 상태의 VRAM 사용량, 유휴 VRAM, 피크(생성 1회 중).
- Nunchaku INT4 FLUX Fill 은 **설치하지 않는다**(5번 항목). 공개 문서 기준 요구 torch·CUDA 버전과 예상 VRAM 을 `추론` 으로 적는다.

### B-4 장비 간 전송 측정 (구간 2 의 전송 부분)
- 1024² PNG 3장(약 `2d_render`·`2d_mask`·`2d_edit` 크기) 을 기존 multipart 규약 형태로 A5000 ↔ 3090 왕복 전송한 시간(10회 평균). 호스트·포트는 `.env` 에서만 읽는다.

---

## 공통: self-hosted 러너

각 워크스테이션에 GitHub Actions self-hosted 러너를 등록한다(라벨 `a5000`, `rtx3090`). 등록 토큰은 저장소 관리자가 발급한다. 러너는 사용자 서비스로 띄우고, 사용하는 파이썬 경로는 `.env` 의 `A5000_PYTHON` / `RTX3090_PYTHON` 으로 준다. 등록이 막히면 보고서에 적고 나머지를 진행한다.

## 완료 조건

- [ ] `check_pins.py` OK, `build/reference` 실행 성공(W39 꺼짐 로그 확인)
- [ ] `legacy/a5000/README.md` 의 (a)–(g) 전부 코드 근거와 함께 기재
- [ ] 구간 1–8 기준선: 자산 2종 × 편집 2건 이상 × 3회, 평균±표준편차, 메모리 4종
- [ ] 동일 코드 2회 실행 차이(Stage 1 coords, Stage 2 feats max|Δ|)
- [ ] 4·6 구간 내부 분해(프로파일 1회)
- [ ] 기존 실측 116–129 s(구간 4–7) 와 같은 자릿수인지 판정
- [ ] conformance 53/53, 서버 360 passed (A5000 에서)
- [ ] 조건 렌더 3회 실행 간 픽셀 차이 표(A-5b)
- [ ] 고객사 스크립트: 원본 실행, GPU fp16 사본(diff 첨부), 실제 규모 1회(A-7)
- [ ] 효과 재산정 표
- [ ] 3090: 상주 VRAM, 전송 시간, 환경 기록
- [ ] 보고서 2개, PR 2개 (본문에 보고서 경로)
