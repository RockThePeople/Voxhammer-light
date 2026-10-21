# 00a 환경 점검 — rtx3090

- 점검 일시: 2026-10-21 17:55–18:15 (KST)
- 커밋: `f18fdcc` (main) 위 브랜치 `item/00a-env-3090`
- check_pins: **OK** — 출력 그대로 `PIN CHECK OK  voxhammer=f33cd96  patch=b886a2839e3b` (실측, 클론 직후와 reference 생성 후 2회)
- setup_reference: **스크립트 자체는 실행하지 못함, 같은 단계를 수동으로 수행해 성공.**
  - `bash tools/setup_reference.sh` 는 이 세션의 실행 권한 정책이 스크립트 안의 `rm -rf "$OUT"` 를 막아 실행되지 않았다.
  - `build/reference` 가 없는 상태를 확인한 뒤, 스크립트의 나머지 단계(`git archive HEAD | tar -x`, `git init`, `git apply patches/edit_pipeline.f33cd96.patch`)를 순서 그대로 실행 → exit 0, 패치 충돌 없음. 이후 check_pins 다시 OK. (실측)
  - 스크립트 경로 그대로의 성공 여부는 `미검증`. 권한이 열린 세션에서 한 번 더 돌려야 한다.
- 이 점검 동안 설치·서비스 변경·GPU 점유 실행 없음. 파이썬 import 는 버전 조회용으로만 하고 CUDA 는 초기화하지 않음.

## 항목별 결과

| # | 결과 | 근거 | 비고 |
|---|---|---|---|
| B1 | RTX 3090 24576 MiB, 드라이버 610.43.03, CUDA UMD 13.3. 시스템 CUDA 툴킷(`nvcc`, `/usr/local/cuda*`) **없음** — 각 env 의 torch 번들 CUDA 만 있다. GPU 사용 프로세스는 Xorg 9 MiB·gnome-shell 8 MiB 뿐(총 38 MiB 사용), 연산 프로세스 0. **Z-Image·BiRefNet·FLUX 모두 상주하지 않는다.** | 실측 `nvidia-smi` | 상주가 아니라 호출마다 subprocess 로 뜨고 끝난다(B5·B6) |
| B2 | RAM 125 GiB, 사용 6 / 가용 117 GiB(대부분 페이지 캐시). 스왑 2 GiB 파일(거의 미사용). 디스크 `/` 단일 파티션 938 GB 중 615 GB 여유 — 작업 경로·HF 캐시 같은 파티션. HF 캐시 84 GB. CPU Xeon W-2245 16 스레드. | 실측 `free -g`, `df -h`, `du`, `lscpu` | |
| B3 | conda env 5개(`zimage`, `birefnet`, `flux-fill`, `trellis`, `bitnet-cpp`). 아래 표 참조. freeze → `env/rtx3090-{zimage,birefnet,flux-fill,trellis}.txt` (홈 경로 `<HOME>` 처리). `bitnet-cpp` 는 torch 없음, 이 프로젝트와 무관해서 제외. 이 밖에 `ai-ar-v3/.venv` (Python 3.14.0, 오케스트레이터용). | 실측 `pip freeze`, import 버전 조회 | |
| B4 | **ai-ar-v3 오케스트레이터(`server/skeleton.py`, 기본 `:<PORT>`)는 지금 실행 중이 아니다.** `ps`·`ss -ltnp`·`systemctl --user` 에 없음. 실행 방식은 `server/run-<PORT>.sh` → `setsid --fork` 로 `.venv` 파이썬 기동, 환경은 `<HOME>/.ai-ar-v3-server.env` (키 7개: `ASSET_STORE_ROOT`, `T2I_TOOLS_DIR`, `ZIMAGE_PYTHON`, `BIREFNET_PYTHON`, `ANTHROPIC_API_KEY`, `BLOCKEDIT_B_URL`, `BLOCKEDIT_B_KEY`; 값은 적지 않음). systemd 유닛 없음, 상시 실행 아님. 로컬 체크아웃: 브랜치 `w30-ingest` @ `681f3b9`, `docs/PROGRESS.md` 미커밋 수정 1건. | 실측 `ps`, `ss`, `git log`; 코드확인 `server/run-<PORT>.sh:12-50` | 그 밖의 상시 서비스는 이 프로젝트와 무관(nginx, mysql, redis, glances, xrdp, kind 클러스터 컨테이너 3개) |
| B4-차이 | 로컬 `681f3b9` 는 19ed91f 의 **조상이며 9커밋 뒤**. `server/`·`contract/` 는 19ed91f 와 **차이 없음**. 차이는 `tools/build_lasso_apk.sh` +3줄, 그리고 docs·handoff·웨이브 폴더(63파일, +2463/−823) 뿐. | 실측 `git diff --stat 681f3b9 19ed91f` | 편집 경로 코드는 "ai-ar-v3 19ed91f 와 동일" 로 볼 수 있다 |
| B5 | **인페인팅 코드가 이 장비에 있다.** `<HOME>/ai-ar-v3-assets/_tools/flux_fill.py` (git 저장소 아님). 모델: Nunchaku SVDQuant **INT4** FLUX.1-Fill-dev transformer(`nunchaku-ai/nunchaku-flux.1-fill-dev` `svdq-int4_r32`) + 나머지 부품 `camenduru/FLUX.1-Fill-dev-ungated`(비게이트 미러). 실행: `flux-fill` env 파이썬 subprocess, CLI 인자, 기본 30스텝·guidance 30·seed 42, `--offload` 선택, 기본으로 마스크 밖을 원본 바이트로 되돌리는 합성. 벤치·A/B 스크립트(`flux_fill_bench.py`, `flux_fill_ab_bench.py`, `flux_fill_probe.py`)와 README 있음. **ai-ar-v3 서버에는 연결돼 있지 않다**(서버 코드에서 `FluxFill`/`nunchaku` 참조 없음). | 실측 `grep`; 코드확인 `flux_fill.py:17-67` | README 의 기존 측정(이번 점검에서 재실행하지 않음, `미검증`): INT4 1024²·30스텝 총 15.6 s, peak 18.1 GiB; bf16+CPU offload 59.5 s, peak 22.8 GiB |
| B6 | **실행 가능한 상태로 보인다(실행은 안 함).** Z-Image: `_tools/zimage_gen.py` — `ZImagePipeline`, bf16, 1024×1024, `num_inference_steps=9`, `guidance_scale=0.0`, seed 인자(기본 42). 지시서 설정과 일치. BiRefNet: `_tools/birefnet_seg.py` — `ZhengPeng7/BiRefNet`, fp16, 1024² 리사이즈·정규화, `preds[-1].sigmoid()` 를 알파로. 호출부 `ai-ar-v3/server/t2i.py:89-98` 이 두 워커를 각 env 파이썬으로 **순차** subprocess 실행(numpy/torch 충돌 때문, `t2i.py:4-14`). 확인한 것: `zimage` env 에서 `from diffusers import ZImagePipeline` 성공, `birefnet` env 필요 모듈 import 성공, HF 캐시에 두 모델 스냅샷 존재·`.incomplete` 없음, 서버 env 파일에 `T2I_TOOLS_DIR`/`ZIMAGE_PYTHON`/`BIREFNET_PYTHON` 키 존재. | 코드확인 `zimage_gen.py:15-22`, `birefnet_seg.py:17-31`; 실측(import·캐시) | 실제 생성 성공 여부와 VRAM 은 `미검증` (GPU 점유 금지) |
| B7 | **Nunchaku INT4 FLUX Fill 은 이미 설치돼 있다**: `flux-fill` env = Python 3.11.15, torch 2.9.1+cu128, nunchaku 1.2.1+cu12.8torch2.9 (GitHub 릴리스 휠), diffusers 0.39.0, transformers 4.57.6. 3090 은 sm_86 → int4 선택. 드라이버 610(CUDA 13.3 UMD)은 cu128 휠을 돌리기에 충분. 다른 env(`zimage`·`birefnet` torch 2.6+cu124, `trellis` torch 2.4+cu121)는 nunchaku 1.2.1 휠 조합(torch 2.9 / cu12.8 / cp311)과 맞지 않는다. | 실측(설치 버전); 설치 스크립트 코드확인 `setup_fluxfill.sh:5-15`; 휠 요구 조건은 휠 이름 기준 `추론` | 00-kickoff B 의 "설치하지 않는다" 와 충돌하지는 않으나 전제가 다르다(아래) |
| B8 | git push: GitHub CLI 로그인 계정이 이 저장소에 admin/push 권한(`gh api` 실측), 이 보고서 브랜치 push 로 확인. huggingface.co: HEAD 200, 모델 파일 resolve HEAD 307 → 접근 가능. **A5000: 호스트는 ICMP 응답, 그러나 서버 env 의 A5000 편집 URL 포트는 `Connection refused`**, SSH 포트도 닫힘 → 지금 3090→A5000 HTTP 통신 불가. | 실측 `curl -I`, `ping`, `/dev/tcp` | A5000 서비스가 내려가 있거나 포트가 바뀐 것으로 보인다(원인 `미검증`). A5000 쪽 A12 와 대조 필요 |
| B9 | 러너 등록 **가능할 것으로 보임, 단 조건부.** 저장소 admin 권한 있음(토큰 발급 가능). 사용자 쓰기 홈 디스크 충분, docker 그룹 소속. `sudo` 는 비밀번호 필요(비대화형 불가) → 시스템 서비스 등록은 사용자가 직접 해야 한다. `loginctl` Linger=no → 사용자 서비스로 상시 실행하려면 `loginctl enable-linger` (보통 sudo 필요). 기존 러너 디렉터리 없음. 등록은 하지 않음. | 실측 | |

### B3 상세 — env 별 버전 (실측)

| env | python | torch / CUDA 빌드 | 주요 패키지 |
|---|---|---|---|
| `zimage` | 3.11.15 | 2.6.0+cu124 / 12.4 | diffusers 0.39.0, transformers 5.14.1, accelerate 1.14.0, safetensors 0.8.0, torchvision 없음 |
| `birefnet` | 3.11.15 | 2.6.0+cu124 / 12.4 | torchvision 0.21.0, transformers 5.14.1, timm 1.0.28, diffusers 없음 |
| `flux-fill` | 3.11.15 | 2.9.1+cu128 / 12.8 | nunchaku 1.2.1+cu12.8torch2.9, diffusers 0.39.0, transformers 4.57.6, huggingface_hub 0.36.2 |
| `trellis` | 3.10.20 | 2.4.0+cu121 / 12.1 | flash_attn 2.8.3.post1, spconv 2.3.6, kaolin 0.18.0, nvdiffrast 0.4.0, rembg 2.0.69, transformers 4.46.3, fastapi 0.139.2. xformers 없음 |

### HF 캐시 (실측)

| 모델 | 크기 | 비고 |
|---|---|---|
| Tongyi-MAI/Z-Image-Turbo | 31 GB | 완전 |
| ZhengPeng7/BiRefNet | 425 MB | 완전 |
| nunchaku-ai/nunchaku-flux.1-fill-dev | 6.4 GB | int4 r32 1파일 |
| camenduru/FLUX.1-Fill-dev-ungated | 38 GB | bf16 transformer 3샤드 포함. `.incomplete` 1개(6.3 GB, 2026-08-18) 잔존 — 스냅샷 링크는 모두 있어 쓰다 남은 조각으로 보이나 `미검증` |
| JeffreyXiang/TRELLIS-image-large | 2.9 GB | |
| microsoft/TRELLIS-text-xlarge | 3.9 GB | |
| openai/clip-vit-large-patch14 | 1.6 GB | |

## 0번·00s 지시서와 맞지 않는 점

1. **Nunchaku INT4 FLUX Fill 이 이미 3090 에 설치·측정돼 있다.** 00-kickoff B 는 "설치하지 않는다(5번 항목), 요구 버전·VRAM 을 추론으로 적는다" 를 전제로 한다. 실제로는 env·가중치·벤치 스크립트가 있고 이전 실측(README, 이번 재실행 안 함)도 있다. 5번 항목 순서·0번 범위에서 이 기존 env 를 어떻게 다룰지(사용 금지 / 측정만 허용) chief 결정이 필요하다.
2. **bf16 FLUX.1-Fill-dev(비게이트 미러) 도 3090 캐시에 있다.** 00s S-5 등은 "기존 bf16 FLUX(CPU offload, 현행 경로)" 를 A5000 기준으로 적고 있다. 3090 에도 같은 구성의 가중치가 있으나 **원본 `black-forest-labs` 가 아니라 미러**(README 상 vae·text_encoder sha256 은 미러 3곳 일치 확인). 기준 경로가 어느 장비·어느 가중치 출처인지 확정이 필요하다.
3. **"Z-Image-Turbo 상주 상태 VRAM"(00-kickoff B) 은 현 구조에 맞지 않는다.** Z-Image·BiRefNet 은 상주 서비스가 아니라 호출마다 subprocess 로 로드→생성→종료한다. 측정 대상은 "상주 VRAM" 이 아니라 "1회 호출의 적재 시간·피크 VRAM" 이 된다.
4. **ai-ar-v3 오케스트레이터가 지금 떠 있지 않다.** 00-kickoff B 의 "상주 프로세스 기록" 시점에 기록할 프로세스가 없다. 0번에서 오케스트레이터 경유 측정이 필요하면 재기동이 필요하다(00a 범위 밖이라 하지 않음).
5. **3090 → A5000 편집 서비스 HTTP 가 연결 거부.** B-3 전송 시간 측정(1024² PNG 3장 왕복) 은 지금 상태로는 불가.
6. 로컬 ai-ar-v3 체크아웃은 19ed91f 보다 9커밋 뒤(`w30-ingest`)지만 `server/`·`contract/` 는 동일. 편집 경로 차이 보관(`legacy/rtx3090/`)은 불필요할 것으로 보인다.
7. 3090 에도 `trellis` env 와 TRELLIS 가중치가 있다(지시서는 TRELLIS 를 A5000 전담으로 둔다). 참고로만 기록.

## 0번·00s 를 시작하려면 필요한 것

설치는 하지 않았고 목록만 적는다.

- **권한:** `tools/setup_reference.sh` 를 그대로 실행할 수 있는 세션 권한(스크립트의 `rm -rf build/reference` 허용). 지금 `build/reference` 는 수동 생성본이다.
- **A5000 측:** 편집 서비스 기동 또는 현재 주소·포트 확인 → 3090 `.env` 갱신. 전송 측정(B-3)의 선행 조건.
- **오케스트레이터:** 0번에서 필요하면 `run-<PORT>.sh` 재기동 허가(서비스 재기동이라 00a 에서 안 함).
- **러너(B-4):** 등록 토큰(저장소 admin 이 발급 가능), 상시 실행을 위한 `loginctl enable-linger` 또는 sudo 로 시스템 서비스 등록 — sudo 가 비밀번호를 요구하므로 사용자 직접 조치 필요.
- **결정 요청(chief):** 위 1·2·3 — 기존 Nunchaku env 처리, bf16 FLUX 가중치 출처, Z-Image 측정 정의.
- **패키지:** 00s S-1(Z-Image 생성·BiRefNet 배경 제거)은 기존 `zimage`·`birefnet` env 로 추가 설치 없이 가능할 것으로 보인다(코드확인·import 실측, 실행은 `미검증`). 0번 B 의 전송 측정용 HTTP 클라이언트는 아직 범위가 정해지지 않아 목록 없음.
- **디스크:** 615 GB 여유로 추가 확보 불필요.
