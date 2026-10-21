# 00a 환경 점검 (읽기 전용) — 0번 착수 전 단계

- 목적: 0번 기준선·00s 시나리오 지시를 실제 환경에 맞추기 위해 두 워크스테이션의 현재 상태를 기록한다.
- 결과 유형: 기록만. **코드·환경·서비스를 바꾸지 않는다.**
- 담당: **a5000** (A절), **rtx3090** (B절)
- 브랜치: `item/00a-env-a5000`, `item/00a-env-3090`
- 산출물: `docs/reports/00a-env-<장비>.md`, `env/<장비>-<env이름>.txt` (pip freeze, 경로·호스트 제거)
- 다음 단계: chief 가 두 보고를 검토해 `ops/tasks/00-kickoff.md`·`00s-scenarios.md` 를 고친 뒤, 각 세션이 pull 해서 0번·00s 를 시작한다. **00a 승인 전에는 0번·00s 를 시작하지 않는다.**

## 금지 (00a 동안)

- 패키지 설치·업데이트, conda/venv 생성·수정 (단, 아래 1의 저장소 클론과 `build/reference` 생성은 허용)
- 실행 중인 서비스 중지·재기동, 포트 점유 변경
- GPU 를 점유하는 실행(모델 로드, 추론, 렌더). `nvidia-smi` 조회는 허용
- 모델·데이터셋 다운로드
- 호스트·IP·포트·키·토큰·홈 경로를 보고서·커밋에 쓰기 → `<EDIT_HOST>`, `<HOME>`, `<PORT>` 같은 자리표시자로 바꾼다

## 공통 절차

1. `git clone --recurse-submodules https://github.com/RockThePeople/Voxhammer-light`
2. `python3 tools/check_pins.py` → OK 여부와 출력 그대로
3. `bash tools/setup_reference.sh` → 성공 여부 (CPU 작업, 설치 없음)
4. 아래 장비별 항목을 조사해 보고서에 적는다. **모르는 항목은 "확인 불가"와 이유를 적는다. 추측으로 채우지 않는다.**
5. 각 항목에 근거를 붙인다: `실측`(명령 실행 결과) / `코드확인`(파일:행) / `미검증`
6. 브랜치에 보고서와 `env/*.txt` 를 커밋하고 PR 을 연다. PR 본문에 보고서 경로를 적는다.

## A. a5000

| # | 항목 | 확인 방법 예시 |
|---|---|---|
| A1 | GPU·드라이버·CUDA, 지금 GPU 를 쓰는 프로세스와 VRAM | `nvidia-smi` |
| A2 | RAM 총량·여유, 스왑, 디스크 여유(작업 경로·HF 캐시 경로) | `free -g`, `df -h` |
| A3 | 현행 편집에 쓰는 파이썬 환경: 이름, python, torch, CUDA 빌드, xformers/flash-attn, spconv, kaolin, nvdiffrast, rembg, diffusers, transformers, safetensors 버전 → `env/a5000-<env>.txt` | `pip freeze`, `python -c "import torch; ..."` |
| A4 | 현행 코드 위치와 실행 방식: 편집 서비스(HTTP), 편집 래퍼, `run_edit` 호출부, Blender 조건 렌더 호출, FLUX 인페인팅 호출. 각각 파일 경로(자리표시자로), 실행 명령, 상시 실행 여부 | `ps`, `systemctl --user`, 실행 스크립트 |
| A5 | 모델 상주 여부: 편집 서비스가 TRELLIS 를 프로세스 시작 시 1회 로드하는가, 요청마다 로드하는가 (0번 R8a 판단 근거) | 코드확인(파일:행) |
| A6 | 인페인팅이 실제로 어디서 도는가: A5000 bf16 FLUX CPU offload 여부, subprocess 여부, 이때 편집 서비스를 내리고 다시 올리는가 | 코드확인 |
| A7 | `run_edit` 호출 인자(cfg, skip_step, re_init, slat_path, mask_cells), `slat_tgt` safetensors 저장 여부, W39 가 켜지는지 | 코드확인 |
| A8 | 조건 렌더에서 5개 시점 중 `2d_render.png` 로 쓰는 시점 번호 | 코드확인 |
| A9 | 현행 코드가 쓰는 VoxHammer 가 f33cd96 + 기준 패치와 같은가 (`git diff` 또는 파일 해시 비교) | 실측 |
| A10 | 모델·도구 위치와 버전: TRELLIS-image-large 체크포인트, FLUX.1-Fill-dev, DINOv2, rembg u2net, Blender 버전·`bpyrenderer` 설치 여부 | 실측 |
| A11 | 자산: moto-b (`assets/moto-b`, `handoff/lasso`) 를 이 장비에서 쓸 수 있는가 | 실측 |
| A12 | 네트워크·권한: `git push` 가능(브랜치 push 로 확인), huggingface.co 접근 가능 여부(HEAD 요청만, 다운로드 금지), 3090 과 통신 가능 여부(포트는 자리표시자) | 실측 |
| A13 | self-hosted 러너 등록 가능 여부(설치 권한, 상시 실행 계정). 등록은 하지 않는다 | 실측 |
| A14 | 이 장비에서 ai-ar-v3 conformance·서버 테스트를 돌릴 수 있는 파이썬이 있는가(설치 없이). 있으면 실행 결과 | 실측 |

## B. rtx3090

| # | 항목 | 확인 방법 예시 |
|---|---|---|
| B1 | GPU·드라이버·CUDA, 지금 GPU 를 쓰는 프로세스와 VRAM(Z-Image, BiRefNet, 기타) | `nvidia-smi` |
| B2 | RAM·디스크 여유 | `free -g`, `df -h` |
| B3 | 파이썬 환경: zimage, birefnet 등 env 별 python·torch·CUDA·diffusers 버전 → `env/rtx3090-<env>.txt` | `pip freeze` |
| B4 | 상시 실행 서비스: ai-ar-v3 오케스트레이터 등, 실행 방식, 실행 중인 커밋. ai-ar-v3 main(19ed91f) 과 다르면 차이 요약 | `ps`, `git -C <경로> log -1` |
| B5 | 인페인팅 코드가 이 장비에 있는가, 있다면 위치·모델·실행 방식 | 실측·코드확인 |
| B6 | Z-Image-Turbo 생성 설정(1024², 9스텝, guidance 0) 과 BiRefNet 배경 제거가 지금 바로 실행 가능한 상태인가(실행은 하지 않는다) | 코드확인 |
| B7 | Nunchaku INT4 FLUX Fill 이 요구하는 torch·CUDA 와 현재 환경 비교(설치하지 않음, 공개 문서 기준, 근거 `추론`) | 문서 |
| B8 | 네트워크·권한: `git push` 가능, huggingface.co 접근 가능 여부(HEAD 만), A5000 과 통신 가능 여부 | 실측 |
| B9 | self-hosted 러너 등록 가능 여부. 등록은 하지 않는다 | 실측 |

## 보고서 양식

`docs/reports/00a-env-<장비>.md`

```
# 00a 환경 점검 — <장비>
- 점검 일시, 커밋, check_pins 결과, setup_reference 결과
## 항목별 결과
| # | 결과 | 근거(실측/코드확인/미검증) | 비고 |
## 0번·00s 지시서와 맞지 않는 점
(예: 현행 인페인팅이 이미 3090 에서 돈다, Blender 가 없다, moto-b 가 없다 …)
## 0번·00s 를 시작하려면 필요한 것
(설치가 필요한 패키지와 이유, 디스크 확보, 권한 등. 설치는 하지 않고 목록만)
```
