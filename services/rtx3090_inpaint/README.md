# rtx3090_inpaint — R9 상주 INT4 FLUX Fill 워커

3090 에서 FLUX.1-Fill-dev(INT4 transformer)를 한 번 올려 두고 HTTP 로 인페인팅을 받는 프로세스다.
- 결과가 바뀌는 항목이라 **기본 꺼짐**이다. `VHL_R9=1` 일 때만 뜬다.
  - INT4 채택은 확정됐다(사용자 2026-10-23).
  - 기본값을 켜짐으로 바꾸는 것은 5번 PR 병합 때 chief 가 지시한다. 그 전까지 `.env` 에 `VHL_R9=1` 을 영구로 넣지 않는다.
- 규약: [services/common/http-contract.md](../common/http-contract.md) §2
- 측정 결과: [docs/reports/05-R9.md](../../docs/reports/05-R9.md)

## 1. 실행 환경

| 항목 | 값 |
|---|---|
| env | conda `flux-fill` (`<HOME>/anaconda3/envs/flux-fill/bin/python`). 기존 env 를 그대로 쓰고 이 워커 때문에 설치한 것은 없다 |
| Python | 3.11.15 |
| torch / CUDA | 2.9.1+cu128 / 12.8 (드라이버 610.43.03, sm_86 → INT4) |
| nunchaku | 1.2.1+cu12.8torch2.9 (GitHub 릴리스 휠. PyPI 의 동명 패키지가 아니다) |
| diffusers / transformers / huggingface_hub | 0.39.0 / 4.57.6 / 0.36.2 |
| 전체 목록 | `env/rtx3090-flux-fill.txt` (pip freeze) |
| 재설치 | `legacy/rtx3090/flux_fill/setup_fluxfill.sh` (env 를 새로 만들 때만) |
| 의존 코드 | 포크 사본 `build/reference-v2` (`REFERENCE_DIR=build/reference-v2 bash tools/setup_reference.sh`). `utils/inpaint.py:inpaint()` 를 그대로 부른다 |

HTTP 서버는 표준 라이브러리 `http.server` 다. fastapi 등은 필요 없다.

지표 계산(R-4)용 venv `<HOME>/venvs/vhl-eval` 은 워커와 무관하다:
- `flux-fill` 파이썬 + `--system-site-packages` 로 만들었고, `lpips==0.1.4`·`scipy==1.17.1` 을 더했다.

## 2. 가중치

| 부품 | 저장소 · snapshot | 비고 |
|---|---|---|
| INT4 transformer | `nunchaku-ai/nunchaku-flux.1-fill-dev` · `bd90903` · `svdq-int4_r32-flux.1-fill-dev.safetensors` (6.77 GB, sha256 `8ef7c319…af68`) | 비게이트 |
| T5·CLIP·VAE·토크나이저·설정 | `black-forest-labs/FLUX.1-Fill-dev` · `358293d` (19개 파일, bf16 transformer 제외) | **게이트**. rockthepeople 토큰으로 받음. 미러와 sha256 동일 |

처음 받을 때(한 번만, 토큰 필요):
```bash
HF_TOKEN=<토큰> <flux-fill python> -c "from huggingface_hub import snapshot_download as s; \
  s('nunchaku-ai/nunchaku-flux.1-fill-dev', allow_patterns=['svdq-int4_r32-*']); \
  s('black-forest-labs/FLUX.1-Fill-dev', allow_patterns=['model_index.json','scheduler/*','text_encoder/*','text_encoder_2/*','tokenizer/*','tokenizer_2/*','vae/*','transformer/config.json'])"
```
워커는 `HF_HUB_OFFLINE=1`(기본)로 HF 캐시(`HF_HOME`, 기본 `<HOME>/.cache/huggingface`)에서만 읽는다. 실행 중에는 네트워크를 쓰지 않는다.

## 3. 환경변수

저장소 루트 `.env` 에 둔다(커밋 금지). `run.sh` 가 읽는다.

| 이름 | 필수 | 기본 | 뜻 |
|---|---|---|---|
| `VHL_R9` | ✓ | (꺼짐) | `1` 이어야 뜬다 |
| `RTX3090_PYTHON` | ✓ (`run.sh`) | – | `flux-fill` env 의 python 경로 |
| `INPAINT_PORT` | – | 8082 | 바인딩 포트 |
| `INPAINT_BIND` | – | `127.0.0.1` | 바인딩 주소. A5000 실험 스크립트가 직접 부를 때만 `0.0.0.0`(X-paper X3, chief 지시 2026-10-26) |
| `RUNS_DIR` | – | `<repo>/runs` | 잡 산출물 `RUNS_DIR/r9-inpaint/<job_id>/`, 로그 `RUNS_DIR/r9-inpaint/worker.log` |
| `REFERENCE_DIR` | – | `build/reference-v2` | 포크 사본 위치 |
| `FLUX_FILL_BASE` | – | `black-forest-labs/FLUX.1-Fill-dev` | T5·CLIP·VAE 저장소. 미러로 바꿀 수 있다(정오표 E11) |
| `FLUX_FILL_TRANSFORMER` | – | `nunchaku-ai/nunchaku-flux.1-fill-dev` | INT4 transformer 저장소 |
| `HF_HUB_OFFLINE` | – | `1` | `0` 이면 실행 중에도 hub 에 접근한다(권장하지 않음) |
| `HF_HOME` | – | HF 기본값 | 가중치 캐시 위치 |
| `HF_TOKEN` | 처음 받을 때만 | – | 원본 FLUX.1-Fill-dev(게이트) 다운로드용. 워커 실행에는 필요 없다 |

## 4. 기동·정지·재기동

```bash
bash services/rtx3090_inpaint/run.sh start     # setsid 로 띄우고 [ready] 까지 기다린다 (실측 약 7.4 s)
bash services/rtx3090_inpaint/run.sh status    # GET /health
bash services/rtx3090_inpaint/run.sh restart   # stop → start
bash services/rtx3090_inpaint/run.sh stop      # SIGTERM, 10 s 뒤 SIGKILL
```

- `start` 는 이미 포트에 떠 있으면 아무것도 하지 않는다. `VHL_R9` 가 `1` 이 아니면 실패한다.
- 프로세스는 `setsid --fork` 로 떠서 셸·세션이 끝나도 산다. 부팅 시 자동 기동은 설정하지 않았다(systemd 유닛 없음).
- 직접 띄울 때(디버그):
  ```bash
  VHL_R9=1 <flux-fill python> -u services/rtx3090_inpaint/worker.py
  ```
- **Z-Image 와 동시에 상주할 수 없다**(R-3 실측 OOM). 생성(Z-Image)을 돌릴 때는 `stop` → 생성 → `start` 로 교대한다. 한 주기 실측 29.9 s 다.

## 5. 상태 확인과 로그

- `GET /health` 응답의 주요 필드:
  - `ok`: 모델 적재 완료
  - `busy`: 실행 중인 job_id
  - `queue`: 대기 수
  - `calls`: 누적 호출
  - `load`: 저장소·snapshot·적재 시간·적재 후 메모리
- `GET /jobs/<id>`: `state`(queued·running·done·failed), 결과 sha256, 마스크 밖 합성 전후 차이, 시간·VRAM
- 로그: `RUNS_DIR/r9-inpaint/worker.log` 에 `[load]`, `[ready]`, `[http]` 요청 줄, 예외 traceback 이 남는다.
- 잡 산출물: `RUNS_DIR/r9-inpaint/<job_id>/{2d_render,2d_mask,2d_edit,2d_edit_raw}.png`, `result.json`. 자동으로 지우지 않는다.

## 6. 동작 요약

- 입력: `image`(원본 렌더 1024² PNG, RGBA 면 흰 배경 합성), `mask`(흰색 = 다시 그릴 영역), `prompt`, `seed`(기본 0)
- 설정: 1024², 50스텝, guidance 30, max_sequence_length 512, CPU generator. 포크 기본값과 같다.
- 출력: 합성 전 `2d_edit_raw`, 마스크 밖을 원본 렌더로 덮은 `2d_edit`(2진 마스크, 페더 없음)
- 실측(R-2·R-4):
  - 호출 약 26.0 s
  - 상주 16.4 GiB, 호출 피크 alloc 18.13 GiB(nvidia-smi 20.1 GiB)
  - **같은 시드에서도 출력이 회차마다 조금씩 다르다**(비결정적)
- 테스트: `python3 tests/cpu/test_r9_composite.py` (합성 6건)
