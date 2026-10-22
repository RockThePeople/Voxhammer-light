# 00 착수 — rtx3090 (B절) — 보고서

- 장비: rtx3090 (RTX 3090 24 GB, 드라이버 610.43.03)
- 브랜치·커밋: `item/00-kickoff-3090`, 기준 main `f3ef6ab`
- 기준 상태: 00a 승인 커밋 main `361127c` (태그 `approved/00a` 미생성)
- 토글: 없음 (결과 유형 "동일". 기록과 측정만 하고 코드 동작은 바꾸지 않는다)
- check_pins: `PIN CHECK OK  voxhammer=f33cd96  patch=b886a2839e3b` (실측, 시작 시·B-0 후)

## 1. 예상

- B-0: 새 `setup_reference.sh` 가 `build/reference-v2` 를 삭제 없이 만들고 스탬프와 커밋을 남긴다.
- B-3 (추론, 측정 전에 적음): Z-Image-Turbo 는 bf16 약 6B 파라미터에 Qwen 계열 텍스트 인코더를 더한 크기라 피크가 20 GiB 안팎, 생성 9스텝은 10 s 안팎, 적재는 페이지 캐시가 따뜻하면 10 s 이하. BiRefNet 은 1–3 GiB, 1 s 미만 추론. 두 워커 모두 Python·torch import 에 몇 초가 든다.
- B-4: 미러의 LFS 파일 내용 sha256 은 HF blob 이름(LFS oid)과 같다.

## 2. 변경 파일

| 파일 | 내용 |
|---|---|
| `legacy/rtx3090/flux_fill/{flux_fill.py, flux_fill_bench.py, flux_fill_ab_bench.py, flux_fill_probe.py}` | 3090 인페인팅 도구 원본 그대로 (sha256 동일) |
| `legacy/rtx3090/flux_fill/{README.flux-fill.md, setup_fluxfill.sh}` | 원본에서 홈 경로 `~/` → `<HOME>/` 만 바꿈 (각 2행) |
| `legacy/rtx3090/flux_fill/README.md` | 출처, 원본 대 보관본 sha256 표, 기존 측정값 `미검증` 표시 |
| `legacy/rtx3090/flux_fill/mirror_sha256.txt` | B-4 bf16 미러 파일 23개 sha256 |
| `reference/rtx3090/measure_call.py`, `reference/rtx3090/_inner.py` | B-3 측정 래퍼 (워커 밖) |
| `docs/reports/00-kickoff-3090.md` | 이 보고서 |

새 의존성 없음. 측정 래퍼는 시스템 `python3`(표준 라이브러리만)과 기존 `zimage`·`birefnet` env 를 그대로 쓴다.

## B-0 기준 사본

`REFERENCE_DIR=build/reference-v2 bash tools/setup_reference.sh` → exit 0, 출력 `reference ready: build/reference-v2`. 사본 커밋 `dbbe9f1 f33cd96 + edit_pipeline patch`, 이후 check_pins OK. `W39_ATTN_MASK` 미설정을 확인했고 스크립트의 경고는 출력되지 않았다. (실측)

00a 에서 손으로 만든 `build/reference` 는 지우지 않고 그대로 두었다. 이후 이 세션은 `build/reference-v2` 만 쓴다.

## B-1·B-2 legacy 보관

지시서대로 `<HOME>/ai-ar-v3-assets/_tools/` 의 FLUX Fill 도구 6개를 `legacy/rtx3090/flux_fill/` 로 복사했다.

- 코드 4개는 바이트가 같다.
- README·설치 스크립트는 홈 경로 자리표시자만 바꿨고 diff 는 4행이다. 원본 대 보관본 sha256 은 `legacy/rtx3090/flux_fill/README.md` 표에 있다. (실측)
- 원본 README 의 측정값은 `미검증` 으로 표시했다.
- 가중치는 포함하지 않았다.

## B-3 호출 1회 비용 (3회 평균 ± 표본 표준편차, 실측)

조건:
- 다른 GPU 연산 프로세스 없음(래퍼가 시작 전에 검사), GPU 기준 사용량 38 MiB(Xorg·gnome-shell).
- Z-Image → BiRefNet 을 순차로 한 쌍씩 3회 돌렸다(ai-ar-v3 `server/t2i.py` 의 실제 순서와 같음). BiRefNet 입력은 같은 회차의 Z-Image 출력이다.
- 프롬프트는 S-1 템플릿에 A-M1 subject 를 넣은 것이다: `a man riding a black motorcycle, full body, 3/4 view, product photography, studio lighting, plain white background, centered, entire subject fully visible with margin, no text, no watermark`. seed 42.
- 워커 sha256: `zimage_gen.py` 94fc5a1e…7629, `birefnet_seg.py` 7f163745…b735. 워커는 수정하지 않았다.
- HF 가중치는 페이지 캐시에 있는 상태(따뜻함)로 보인다. 첫 회만 적재가 약 1 s 길다. 콜드 캐시는 재지 않았다.

측정 방법:
- 래퍼(`reference/rtx3090/measure_call.py`)가 워커를 subprocess 로 띄워 벽시계 시간을 잰다.
- 그 subprocess 안의 `_inner.py` 는 워커를 `runpy` 로 그대로 실행한다. 라이브러리 진입점에만 훅을 건다: Z-Image 는 `from_pretrained`·`.to`·`__call__`, BiRefNet 은 `from_pretrained` 와 forward pre/post hook.
- 경계마다 `torch.cuda.synchronize()` 후 시각을 기록한다. 생성 직전에 `reset_peak_memory_stats()` 를 부른다.
- nvidia-smi 는 0.1 s 간격으로 샘플링한다.

| 구간 (s) | Z-Image-Turbo | BiRefNet |
|---|---|---|
| **subprocess 기동~종료 (벽시계)** | **21.92 ± 0.69** (22.71, 21.57, 21.47) | **7.56 ± 1.06** (8.78, 6.93, 6.96) |
| 기동~적재 시작 (인터프리터·torch·diffusers/transformers import) | 3.74 ± 0.02 | 2.97 ± 0.49 |
| 모델 적재 (`from_pretrained` + GPU 전송) | 6.87 ± 0.59 | 2.56 ± 0.39 |
| 생성 / 추론 | 10.00 ± 0.06 (9스텝, 약 1.0 s/it) | 0.78 ± 0.16 |
| 생성 후~종료 (PNG 저장·정리) | 1.31 ± 0.06 | 1.25 ± 0.02 |

| 메모리 | Z-Image-Turbo | BiRefNet |
|---|---|---|
| `max_memory_allocated` 적재 직후 | 19.28 GiB | 0.44 GiB |
| `max_memory_allocated` 생성 중 | **21.67 GiB** | 1.58 GiB |
| nvidia-smi 프로세스 used_memory 최대 | 23,814 MiB (23.26 GiB) | 2,676 MiB |
| nvidia-smi GPU 전체 memory.used 최대 | 23,860 MiB / 24,576 | 2,722 MiB |

메모리 값은 3회 모두 같았다(표준편차 0).

- 계측 JSON: `runs/b3-call-cost-20261022-121854/b3_call_cost.json` (git 제외), sha256 `da60e638a48160ae202116a3b17150c778f405e88a6e27cb19258612f5b99c40`
- 출력 재현성: 같은 시드에서 Z-Image 출력 3회가 바이트 동일했다(`ba83d6d4…8119`). BiRefNet 출력 3회도 바이트 동일했다(`1909de04…c64f`). (실측)
- 효능: 출력 1회차를 육안으로 확인했다. 검은 오토바이에 탄 사람이 생성됐고 배경이 깔끔히 분리됐다. **다만 측면 뷰이며, 프롬프트의 "3/4 view" 를 따르지 않았다.** S-1 에 영향이 있다(아래 8절). (실측·육안)

**R9 판단 근거 (추론).** Z-Image 한 프로세스가 nvidia-smi 기준 23.3 GiB 로 카드를 거의 다 쓴다. INT4 FLUX Fill 의 기존 기록 피크는 18.1 GiB(`미검증`)다. 따라서 두 모델을 지금 설정 그대로 3090 에 **동시 상주시키는 것은 불가능**하다. 상주시키려면 다음 중 하나가 필요하다:
- Z-Image 측의 offload 또는 양자화
- 텍스트 인코더 해제
- 순차 로드/언로드 (Z-Image 적재 비용 약 6.9 s, 프로세스 기동 포함 약 10.6 s)

## B-4 bf16 FLUX.1-Fill-dev 미러 sha256 (실측)

- 대상: `camenduru/FLUX.1-Fill-dev-ungated` 스냅샷 `da3981208deaf7da0e80f83d1ae892c23c8f7310`(refs/main)의 파일 23개. 전체 목록은 `legacy/rtx3090/flux_fill/mirror_sha256.txt`.
- **LFS 파일 9개 모두, 내용 sha256 이 HF 캐시 blob 이름(LFS oid)과 일치했다.** 손상이나 부분 다운로드가 없다.
- 00a 에서 본 `.incomplete` 조각(6.3 GB)은 스냅샷이 참조하지 않는 잔여물이다.
- `flux-fill` env 와 가중치는 사용하지 않았고 바꾸지도 않았다.

| 파일 | 바이트 | sha256 |
|---|---|---|
| transformer 00001-of-00003 | 9,985,006,384 | `4cef6b166b2163ba9a01ccb68bf53054556852b69b32c0565dfe0741d28982d7` |
| transformer 00002-of-00003 | 9,949,328,904 | `cd99fdd45ac404a9c98c939bdc986647cd5160ca48b8e679e81d3d59c2b9a1e0` |
| transformer 00003-of-00003 | 3,870,584,832 | `ed878852603579083b4a4ea80544504faca9f42ebd41aaa103c3023ee2ebfbd5` |
| text_encoder_2 00001-of-00002 | 4,994,582,224 | `ec87bffd1923e8b2774a6d240c922a41f6143081d52cf83b8fe39e9d838c893e` |
| text_encoder_2 00002-of-00002 | 4,530,066,360 | `a5640855b301fcdbceddfa90ae8066cd9414aff020552a201a255ecf2059da00` |
| text_encoder | 246,144,352 | `893d67a23f4693ed42cdab4cbad7fe3e727cf59609c40da28a46b5470f9ed082` |
| vae | 335,306,212 | `8c717328c8ad41faab2ccfd52ae17332505c6833cf176aad56e7b58f2c4d4c94` |
| tokenizer_2/spiece.model | 791,656 | `d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86` |

원본 `black-forest-labs/FLUX.1-Fill-dev` 와의 대조는 처음 제출할 때 하지 않았다. 아래는 사용자 지시(2026-10-22)로 추가했다.

### B-4 추가: 원본 저장소 대조 (2026-10-22, 사용자 지시, 실측)

- 접근 확인 (HEAD 요청):
  - 토큰 없이: 401 `GatedRepo`
  - 토큰으로(HF 계정 `rockthepeople`, fine-grained 토큰, 3090 로컬 `.env` 에만 있음): 200/302
  - 결론: 이 계정은 **라이선스 동의를 마쳤고 원본에 접근할 수 있다.**
- 대조 방법:
  - 미러 스냅샷 파일 23개 각각에 대해 원본 `main`(커밋 `358293da0354175698b67ec8299acf928313a78a`)의 같은 경로로 **HEAD 요청만** 보냈다. 다운로드는 하지 않았다.
  - LFS 파일 9개: 응답의 `X-Linked-Etag`(LFS sha256)를 위 표의 미러 내용 sha256 과 비교했다.
  - 일반 파일 14개: 응답의 `ETag`(git blob sha1)를 미러 파일로 계산한 git blob sha1 과 비교했다.
- 결과: **23/23 일치. 미러는 원본 main 과 파일 단위로 같다.** 원본에만 있는 파일(예: 단일 파일 체크포인트)은 대조 범위 밖이다.
- 영향: 3090 bf16 경로가 "비게이트 미러" 라는 우려는 내용상 해소됐다. 출처 표기는 그대로 미러로 둔다. 원본 경로로 바꿀지는 chief 가 판단한다.

## B-5

0번에서 제외했다(지시서). 오케스트레이터는 기동하지 않았고 포트도 열지 않았다.

## 3·4·5절 (구간 1–8 기준선, 동등성)

3090 에는 해당이 없다. 구간 1–8 기준선과 동등성은 A5000 A절에서 다룬다. 3090 측정은 위 B-3 이 전부다.

## 6. 예상과의 차이와 원인

- Z-Image 피크(21.67 GiB alloc, 프로세스 23.3 GiB)는 예상한 "20 GiB 안팎"보다 컸다. 적재 직후에 이미 19.28 GiB 다. 텍스트 인코더와 transformer 를 모두 bf16 으로 GPU 에 올리고 `low_cpu_mem_usage=False` 이기 때문이다(코드확인 `zimage_gen.py:16-19`). 생성 중 추가분 2.4 GiB 는 활성값과 VAE 디코드로 본다(`추론`, 분해하지 않았다).
- 생성 시간(10.0 s)과 BiRefNet(0.78 s)은 예상 범위였다.
- 호출 하나에서 모델과 무관한 고정비가 크다. Z-Image 의 import 와 종료가 약 5 s(벽시계의 23 %)이고, BiRefNet 은 약 4.2 s(56 %)다. 상주화하면 없어지는 비용이다.

## 7. 실행 정보

- 자산: 없음 (생성 측정이라 입력 자산·마스크가 없다). 프롬프트와 시드는 B-3 절.
- 실행 횟수: Z-Image·BiRefNet 각 3회, 워밍업 없음(첫 회 포함).
- 계측 JSON sha256: `da60e638a48160ae202116a3b17150c778f405e88a6e27cb19258612f5b99c40`
- 환경: `zimage` env(torch 2.6.0+cu124, diffusers 0.39.0), `birefnet` env(torch 2.6.0+cu124, transformers 5.14.1). freeze 는 `env/rtx3090-*.txt`(00a).

## 8. 남은 문제·요청

- **S-1 관련 (chief 판단 요청):** 측정에 쓴 A-M1 프롬프트("3/4 view" 포함)가 seed 42 에서 측면 뷰로 나왔다.
  - S-1 은 "subject 와 다르게 나오면 시드만 바꿔 다시" 라고 정한다. 시점 불일치도 그 사유에 들어가는지 정해 주기 바란다.
  - S-1 은 별도 브랜치(`item/00s-scenarios-3090`)에서 진행하면서 이 점을 기록한다.
- 개선 기회(이 항목에서는 구현하지 않음):
  - Z-Image·BiRefNet 상주화로 호출당 고정비 약 5 s / 4 s 를 줄일 수 있다.
  - Z-Image 피크가 23 GiB 라 3090 에서 INT4 FLUX 와 동시 상주할 수 없다. R9 설계 시 순차 로드나 offload 를 검토해야 한다.
- self-hosted 러너: 보류(사용자 sudo 필요).
