# FLUX.1-Fill-dev (nunchaku SVDQuant INT4) — 3090 로컬 인페인트

## 쓰는 법

```bash
<HOME>/anaconda3/envs/flux-fill/bin/python <HOME>/ai-ar-v3-assets/_tools/flux_fill.py \
    --image <원본.png> --mask <마스크.png> --prompt "<채울 내용>" --out <결과.png> \
    [--steps 30] [--guidance 30] [--seed 42] [--offload]
```

**마스크 규약: 흰색(255) = 다시 그릴 영역, 검은색 = 보존 영역.**
기본으로 **마스크 밖을 원본 바이트로 되돌린다**(`--no-composite` 로 끌 수 있다).

## 왜 합성이 기본인가

FluxFill 은 잠재공간 전체를 디코드한다. 그래서 **건드리지 않은 영역도 VAE 왕복으로
전부 미세하게 바뀐다.** 실측:

| | 합성 전 | 합성 후 |
|---|---|---|
| 마스크 안 화소차>16 | 56.5% | 56.7% |
| 마스크 밖 화소차>16 | 1.72% | 0% |
| **마스크 밖 바이트 동일** | **0.01%** | **100.00%** |

"예외가 안 났다 ≠ 안전하다" 그대로다. 1.72% 는 눈에 안 보이지만 보존(B) 기준으로는 실패다.

## 실측 (RTX 3090 · 1024² · 30 steps)

```
적재 5s · 추론 16s · 총 21s · peak VRAM 18.1 GiB / 24 GiB
```

## 설치된 것

| | |
|---|---|
| 환경 | conda `flux-fill` (py3.11 · torch 2.9.1+cu128) — 기존 zimage/trellis/birefnet 불변 |
| 엔진 | nunchaku **1.2.1+cu12.8torch2.9** (GitHub 릴리스 휠) |
| 정밀도 | **int4** — 3090 은 sm_86 이라 자동 선택된다 (fp4 는 50 시리즈용) |
| transformer | `nunchaku-ai/nunchaku-flux.1-fill-dev` svdq-int4_r32 · **6.77 GB** (원본 23.8 GB) |
| 나머지 부품 | `camenduru/FLUX.1-Fill-dev-ungated` · 9.5 GB |

재설치: `bash <HOME>/ai-ar-v3-assets/_tools/setup_fluxfill.sh`

## ⚠️ 두 가지 주의

**① PyPI 의 `nunchaku` 는 동명이인이다** — "piecewise linear segmentation" 라이브러리다.
`pip install nunchaku` 하면 엉뚱한 것이 깔린다. **GitHub 릴리스 휠로만 설치한다.**

**② 부품 출처가 원본이 아니다.** `black-forest-labs/FLUX.1-Fill-dev` 는 게이트라
토큰 없이 401 이다. 그래서 비게이트 미러를 썼고, **서로 무관한 미러 3곳의 LFS sha256 을
대조해 vae·text_encoder·text_encoder_2 가 전부 일치**하는 것을 확인하고 채택했다.

```
vae                              8c717328c8ad41fa…  3미러 일치
text_encoder                     893d67a23f4693ed…  3미러 일치
text_encoder_2 (1/2)             ec87bffd1923e8b2…  3미러 일치
text_encoder_2 (2/2)             a5640855b301fcdb…  3미러 일치
```

HF 토큰이 생기면 `FLUX_FILL_BASE=black-forest-labs/FLUX.1-Fill-dev` 로 바꾸면 원본을 쓴다.

## 🔴 라이선스 — 상업적 사용 불가

FLUX.1-Fill-dev 는 **flux-1-dev-non-commercial-license** 다. 연구·평가는 되고
**상업 배포는 안 된다.** 이 프로젝트가 제품으로 가면 이 경로는 못 쓴다 —
Z-Image-Turbo(Apache-2.0)나 다른 백본으로 갈아타야 한다. 지금은 실험 도구로만 쓴다.

---

## 원본 bf16 대조 실측 (RTX 3090 · 1024² · 30스텝 · 같은 이미지·마스크·프롬프트)

원본 transformer 도 설치돼 있다 (`camenduru/FLUX.1-Fill-dev-ungated` `transformer/`, 22.17 GiB).
`flux_fill_ab_bench.py --backend int4|bf16` 로 둘을 같은 프로토콜(워밍업 1회 버리고 3회)로 잰다.

| | INT4 (nunchaku) | 원본 bf16 | 배수 |
|---|---|---|---|
| 가중치 | **6.31 GiB** | 22.17 GiB | **3.5× 작다** |
| 적재 | 4.7 s | 1.8 s* | |
| **호출 총시간** | **15.62 s** | **59.50 s** | **3.8× 빠르다** |
| 디노이즈 스텝당 | **495 ms** | 1,295 ms | **2.6× 빠르다** |
| peak VRAM | 18.1 GiB | 22.8 GiB | |

\* bf16 적재가 짧아 보이는 건 `enable_model_cpu_offload()` 라 GPU 전송이 적재가 아니라
**매 호출**로 미뤄지기 때문이다. 그 전송이 총시간 59.50s 중 약 20s 다.

⚠️ **원본은 3090 에 통째로 안 올라간다.** transformer 22.17 GiB + T5 9.5 GiB + VAE 를
더하면 24 GiB 를 넘는다. 그래서 원본만 `enable_model_cpu_offload()` 로 돌렸다.
디노이즈 동안엔 transformer 가 GPU 에 상주하므로 **스텝당 1,295 ms 는 공정한 연산 비교**고,
총시간 59.50s 에는 매 호출 22 GiB PCIe 전송이 포함돼 있다. 즉 **2.6× 는 순수 연산 이득,
3.8× 는 이 기기에서 실제로 체감하는 이득**이다.

### 품질 — 양자화 손실은 눈에 안 띈다, 하지만 같은 그림은 아니다

| | 금색 화소 | 마스크 안 변화 | 마스크 밖 바이트 동일 |
|---|---|---|---|
| INT4 | 12.6% | 89.0% | 100.00% |
| bf16 | 11.6% | 87.7% | 100.00% |

둘 다 아이언맨 헬멧을 제대로 그렸고 육안 차이가 없다. 다만 **같은 시드·같은 프롬프트인데
결과가 같지 않다** — 마스크 안 평균 화소차 13.2, 17.3% 화소가 >16 차이다. 양자화가 샘플링
궤적을 바꾼다. **INT4 로 뽑은 그림을 bf16 으로 재현할 수 없다.** 재현성이 필요한 실험에서는
어느 쪽으로 쟀는지를 반드시 기록해야 한다.
