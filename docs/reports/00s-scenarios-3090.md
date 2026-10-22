# 00s 시나리오 준비 — rtx3090 (S-1 트랙 A 입력 이미지)

- 장비: rtx3090 (RTX 3090 24 GB, 드라이버 610.43.03)
- 브랜치·커밋: `item/00s-scenarios-3090`, 기준 main `f3ef6ab`
- 통합 보고서는 a5000 이 쓰는 `docs/reports/00s-scenarios.md` 다. 이 문서는 그 보고서의 S-1 절 원본이다.
- check_pins: OK (실측)

## 1. 예상

- 기존 3090 경로(Z-Image-Turbo 1024²·9스텝·guidance 0 → BiRefNet)는 B-3 에서 같은 시드면 바이트 동일한 출력을 냈다. 그래서 같은 프롬프트·시드로 다시 돌려도 결과가 같아야 한다.
- 템플릿의 "3/4 view" 는 Z-Image 가 지키지 않을 수 있다. 00-kickoff B-3 에서 오토바이가 측면 뷰로 나왔다.

## 2. 변경 파일

| 파일 | 내용 |
|---|---|
| `tools/s1_track_a_images.py` | S-1 생성 스크립트. 기존 워커 `zimage_gen.py`·`birefnet_seg.py` 를 수정 없이 순차 subprocess 로 호출한다. 프롬프트는 요구사항서 6장 템플릿 원문을 쓴다 |
| `scenarios/track_a_inputs.json` | 자산 3종의 생성 파라미터, 환경 버전, 파일 sha256 |
| `docs/reports/00s-scenarios-3090.md` | 이 보고서 |

새 의존성은 없다. 이미지는 git 에 넣지 않았고 `ASSETS_DIR/track_a/<asset>/seed42/` 에 두었다. 이 장비에는 `ASSETS_DIR` 가 정해져 있지 않았으므로, git 제외 대상인 로컬 `.env` 에 `ASSETS_DIR=<HOME>/voxhammer-light-assets` 로 정했다.

## 3. 생성 설정 (코드확인·실측)

| 항목 | 값 |
|---|---|
| t2i | `Tongyi-MAI/Z-Image-Turbo`, bf16, 1024×1024, `num_inference_steps=9`, `guidance_scale=0.0`, CUDA generator seed 42 (`zimage_gen.py` sha256 `94fc5a1e…7629`) |
| 배경 제거 | `ZhengPeng7/BiRefNet`, fp16, 1024² 리사이즈·ImageNet 정규화, `preds[-1].sigmoid()` → 알파 (`birefnet_seg.py` sha256 `7f163745…b735`) |
| 프롬프트 | `{subject}, full body, 3/4 view, product photography, studio lighting, plain white background, centered, entire subject fully visible with margin, no text, no watermark` 에서 `{subject}` 만 바꿈 |
| 환경 | `zimage` env: Python 3.11.15, torch 2.6.0+cu124, diffusers 0.39.0, transformers 5.14.1. `birefnet` env: torch 2.6.0+cu124, torchvision 0.21.0, transformers 5.14.1 |
| 실행 시각 | 2026-10-22 12:23–12:25 KST. 00-kickoff B-3 가 끝난 뒤에 실행했다. 실행 전에 다른 GPU 연산 프로세스가 없는 것을 확인했다 |

## 4. 결과 (실측)

| 자산 | subject | 시나리오 | 시드 | `image_rgb.png` sha256 | `input.png` (RGBA) sha256 | 전경 비율 · bbox (px) |
|---|---|---|---|---|---|---|
| moto | a man riding a black motorcycle | A-M1, A-M2 | 42 | `ba83d6d489ea414ac5c0d77c34257e8240e69297f610478c895579b37ade8119` | `1909de04ad14ec999d8541352a700198dc5c1d960b717b549664b1c507c8c64f` | 21.7 % · x 130–958, y 181–854 |
| snowman | a snowman with a carrot nose, stick arms and a red scarf | A-S1, A-S2 | 42 | `cb00015b6ed4521d9fd3bd330e5fb6fb38505ce0ba012c2266e578d1e7ca89f6` | `18060f3b0950517a5a51c98df3406ce24087ad3facfcc48d7b3a1000a6dabd34` | 20.7 % · x 221–816, y 213–961 |
| chair | a wooden armchair with armrests | A-C1, A-C2 | **43** | `75b52e27d60fbe6a16f68cee1d273e46910f95bdd9d50fa53f969f24dbdef017` | `c967a054b08e338c921afa39effb2e9ec24cfb893cf01588d64a7c61cdd018fe` | 13.0 % · x 237–794, y 246–968 |
| ~~chair~~ (기각) | 〃 | – | 42 | `c9c90c65a7f44fc5df226371c018c34b2bbd6c7083a055747a96d3d4a2ddbab8` | `129c302d82cc11d36e10197d08a7669ef97717d9d2024844d701153fcdbfc433` | 14.6 % · x 237–781, y 168–935 |

- 재현성: moto 의 두 파일 sha256 은 00-kickoff B-3 의 3회 출력과 같다. 같은 프롬프트·시드로 4회 생성해 4회 모두 바이트가 같았다는 뜻이다. (실측)
- 전경 비율은 알파 > 127 인 화소의 비율이다. 세 자산 모두 프레임 가장자리에 닿지 않는다(최소 여백 63 px, snowman 아래쪽).

## 5. 육안 확인 (실측·육안, RGB 와 RGBA 를 회색 배경에 합성해 봄)

| 자산 | 부품 | 시점 | 배경 분리 | 판정 |
|---|---|---|---|---|
| moto | 헬멧 쓴 라이더 머리, 앞·뒷바퀴, 차체 모두 있음 | **측면**(3/4 아님) | 깨끗함 | subject 일치. 시드 유지 |
| snowman | 당근 코, 막대 팔 2개, 빨간 목도리, 눈덩이 2단, 머리 위 빈 공간 있음 | **정면**(3/4 아님) | 깨끗함. 바닥의 눈 부스러기는 제거됨 | subject 일치. 시드 유지 |
| chair seed 42 | 등받이(위 가로대 + **세로 살 3개**), 양쪽 팔걸이, 좌판, 다리 4개 | 3/4 | 깨끗함 | subject 일치. 그러나 A-C1 효능 문제로 기각(아래) |
| chair seed 43 | 등받이(**살 없는 곡면 판 1장**, 아래는 비어 있음), 양쪽 팔걸이, 좌판, 다리 4개 | 3/4 | 깨끗함 | subject 일치. **채택** |

S-1 규칙("subject 와 다르게 나오면, 예컨대 부품이 빠지면 시드만 바꿔 다시")에 해당하는 자산은 없다. moto·snowman 은 시점이 3/4 가 아니지만 chief 지시(2026-10-22)에 따라 seed 42 를 유지했다.

chair 는 chief 지시(2026-10-22, PR #4)에 따라 다시 골랐다.
- 규칙: seed 43 부터 순서대로 최대 5개를 만들고, 등받이가 판이거나 가로 살인 첫 시드를 쓴다.
- seed 43 이 첫 생성에서 조건을 만족했다. 그래서 **seed 43 을 채택했고 44–47 은 만들지 않았다.**
- 프롬프트는 바꾸지 않았다.
- seed 42 기록은 `scenarios/track_a_inputs.json` 의 `rejected.chair` 에 남겼다. (실측·육안)

## 6. 예상과의 차이와 확인 요청 (chief)

> 1·2·3 은 chief 가 결정했다(2026-10-22, PR #4 댓글).
> 1: 시점은 그대로 둔다.
> 2: chair 는 seed 43 으로 바꿨다(5절).
> 3: `xfer/track-a-inputs` 브랜치로 git 전달한다.
> 아래는 처음 제출할 때의 기록이다.

1. **"3/4 view" 를 따르지 않은 자산이 2개다.** moto 는 측면, snowman 은 정면이다.
   - S-1 규칙상 시점 불일치는 재생성 사유가 아니라고 보고 시드를 유지했다.
   - 시점 불일치도 재생성 사유로 볼지 정해 주기 바란다. 재생성한다면 바꿀 수 있는 것은 시드뿐이다. 프롬프트는 확정이라 바꿀 수 없다.
   - 3D 생성 품질에 미치는 영향은 S-2(TRELLIS) 결과를 봐야 알 수 있다(`미검증`). 오토바이는 측면에서 반대쪽이 안 보이고, 눈사람은 정면에서 깊이 정보가 적다.
2. **A-C1 편집의 효능 판정이 어려울 수 있다.** 생성된 의자의 등받이에 이미 세로 나무 살이 있다(위 가로대와 세로 살 3개). A-C1 프롬프트는 "a tall backrest with vertical wooden slats" 다. 편집 후에도 원본과 비슷해 보여 효능 지표(마스크 안 편집량, CLIP-T)가 낮게 나올 수 있다(`추론`).
   - 시나리오 정의는 확정이므로 바꾸지 않았다.
   - 시드를 바꿔 등받이가 판 형태인 의자를 고를지 결정이 필요하다. 다만 S-1 규칙의 재생성 사유는 아니다.
3. **이미지를 A5000 으로 넘기는 방법이 정해져 있지 않다.** S-2 는 A5000 에서 이 이미지를 입력으로 쓴다.
   - 00a 기준으로 3090 → A5000 HTTP 는 연결 거부, SSH 포트도 닫혀 있다.
   - 0번 동안은 새 포트·서비스를 열지 않는다.
   - 전달 경로를 지시해 주기 바란다. 받은 쪽은 위 sha256 으로 검증하면 된다.

## 7. 실행 정보

- 실행 횟수: moto·snowman 은 seed 42 로 1회, chair 는 seed 42 와 43 으로 1회씩. moto 는 같은 설정으로 B-3 에서 3회 더 실행했다.
- 산출 위치: `ASSETS_DIR/track_a/<asset>/seed<N>/{image_rgb.png,input.png,params.json}`(채택본: moto·snowman seed42, chair seed43) (git 제외). params 는 `scenarios/track_a_inputs.json` 에 그대로 옮겼다.
