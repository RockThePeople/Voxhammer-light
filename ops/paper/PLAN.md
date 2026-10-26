# VoxHammer 경량화 논문 — 추가 실험 계획

Oct 26, 2026 · @Alex

논문에 넣을 품질 결과를 얻으려면 실험 4개(X0, X2, X3, X4)가 필요하다. 편집 시간은 기존 실측 자료(구간 실측 PR #5, E2a PR #16, R9 PR #7)를 그대로 쓰므로 이 계획에서 뺐다. X0은 이미 저장된 결과 GLB를 새 메트릭으로 다시 채점하는 작업이라 편집을 다시 돌리지 않는다. 비교 기준은 원저자 경로(VoxHammer 원 구현, 25 스텝, 편집마다 150시점 재등록, bf16 FLUX.1 Fill)이다.

## 1. 검증할 주장

| ID | 주장 | 증거가 되는 결과 | 담당 실험 |
| --- | --- | --- | --- |
| C2 | 시간을 줄여도 단발 편집 품질이 유지된다 | 보존 영역·전체 품질·조건 정합 메트릭의 차이가 허용치 안 | X0, X2 |
| C3 | 재등록이 없어 연속 편집에서 보존 영역 품질이 더 높다 | 회차가 늘어도 보존 영역 메트릭이 v0 대비 유지되고, 원저자 경로는 회차마다 떨어진다 | X3, X4 |

C2와 C3는 서로 다른 주장이다. C2는 "품질이 같다"(동등성)를, C3는 "품질이 더 좋다"(우위)를 보여야 하므로 통계 검정도 다르게 한다(3절).

## 2. 비교 메트릭

VoxHammer의 세 범주(보존 영역, 전체 3D 품질, 조건 정합)를 기본 틀로 쓰고, 3DEditFormer의 형상 메트릭(NC, F1@0.01)과 계산 규약을 더한다. 3DEditFormer는 정답 편집 메시와 비교하지만 우리 데이터에는 정답 편집 메시가 없으므로, 형상 메트릭은 VoxHammer처럼 **원본 대비 보존 영역에서만** 계산한다.

| 범주 | 메트릭 | 계산 방법 (본 연구 규약) | 차용 출처 | 쓰는 실험 |
| --- | --- | --- | --- | --- |
| 보존 영역 형상 | CD ↓ | 메시를 단위 큐브로 정규화하고 표면에서 100,000점 균일 샘플. 편집 영역(마스크 ∪ halo 2) 밖의 점끼리 평균 최근접 거리 | VoxHammer, 3DEditFormer | X0, X2, X3, X4 |
| 보존 영역 형상 | NC ↑ | 같은 점 쌍에서 표면 법선 내적의 절댓값 평균 | 3DEditFormer | X0, X2, X3, X4 |
| 보존 영역 형상 | F1@0.01 ↑ | 거리 0.01 이하인 점의 정밀도·재현율 조화평균 | 3DEditFormer | X0, X2, X3, X4 |
| 보존 영역 형상 | 점유 IoU, 소실·신규 셀 | 64³ 격자의 R\_keep 안에서 점유 비교 (기존 자체 지표) | 자체 | 전부 |
| 보존 영역 외관 | PSNR(M) ↑, SSIM(M) ↑, LPIPS(M) ↓ | Edit3D-Bench `render.py`로 16시점 렌더. 편집 영역을 각 시점에 투영한 실루엣을 제외한 화소만 비교 | VoxHammer (3DEditFormer는 10시점) | X0, X2, X3, X4 |
| 전체 3D 품질 | FID ↓ | 편집 결과 렌더 집합과 원본 렌더 집합의 Inception 특징 분포 거리 | VoxHammer | X2 (표본 60건 이상일 때만) |
| 전체 3D 품질 | FVD ↓ | 회전 렌더 시퀀스를 동영상으로 보고 계산 | VoxHammer | X2 (같은 조건) |
| 조건 정합 | DINO-I ↑ | 편집 결과 렌더와 2D 편집 조건 이미지의 DINOv2 특징 코사인 유사도 (편집 시점) | VoxHammer, 3DEditFormer | X0, X2, X3 |
| 조건 정합 | CLIP-T ↑ | 편집 결과 렌더와 편집 프롬프트의 CLIP 유사도 | VoxHammer | X0, X2, X3 |
| 사람 평가 (선택) | 선호 비율 | 5회차 결과를 원저자 경로와 나란히 보여 주고 보존 상태가 나은 쪽 선택 (2지선다) | VoxHammer 사용자 평가(30명) 형식 | X3 |

**메트릭 사용 시 주의.** FID·FVD는 표본 수에 민감하므로 14건 시나리오에서는 보고하지 않고 Edit3D-Bench 60건(X2-B)에서만 쓴다. 무편집 왕복만으로도 마스크 밖 PSNR이 25.9–35.3 dB이므로, 다른 논문 수치와 절대값을 비교하지 않고 같은 규약으로 잰 원저자 경로 대비 차이만 보고한다.

## 3. 실험 목록

경로 이름은 다음과 같이 고정한다. **P0** 원저자 경로(편집마다 재등록, 25 스텝, bf16 인페인팅). **P1** 우리 경로(저장 SLat 재사용, 15 스텝, INT4 인페인팅, 이후 항목은 구현되는 대로 추가). **P2** 절제 경로: P1과 같되 매 편집 전에 150시점 재등록을 켠다. P1과 P2의 차이는 재등록 하나뿐이다.

### X0. 기존 결과 재채점 (편집 재실행 없음)

- **목적:** C2의 첫 근거를 가장 싸게 얻는다.
- **입력:** 저장된 결과 GLB. 14건 × 3회 × {25/25 기준, 15/15 E2a}, 무편집 왕복 결과.
- **측정:** 2절의 CD·NC·F1@0.01(보존 영역), PSNR·SSIM·LPIPS(M), DINO-I, CLIP-T. 지금까지 없던 NC·F1·보존 영역 CD가 추가된다.
- **산출:** 논문 표 "단발 품질, 25 스텝 대 15 스텝".

### X2. 단발 편집 품질 동등성 (C2)

- **X2-A 3D만 비교:** 조건 이미지를 동결본으로 고정하고 P0와 P1의 3D 경로만 비교한다. 14건 × 3회. 인페인팅 차이를 배제하고 W34·E2a·R4 등의 영향만 본다.
- **X2-B 공개 벤치마크:** Edit3D-Bench에서 GSO 30건, PartObjaverse-Tiny 30건(모델당 프롬프트 1개)을 고정 시드로 뽑는다. 데이터셋이 제공하는 2D 편집 이미지와 3D 마스크를 그대로 쓴다. P0·P1 각 1회. FID·FVD는 이 실험에서만 보고한다. P1의 최초 SLat은 1회 등록으로 만들고 그 시간은 따로 적는다.
- **X2-C 종단:** 각 경로가 자기 인페인팅을 쓴다(P0 bf16, P1 INT4). 트랙 A 6건 × 3회. 지금까지 측정하지 않은 INT4 조건의 3D 품질이 여기서 나온다.
- **판정:** 시나리오별 쌍 차이에 대해 동등성 검정(TOST)을 하고, 동등성 한계는 기존 절대 허용치(ΔPSNR −0.5 dB, ΔLPIPS +0.005, ΔDINO-I·ΔCLIP-T −0.015, ΔCD +0.002)를 쓴다. 개선된 칸 수도 함께 보고한다.

### X3. 연속 5회 편집 (C3, 핵심)

- **비교:** P0, P1, P2 세 경로. P1과 P2의 차이가 재등록의 효과이고, P0와 P1의 차이가 전체 경로의 효과이다.
- **자산:** 3종. 자체 생성 경로(Z-Image-Turbo → BiRefNet → TRELLIS)로 새로 만드는 오프로드 지프, 나무 오두막, 모험가 피규어. 프롬프트는 아래 "X3 자산과 프롬프트"에 있다.
- **편집:** 자산마다 5회차. 각 자산을 교체 3·추가 1·제거 1로 구성한다. 마스크는 v0의 64³ 셀 집합으로 정의해 세 경로에 같게 준다.
- **조건 이미지:** 각 경로가 자기 직전 결과를 렌더해 같은 프롬프트·시드로 인페인팅한다. 결정적인 bf16 경로를 쓰고, 회차별 승인 이미지를 모두 저장한다.
- **측정 (회차 k = 1…5):** v0 대비, 지금까지 편집한 모든 마스크 ∪ halo 2의 바깥에서 CD·NC·F1@0.01, PSNR·SSIM·LPIPS(M), 점유 IoU·소실·신규 셀. 직전 회차 대비 같은 지표. 그 회차 편집 영역의 DINO-I·CLIP-T.
- **판정:** 5회차에서 P1이 P0보다 보존 지표가 좋다는 단측 부호순위 검정(자산 × 반복 단위). 회차–지표 곡선을 그림으로 제시한다.
- **캡처:** 1·3·5회차를 편집 시점과 3/4 시점에서 P0 | P1 나란히.

**X3 자산과 프롬프트.** 최초 생성은 Z-Image-Turbo, 편집은 FLUX.1 Fill bf16에 그대로 넣는다. 편집 프롬프트는 마스크 안에 보일 내용을 적은 것이며, 제거 회차는 부위가 사라진 뒤의 모습으로 썼다. 생성 후 5회차 누적 마스크가 점유 셀의 40%를 넘으면 해당 회차를 더 작은 부위로 바꾼다.

*자산 1: 오프로드 지프*

```text
A compact off-road jeep, three-quarter front view, centered, full object visible, chunky all-terrain tires, a black front winch bumper, a side air-intake snorkel on the right front pillar, a spare tire mounted on the rear door, olive green paint, simple clean 3D game asset style, pure white background, no shadow
```

1. (교체, 앞 범퍼) a heavy chrome steel bull bar front bumper on the jeep
2. (교체·외관, 보닛) the jeep hood painted bright red with two white racing stripes
3. (교체, 뒤 스페어 타이어) a rear rack holding two red jerry cans on the back of the jeep
4. (추가, 지붕 위) a black roof rack with a brown leather travel bag strapped on top of the jeep roof
5. (제거, 스노클) the smooth right front pillar of the jeep with no snorkel, olive green paint

*자산 2: 나무 오두막*

```text
A small stylized wooden cottage, three-quarter front view, centered, full object visible, a stone chimney on the roof, a plain wooden front door, two square windows on the front wall, a small front porch with a wooden railing, brown shingled roof, simple clean 3D game asset style, pure white background, no shadow
```

1. (교체, 현관문) an arched red wooden front door with iron hinges
2. (교체·외관, 왼쪽 창) a round stained glass window with blue and yellow panes
3. (추가, 오른쪽 창 아래) a wooden flower box with red flowers under the window
4. (교체, 현관 난간) a white picket fence railing around the porch
5. (제거, 굴뚝) a plain brown shingled roof with no chimney

*자산 3: 모험가 피규어*

```text
A chibi adventurer figurine standing, three-quarter front view, centered, full body visible, wearing a brown leather hat, holding a wooden sword in the right hand, a round wooden shield on the left arm, brown leather boots, green tunic, simple clean 3D game asset style, pure white background, no shadow
```

1. (교체, 모자) a tall blue pointed wizard hat with yellow stars
2. (교체, 오른손의 칼) a glowing brass lantern held in the right hand
3. (추가, 오른쪽 어깨 위) a small blue bird perched on the right shoulder
4. (교체·외관, 부츠) red sneakers with white soles
5. (제거, 왼팔의 방패) the bare left arm in a green tunic sleeve, no shield

### X4. 무편집 재등록 체인 (C3 보조)

- **목적:** 편집 효과를 빼고 재등록 왕복 손실만 누적시켜 본다.
- **방법:** 빈 마스크로 5회 반복. P0는 매 회 GLB → 150시점 재등록 → 디코딩 → GLB. P1은 저장 SLat을 그대로 다음 회차로 넘긴다.
- **자산:** 14건의 원본 자산 11종(트랙 A 3, 트랙 B 8).
- **측정:** 회차별 v0 대비 전체 영역 CD·NC·F1·PSNR·SSIM·LPIPS·점유 IoU.
- **기대 판정:** P1은 회차와 무관하게 일정하고, P0는 1회차에서 이미 점유 IoU 0.800–0.959로 떨어진 뒤 회차마다 더 떨어지는지를 본다. 떨어지지 않으면 그 결과도 그대로 보고한다.

### 공통 규칙

- 장비: 3D 편집은 A5000, 인페인팅은 3090.
- 반복: 시나리오당 3회(seed 0). Stage 2 비결정성 때문에 결과는 3회 평균과 범위로 보고한다.
- 렌더러: 모든 렌더 메트릭은 Edit3D-Bench `render.py` 하나로 계산한다.
- 편집 시간은 이 계획에서 재지 않는다. 논문의 시간 수치는 기존 실측 자료를 쓴다.

## 4. 실행 순서와 소요 시간

A5000 기준 총 약 32시간이 예상되며, 편집을 다시 돌리지 않는 X0과 재등록 손실만 보는 X4를 먼저 하면 C2·C3의 첫 근거를 약 4시간 안에 얻는다. 소요 시간은 기존 구간 실측의 편집당 시간에 실행 횟수를 곱한 추정이다.

| 순서 | 실험 | 선행 조건 | 장비 | 예상 시간 (h) | 계산 근거 |
| --- | --- | --- | --- | --- | --- |
| 1 | X0 재채점 | 채점 스크립트(NC·F1·보존 CD, 2D 실루엣 마스크) | A5000 | 1–2 | 결과 GLB 약 100개 × 16시점 렌더 |
| 2 | X4 무편집 체인 | 없음 | A5000 | 2.5 | P0 약 140 s × 5회 × 11자산 |
| 3 | X2-A 3D만 비교 | 현재 P1 상태 | A5000 | 4 | (274 + 86) s × 42회 |
| 4 | X3 연속 편집 | P2 토글(우리 경로에 재등록 켜기), 자산 3종 생성, 회차별 마스크 확정 | A5000 + 3090 | 11 (반복 1회면 4) | 자산당 P0 32분 + P1 17분 + P2 27분, × 3자산 × 3회 |
| 5 | X2-B Edit3D-Bench 60건 | 60건 추출 스크립트, P1 최초 등록 | A5000 | 10 | (274 + 86) s × 60건 + 최초 등록 120 s × 60건 + 채점 |
| 6 | X2-C 종단 | 없음 | A5000 + 3090 | 2.6 | (388 + 140) s × 18회 |

**결정이 필요한 것.**

- X3의 회차별 마스크(64³ 셀)를 확정해야 한다. 프롬프트는 정해졌으므로, 수행사가 생성된 자산 위에서 마스크 초안을 만들고 사용자가 승인한다.
- X3의 반복 횟수(3회, 11시간 / 1회, 4시간)를 정해야 한다. Stage 2가 비결정적이므로 검정을 하려면 3회가 필요하다.
- 사람 평가(선택 항목)를 할지 정해야 한다. 하지 않으면 논문에서 "작업자 육안 판정"으로 쓴다.

## 5. 논문 반영

실험이 끝나면 논문 초안의 추정값과 "미측정" 표기를 아래 표·그림으로 바꾼다.

| 논문 위치 | 표·그림 | 채울 실험 | 지금 초안의 상태 |
| --- | --- | --- | --- |
| 5.3절 | 표 4에 NC·F1·보존 CD·SSIM(M)·LPIPS(M) 열 추가 | X0 | PSNR·DINO-I·CLIP-T 일부만 |
| 새 5.6절 | 표: 단발 품질 P0 대 P1 (VoxHammer 표 1과 같은 세 범주) | X2-A, X2-B | 없음 |
| 5.4절 | INT4 조건의 3D 품질 | X2-C | "측정하지 않았다" |
| 새 5.7절 | 그림: 회차별 보존 지표 곡선 (P0, P1, P2) | X3, X4 | 7장 향후 과제로만 언급 |
| 새 5.7절 | 그림: 1·3·5회차 캡처, 원저자 대 우리 | X3 | 없음 |

**논문 제목과 기여 문장 조정.** X3·X4에서 P1의 우위가 확인되면 기여 4번("보존 셀이 판본 사이에서 비트 단위로 유지된다")을 "연속 5회 편집에서 보존 영역 CD·PSNR(M)이 원저자 경로보다 높다"는 측정 결과로 바꾼다. 확인되지 않으면 기여에서 빼고 한계에 남긴다.

## 6. 출처

- [VoxHammer: Training-Free Precise and Coherent 3D Editing in Native 3D Space](https://arxiv.org/html/2508.19247v1) (3DV 2026). 표 1의 세 범주(Unedited Region Preservation: CD, PSNR·SSIM·LPIPS (M) / Overall 3D Quality: FID, FVD / Condition Alignment: DINO-I, CLIP-T), 사용자 평가 30명, 부록 표 4의 편집당 시간(A100에서 133 s).
- [Towards Scalable and Consistent 3D Editing (3DEditFormer)](https://arxiv.org/html/2510.02994v1) (ICML 2026). 3D 메트릭 CD·NC·F1@0.01(메시당 100,000점 샘플), 2D 메트릭 PSNR·SSIM·LPIPS·DINO-I(10개 고정 시점 렌더), 마스크 크기를 9·18·27% 키운 VoxHammer 비교.
- [RockThePeople/Voxhammer-light](https://github.com/RockThePeople/Voxhammer-light): 표 2 구간 실측(PR #5), 14건 시나리오(PR #10), E2a(PR #16), R9(PR #7). 4절의 소요 시간은 이 값으로 계산한 추정이다.
