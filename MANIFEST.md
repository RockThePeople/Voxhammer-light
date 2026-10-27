# paper-results — 논문 캡처·표 (X-paper, a5000)

> 사진을 어떤 기준으로 비교해 보는지는 [README.md](README.md)(안내서)에 있다.

생성: `tools/paper/capture/{capture,bpy_render,build,finish}.py`, 저장소 `item/X-paper-a5000` 커밋 `5b42eeb0c9301bf3b99025a17227d44644efcae3`(캡처 도구). 새 편집·새 채점 없음 — 보고서 `0531459` 기준 기존 출력만 다시 찍었다.

## 렌더 설정

- 렌더러: Blender Cycles(Edit3D-Bench `render.py` 와 같은 bpyrenderer 설정 — 64 샘플, 디노이즈, 확산·광택 반사 1회), GPU(OptiX). 1024×1024.
- 조명: HDRI `brown_photostudio_02_1k.exr`(Bench 와 같음), 바닥 없음(그림자 없음).
- shaded = 고정 조명, glTF 재질 그대로(Blender 기본 색 변환). albedo = 재질의 Base Color(텍스처·정점색 연결)를 Emission(세기 1)으로, 월드 조명 0, 색 변환 Standard.
- 배경: 투명 RGBA 로 찍고 알파로 흰색(255,255,255) 위에 합성한 불투명 RGB PNG(가장자리 검은 테두리 없음). 이미지 안에 글자 없음(라벨은 sheets/ 의 시트에만).
- crop = 3D 마스크 셀 bbox 8 꼭짓점을 같은 카메라로 투영한 상자 + 양쪽 15% 여유, 정사각형, 1024² 로 확대(X4 제외). X3 원본 r0 는 회차 마스크마다 crop_r<k>.png.
- P1·P2(정점색 R4 메시)는 output.glb 를 N 좌표로 옮긴 뒤 (iv) 보정(`editcore/render.py:vertex_color_render_glb`) — view.glb 와 같은 보정이다. view.glb 를 trimesh 로 옮기면 보정 재질·선형 float 정점색이 빠져 검게 나와서 이렇게 했다. P0 는 to_glb 출력.

## 카메라·좌표

- 카메라: Bench/포크 정의 — 반지름 1.8, 원점 주시, 렌즈 35 mm·센서 32 mm. az = Bench 방위(Blender 방위 = az − 90°). 정면은 원본 r0 의 고도 10° 8방향 렌더에서 정했다(cameras.json).
- 캐릭터 = 정면(az = 정면, 고도 10°), 사물 = 3/4 정면(az = 정면 + 45°, 고도 25°). 모든 경로·회차·실행에 같은 카메라.
- 좌표: 자산마다 v0 메시 bbox 로 정한 N 좌표(중심 원점, 최대 변 1; `tools/paper_metrics.py` AssetFrame) 하나에 모든 결과를 옮긴다. raw(R4·editcore) → v0: (p − raw_c)·raw_s(트랙 A·X3 gen.json, 트랙 B·X2-B 항등) → N: (p − t0_c)·t0_s.
- P0: X3 = round.json 의 누적 변환 C_k(v0 → 그 회차 입력 좌표)의 역행렬로 v0 에 되돌림(output_v0.glb). X2-A·X2-B = 입력에 건 정규화 T(prep.json) 좌표 = N(같은 v0 bbox 변환). X4 = 채점과 같이 N 으로 본다(회차마다의 재정규화 표류는 되돌리지 않음 — 실루엣 확인 참조).

| 자산 | 분류 | 정면 az | 카메라 az | 고도 | 근거 |
|---|---|---|---|---|---|
| moto | 사물 | 0 | 45 | 25 | 전조등·라이더 정면 |
| snowman | 캐릭터 | 180 | 180 | 10 | 얼굴·당근 코 |
| chair | 사물 | 0 | 45 | 25 | 좌석 정면 |
| jeep | 사물 | 0 | 45 | 25 | 그릴 |
| cottage | 사물 | 90 | 135 | 25 | 현관문 |
| adventurer | 캐릭터 | 45 | 45 | 10 | 얼굴 정면(X3 마스크 작업의 포크 시점 1) |
| B1 | 사물 | 0 | 45 | 25 | 바구니 앞면 |
| B2 | 사물 | 0 | 45 | 25 | 디저트 앞면 |
| B3 | 캐릭터 | 0 | 0 | 10 | 얼굴 |
| B4 | 사물 | 45 | 90 | 25 | 선풍기 날개 면 |
| B5 | 사물 | 0 | 45 | 25 | 동물들이 보는 쪽 |
| B6 | 캐릭터 | 0 | 0 | 10 | 얼굴 |
| B7 | 사물 | 0 | 45 | 25 | 탁자(대칭, 방위 0) |
| B8 | 사물 | 0 | 45 | 25 | 탑 문 |
| X2B-1 | 캐릭터 | 0 | 0 | 10 | 얼굴 |
| X2B-2 | 사물 | 0 | 45 | 25 | 램프(대칭, 방위 0) |
| X2B-3 | 캐릭터 | 0 | 0 | 10 | 얼굴 |

**좌표 변환(자산 N 좌표):**

| 자산 | v0 원본 | raw_c | raw_s | t0_c | t0_s |
|---|---|---|---|---|---|
| moto | `$ASSETS_DIR/track_a/moto/gen/model.glb` | [-0.000733, -0.000571, 0.000105] | 0.999187 | [-0.0, 0.0, 0.0] | 1.000000 |
| snowman | `$ASSETS_DIR/track_a/snowman/gen/model.glb` | [-9.1e-05, -0.000522, -0.001796] | 1.000787 | [0.0, 0.0, -0.0] | 1.000000 |
| chair | `$ASSETS_DIR/track_a/chair/gen/model.glb` | [-0.000571, -0.000299, 0.002338] | 0.999415 | [0.0, 0.0, 0.0] | 1.000000 |
| jeep | `$ASSETS_DIR/x3/track_a/jeep/gen/model.glb` | [0.000753, -0.001983, -0.003516] | 1.006067 | [-0.0, 0.0, 0.0] | 1.000000 |
| cottage | `$ASSETS_DIR/x3/track_a/cottage/gen/model.glb` | [-0.002229, -0.00023, 0.000307] | 0.999516 | [0.0, 0.0, -0.0] | 1.000000 |
| adventurer | `$ASSETS_DIR/x3/track_a/adventurer/gen/model.glb` | [-0.004236, -8.4e-05, 2e-06] | 1.000236 | [0.0, 0.0, -0.0] | 1.000000 |
| B1 | `$ASSETS_DIR/Edit3D-Bench/data/GSO/RJ_Rabbit_Easter_Basket_Blue/source_model/model.glb` | [0.0, 0.0, 0.0] | 1.000000 | [-0.000833, 0.002178, -0.002196] | 1.005015 |
| B2 | `$ASSETS_DIR/Edit3D-Bench/data/PartObjaverse-Tiny/0c4e4a0b2a374553ae762bb33e8dd2d0/source_model/model.glb` | [0.0, 0.0, 0.0] | 1.000000 | [-0.00092, -0.000194, 0.001497] | 0.999812 |
| B3 | `$ASSETS_DIR/Edit3D-Bench/data/GSO/SpiderMan_Titan_Hero_12Inch_Action_Figure_5Hnn4mtkFsP/source_model/model.glb` | [0.0, 0.0, 0.0] | 1.000000 | [-0.000362, -0.0001, 0.002411] | 1.000452 |
| B4 | `$ASSETS_DIR/Edit3D-Bench/data/PartObjaverse-Tiny/dcf35ae14df947a9a700b8bcff8db57f/source_model/model.glb` | [0.0, 0.0, 0.0] | 1.000000 | [-0.000372, -0.000149, 0.001465] | 1.000981 |
| B5 | `$ASSETS_DIR/Edit3D-Bench/data/GSO/Squirtin_Barnyard_Friends_4pk/source_model/model.glb` | [0.0, 0.0, 0.0] | 1.000000 | [0.000234, -0.000511, 0.000964] | 1.001156 |
| B6 | `$ASSETS_DIR/Edit3D-Bench/data/PartObjaverse-Tiny/97582e1d16f6471bac014081535c26ad/source_model/model.glb` | [0.0, 0.0, 0.0] | 1.000000 | [0.000417, -0.000344, 5e-06] | 0.999983 |
| B7 | `$ASSETS_DIR/Edit3D-Bench/data/GSO/3D_Dollhouse_TablePurple/source_model/model.glb` | [0.0, 0.0, 0.0] | 1.000000 | [0.000449, -0.00044, -3e-06] | 1.001287 |
| B8 | `$ASSETS_DIR/Edit3D-Bench/data/PartObjaverse-Tiny/5fb5b1704b464f5c94b667f1b630bb39/source_model/model.glb` | [0.0, 0.0, 0.0] | 1.000000 | [0.001717, -0.001045, 0.001153] | 1.001219 |
| X2B-1 | `$ASSETS_DIR/Edit3D-Bench/data/GSO/3D_Dollhouse_Happy_Brother/source_model/model.glb` | [0.0, 0.0, 0.0] | 1.000000 | [8e-05, 0.000987, -0.000509] | 1.001753 |
| X2B-2 | `$ASSETS_DIR/Edit3D-Bench/data/GSO/3D_Dollhouse_Lamp/source_model/model.glb` | [0.0, 0.0, 0.0] | 1.000000 | [-0.000467, -6.7e-05, -3.8e-05] | 0.999811 |
| X2B-3 | `$ASSETS_DIR/Edit3D-Bench/data/PartObjaverse-Tiny/00200996b8f34f55a2dd2f44d316d107/source_model/model.glb` | [0.0, 0.0, 0.0] | 1.000000 | [-0.000747, 0.000393, -0.000715] | 0.999080 |

**X3 P0 누적 변환 C_k(v0 → 그 회차 좌표, 렌더는 C_k⁻¹ 적용):** `x3/<자산>/p0/r<k>/` 의 원본 round.json 값.

- jeep r1: scale 1.000000, translate [0.0, 0.0, 0.0]
- jeep r2: scale 1.000751, translate [0.001248, -0.000357, 0.000565]
- jeep r3: scale 1.001228, translate [0.001374, -0.000299, 0.000855]
- jeep r4: scale 1.005959, translate [0.034517, -0.000295, -0.000823]
- jeep r5: scale 1.006371, translate [0.032057, -0.02985, -0.000776]
- cottage r1: scale 1.000000, translate [0.0, 0.0, 0.0]
- cottage r2: scale 0.999652, translate [0.000381, 0.000206, 0.000139]
- cottage r3: scale 0.999337, translate [0.00073, 0.00031, -6.3e-05]
- cottage r4: scale 0.999264, translate [0.000937, 0.000532, -0.000319]
- cottage r5: scale 0.998577, translate [0.00059, 0.000656, -0.000719]
- adventurer r1: scale 1.000000, translate [-0.0, 0.0, 0.0]
- adventurer r2: scale 1.000269, translate [0.000549, 0.000195, -9.7e-05]
- adventurer r3: scale 1.000309, translate [-0.000171, 0.000313, -0.004384]
- adventurer r4: scale 1.000678, translate [-0.00101, 0.00063, -0.004912]
- adventurer r5: scale 1.005339, translate [-0.000969, -0.001363, -0.004699]

**X2-A·X2-B P0 정규화 T(prep.json):**

- p0/A-C1: scale 1.000000, translate [-0.0, -0.0, -0.0]
- p0/A-C2: scale 1.000000, translate [-0.0, -0.0, -0.0]
- p0/A-M1: scale 1.000000, translate [0.0, -0.0, -0.0]
- p0/A-M2: scale 1.000000, translate [0.0, -0.0, -0.0]
- p0/A-S1: scale 1.000000, translate [-0.0, -0.0, 0.0]
- p0/A-S2: scale 1.000000, translate [-0.0, -0.0, 0.0]
- p0/B1: scale 1.005015, translate [0.000837, -0.002189, 0.002207]
- p0/B2: scale 0.999812, translate [0.000919, 0.000194, -0.001496]
- p0/B3: scale 1.000452, translate [0.000363, 0.0001, -0.002412]
- p0/B4: scale 1.000981, translate [0.000373, 0.000149, -0.001466]
- p0/B5: scale 1.001156, translate [-0.000234, 0.000512, -0.000965]
- p0/B6: scale 0.999983, translate [-0.000417, 0.000344, -5e-06]
- p0/B7: scale 1.001287, translate [-0.00045, 0.00044, 3e-06]
- p0/B8: scale 1.001219, translate [-0.001719, 0.001047, -0.001155]
- p0/X2B-1: scale 1.001753, translate [-8.1e-05, -0.000989, 0.00051]
- p0/X2B-2: scale 0.999811, translate [0.000467, 6.7e-05, 3.8e-05]
- p0/X2B-3: scale 0.999080, translate [0.000747, -0.000392, 0.000714]

## 폴더 → 실험·표

| 폴더 | 내용 | 보고서 |
|---|---|---|
| `x3/<자산>/{original/r0, p0, p1, p2}/r<k>/` | X3 연속 5회 편집(P1·P2 = 봉합본), `cond/r<k>/<경로>/` 조건 이미지·prompt.txt | 4절, 요약 S1·S2 |
| `stitch/<자산>/{p1,p2}_prestitch/`, `stitch/jeep/{p1_skirt, p1_stitch_skirt, p1_fulldecode}/` | 봉합 전·치마 벽 시험·통째 디코딩 | 10·10-1절 |
| `x4/<자산>/{p0/r0–r5, p1/r1-5}/` | 무편집 재등록(P1 r1–r5 는 같은 메시라 1회) | 2절, 요약 S5 |
| `x0/<시나리오>/{original, base25/r1, c15_glb/r1, c15_cbin/r1}/` | X0 r1(c15_cbin = R4 + 봉합, 채택본) | 3절, 요약 S4 |
| `x2a/<시나리오>/{original, p0/r1–r2, p1_glb/r1–r3, p1_cbin/r1–r3}/`, `cond/` | X2-A, 동결 조건 이미지 | 5절, 요약 S6 |
| `x2b/<건>/{original, p0, p1}/`, `cond/` | X2-B, 사용한 조건 이미지(데이터셋 2D 또는 FLUX) | 요약 S7 |
| `eval/` | 사람 평가 x3-eval: 평가자가 본 익명 칸 이미지(items/, 칸 A·B·C 그대로), 페이지 캡처(x3_eval_page.png — 평가 뒤에 찍어 저장된 선택이 파란 버튼으로 보인다), items.json, choices.json, blind_key.json(칸 ↔ 경로) | 요약 S3 |
| `sheets/` | 경로 × 회차 시트(라벨), 실루엣 확인(빨강 = 원본만, 파랑 = 결과만, 회색 = 겹침), 시점 확인 | — |
| `img/` | 보고서 그림 원본 해상도 | 각 절 |
| `tables/` | 요약 S1–S7 CSV(S1 빈 자리 열 제거), `raw/` 원값 JSON(경로는 $RUNS_DIR 등으로 바꿈) | 요약 |

## 실루엣 확인(원본 대 각 경로 r1, 알파 IoU)

- 121 쌍, 최소 0.815, 0.90 미만 24 쌍(시점 어긋남 판정 기준 0.90).

- 0.90 미만 쌍은 모두 편집량이 큰 시나리오다. 편집 마스크(∪ halo 2) 투영 화소를 빼고 다시 재면 최소 0.916 — **시점 어긋남 0 건**(낮은 값은 편집으로 실루엣이 바뀐 몫).

| 실험 | 자산·시나리오 | 경로 | 알파 IoU | 편집 영역 밖 알파 IoU |
|---|---|---|---|---|
| x0 | B4 | base25 | 0.815 | 0.916 |
| x2a | B4 | p0 | 0.823 | 0.931 |
| x0 | B4 | c15_glb | 0.831 | 0.934 |
| x2a | B4 | p1_glb | 0.831 | 0.934 |
| x0 | B4 | c15_cbin | 0.831 | 0.935 |
| x2a | B4 | p1_cbin | 0.831 | 0.935 |
| x2a | A-C1 | p0 | 0.843 | 0.971 |
| x0 | A-C1 | c15_glb | 0.845 | 0.984 |
| x2a | A-C1 | p1_glb | 0.845 | 0.984 |
| x0 | A-C1 | c15_cbin | 0.846 | 0.985 |
| x2a | A-C1 | p1_cbin | 0.846 | 0.985 |
| x0 | A-C1 | base25 | 0.854 | 0.989 |
| x0 | A-S1 | c15_cbin | 0.875 | 0.999 |
| x2a | A-S1 | p1_cbin | 0.875 | 0.999 |
| x0 | A-S1 | c15_glb | 0.875 | 0.999 |
| x2a | A-S1 | p1_glb | 0.875 | 0.999 |
| x0 | B5 | base25 | 0.888 | 0.970 |
| x2a | B5 | p0 | 0.889 | 0.974 |
| x2b | X2B-2 | p1 | 0.889 | 0.995 |
| x2b | X2B-2 | p0 | 0.892 | 0.993 |
| x0 | B5 | c15_glb | 0.894 | 0.982 |
| x2a | B5 | p1_glb | 0.894 | 0.982 |
| x0 | B5 | c15_cbin | 0.894 | 0.983 |
| x2a | B5 | p1_cbin | 0.894 | 0.983 |

**X4 P1 r1–r5 output.glb sha256 동일 여부:** moto 다름, snowman 다름, chair 다름, B1 다름, B2 다름, B3 다름, B4 다름, B5 다름, B6 다름, B7 다름, B8 다름

## 없음(기존 출력이 없어 만들지 않은 것)

- 없음

## 편집 영역 가려짐(그 시점에서 편집 영역이 보이지 않음, 시점은 늘리지 않음)

판정: 원본 메시를 그 카메라로 256² 레이캐스팅해 마스크(∪1셀) 셀에 닿는 화소가 50 미만.

- x3/cottage/r2: 보이는 화소 0

## 모델 원본(메시 sha256)

| 실험 | 결과 폴더 | 원본 메시 | 좌표 | sha256 | 비고 |
|---|---|---|---|---|---|
| x3 | `x3/jeep/original/r0` | `$ASSETS_DIR/x3/track_a/jeep/gen/model.glb` | v0 | `228ea8cbe66f2297…` | 생성 텍스처 GLB(A_origin) |
| x3 | `x3/jeep/p0/r1` | `$RUNS_DIR/paper/x3/p0/jeep/r1/output_v0.glb` | v0 | `ebe5b707ebd1da84…` | P0 to_glb, round.json C_k⁻¹ 로 v0 로 되돌림 |
| x3 | `x3/jeep/p1/r1` | `$RUNS_DIR/paper/x3/stitched/p1/jeep/r1/output.glb` | raw + (iv) | `68da8b970f7e3c93…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/jeep/p2/r1` | `$RUNS_DIR/paper/x3/stitched/p2/jeep/r1/output.glb` | raw + (iv) | `81fa3f79a2745c73…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/jeep/p0/r2` | `$RUNS_DIR/paper/x3/p0/jeep/r2/output_v0.glb` | v0 | `92744286c0e49714…` | P0 to_glb, round.json C_k⁻¹ 로 v0 로 되돌림 |
| x3 | `x3/jeep/p1/r2` | `$RUNS_DIR/paper/x3/stitched/p1/jeep/r2/output.glb` | raw + (iv) | `e0a0aaaafdbbc833…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/jeep/p2/r2` | `$RUNS_DIR/paper/x3/stitched/p2/jeep/r2/output.glb` | raw + (iv) | `88329e3a733daf2b…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/jeep/p0/r3` | `$RUNS_DIR/paper/x3/p0/jeep/r3/output_v0.glb` | v0 | `f40a03db972a35a2…` | P0 to_glb, round.json C_k⁻¹ 로 v0 로 되돌림 |
| x3 | `x3/jeep/p1/r3` | `$RUNS_DIR/paper/x3/stitched/p1/jeep/r3/output.glb` | raw + (iv) | `06923d53709849b8…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/jeep/p2/r3` | `$RUNS_DIR/paper/x3/stitched/p2/jeep/r3/output.glb` | raw + (iv) | `5fc70cbd09a67a76…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/jeep/p0/r4` | `$RUNS_DIR/paper/x3/p0/jeep/r4/output_v0.glb` | v0 | `9adaaf1ea03d68a9…` | P0 to_glb, round.json C_k⁻¹ 로 v0 로 되돌림 |
| x3 | `x3/jeep/p1/r4` | `$RUNS_DIR/paper/x3/stitched/p1/jeep/r4/output.glb` | raw + (iv) | `3595226dbda61eaa…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/jeep/p2/r4` | `$RUNS_DIR/paper/x3/stitched/p2/jeep/r4/output.glb` | raw + (iv) | `2a472815358a50d5…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/jeep/p0/r5` | `$RUNS_DIR/paper/x3/p0/jeep/r5/output_v0.glb` | v0 | `76e0931bea901303…` | P0 to_glb, round.json C_k⁻¹ 로 v0 로 되돌림 |
| x3 | `x3/jeep/p1/r5` | `$RUNS_DIR/paper/x3/stitched/p1/jeep/r5/output.glb` | raw + (iv) | `69479ee1868c60df…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/jeep/p2/r5` | `$RUNS_DIR/paper/x3/stitched/p2/jeep/r5/output.glb` | raw + (iv) | `b23997fcf001c41b…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/cottage/original/r0` | `$ASSETS_DIR/x3/track_a/cottage/gen/model.glb` | v0 | `326959e830023df1…` | 생성 텍스처 GLB(A_origin) |
| x3 | `x3/cottage/p0/r1` | `$RUNS_DIR/paper/x3/p0/cottage/r1/output_v0.glb` | v0 | `1b7b209ab83b8720…` | P0 to_glb, round.json C_k⁻¹ 로 v0 로 되돌림 |
| x3 | `x3/cottage/p1/r1` | `$RUNS_DIR/paper/x3/stitched/p1/cottage/r1/output.glb` | raw + (iv) | `fa64bbb6078ae65b…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/cottage/p2/r1` | `$RUNS_DIR/paper/x3/stitched/p2/cottage/r1/output.glb` | raw + (iv) | `425cef9194ebaaac…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/cottage/p0/r2` | `$RUNS_DIR/paper/x3/p0/cottage/r2/output_v0.glb` | v0 | `1410dbfd5fe03fca…` | P0 to_glb, round.json C_k⁻¹ 로 v0 로 되돌림 |
| x3 | `x3/cottage/p1/r2` | `$RUNS_DIR/paper/x3/stitched/p1/cottage/r2/output.glb` | raw + (iv) | `cd14062241a33717…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/cottage/p2/r2` | `$RUNS_DIR/paper/x3/stitched/p2/cottage/r2/output.glb` | raw + (iv) | `4306880444a1e9aa…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/cottage/p0/r3` | `$RUNS_DIR/paper/x3/p0/cottage/r3/output_v0.glb` | v0 | `f85e439c8096485c…` | P0 to_glb, round.json C_k⁻¹ 로 v0 로 되돌림 |
| x3 | `x3/cottage/p1/r3` | `$RUNS_DIR/paper/x3/stitched/p1/cottage/r3/output.glb` | raw + (iv) | `7f38fdb0d02444ea…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/cottage/p2/r3` | `$RUNS_DIR/paper/x3/stitched/p2/cottage/r3/output.glb` | raw + (iv) | `a1a9183d95e66db7…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/cottage/p0/r4` | `$RUNS_DIR/paper/x3/p0/cottage/r4/output_v0.glb` | v0 | `49573d60d7d86770…` | P0 to_glb, round.json C_k⁻¹ 로 v0 로 되돌림 |
| x3 | `x3/cottage/p1/r4` | `$RUNS_DIR/paper/x3/stitched/p1/cottage/r4/output.glb` | raw + (iv) | `dab9c0b99a3d5933…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/cottage/p2/r4` | `$RUNS_DIR/paper/x3/stitched/p2/cottage/r4/output.glb` | raw + (iv) | `c9355ace0d28493d…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/cottage/p0/r5` | `$RUNS_DIR/paper/x3/p0/cottage/r5/output_v0.glb` | v0 | `9520efaeb5dcfa05…` | P0 to_glb, round.json C_k⁻¹ 로 v0 로 되돌림 |
| x3 | `x3/cottage/p1/r5` | `$RUNS_DIR/paper/x3/stitched/p1/cottage/r5/output.glb` | raw + (iv) | `ed7aaa6aef8013d2…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/cottage/p2/r5` | `$RUNS_DIR/paper/x3/stitched/p2/cottage/r5/output.glb` | raw + (iv) | `4c07f6bd1a59cf02…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/adventurer/original/r0` | `$ASSETS_DIR/x3/track_a/adventurer/gen/model.glb` | v0 | `59dcaa7a5d9d8eef…` | 생성 텍스처 GLB(A_origin) |
| x3 | `x3/adventurer/p0/r1` | `$RUNS_DIR/paper/x3/p0/adventurer/r1/output_v0.glb` | v0 | `3a91bfa952079e37…` | P0 to_glb, round.json C_k⁻¹ 로 v0 로 되돌림 |
| x3 | `x3/adventurer/p1/r1` | `$RUNS_DIR/paper/x3/stitched/p1/adventurer/r1/output.glb` | raw + (iv) | `f92aa6817628cb02…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/adventurer/p2/r1` | `$RUNS_DIR/paper/x3/stitched/p2/adventurer/r1/output.glb` | raw + (iv) | `57f24151d6191335…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/adventurer/p0/r2` | `$RUNS_DIR/paper/x3/p0/adventurer/r2/output_v0.glb` | v0 | `0d6c875e4289afa3…` | P0 to_glb, round.json C_k⁻¹ 로 v0 로 되돌림 |
| x3 | `x3/adventurer/p1/r2` | `$RUNS_DIR/paper/x3/stitched/p1/adventurer/r2/output.glb` | raw + (iv) | `3dbc1ad1e1cd4019…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/adventurer/p2/r2` | `$RUNS_DIR/paper/x3/stitched/p2/adventurer/r2/output.glb` | raw + (iv) | `00463e19a91d6e18…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/adventurer/p0/r3` | `$RUNS_DIR/paper/x3/p0/adventurer/r3/output_v0.glb` | v0 | `3d317e4f84bbf761…` | P0 to_glb, round.json C_k⁻¹ 로 v0 로 되돌림 |
| x3 | `x3/adventurer/p1/r3` | `$RUNS_DIR/paper/x3/stitched/p1/adventurer/r3/output.glb` | raw + (iv) | `9890bf4c254a0286…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/adventurer/p2/r3` | `$RUNS_DIR/paper/x3/stitched/p2/adventurer/r3/output.glb` | raw + (iv) | `a8f5800493ff468e…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/adventurer/p0/r4` | `$RUNS_DIR/paper/x3/p0/adventurer/r4/output_v0.glb` | v0 | `b2026a3e7bcc95ca…` | P0 to_glb, round.json C_k⁻¹ 로 v0 로 되돌림 |
| x3 | `x3/adventurer/p1/r4` | `$RUNS_DIR/paper/x3/stitched/p1/adventurer/r4/output.glb` | raw + (iv) | `1439728a8edbe6b5…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/adventurer/p2/r4` | `$RUNS_DIR/paper/x3/stitched/p2/adventurer/r4/output.glb` | raw + (iv) | `1297ed184726c1c3…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/adventurer/p0/r5` | `$RUNS_DIR/paper/x3/p0/adventurer/r5/output_v0.glb` | v0 | `8d72e39d49b74d21…` | P0 to_glb, round.json C_k⁻¹ 로 v0 로 되돌림 |
| x3 | `x3/adventurer/p1/r5` | `$RUNS_DIR/paper/x3/stitched/p1/adventurer/r5/output.glb` | raw + (iv) | `ebbdc676f0710b6a…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| x3 | `x3/adventurer/p2/r5` | `$RUNS_DIR/paper/x3/stitched/p2/adventurer/r5/output.glb` | raw + (iv) | `785eb8e4e6eddf02…` | 봉합본 output.glb + (iv) 보정(= view.glb) |
| stitch | `stitch/jeep/p1_prestitch/r1` | `$RUNS_DIR/paper/x3/p1/jeep/r1/output.glb` | raw + (iv) | `d9c489537991a4c3…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/jeep/p1_prestitch/r2` | `$RUNS_DIR/paper/x3/p1/jeep/r2/output.glb` | raw + (iv) | `2dc00544e6dfda3f…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/jeep/p1_prestitch/r3` | `$RUNS_DIR/paper/x3/p1/jeep/r3/output.glb` | raw + (iv) | `ca7295bfb77c7582…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/jeep/p1_prestitch/r4` | `$RUNS_DIR/paper/x3/p1/jeep/r4/output.glb` | raw + (iv) | `0a920c86df4a65d8…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/jeep/p1_prestitch/r5` | `$RUNS_DIR/paper/x3/p1/jeep/r5/output.glb` | raw + (iv) | `e2e3c14043c11ea9…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/jeep/p2_prestitch/r1` | `$RUNS_DIR/paper/x3/p2/jeep/r1/output.glb` | raw + (iv) | `bf008d173d4f5ea4…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/jeep/p2_prestitch/r2` | `$RUNS_DIR/paper/x3/p2/jeep/r2/output.glb` | raw + (iv) | `7c97b61603b75c9d…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/jeep/p2_prestitch/r3` | `$RUNS_DIR/paper/x3/p2/jeep/r3/output.glb` | raw + (iv) | `9496ed213e952480…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/jeep/p2_prestitch/r4` | `$RUNS_DIR/paper/x3/p2/jeep/r4/output.glb` | raw + (iv) | `c74fea76d9f7d708…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/jeep/p2_prestitch/r5` | `$RUNS_DIR/paper/x3/p2/jeep/r5/output.glb` | raw + (iv) | `e0a53afb8a729ab2…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/cottage/p1_prestitch/r1` | `$RUNS_DIR/paper/x3/p1/cottage/r1/output.glb` | raw + (iv) | `d641250500d25904…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/cottage/p1_prestitch/r2` | `$RUNS_DIR/paper/x3/p1/cottage/r2/output.glb` | raw + (iv) | `6d0d8bea25b15f75…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/cottage/p1_prestitch/r3` | `$RUNS_DIR/paper/x3/p1/cottage/r3/output.glb` | raw + (iv) | `7b5fa38e943e7e8d…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/cottage/p1_prestitch/r4` | `$RUNS_DIR/paper/x3/p1/cottage/r4/output.glb` | raw + (iv) | `d1803c6ca692af20…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/cottage/p1_prestitch/r5` | `$RUNS_DIR/paper/x3/p1/cottage/r5/output.glb` | raw + (iv) | `9eca1cd5d3c878e5…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/cottage/p2_prestitch/r1` | `$RUNS_DIR/paper/x3/p2/cottage/r1/output.glb` | raw + (iv) | `adda71b38a06e743…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/cottage/p2_prestitch/r2` | `$RUNS_DIR/paper/x3/p2/cottage/r2/output.glb` | raw + (iv) | `f882a354af785526…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/cottage/p2_prestitch/r3` | `$RUNS_DIR/paper/x3/p2/cottage/r3/output.glb` | raw + (iv) | `dd2a109d27b03f55…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/cottage/p2_prestitch/r4` | `$RUNS_DIR/paper/x3/p2/cottage/r4/output.glb` | raw + (iv) | `f37336438108b34f…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/cottage/p2_prestitch/r5` | `$RUNS_DIR/paper/x3/p2/cottage/r5/output.glb` | raw + (iv) | `1769aaa24320f0b1…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/adventurer/p1_prestitch/r1` | `$RUNS_DIR/paper/x3/p1/adventurer/r1/output.glb` | raw + (iv) | `2fdf328400b583ba…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/adventurer/p1_prestitch/r2` | `$RUNS_DIR/paper/x3/p1/adventurer/r2/output.glb` | raw + (iv) | `0969b88e0289c5c8…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/adventurer/p1_prestitch/r3` | `$RUNS_DIR/paper/x3/p1/adventurer/r3/output.glb` | raw + (iv) | `035af3fae946739f…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/adventurer/p1_prestitch/r4` | `$RUNS_DIR/paper/x3/p1/adventurer/r4/output.glb` | raw + (iv) | `44d84f7e6b11162c…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/adventurer/p1_prestitch/r5` | `$RUNS_DIR/paper/x3/p1/adventurer/r5/output.glb` | raw + (iv) | `e9ebd36e57d7b6c2…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/adventurer/p2_prestitch/r1` | `$RUNS_DIR/paper/x3/p2/adventurer/r1/output.glb` | raw + (iv) | `ece800f3f7379ebe…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/adventurer/p2_prestitch/r2` | `$RUNS_DIR/paper/x3/p2/adventurer/r2/output.glb` | raw + (iv) | `31624c5b590886dc…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/adventurer/p2_prestitch/r3` | `$RUNS_DIR/paper/x3/p2/adventurer/r3/output.glb` | raw + (iv) | `1046687f9937dc65…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/adventurer/p2_prestitch/r4` | `$RUNS_DIR/paper/x3/p2/adventurer/r4/output.glb` | raw + (iv) | `e979433f096ae76c…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/adventurer/p2_prestitch/r5` | `$RUNS_DIR/paper/x3/p2/adventurer/r5/output.glb` | raw + (iv) | `423ca75c0bb0ba55…` | 봉합 전(스냅만) view.glb |
| stitch | `stitch/jeep/p1_skirt/r1` | `$RUNS_DIR/paper/x3/seamvar/skirt/p1/jeep/r1/output.glb` | raw + (iv) | `7365bc2639106cb7…` | 10-1절 시험 조합 skirt |
| stitch | `stitch/jeep/p1_skirt/r2` | `$RUNS_DIR/paper/x3/seamvar/skirt/p1/jeep/r2/output.glb` | raw + (iv) | `9399cdda258d5b34…` | 10-1절 시험 조합 skirt |
| stitch | `stitch/jeep/p1_skirt/r3` | `$RUNS_DIR/paper/x3/seamvar/skirt/p1/jeep/r3/output.glb` | raw + (iv) | `58645101a7a36e53…` | 10-1절 시험 조합 skirt |
| stitch | `stitch/jeep/p1_skirt/r4` | `$RUNS_DIR/paper/x3/seamvar/skirt/p1/jeep/r4/output.glb` | raw + (iv) | `da10658fc7a32662…` | 10-1절 시험 조합 skirt |
| stitch | `stitch/jeep/p1_skirt/r5` | `$RUNS_DIR/paper/x3/seamvar/skirt/p1/jeep/r5/output.glb` | raw + (iv) | `746750b945325d72…` | 10-1절 시험 조합 skirt |
| stitch | `stitch/jeep/p1_stitch_skirt/r1` | `$RUNS_DIR/paper/x3/seamvar/both/p1/jeep/r1/output.glb` | raw + (iv) | `2fb89f066446e099…` | 10-1절 시험 조합 stitch_skirt |
| stitch | `stitch/jeep/p1_stitch_skirt/r2` | `$RUNS_DIR/paper/x3/seamvar/both/p1/jeep/r2/output.glb` | raw + (iv) | `712317dd22f07029…` | 10-1절 시험 조합 stitch_skirt |
| stitch | `stitch/jeep/p1_stitch_skirt/r3` | `$RUNS_DIR/paper/x3/seamvar/both/p1/jeep/r3/output.glb` | raw + (iv) | `7b68e1e2b1e950b8…` | 10-1절 시험 조합 stitch_skirt |
| stitch | `stitch/jeep/p1_stitch_skirt/r4` | `$RUNS_DIR/paper/x3/seamvar/both/p1/jeep/r4/output.glb` | raw + (iv) | `ffc1938581dd8e30…` | 10-1절 시험 조합 stitch_skirt |
| stitch | `stitch/jeep/p1_stitch_skirt/r5` | `$RUNS_DIR/paper/x3/seamvar/both/p1/jeep/r5/output.glb` | raw + (iv) | `e37e5eee89587d24…` | 10-1절 시험 조합 stitch_skirt |
| stitch | `stitch/jeep/p1_fulldecode/r1` | `$RUNS_DIR/paper/x3/c4/fulldecode/jeep/r1/output.glb` | raw + (iv) | `dd74a9d2cecc32df…` | 승계 없이 통째 디코딩(C4 3) |
| stitch | `stitch/jeep/p1_fulldecode/r2` | `$RUNS_DIR/paper/x3/c4/fulldecode/jeep/r2/output.glb` | raw + (iv) | `cf540580d80860d0…` | 승계 없이 통째 디코딩(C4 3) |
| stitch | `stitch/jeep/p1_fulldecode/r3` | `$RUNS_DIR/paper/x3/c4/fulldecode/jeep/r3/output.glb` | raw + (iv) | `14b06ed486279efc…` | 승계 없이 통째 디코딩(C4 3) |
| stitch | `stitch/jeep/p1_fulldecode/r4` | `$RUNS_DIR/paper/x3/c4/fulldecode/jeep/r4/output.glb` | raw + (iv) | `7d3745054f935301…` | 승계 없이 통째 디코딩(C4 3) |
| stitch | `stitch/jeep/p1_fulldecode/r5` | `$RUNS_DIR/paper/x3/c4/fulldecode/jeep/r5/output.glb` | raw + (iv) | `ec7b94d4fbbddd73…` | 승계 없이 통째 디코딩(C4 3) |
| x4 | `x4/moto/p0/r0` | `$ASSETS_DIR/track_a/moto/gen/model.glb` | v0 | `ea993b04070bdcfe…` | 원본(r0) |
| x4 | `x4/moto/p0/r1` | `$RUNS_DIR/paper/x4/p0/moto/r1/output.glb` | n | `0a5eb7bb3d88dda0…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/moto/p0/r2` | `$RUNS_DIR/paper/x4/p0/moto/r2/output.glb` | n | `ebbd1729d621896c…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/moto/p0/r3` | `$RUNS_DIR/paper/x4/p0/moto/r3/output.glb` | n | `cbd975c4b1951a17…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/moto/p0/r4` | `$RUNS_DIR/paper/x4/p0/moto/r4/output.glb` | n | `4104f9d54df0cfe8…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/moto/p0/r5` | `$RUNS_DIR/paper/x4/p0/moto/r5/output.glb` | n | `f0ac90da084ba44a…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/moto/p1/r1-5` | `$RUNS_DIR/paper/x4/p1/moto/r1/output.glb` | raw + (iv) | `4e7ffe2ef0134fb8…` | P1 view.glb(r1–r5 같은 메시) |
| x4 | `x4/snowman/p0/r0` | `$ASSETS_DIR/track_a/snowman/gen/model.glb` | v0 | `e072dca0f94291d6…` | 원본(r0) |
| x4 | `x4/snowman/p0/r1` | `$RUNS_DIR/paper/x4/p0/snowman/r1/output.glb` | n | `e8f718128abc8c35…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/snowman/p0/r2` | `$RUNS_DIR/paper/x4/p0/snowman/r2/output.glb` | n | `12d3ce82e303646e…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/snowman/p0/r3` | `$RUNS_DIR/paper/x4/p0/snowman/r3/output.glb` | n | `5158e3d05503be4b…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/snowman/p0/r4` | `$RUNS_DIR/paper/x4/p0/snowman/r4/output.glb` | n | `fa98c6de31577c65…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/snowman/p0/r5` | `$RUNS_DIR/paper/x4/p0/snowman/r5/output.glb` | n | `8cbce56676b0b708…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/snowman/p1/r1-5` | `$RUNS_DIR/paper/x4/p1/snowman/r1/output.glb` | raw + (iv) | `23c62840a82e7f58…` | P1 view.glb(r1–r5 같은 메시) |
| x4 | `x4/chair/p0/r0` | `$ASSETS_DIR/track_a/chair/gen/model.glb` | v0 | `67595c621826fee7…` | 원본(r0) |
| x4 | `x4/chair/p0/r1` | `$RUNS_DIR/paper/x4/p0/chair/r1/output.glb` | n | `5890e7a22030801f…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/chair/p0/r2` | `$RUNS_DIR/paper/x4/p0/chair/r2/output.glb` | n | `4c235ecff3c612a1…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/chair/p0/r3` | `$RUNS_DIR/paper/x4/p0/chair/r3/output.glb` | n | `37fc3e63faf438c7…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/chair/p0/r4` | `$RUNS_DIR/paper/x4/p0/chair/r4/output.glb` | n | `965b0ed781fdbff8…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/chair/p0/r5` | `$RUNS_DIR/paper/x4/p0/chair/r5/output.glb` | n | `71cdbe0db5d65211…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/chair/p1/r1-5` | `$RUNS_DIR/paper/x4/p1/chair/r1/output.glb` | raw + (iv) | `0ffc306dcac60a4b…` | P1 view.glb(r1–r5 같은 메시) |
| x4 | `x4/B1/p0/r0` | `$ASSETS_DIR/Edit3D-Bench/data/GSO/RJ_Rabbit_Easter_Basket_Blue/source_model/model.glb` | v0 | `2216fce000dd0910…` | 원본(r0) |
| x4 | `x4/B1/p0/r1` | `$RUNS_DIR/paper/x4/p0/B1/r1/output.glb` | n | `2b3c8f553400ad38…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B1/p0/r2` | `$RUNS_DIR/paper/x4/p0/B1/r2/output.glb` | n | `b5192ed855346d3b…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B1/p0/r3` | `$RUNS_DIR/paper/x4/p0/B1/r3/output.glb` | n | `dd57e5a3b1dabbda…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B1/p0/r4` | `$RUNS_DIR/paper/x4/p0/B1/r4/output.glb` | n | `091047ee9ae94283…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B1/p0/r5` | `$RUNS_DIR/paper/x4/p0/B1/r5/output.glb` | n | `a46ef8066450729b…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B1/p1/r1-5` | `$RUNS_DIR/paper/x4/p1/B1/r1/output.glb` | raw + (iv) | `a08513da82d998d8…` | P1 view.glb(r1–r5 같은 메시) |
| x4 | `x4/B2/p0/r0` | `$ASSETS_DIR/Edit3D-Bench/data/PartObjaverse-Tiny/0c4e4a0b2a374553ae762bb33e8dd2d0/source_model/model.glb` | v0 | `d9f6045824219335…` | 원본(r0) |
| x4 | `x4/B2/p0/r1` | `$RUNS_DIR/paper/x4/p0/B2/r1/output.glb` | n | `0009d93916e050cf…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B2/p0/r2` | `$RUNS_DIR/paper/x4/p0/B2/r2/output.glb` | n | `f0c644853c57d0b0…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B2/p0/r3` | `$RUNS_DIR/paper/x4/p0/B2/r3/output.glb` | n | `22bad20f3469aeca…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B2/p0/r4` | `$RUNS_DIR/paper/x4/p0/B2/r4/output.glb` | n | `be586afd06b9bbd6…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B2/p0/r5` | `$RUNS_DIR/paper/x4/p0/B2/r5/output.glb` | n | `c7b56022f4a73273…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B2/p1/r1-5` | `$RUNS_DIR/paper/x4/p1/B2/r1/output.glb` | raw + (iv) | `8b6ac6197ed66833…` | P1 view.glb(r1–r5 같은 메시) |
| x4 | `x4/B3/p0/r0` | `$ASSETS_DIR/Edit3D-Bench/data/GSO/SpiderMan_Titan_Hero_12Inch_Action_Figure_5Hnn4mtkFsP/source_model/model.glb` | v0 | `8b33108dcdf66441…` | 원본(r0) |
| x4 | `x4/B3/p0/r1` | `$RUNS_DIR/paper/x4/p0/B3/r1/output.glb` | n | `3f29072348d7cdea…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B3/p0/r2` | `$RUNS_DIR/paper/x4/p0/B3/r2/output.glb` | n | `3d8b224d01fd1c42…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B3/p0/r3` | `$RUNS_DIR/paper/x4/p0/B3/r3/output.glb` | n | `c1df49f2a65dd896…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B3/p0/r4` | `$RUNS_DIR/paper/x4/p0/B3/r4/output.glb` | n | `71832549c757f86b…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B3/p0/r5` | `$RUNS_DIR/paper/x4/p0/B3/r5/output.glb` | n | `211ba3af221c90f1…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B3/p1/r1-5` | `$RUNS_DIR/paper/x4/p1/B3/r1/output.glb` | raw + (iv) | `3995b05722e09512…` | P1 view.glb(r1–r5 같은 메시) |
| x4 | `x4/B4/p0/r0` | `$ASSETS_DIR/Edit3D-Bench/data/PartObjaverse-Tiny/dcf35ae14df947a9a700b8bcff8db57f/source_model/model.glb` | v0 | `0d431158b0695c2d…` | 원본(r0) |
| x4 | `x4/B4/p0/r1` | `$RUNS_DIR/paper/x4/p0/B4/r1/output.glb` | n | `cacae0528502f3d2…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B4/p0/r2` | `$RUNS_DIR/paper/x4/p0/B4/r2/output.glb` | n | `f902e88385a77a31…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B4/p0/r3` | `$RUNS_DIR/paper/x4/p0/B4/r3/output.glb` | n | `281f21c9029b6134…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B4/p0/r4` | `$RUNS_DIR/paper/x4/p0/B4/r4/output.glb` | n | `1db1da70ec273d84…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B4/p0/r5` | `$RUNS_DIR/paper/x4/p0/B4/r5/output.glb` | n | `1734ac353785f2e7…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B4/p1/r1-5` | `$RUNS_DIR/paper/x4/p1/B4/r1/output.glb` | raw + (iv) | `834cdba14951f4d3…` | P1 view.glb(r1–r5 같은 메시) |
| x4 | `x4/B5/p0/r0` | `$ASSETS_DIR/Edit3D-Bench/data/GSO/Squirtin_Barnyard_Friends_4pk/source_model/model.glb` | v0 | `9fa2730506afa339…` | 원본(r0) |
| x4 | `x4/B5/p0/r1` | `$RUNS_DIR/paper/x4/p0/B5/r1/output.glb` | n | `de0249dad302ec7b…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B5/p0/r2` | `$RUNS_DIR/paper/x4/p0/B5/r2/output.glb` | n | `16c26f0e33a02424…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B5/p0/r3` | `$RUNS_DIR/paper/x4/p0/B5/r3/output.glb` | n | `8854949fa8852887…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B5/p0/r4` | `$RUNS_DIR/paper/x4/p0/B5/r4/output.glb` | n | `f6723dbf55bf5d4a…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B5/p0/r5` | `$RUNS_DIR/paper/x4/p0/B5/r5/output.glb` | n | `720ab57a858a420b…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B5/p1/r1-5` | `$RUNS_DIR/paper/x4/p1/B5/r1/output.glb` | raw + (iv) | `7c4b5b0d9d966dfc…` | P1 view.glb(r1–r5 같은 메시) |
| x4 | `x4/B6/p0/r0` | `$ASSETS_DIR/Edit3D-Bench/data/PartObjaverse-Tiny/97582e1d16f6471bac014081535c26ad/source_model/model.glb` | v0 | `a0b1d50d5c781ba7…` | 원본(r0) |
| x4 | `x4/B6/p0/r1` | `$RUNS_DIR/paper/x4/p0/B6/r1/output.glb` | n | `bd7b8dc4b9be7148…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B6/p0/r2` | `$RUNS_DIR/paper/x4/p0/B6/r2/output.glb` | n | `16878a2eded587fa…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B6/p0/r3` | `$RUNS_DIR/paper/x4/p0/B6/r3/output.glb` | n | `ed539a1e948013c4…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B6/p0/r4` | `$RUNS_DIR/paper/x4/p0/B6/r4/output.glb` | n | `24cb760a5d89014b…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B6/p0/r5` | `$RUNS_DIR/paper/x4/p0/B6/r5/output.glb` | n | `e52ca83d3b3e9246…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B6/p1/r1-5` | `$RUNS_DIR/paper/x4/p1/B6/r1/output.glb` | raw + (iv) | `ad1677ff3723d0dc…` | P1 view.glb(r1–r5 같은 메시) |
| x4 | `x4/B7/p0/r0` | `$ASSETS_DIR/Edit3D-Bench/data/GSO/3D_Dollhouse_TablePurple/source_model/model.glb` | v0 | `8afad6ac7d16aa6f…` | 원본(r0) |
| x4 | `x4/B7/p0/r1` | `$RUNS_DIR/paper/x4/p0/B7/r1/output.glb` | n | `61b811c2c0820eb8…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B7/p0/r2` | `$RUNS_DIR/paper/x4/p0/B7/r2/output.glb` | n | `f35f6a4353eb4c0a…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B7/p0/r3` | `$RUNS_DIR/paper/x4/p0/B7/r3/output.glb` | n | `004c5822d0879370…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B7/p0/r4` | `$RUNS_DIR/paper/x4/p0/B7/r4/output.glb` | n | `4b4bd78f78565fe0…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B7/p0/r5` | `$RUNS_DIR/paper/x4/p0/B7/r5/output.glb` | n | `e8aba2c23825e9a5…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B7/p1/r1-5` | `$RUNS_DIR/paper/x4/p1/B7/r1/output.glb` | raw + (iv) | `8629ff713e882369…` | P1 view.glb(r1–r5 같은 메시) |
| x4 | `x4/B8/p0/r0` | `$ASSETS_DIR/Edit3D-Bench/data/PartObjaverse-Tiny/5fb5b1704b464f5c94b667f1b630bb39/source_model/model.glb` | v0 | `3190d9403dc02279…` | 원본(r0) |
| x4 | `x4/B8/p0/r1` | `$RUNS_DIR/paper/x4/p0/B8/r1/output.glb` | n | `c422bed335122fc2…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B8/p0/r2` | `$RUNS_DIR/paper/x4/p0/B8/r2/output.glb` | n | `95e636facd030f5b…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B8/p0/r3` | `$RUNS_DIR/paper/x4/p0/B8/r3/output.glb` | n | `77e2c5b09b16b08e…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B8/p0/r4` | `$RUNS_DIR/paper/x4/p0/B8/r4/output.glb` | n | `4100237f5978bb55…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B8/p0/r5` | `$RUNS_DIR/paper/x4/p0/B8/r5/output.glb` | n | `0a8ac81408612360…` | P0 to_glb(N 좌표, x4_score 와 같음) |
| x4 | `x4/B8/p1/r1-5` | `$RUNS_DIR/paper/x4/p1/B8/r1/output.glb` | raw + (iv) | `6a9b8a2ec9a68d76…` | P1 view.glb(r1–r5 같은 메시) |
| x0 | `x0/A-M1/original` | `$ASSETS_DIR/track_a/moto/gen/model.glb` | v0 | `ea993b04070bdcfe…` | 원본 v0 |
| x0 | `x0/A-M1/base25/r1` | `$RUNS_DIR/01-port/track_a-A-M1/port-r1/output.glb` | raw | `d2cc038ff0d028c6…` | 25 스텝 to_glb(0-1 이식본 r1) |
| x0 | `x0/A-M1/c15_glb/r1` | `$RUNS_DIR/2026-10-25/track_a-A-M1/e2a-c15/r1/output.glb` | raw | `354ca8f4d16dc075…` | 15 스텝 to_glb(E2a c15 r1) |
| x0 | `x0/A-M1/c15_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/A-M1/r1/output.glb` | raw + (iv) | `58d37ae303f5427e…` | 15 스텝 R4 .cbin + 봉합(채택), 같은 slat_tgt |
| x0 | `x0/A-M2/original` | `$ASSETS_DIR/track_a/moto/gen/model.glb` | v0 | `ea993b04070bdcfe…` | 원본 v0 |
| x0 | `x0/A-M2/base25/r1` | `$RUNS_DIR/01-port/track_a-A-M2/port-r1/output.glb` | raw | `bbe72daed6dd5483…` | 25 스텝 to_glb(0-1 이식본 r1) |
| x0 | `x0/A-M2/c15_glb/r1` | `$RUNS_DIR/2026-10-25/track_a-A-M2/e2a-c15/r1/output.glb` | raw | `3db06a2aea95c04d…` | 15 스텝 to_glb(E2a c15 r1) |
| x0 | `x0/A-M2/c15_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/A-M2/r1/output.glb` | raw + (iv) | `fe65470e67be91c6…` | 15 스텝 R4 .cbin + 봉합(채택), 같은 slat_tgt |
| x0 | `x0/A-S1/original` | `$ASSETS_DIR/track_a/snowman/gen/model.glb` | v0 | `e072dca0f94291d6…` | 원본 v0 |
| x0 | `x0/A-S1/base25/r1` | `$RUNS_DIR/01-port/track_a-A-S1/port-r1/output.glb` | raw | `abfad41cd9c5a25d…` | 25 스텝 to_glb(0-1 이식본 r1) |
| x0 | `x0/A-S1/c15_glb/r1` | `$RUNS_DIR/2026-10-25/track_a-A-S1/e2a-c15/r1/output.glb` | raw | `d5c2af8aeca2a71f…` | 15 스텝 to_glb(E2a c15 r1) |
| x0 | `x0/A-S1/c15_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/A-S1/r1/output.glb` | raw + (iv) | `73273917d0a8720e…` | 15 스텝 R4 .cbin + 봉합(채택), 같은 slat_tgt |
| x0 | `x0/A-S2/original` | `$ASSETS_DIR/track_a/snowman/gen/model.glb` | v0 | `e072dca0f94291d6…` | 원본 v0 |
| x0 | `x0/A-S2/base25/r1` | `$RUNS_DIR/01-port/track_a-A-S2/port-r1/output.glb` | raw | `cfe25093e2f0c32b…` | 25 스텝 to_glb(0-1 이식본 r1) |
| x0 | `x0/A-S2/c15_glb/r1` | `$RUNS_DIR/2026-10-25/track_a-A-S2/e2a-c15/r1/output.glb` | raw | `8e5596854bcd73d3…` | 15 스텝 to_glb(E2a c15 r1) |
| x0 | `x0/A-S2/c15_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/A-S2/r1/output.glb` | raw + (iv) | `cd49d09517665643…` | 15 스텝 R4 .cbin + 봉합(채택), 같은 slat_tgt |
| x0 | `x0/A-C1/original` | `$ASSETS_DIR/track_a/chair/gen/model.glb` | v0 | `67595c621826fee7…` | 원본 v0 |
| x0 | `x0/A-C1/base25/r1` | `$RUNS_DIR/01-port/track_a-A-C1/port-r1/output.glb` | raw | `93fe25894038679b…` | 25 스텝 to_glb(0-1 이식본 r1) |
| x0 | `x0/A-C1/c15_glb/r1` | `$RUNS_DIR/2026-10-25/track_a-A-C1/e2a-c15/r1/output.glb` | raw | `eac5447f0eb5e122…` | 15 스텝 to_glb(E2a c15 r1) |
| x0 | `x0/A-C1/c15_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/A-C1/r1/output.glb` | raw + (iv) | `227cbfd581dd48a5…` | 15 스텝 R4 .cbin + 봉합(채택), 같은 slat_tgt |
| x0 | `x0/A-C2/original` | `$ASSETS_DIR/track_a/chair/gen/model.glb` | v0 | `67595c621826fee7…` | 원본 v0 |
| x0 | `x0/A-C2/base25/r1` | `$RUNS_DIR/01-port/track_a-A-C2/port-r1/output.glb` | raw | `ca9f2396f85cac2f…` | 25 스텝 to_glb(0-1 이식본 r1) |
| x0 | `x0/A-C2/c15_glb/r1` | `$RUNS_DIR/2026-10-25/track_a-A-C2/e2a-c15/r1/output.glb` | raw | `481f0be4155572f0…` | 15 스텝 to_glb(E2a c15 r1) |
| x0 | `x0/A-C2/c15_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/A-C2/r1/output.glb` | raw + (iv) | `d3a865475996444a…` | 15 스텝 R4 .cbin + 봉합(채택), 같은 slat_tgt |
| x0 | `x0/B1/original` | `$ASSETS_DIR/Edit3D-Bench/data/GSO/RJ_Rabbit_Easter_Basket_Blue/source_model/model.glb` | v0 | `2216fce000dd0910…` | 원본 v0 |
| x0 | `x0/B1/base25/r1` | `$RUNS_DIR/01-port/track_b-B1/port-r1/output.glb` | raw | `12bb6bcf14cb6d58…` | 25 스텝 to_glb(0-1 이식본 r1) |
| x0 | `x0/B1/c15_glb/r1` | `$RUNS_DIR/2026-10-25/track_b-B1/e2a-c15/r1/output.glb` | raw | `8c328753f828554e…` | 15 스텝 to_glb(E2a c15 r1) |
| x0 | `x0/B1/c15_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/B1/r1/output.glb` | raw + (iv) | `9a2b3d6e7d00a34a…` | 15 스텝 R4 .cbin + 봉합(채택), 같은 slat_tgt |
| x0 | `x0/B2/original` | `$ASSETS_DIR/Edit3D-Bench/data/PartObjaverse-Tiny/0c4e4a0b2a374553ae762bb33e8dd2d0/source_model/model.glb` | v0 | `d9f6045824219335…` | 원본 v0 |
| x0 | `x0/B2/base25/r1` | `$RUNS_DIR/01-port/track_b-B2/port-r1/output.glb` | raw | `42a75801d1769dbd…` | 25 스텝 to_glb(0-1 이식본 r1) |
| x0 | `x0/B2/c15_glb/r1` | `$RUNS_DIR/2026-10-25/track_b-B2/e2a-c15/r1/output.glb` | raw | `5ec99d4e62dfadf3…` | 15 스텝 to_glb(E2a c15 r1) |
| x0 | `x0/B2/c15_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/B2/r1/output.glb` | raw + (iv) | `d802cd53e8a4849f…` | 15 스텝 R4 .cbin + 봉합(채택), 같은 slat_tgt |
| x0 | `x0/B3/original` | `$ASSETS_DIR/Edit3D-Bench/data/GSO/SpiderMan_Titan_Hero_12Inch_Action_Figure_5Hnn4mtkFsP/source_model/model.glb` | v0 | `8b33108dcdf66441…` | 원본 v0 |
| x0 | `x0/B3/base25/r1` | `$RUNS_DIR/01-port/track_b-B3/port-r1/output.glb` | raw | `c9fd395aafcdab5d…` | 25 스텝 to_glb(0-1 이식본 r1) |
| x0 | `x0/B3/c15_glb/r1` | `$RUNS_DIR/2026-10-25/track_b-B3/e2a-c15/r1/output.glb` | raw | `2882e935eb59286e…` | 15 스텝 to_glb(E2a c15 r1) |
| x0 | `x0/B3/c15_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/B3/r1/output.glb` | raw + (iv) | `b8c314d007465d20…` | 15 스텝 R4 .cbin + 봉합(채택), 같은 slat_tgt |
| x0 | `x0/B4/original` | `$ASSETS_DIR/Edit3D-Bench/data/PartObjaverse-Tiny/dcf35ae14df947a9a700b8bcff8db57f/source_model/model.glb` | v0 | `0d431158b0695c2d…` | 원본 v0 |
| x0 | `x0/B4/base25/r1` | `$RUNS_DIR/01-port/track_b-B4/port-r1/output.glb` | raw | `87a8678a6da84cd4…` | 25 스텝 to_glb(0-1 이식본 r1) |
| x0 | `x0/B4/c15_glb/r1` | `$RUNS_DIR/2026-10-25/track_b-B4/e2a-c15/r1/output.glb` | raw | `c6a581905c9eea97…` | 15 스텝 to_glb(E2a c15 r1) |
| x0 | `x0/B4/c15_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/B4/r1/output.glb` | raw + (iv) | `c5c218811254e42b…` | 15 스텝 R4 .cbin + 봉합(채택), 같은 slat_tgt |
| x0 | `x0/B5/original` | `$ASSETS_DIR/Edit3D-Bench/data/GSO/Squirtin_Barnyard_Friends_4pk/source_model/model.glb` | v0 | `9fa2730506afa339…` | 원본 v0 |
| x0 | `x0/B5/base25/r1` | `$RUNS_DIR/01-port/track_b-B5/port-r1/output.glb` | raw | `40ff7cc822ba610e…` | 25 스텝 to_glb(0-1 이식본 r1) |
| x0 | `x0/B5/c15_glb/r1` | `$RUNS_DIR/2026-10-25/track_b-B5/e2a-c15/r1/output.glb` | raw | `adeb32293baef7e8…` | 15 스텝 to_glb(E2a c15 r1) |
| x0 | `x0/B5/c15_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/B5/r1/output.glb` | raw + (iv) | `9c737b9ee3f6bfa7…` | 15 스텝 R4 .cbin + 봉합(채택), 같은 slat_tgt |
| x0 | `x0/B6/original` | `$ASSETS_DIR/Edit3D-Bench/data/PartObjaverse-Tiny/97582e1d16f6471bac014081535c26ad/source_model/model.glb` | v0 | `a0b1d50d5c781ba7…` | 원본 v0 |
| x0 | `x0/B6/base25/r1` | `$RUNS_DIR/01-port/track_b-B6/port-r1/output.glb` | raw | `0d3000f28062554b…` | 25 스텝 to_glb(0-1 이식본 r1) |
| x0 | `x0/B6/c15_glb/r1` | `$RUNS_DIR/2026-10-25/track_b-B6/e2a-c15/r1/output.glb` | raw | `68b7fdceb3a44dee…` | 15 스텝 to_glb(E2a c15 r1) |
| x0 | `x0/B6/c15_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/B6/r1/output.glb` | raw + (iv) | `36e02a432ffb5056…` | 15 스텝 R4 .cbin + 봉합(채택), 같은 slat_tgt |
| x0 | `x0/B7/original` | `$ASSETS_DIR/Edit3D-Bench/data/GSO/3D_Dollhouse_TablePurple/source_model/model.glb` | v0 | `8afad6ac7d16aa6f…` | 원본 v0 |
| x0 | `x0/B7/base25/r1` | `$RUNS_DIR/01-port/track_b-B7/port-r1/output.glb` | raw | `8a69b662fa0b094a…` | 25 스텝 to_glb(0-1 이식본 r1) |
| x0 | `x0/B7/c15_glb/r1` | `$RUNS_DIR/2026-10-25/track_b-B7/e2a-c15/r1/output.glb` | raw | `da7abd364055bad7…` | 15 스텝 to_glb(E2a c15 r1) |
| x0 | `x0/B7/c15_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/B7/r1/output.glb` | raw + (iv) | `f0f8dfbcd235e27c…` | 15 스텝 R4 .cbin + 봉합(채택), 같은 slat_tgt |
| x0 | `x0/B8/original` | `$ASSETS_DIR/Edit3D-Bench/data/PartObjaverse-Tiny/5fb5b1704b464f5c94b667f1b630bb39/source_model/model.glb` | v0 | `3190d9403dc02279…` | 원본 v0 |
| x0 | `x0/B8/base25/r1` | `$RUNS_DIR/01-port/track_b-B8/port-r1/output.glb` | raw | `43a5ab116bbb6f80…` | 25 스텝 to_glb(0-1 이식본 r1) |
| x0 | `x0/B8/c15_glb/r1` | `$RUNS_DIR/2026-10-25/track_b-B8/e2a-c15/r1/output.glb` | raw | `0e45a545a6292f4b…` | 15 스텝 to_glb(E2a c15 r1) |
| x0 | `x0/B8/c15_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/B8/r1/output.glb` | raw + (iv) | `559d918e01a295ed…` | 15 스텝 R4 .cbin + 봉합(채택), 같은 slat_tgt |
| x2a | `x2a/A-M1/original` | `$ASSETS_DIR/track_a/moto/gen/model.glb` | v0 | `ea993b04070bdcfe…` | 원본 v0 |
| x2a | `x2a/A-M1/p0/r1` | `$RUNS_DIR/paper/x2a/p0/A-M1/r1/output.glb` | n | `c81ecc4f0d23d4f9…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/A-M1/p0/r2` | `$RUNS_DIR/paper/x2a/p0/A-M1/r2/output.glb` | n | `00a9c4e5f85664fa…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/A-M1/p1_glb/r1` | `$RUNS_DIR/2026-10-25/track_a-A-M1/e2a-c15/r1/output.glb` | raw | `354ca8f4d16dc075…` | P1-glb(E2a c15) |
| x2a | `x2a/A-M1/p1_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/A-M1/r1/output.glb` | raw + (iv) | `58d37ae303f5427e…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/A-M1/p1_glb/r2` | `$RUNS_DIR/2026-10-25/track_a-A-M1/e2a-c15/r2/output.glb` | raw | `bd0443ec0516662c…` | P1-glb(E2a c15) |
| x2a | `x2a/A-M1/p1_cbin/r2` | `$RUNS_DIR/paper/x0/cbin_stitched/A-M1/r2/output.glb` | raw + (iv) | `95b8a51bf22bb93a…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/A-M1/p1_glb/r3` | `$RUNS_DIR/2026-10-25/track_a-A-M1/e2a-c15/r3/output.glb` | raw | `92f8f21758613a72…` | P1-glb(E2a c15) |
| x2a | `x2a/A-M1/p1_cbin/r3` | `$RUNS_DIR/paper/x0/cbin_stitched/A-M1/r3/output.glb` | raw + (iv) | `98255ff57b07eab4…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/A-M2/original` | `$ASSETS_DIR/track_a/moto/gen/model.glb` | v0 | `ea993b04070bdcfe…` | 원본 v0 |
| x2a | `x2a/A-M2/p0/r1` | `$RUNS_DIR/paper/x2a/p0/A-M2/r1/output.glb` | n | `a08bca4d63cc3909…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/A-M2/p0/r2` | `$RUNS_DIR/paper/x2a/p0/A-M2/r2/output.glb` | n | `a5c7e43a7f761e15…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/A-M2/p1_glb/r1` | `$RUNS_DIR/2026-10-25/track_a-A-M2/e2a-c15/r1/output.glb` | raw | `3db06a2aea95c04d…` | P1-glb(E2a c15) |
| x2a | `x2a/A-M2/p1_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/A-M2/r1/output.glb` | raw + (iv) | `fe65470e67be91c6…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/A-M2/p1_glb/r2` | `$RUNS_DIR/2026-10-25/track_a-A-M2/e2a-c15/r2/output.glb` | raw | `7720b051508847b6…` | P1-glb(E2a c15) |
| x2a | `x2a/A-M2/p1_cbin/r2` | `$RUNS_DIR/paper/x0/cbin_stitched/A-M2/r2/output.glb` | raw + (iv) | `d2657c0b045c3ccf…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/A-M2/p1_glb/r3` | `$RUNS_DIR/2026-10-25/track_a-A-M2/e2a-c15/r3/output.glb` | raw | `2e6394acd8b87a2d…` | P1-glb(E2a c15) |
| x2a | `x2a/A-M2/p1_cbin/r3` | `$RUNS_DIR/paper/x0/cbin_stitched/A-M2/r3/output.glb` | raw + (iv) | `1e1cf2beb8d93630…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/A-S1/original` | `$ASSETS_DIR/track_a/snowman/gen/model.glb` | v0 | `e072dca0f94291d6…` | 원본 v0 |
| x2a | `x2a/A-S1/p0/r1` | `$RUNS_DIR/paper/x2a/p0/A-S1/r1/output.glb` | n | `501e319460dcacce…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/A-S1/p0/r2` | `$RUNS_DIR/paper/x2a/p0/A-S1/r2/output.glb` | n | `73d07adbe889ff00…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/A-S1/p1_glb/r1` | `$RUNS_DIR/2026-10-25/track_a-A-S1/e2a-c15/r1/output.glb` | raw | `d5c2af8aeca2a71f…` | P1-glb(E2a c15) |
| x2a | `x2a/A-S1/p1_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/A-S1/r1/output.glb` | raw + (iv) | `73273917d0a8720e…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/A-S1/p1_glb/r2` | `$RUNS_DIR/2026-10-25/track_a-A-S1/e2a-c15/r2/output.glb` | raw | `2f312c8053077dc3…` | P1-glb(E2a c15) |
| x2a | `x2a/A-S1/p1_cbin/r2` | `$RUNS_DIR/paper/x0/cbin_stitched/A-S1/r2/output.glb` | raw + (iv) | `6db1feb758d3df61…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/A-S1/p1_glb/r3` | `$RUNS_DIR/2026-10-25/track_a-A-S1/e2a-c15/r3/output.glb` | raw | `ebed3efc3d1a34df…` | P1-glb(E2a c15) |
| x2a | `x2a/A-S1/p1_cbin/r3` | `$RUNS_DIR/paper/x0/cbin_stitched/A-S1/r3/output.glb` | raw + (iv) | `97401e5b581c3e15…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/A-S2/original` | `$ASSETS_DIR/track_a/snowman/gen/model.glb` | v0 | `e072dca0f94291d6…` | 원본 v0 |
| x2a | `x2a/A-S2/p0/r1` | `$RUNS_DIR/paper/x2a/p0/A-S2/r1/output.glb` | n | `ba358e5b51b73ce3…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/A-S2/p0/r2` | `$RUNS_DIR/paper/x2a/p0/A-S2/r2/output.glb` | n | `114a12cc63e12456…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/A-S2/p1_glb/r1` | `$RUNS_DIR/2026-10-25/track_a-A-S2/e2a-c15/r1/output.glb` | raw | `8e5596854bcd73d3…` | P1-glb(E2a c15) |
| x2a | `x2a/A-S2/p1_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/A-S2/r1/output.glb` | raw + (iv) | `cd49d09517665643…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/A-S2/p1_glb/r2` | `$RUNS_DIR/2026-10-25/track_a-A-S2/e2a-c15/r2/output.glb` | raw | `745823ecf042dcc6…` | P1-glb(E2a c15) |
| x2a | `x2a/A-S2/p1_cbin/r2` | `$RUNS_DIR/paper/x0/cbin_stitched/A-S2/r2/output.glb` | raw + (iv) | `83eafe878f141b4a…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/A-S2/p1_glb/r3` | `$RUNS_DIR/2026-10-25/track_a-A-S2/e2a-c15/r3/output.glb` | raw | `e2ec0d9670e306cb…` | P1-glb(E2a c15) |
| x2a | `x2a/A-S2/p1_cbin/r3` | `$RUNS_DIR/paper/x0/cbin_stitched/A-S2/r3/output.glb` | raw + (iv) | `46ab301977d38c05…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/A-C1/original` | `$ASSETS_DIR/track_a/chair/gen/model.glb` | v0 | `67595c621826fee7…` | 원본 v0 |
| x2a | `x2a/A-C1/p0/r1` | `$RUNS_DIR/paper/x2a/p0/A-C1/r1/output.glb` | n | `493448e2ecf2102f…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/A-C1/p0/r2` | `$RUNS_DIR/paper/x2a/p0/A-C1/r2/output.glb` | n | `77323ae1ff9b5b5d…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/A-C1/p1_glb/r1` | `$RUNS_DIR/2026-10-25/track_a-A-C1/e2a-c15/r1/output.glb` | raw | `eac5447f0eb5e122…` | P1-glb(E2a c15) |
| x2a | `x2a/A-C1/p1_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/A-C1/r1/output.glb` | raw + (iv) | `227cbfd581dd48a5…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/A-C1/p1_glb/r2` | `$RUNS_DIR/2026-10-25/track_a-A-C1/e2a-c15/r2/output.glb` | raw | `9f9b00eb36024610…` | P1-glb(E2a c15) |
| x2a | `x2a/A-C1/p1_cbin/r2` | `$RUNS_DIR/paper/x0/cbin_stitched/A-C1/r2/output.glb` | raw + (iv) | `56707d3e1cc51d86…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/A-C1/p1_glb/r3` | `$RUNS_DIR/2026-10-25/track_a-A-C1/e2a-c15/r3/output.glb` | raw | `e86c91c6debf0086…` | P1-glb(E2a c15) |
| x2a | `x2a/A-C1/p1_cbin/r3` | `$RUNS_DIR/paper/x0/cbin_stitched/A-C1/r3/output.glb` | raw + (iv) | `4d75a8d7c78cd7c1…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/A-C2/original` | `$ASSETS_DIR/track_a/chair/gen/model.glb` | v0 | `67595c621826fee7…` | 원본 v0 |
| x2a | `x2a/A-C2/p0/r1` | `$RUNS_DIR/paper/x2a/p0/A-C2/r1/output.glb` | n | `18270e11fd50e349…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/A-C2/p0/r2` | `$RUNS_DIR/paper/x2a/p0/A-C2/r2/output.glb` | n | `a3fffaae90aa57a2…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/A-C2/p1_glb/r1` | `$RUNS_DIR/2026-10-25/track_a-A-C2/e2a-c15/r1/output.glb` | raw | `481f0be4155572f0…` | P1-glb(E2a c15) |
| x2a | `x2a/A-C2/p1_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/A-C2/r1/output.glb` | raw + (iv) | `d3a865475996444a…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/A-C2/p1_glb/r2` | `$RUNS_DIR/2026-10-25/track_a-A-C2/e2a-c15/r2/output.glb` | raw | `2e8af3bd67b823c8…` | P1-glb(E2a c15) |
| x2a | `x2a/A-C2/p1_cbin/r2` | `$RUNS_DIR/paper/x0/cbin_stitched/A-C2/r2/output.glb` | raw + (iv) | `3f206cee0f7b8f67…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/A-C2/p1_glb/r3` | `$RUNS_DIR/2026-10-25/track_a-A-C2/e2a-c15/r3/output.glb` | raw | `a470239cc86fb26c…` | P1-glb(E2a c15) |
| x2a | `x2a/A-C2/p1_cbin/r3` | `$RUNS_DIR/paper/x0/cbin_stitched/A-C2/r3/output.glb` | raw + (iv) | `5df8234d27d8a78e…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/B1/original` | `$ASSETS_DIR/Edit3D-Bench/data/GSO/RJ_Rabbit_Easter_Basket_Blue/source_model/model.glb` | v0 | `2216fce000dd0910…` | 원본 v0 |
| x2a | `x2a/B1/p0/r1` | `$RUNS_DIR/paper/x2a/p0/B1/r1/output.glb` | n | `6da65e3825a199fb…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/B1/p0/r2` | `$RUNS_DIR/paper/x2a/p0/B1/r2/output.glb` | n | `05f5fc35ec847a2b…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/B1/p1_glb/r1` | `$RUNS_DIR/2026-10-25/track_b-B1/e2a-c15/r1/output.glb` | raw | `8c328753f828554e…` | P1-glb(E2a c15) |
| x2a | `x2a/B1/p1_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/B1/r1/output.glb` | raw + (iv) | `9a2b3d6e7d00a34a…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/B1/p1_glb/r2` | `$RUNS_DIR/2026-10-25/track_b-B1/e2a-c15/r2/output.glb` | raw | `ad06c8573459dc23…` | P1-glb(E2a c15) |
| x2a | `x2a/B1/p1_cbin/r2` | `$RUNS_DIR/paper/x0/cbin_stitched/B1/r2/output.glb` | raw + (iv) | `8df92b2b3e43cc3b…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/B1/p1_glb/r3` | `$RUNS_DIR/2026-10-25/track_b-B1/e2a-c15/r3/output.glb` | raw | `757a68d9951bdbe1…` | P1-glb(E2a c15) |
| x2a | `x2a/B1/p1_cbin/r3` | `$RUNS_DIR/paper/x0/cbin_stitched/B1/r3/output.glb` | raw + (iv) | `68264f70ab3345cc…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/B2/original` | `$ASSETS_DIR/Edit3D-Bench/data/PartObjaverse-Tiny/0c4e4a0b2a374553ae762bb33e8dd2d0/source_model/model.glb` | v0 | `d9f6045824219335…` | 원본 v0 |
| x2a | `x2a/B2/p0/r1` | `$RUNS_DIR/paper/x2a/p0/B2/r1/output.glb` | n | `b62d3ec9c7d8c696…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/B2/p0/r2` | `$RUNS_DIR/paper/x2a/p0/B2/r2/output.glb` | n | `5e48645bacf4590d…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/B2/p1_glb/r1` | `$RUNS_DIR/2026-10-25/track_b-B2/e2a-c15/r1/output.glb` | raw | `5ec99d4e62dfadf3…` | P1-glb(E2a c15) |
| x2a | `x2a/B2/p1_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/B2/r1/output.glb` | raw + (iv) | `d802cd53e8a4849f…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/B2/p1_glb/r2` | `$RUNS_DIR/2026-10-25/track_b-B2/e2a-c15/r2/output.glb` | raw | `ddca708045b41ad2…` | P1-glb(E2a c15) |
| x2a | `x2a/B2/p1_cbin/r2` | `$RUNS_DIR/paper/x0/cbin_stitched/B2/r2/output.glb` | raw + (iv) | `3f2c7ee4fd5a481d…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/B2/p1_glb/r3` | `$RUNS_DIR/2026-10-25/track_b-B2/e2a-c15/r3/output.glb` | raw | `f60d12ff4bf31272…` | P1-glb(E2a c15) |
| x2a | `x2a/B2/p1_cbin/r3` | `$RUNS_DIR/paper/x0/cbin_stitched/B2/r3/output.glb` | raw + (iv) | `ba0e9843f8d52784…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/B3/original` | `$ASSETS_DIR/Edit3D-Bench/data/GSO/SpiderMan_Titan_Hero_12Inch_Action_Figure_5Hnn4mtkFsP/source_model/model.glb` | v0 | `8b33108dcdf66441…` | 원본 v0 |
| x2a | `x2a/B3/p0/r1` | `$RUNS_DIR/paper/x2a/p0/B3/r1/output.glb` | n | `65beaf2fc335ef7a…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/B3/p0/r2` | `$RUNS_DIR/paper/x2a/p0/B3/r2/output.glb` | n | `d8d50b5469d4792a…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/B3/p1_glb/r1` | `$RUNS_DIR/2026-10-25/track_b-B3/e2a-c15/r1/output.glb` | raw | `2882e935eb59286e…` | P1-glb(E2a c15) |
| x2a | `x2a/B3/p1_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/B3/r1/output.glb` | raw + (iv) | `b8c314d007465d20…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/B3/p1_glb/r2` | `$RUNS_DIR/2026-10-25/track_b-B3/e2a-c15/r2/output.glb` | raw | `7bee9ba0bd12b9fd…` | P1-glb(E2a c15) |
| x2a | `x2a/B3/p1_cbin/r2` | `$RUNS_DIR/paper/x0/cbin_stitched/B3/r2/output.glb` | raw + (iv) | `7d603b41b10ef656…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/B3/p1_glb/r3` | `$RUNS_DIR/2026-10-25/track_b-B3/e2a-c15/r3/output.glb` | raw | `c1082512c2750b15…` | P1-glb(E2a c15) |
| x2a | `x2a/B3/p1_cbin/r3` | `$RUNS_DIR/paper/x0/cbin_stitched/B3/r3/output.glb` | raw + (iv) | `ccf588d127272111…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/B4/original` | `$ASSETS_DIR/Edit3D-Bench/data/PartObjaverse-Tiny/dcf35ae14df947a9a700b8bcff8db57f/source_model/model.glb` | v0 | `0d431158b0695c2d…` | 원본 v0 |
| x2a | `x2a/B4/p0/r1` | `$RUNS_DIR/paper/x2a/p0/B4/r1/output.glb` | n | `c9ec251dbbe35b5c…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/B4/p0/r2` | `$RUNS_DIR/paper/x2a/p0/B4/r2/output.glb` | n | `710ee57738e57bb8…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/B4/p1_glb/r1` | `$RUNS_DIR/2026-10-25/track_b-B4/e2a-c15/r1/output.glb` | raw | `c6a581905c9eea97…` | P1-glb(E2a c15) |
| x2a | `x2a/B4/p1_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/B4/r1/output.glb` | raw + (iv) | `c5c218811254e42b…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/B4/p1_glb/r2` | `$RUNS_DIR/2026-10-25/track_b-B4/e2a-c15/r2/output.glb` | raw | `f42c2e5f5cbcacf9…` | P1-glb(E2a c15) |
| x2a | `x2a/B4/p1_cbin/r2` | `$RUNS_DIR/paper/x0/cbin_stitched/B4/r2/output.glb` | raw + (iv) | `21e5e6bd94e4cf9d…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/B4/p1_glb/r3` | `$RUNS_DIR/2026-10-25/track_b-B4/e2a-c15/r3/output.glb` | raw | `2ebdb8d719b3518e…` | P1-glb(E2a c15) |
| x2a | `x2a/B4/p1_cbin/r3` | `$RUNS_DIR/paper/x0/cbin_stitched/B4/r3/output.glb` | raw + (iv) | `d9902a34230c2e59…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/B5/original` | `$ASSETS_DIR/Edit3D-Bench/data/GSO/Squirtin_Barnyard_Friends_4pk/source_model/model.glb` | v0 | `9fa2730506afa339…` | 원본 v0 |
| x2a | `x2a/B5/p0/r1` | `$RUNS_DIR/paper/x2a/p0/B5/r1/output.glb` | n | `0880775696eb355d…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/B5/p0/r2` | `$RUNS_DIR/paper/x2a/p0/B5/r2/output.glb` | n | `5ad8aabf08984069…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/B5/p1_glb/r1` | `$RUNS_DIR/2026-10-25/track_b-B5/e2a-c15/r1/output.glb` | raw | `adeb32293baef7e8…` | P1-glb(E2a c15) |
| x2a | `x2a/B5/p1_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/B5/r1/output.glb` | raw + (iv) | `9c737b9ee3f6bfa7…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/B5/p1_glb/r2` | `$RUNS_DIR/2026-10-25/track_b-B5/e2a-c15/r2/output.glb` | raw | `a3b5b08694723a1d…` | P1-glb(E2a c15) |
| x2a | `x2a/B5/p1_cbin/r2` | `$RUNS_DIR/paper/x0/cbin_stitched/B5/r2/output.glb` | raw + (iv) | `4e8efcfcd82e05eb…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/B5/p1_glb/r3` | `$RUNS_DIR/2026-10-25/track_b-B5/e2a-c15/r3/output.glb` | raw | `564f8ea26a447887…` | P1-glb(E2a c15) |
| x2a | `x2a/B5/p1_cbin/r3` | `$RUNS_DIR/paper/x0/cbin_stitched/B5/r3/output.glb` | raw + (iv) | `429e89134dd8c59e…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/B6/original` | `$ASSETS_DIR/Edit3D-Bench/data/PartObjaverse-Tiny/97582e1d16f6471bac014081535c26ad/source_model/model.glb` | v0 | `a0b1d50d5c781ba7…` | 원본 v0 |
| x2a | `x2a/B6/p0/r1` | `$RUNS_DIR/paper/x2a/p0/B6/r1/output.glb` | n | `9774a0c99a1fc476…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/B6/p0/r2` | `$RUNS_DIR/paper/x2a/p0/B6/r2/output.glb` | n | `f40f4c0724c506d9…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/B6/p1_glb/r1` | `$RUNS_DIR/2026-10-25/track_b-B6/e2a-c15/r1/output.glb` | raw | `68b7fdceb3a44dee…` | P1-glb(E2a c15) |
| x2a | `x2a/B6/p1_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/B6/r1/output.glb` | raw + (iv) | `36e02a432ffb5056…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/B6/p1_glb/r2` | `$RUNS_DIR/2026-10-25/track_b-B6/e2a-c15/r2/output.glb` | raw | `a39f37211f88daae…` | P1-glb(E2a c15) |
| x2a | `x2a/B6/p1_cbin/r2` | `$RUNS_DIR/paper/x0/cbin_stitched/B6/r2/output.glb` | raw + (iv) | `abc936642b12c908…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/B6/p1_glb/r3` | `$RUNS_DIR/2026-10-25/track_b-B6/e2a-c15/r3/output.glb` | raw | `e3b6e83bae2a64b4…` | P1-glb(E2a c15) |
| x2a | `x2a/B6/p1_cbin/r3` | `$RUNS_DIR/paper/x0/cbin_stitched/B6/r3/output.glb` | raw + (iv) | `7da0ce3f49a2f8cb…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/B7/original` | `$ASSETS_DIR/Edit3D-Bench/data/GSO/3D_Dollhouse_TablePurple/source_model/model.glb` | v0 | `8afad6ac7d16aa6f…` | 원본 v0 |
| x2a | `x2a/B7/p0/r1` | `$RUNS_DIR/paper/x2a/p0/B7/r1/output.glb` | n | `06ab0cd5d607be26…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/B7/p0/r2` | `$RUNS_DIR/paper/x2a/p0/B7/r2/output.glb` | n | `8332ace33e81c407…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/B7/p1_glb/r1` | `$RUNS_DIR/2026-10-25/track_b-B7/e2a-c15/r1/output.glb` | raw | `da7abd364055bad7…` | P1-glb(E2a c15) |
| x2a | `x2a/B7/p1_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/B7/r1/output.glb` | raw + (iv) | `f0f8dfbcd235e27c…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/B7/p1_glb/r2` | `$RUNS_DIR/2026-10-25/track_b-B7/e2a-c15/r2/output.glb` | raw | `e94221e95693ffa8…` | P1-glb(E2a c15) |
| x2a | `x2a/B7/p1_cbin/r2` | `$RUNS_DIR/paper/x0/cbin_stitched/B7/r2/output.glb` | raw + (iv) | `d2315294fad85d75…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/B7/p1_glb/r3` | `$RUNS_DIR/2026-10-25/track_b-B7/e2a-c15/r3/output.glb` | raw | `c428a4a99af372ef…` | P1-glb(E2a c15) |
| x2a | `x2a/B7/p1_cbin/r3` | `$RUNS_DIR/paper/x0/cbin_stitched/B7/r3/output.glb` | raw + (iv) | `be28aa9d017b6982…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/B8/original` | `$ASSETS_DIR/Edit3D-Bench/data/PartObjaverse-Tiny/5fb5b1704b464f5c94b667f1b630bb39/source_model/model.glb` | v0 | `3190d9403dc02279…` | 원본 v0 |
| x2a | `x2a/B8/p0/r1` | `$RUNS_DIR/paper/x2a/p0/B8/r1/output.glb` | n | `b27a6d5feffd3d9e…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/B8/p0/r2` | `$RUNS_DIR/paper/x2a/p0/B8/r2/output.glb` | n | `4870d1c3f1ae5c9d…` | P0 to_glb(T(v0) = N 좌표, prep.json T) |
| x2a | `x2a/B8/p1_glb/r1` | `$RUNS_DIR/2026-10-25/track_b-B8/e2a-c15/r1/output.glb` | raw | `0e45a545a6292f4b…` | P1-glb(E2a c15) |
| x2a | `x2a/B8/p1_cbin/r1` | `$RUNS_DIR/paper/x0/cbin_stitched/B8/r1/output.glb` | raw + (iv) | `559d918e01a295ed…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/B8/p1_glb/r2` | `$RUNS_DIR/2026-10-25/track_b-B8/e2a-c15/r2/output.glb` | raw | `65928df16562dbbe…` | P1-glb(E2a c15) |
| x2a | `x2a/B8/p1_cbin/r2` | `$RUNS_DIR/paper/x0/cbin_stitched/B8/r2/output.glb` | raw + (iv) | `9694acd84abad8c8…` | P1-cbin(R4 + 봉합) |
| x2a | `x2a/B8/p1_glb/r3` | `$RUNS_DIR/2026-10-25/track_b-B8/e2a-c15/r3/output.glb` | raw | `7e0907d52ba50bb8…` | P1-glb(E2a c15) |
| x2a | `x2a/B8/p1_cbin/r3` | `$RUNS_DIR/paper/x0/cbin_stitched/B8/r3/output.glb` | raw + (iv) | `19296999e39e2f90…` | P1-cbin(R4 + 봉합) |
| x2b | `x2b/X2B-1/original` | `$ASSETS_DIR/Edit3D-Bench/data/GSO/3D_Dollhouse_Happy_Brother/source_model/model.glb` | v0 | `ab019e91b349bac9…` | 데이터셋 원본 GLB |
| x2b | `x2b/X2B-1/p0` | `$RUNS_DIR/paper/x2b/p0/X2B-1/r1/output.glb` | n | `cdc7d8d7187ee012…` | P0 to_glb(N) |
| x2b | `x2b/X2B-1/p1` | `$RUNS_DIR/paper/x2b/p1/X2B-1/r1/output.glb` | raw + (iv) | `f2431499704fcacc…` | P1 R4 + 봉합 view.glb |
| x2b | `x2b/X2B-2/original` | `$ASSETS_DIR/Edit3D-Bench/data/GSO/3D_Dollhouse_Lamp/source_model/model.glb` | v0 | `f9db2e0048fe6ac2…` | 데이터셋 원본 GLB |
| x2b | `x2b/X2B-2/p0` | `$RUNS_DIR/paper/x2b/p0/X2B-2/r1/output.glb` | n | `b8f7748ff94e70d8…` | P0 to_glb(N) |
| x2b | `x2b/X2B-2/p1` | `$RUNS_DIR/paper/x2b/p1/X2B-2/r1/output.glb` | raw + (iv) | `a8b5be31def49da3…` | P1 R4 + 봉합 view.glb |
| x2b | `x2b/X2B-3/original` | `$ASSETS_DIR/Edit3D-Bench/data/PartObjaverse-Tiny/00200996b8f34f55a2dd2f44d316d107/source_model/model.glb` | v0 | `e673cba2cc5391a3…` | 데이터셋 원본 GLB |
| x2b | `x2b/X2B-3/p0` | `$RUNS_DIR/paper/x2b/p0/X2B-3/r1/output.glb` | n | `cdc9103d43cf6ce7…` | P0 to_glb(N) |
| x2b | `x2b/X2B-3/p1` | `$RUNS_DIR/paper/x2b/p1/X2B-3/r1/output.glb` | raw + (iv) | `65127991aed9b3db…` | P1 R4 + 봉합 view.glb |

## 표·대조

- CSV 10개, 원값 JSON 17개(tables/raw/, 원본 sha256 은 tables/raw_sources.json).
- 보고서 요약 표 대 같은 원값으로 다시 만든 표: 1245 칸 대조, 불일치 0 칸.
- 모든 파일의 sha256: `SHA256SUMS`.

