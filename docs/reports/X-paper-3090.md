# X 논문 실험 — rtx3090 — 보고서 (진행 중)

- 장비: rtx3090. 지시: 사용자 2026-10-26 10:27(논문 실험 최우선). 지시서 main `60ebce5` `ops/tasks/X-paper.md`, 계획 원문 `ops/paper/PLAN.md`.
- 브랜치: `item/X-paper-3090` (main `60ebce5` 에서 분기)
- 3090 범위:
  - (a) A5000 → 3090 접속 시험
  - (b) X3 자산 이미지 생성 → 사용자 선택
  - (c) X3 P1·P2 인페인팅(INT4 워커)
  - (d) X2-C 의 3090 쪽(04L 종단, chief 지시 뒤)
- 수치 표시: 실측 / 코드확인 / 추론 / 미검증

## 0. 지시 사이의 차이 (확인 요청)

1. **P0 인페인팅 장비.**
   - 사용자 메시지(지시서 `e253a49` 인용)는 "X3 조건 이미지는 세 경로(P0·P1·P2) 모두 3090 INT4" 다.
   - main 의 최신 지시서(`60ebce5` "제안 경로(P1·P2) INT4, 원저자(P0) bf16")는 **P0 = A5000 bf16 + CPU offload, P1·P2 = 3090 INT4** 다.
   - 더 늦은 main 을 따라 **3090 은 P1·P2 만 인페인팅한다.** P0 도 3090 INT4 로 해야 하면 알려 달라.
2. **04L 종단은 이미 한 번 돌았다.**
   - chief 답(4절, 보고 `8225e3b` 기준)보다 뒤인 10:43 에 A5000 R8a 워커가 떴고, 지시 3("워커가 뜨면 바로 종단")에 따라 트랙 A 6 × 3 을 돌렸다(`item/05-R9b-3090` `06eff35`, `docs/reports/04L-link-3090.md`, 18/18 성공, 편집 1회 93.3 ± 2.6 s).
   - 그때는 4절 결정 2(반복마다 `<id>-e2e-r1..r3` 복제 자산)가 정해지기 전이라, A5000 시드 자산 `vhl-ta-*` v1 을 3회 모두 부모로 썼다. 편집은 staging 만 만들고 커밋하지 않아 409 는 나지 않았다.
   - X2-C 에서 복제 자산으로 다시 돌린다(`services/a5000_edit/assets_track_a.json` 이 오면).
   - 워커 감시는 멈췄다(감시 작업 종료).

## 1. (a) A5000 → 3090 접속 시험 — 리스너 대기 (실측)

- 임시 리스너: `python3 -m http.server 18090 --bind 0.0.0.0`(3090, 2026-10-26 11:18:32 기동, `setsid` 로 상주). 응답 본문 `rtx3090 listener ok <기동 시각>`.
- 3090 자체 확인: `curl http://127.0.0.1:18090/` → 200.
- **A5000 에서 할 시험:** `curl -v http://<3090>:18090/`. `<3090>` 은 3090 의 LAN 주소이고 chief 에 따로 전한다(문서에 쓰지 않는다).
- 결과: **접속 성공(실측).** 리스너 로그에 `<A5000> - - [26/Oct/2026 12:51:14] "GET / HTTP/1.1" 200` 이 남았다. A5000 → 3090 방향이 열려 있다.
- 따라서 X3 INT4 워커를 `0.0.0.0:8082` 로 열었다(3절). 18090 리스너는 chief 지시대로 유지하며, seed 0 자산 파일 전달에도 쓴다(2절).
- 접속되면 X3 용 INT4 워커를 0.0.0.0 으로 열어 A5000 실험 스크립트가 직접 부르게 한다. 막히면 3090 이 루프를 돈다(A5000 렌더 수신 → 인페인팅 → 2d_edit 업로드).

## 2. (b) X3 자산 이미지 (실측)

### 예상

- 같은 경로(Z-Image 1024²·9스텝 → BiRefNet)는 같은 시드에서 바이트 동일한 결과를 낸다(00 B-3).
- 원문이 구도("three-quarter front view")를 적었지만 Z-Image 가 따르지 않을 수 있다. 00s S-1 에서 moto 는 측면, snowman 은 정면이었다.

### 실행

- INT4 워커를 내린 뒤(GPU 38 MiB 확인) `tools/x3_assets.py` 로 3 자산 × 시드 0–3 = 12장을 만들었다(2026-10-26 11:2x).
- 프롬프트는 `ops/paper/PLAN.md` 원문 그대로다(템플릿 없음).
- Z-Image-Turbo bf16 1024², 9스텝, guidance 0 → BiRefNet fp16. 워커 `zimage_gen.py`·`birefnet_seg.py` 를 수정 없이 썼다(기존 3090 생성 경로).
- 산출(git 제외): `runs/paper/x3/assets/<asset>/seed<N>/{image_rgb.png, input.png, params.json}`, `index.json`, `sheet.png`
  - `index.json` sha256 `793fd046985a35a961e15d8a525197f02d894fbc0040f6b256732de9e6d2e094`
  - `sheet.png`(12장, 배경 제거본을 회색 위에) sha256 `afe5943ffb3255aec76716691ed115dd3631204adf2c1a3bd6b34001e6124dc6`

| 자산 | 시드 | `image_rgb.png` sha256 | `input.png`(배경 제거) sha256 |
|---|---|---|---|
| jeep | 0 | `89c62c64eec6518681c40ca44e22c40d06317efad7ec45f4e0df4a2fd1c9d83a` | `979f1418d20e12095b0274b052a8c316d0d5beba4405198872fc741c853bd6a3` |
| jeep | 1 | `6735b2167efafcc47bc54e92bb6c566c41445d285075ff8a4fd9a3e3621c59df` | `f543be3afbc22c7384a290d01d4dad91003bffcc063584452cbd906d9d740f32` |
| jeep | 2 | `de96e309648aee34cb336b1d69ad6820229c51b528fe220774d2d45d2738d982` | `177396c9a03e6119829b4e718a2dc99c604f855da71fa7a0a2623ba3757b5797` |
| jeep | 3 | `9cacde4b80dcb4fe2443f54429cf4607950cbf041fcf4d05a4334a2bf924fa5e` | `51dc2b21984191b016d345322fefee9dc7168bc8488e3457aa1034edc7e40904` |
| cottage | 0 | `6402bd081f828e66dfbf2206907efa623d99f8d759b18cf5af5b0ff494797498` | `1fb5c801e665dda89aff4df515eebee9286f9815cf326c6ede069a1f8ae345f6` |
| cottage | 1 | `5963406876e10bfb1b9868745d9f8f13e950eadeb5c186d9d14901a7d33a30b0` | `9a4be277f87af8e9534eb1dd6cf58f6ea57fc99ab20b6b2cec53b94d41bf68da` |
| cottage | 2 | `1ea08f1d7c02ac5903e011f0b20c5e85a5ed586814efb18757db1a027b788515` | `b3dc61ceb04c50057734097d69f98dedcd903cd51cdb5e253be2cda68a317d95` |
| cottage | 3 | `28fb846a0f0f3f5e9993f41605f89ab8b4804a9422afb4d8e72ea2e5579de7e9` | `6c219bcf3ac1c5837ca76fc6da5b084faf0825a13d899093c7c37d7c0e233622` |
| adventurer | 0 | `99bb6891135c8195f16b11866ffe23d999de223a4907b73fc0641b26033fc4a7` | `8d9be0a2f9886b58e1efa51dac91052dee6c54bfeb338ee17c9525f4bc51dbb5` |
| adventurer | 1 | `148583f8d0f9607b22e8d44238213c2be33af17eff66368716ddc5d392693cb0` | `43c7876cd46e67ed434811ffc1a565bdf337ea22829d58c67125397245a4a0d1` |
| adventurer | 2 | `26c5937b77027c7578e0b4693f96073861b9c7ca5b118426f929d3b0261e28d7` | `a04b8c08dffa8465034ea2ff26ce26e40a9532f4086fab1211514242407f132e` |
| adventurer | 3 | `cda3ed20190f881748781fd1bfb96ae6356c8b133fe215c6ef0d34774745e78d` | `36f194b4cf62a03cbf87a2a75622831a3dbf0ac1fbd383c1e96ac57950c26115` |

### 육안 메모 (편집 계획 부위가 보이는지 — 선택 참고, 판단은 사용자)

| 자산 | 시드 0 | 시드 1 | 시드 2 | 시드 3 |
|---|---|---|---|---|
| jeep (범퍼·보닛·뒤 스페어·지붕·오른쪽 스노클) | 3/4, 앞 윈치 범퍼, 소프트탑 일부. 스노클·스페어 안 보임 | 3/4, 지붕 없음(오픈), 뒤쪽 스페어 보임 | 정면에 가까움, 하드탑 지붕, 옆 스페어, 불바형 범퍼. 스노클 안 보임 | 정면, **스노클 보임**(차량 오른쪽 기둥), 옆 스페어, 지붕 |
| cottage (문·왼쪽 창·오른쪽 창·현관 난간·굴뚝) | 굴뚝, 문, 앞 창 1 + 옆 창, 난간 왼쪽만 | 굴뚝, 문, 앞 창 1, 난간 | 굴뚝, 문, 앞 창 1 + 옆 창, 난간 | **굴뚝, 앞 창 2, 문, 양쪽 난간** — 원문과 가장 가깝다 |
| adventurer (모자·오른손 칼·오른쪽 어깨·부츠·왼팔 방패) | 4장 모두 모자·오른손 칼·왼팔 방패·부츠·녹색 튜닉 | 〃 (얼굴이 더 굳음) | 〃 | 〃 |

- jeep 은 원문 다섯 부위가 한 장에 다 보이는 시드가 없다. 지붕 위 추가(4회차)는 지붕이 있는 시드(0·2·3)가 낫다. 스노클 제거(5회차)는 스노클이 보이는 시드(3)만 효능을 볼 수 있다(`추론`).
- 3/4 구도는 jeep 0·1, cottage 전부, adventurer 전부가 따랐다. jeep 2·3 은 정면에 가깝다.

**사용자 결정(2026-10-26 12:37, chief 답 main `8b3f107`):** 무작위로 고르지 않는다. **세 자산 모두 seed 0** 을 쓴다. 편집은 원문 1→5 순서·원문 프롬프트 그대로다.

### A5000 전달 (3D 생성 입력)

| 자산 | 파일 | sha256 | 내려받기 (A5000 에서) |
|---|---|---|---|
| jeep | `input.png`(배경 제거, TRELLIS 입력) | `979f1418d20e12095b0274b052a8c316d0d5beba4405198872fc741c853bd6a3` | `http://<3090>:18090/x3/jeep/input.png` |
| jeep | `image_rgb.png` | `89c62c64eec6518681c40ca44e22c40d06317efad7ec45f4e0df4a2fd1c9d83a` | `…/x3/jeep/image_rgb.png` |
| cottage | `input.png` | `1fb5c801e665dda89aff4df515eebee9286f9815cf326c6ede069a1f8ae345f6` | `…/x3/cottage/input.png` |
| cottage | `image_rgb.png` | `6402bd081f828e66dfbf2206907efa623d99f8d759b18cf5af5b0ff494797498` | `…/x3/cottage/image_rgb.png` |
| adventurer | `input.png` | `8d9be0a2f9886b58e1efa51dac91052dee6c54bfeb338ee17c9525f4bc51dbb5` | `…/x3/adventurer/input.png` |
| adventurer | `image_rgb.png` | `99bb6891135c8195f16b11866ffe23d999de223a4907b73fc0641b26033fc4a7` | `…/x3/adventurer/image_rgb.png` |

- 3090 경로: `runs/paper/x3/assets/<자산>/seed0/` (생성 파라미터 `params.json` 포함).
- 18090 리스너에 같은 파일과 `x3/SHA256SUMS` 를 올렸다. A5000 이 위 URL 로 받고 sha256 을 대조하면 된다. 3090 에서 받은 파일의 해시가 원본과 같은 것을 확인했다(실측).
- git 에는 커밋하지 않았다(지시: `runs/` 에 두고 sha256 만).

## 3. (c) X3 인페인팅 — 워커 기동 완료, A5000 호출 대기

- **P0 는 A5000 bf16, 3090 은 P1·P2 만 INT4**(chief 답 2, main `60ebce5` 확인).
- INT4 워커 재기동(2026-10-26 12:5x, 실측):
  - `INPAINT_BIND=0.0.0.0 VHL_R9=1 bash services/rtx3090_inpaint/run.sh start`
  - 로그 `[ready] 0.0.0.0:8082`. 3090 LAN 주소로 `/health` 200 을 받았다.
  - 원본 `358293d` + INT4 `bd90903`. R9 설정: 1024², 50스텝, guidance 30, seed 는 요청(0), 마스크 밖 합성 켬.
- A5000 실험 스크립트가 `POST http://<3090>:8082/inpaint?wait=1` (규약 §2, multipart image·mask·prompt·seed)로 직접 부른다. 결과는 `GET /jobs/<id>/2d_edit.png`.
- 워커 변경: `INPAINT_BIND` 를 추가했다(기본 127.0.0.1 그대로, 열 때만 0.0.0.0).
- **호출별 기록:** 워커를 `RUNS_DIR=runs/paper/x3` 로 다시 띄웠다(호출 0 상태에서 재기동). 호출마다 `runs/paper/x3/r9-inpaint/<job_id>/` 에 다음이 남는다:
  - `2d_render.png`·`2d_mask.png`·`2d_edit.png`(합성 후)·`2d_edit_raw.png`(합성 전)
  - `result.json`: 네 파일 sha256, 입력 sha256, 프롬프트·시드, 시간·VRAM
  - 로그 `runs/paper/x3/r9-inpaint/worker.log`
- 인증은 없다(chief 답: 보안 조치 하지 않음). LAN 안에서 누구나 부를 수 있다.
- 호출마다 `2d_render`·`2d_mask`·`2d_edit` 와 해시를 `runs/paper/x3/` 에 남긴다.

## 4. 변경 파일

- `tools/x3_assets.py`: X3 자산 이미지 생성(원문 프롬프트, Z-Image → BiRefNet, sha256 색인·시트)
- `services/rtx3090_inpaint/worker.py`·`README.md`: `INPAINT_BIND`(기본 127.0.0.1)
- `docs/reports/X-paper-3090.md`: 이 보고서
