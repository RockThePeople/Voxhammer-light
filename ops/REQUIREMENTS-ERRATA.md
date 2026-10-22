# REQUIREMENTS 정오표

`ops/REQUIREMENTS.md` 는 고객사 원문이므로 고치지 않는다. 고객사가 회신으로 정정하거나 해석을 확정한 내용은 이 문서에 쌓는다. 원문과 이 문서가 다르면 **이 문서가 우선한다.** 각 항목에는 원문 위치, 정정 내용, 근거 회신 날짜를 적는다.

| # | 원문 위치 | 원문 요지 | 정정·확정 | 근거 |
|---|---|---|---|---|
| E1 | 6장 「평가 자산과 편집 시나리오」 첫 단락 (338행) | "moto-b는 SLat feats가 없어 0번 기준선(원저자 경로 시간 측정)에만 씁니다." | **고객사 오류.** A5000 자산 저장소에 feats 가 있는 moto-b SLat(`v3-moto-b/v1/slat.safetensors`, 9,150셀, denormalized)이 있다(00a A5000 실측). 보조 측정(W34 SLat 재사용 경로)에 쓰는 것을 허용한다. 사용 전 확인 절차는 `ops/tasks/00-kickoff.md` A-6. 확인을 통과하면 트랙 A 보조 자산으로 쓸 수 있다(트랙 A 6건·14건 계산에는 넣지 않는다). W34 경로 시간은 참고값이며 비교 기준이 아니다. 0번 시간 기준선은 원문대로 원저자 경로다. | 고객사 회신 A2, 2026-10-22 |
| E2 | 6장 「품질 지표」 R_keep "(편집 마스크 ⊕ halo 2)" | halo 2 팽창 방식 미지정 | `deltacontract.dilate_cells(cells, 2)`(ai-ar-v3 `19ed91f`, `contract/python/deltacontract/coords.py:406`, 26-이웃, 64³ 밖 셀 제거)를 **그대로 호출**한다. 새로 구현하지 않는다. | 고객사 회신 A6(b), 2026-10-22 |
| E3 | 2장·5번 R9 FLUX 가중치 | 출처 미지정 | 원본 `black-forest-labs/FLUX.1-Fill-dev` 만 쓴다. 미러는 쓰지 않는다. 다운로드 계정 rockthepeople, 토큰은 고객사가 장비에 직접 설정. | 고객사 회신 A3·추가 1), 2026-10-22 |
| E4 | 4번 R8a·5번 R9 계약 | `/v2/trellis/edit` 대체 여부 미정 | (가)안. 엔드포인트·잡 조회·응답 형식(chunks, to_version)·409 VERSION_CONFLICT 유지. 편집 요청에 `2d_render`·`2d_mask`·`2d_edit` multipart 추가, 없으면 이유와 함께 4xx. A5000 에 렌더 엔드포인트 추가(입력 asset_id·base_version·3D 마스크·R10 고정 시점 → 출력 2d_render·2d_mask·두 해시·base_version). 3090 오케스트레이터는 렌더 요청 → 로컬 인페인팅 → 이미지 3종+렌더 해시로 편집 요청 순서로 바꾸고, t2i 조건 생성 부분만 교체. A5000 은 렌더 해시·base_version 이 자기 렌더와 다르면 4xx. 통신은 3090 → A5000 한 방향. ai-ar-v3 반영은 수행사가 계약 변경 제안+패치(deltacontract 스키마·오케스트레이터·테스트)를 ai-ar-v3 PR 로 내고(conformance 전부·서버 테스트 전부 통과), 병합은 고객사. | 고객사 회신 A4·추가 4), 2026-10-22 |
| E5 | 조건 이미지 시드 | — | 시드 0(VoxHammer `utils/inpaint.py` 와 같은 값). R9 비교도 같은 시드. | 고객사 회신 A6(c), 2026-10-22 |
| E6 | R9 일정 | 1–9 선행 0-1 | R9 는 0번 승인 뒤 별도 브랜치·PR 로 병행하되 1–4번 일정·검증을 늦추지 않는다. 제출·승인과 3D 품질 평가는 4번 승인 뒤, TRELLIS 는 직전 승인 상태 고정. | 고객사 회신 A1, 2026-10-22 |
