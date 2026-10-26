# A5000 상주 편집·렌더 워커 (04L)

3090 오케스트레이터 → A5000 한 방향(정오표 E4). 규약은 `services/common/http-contract.md` §3(계약 3.28.0).

```
bash services/a5000_edit/run.sh          # 0.0.0.0:8082 (EDIT_PORT), 모델 상주 로드 후 요청 받음
curl -H "X-Blockedit-Key: $TRELLIS_API_KEY" http://<EDIT_HOST>:8082/v2/trellis/health
```

- 인증: `X-Blockedit-Key` = `TRELLIS_API_KEY`(.env, 없으면 `dev-key`). 3090 쪽 키와 같아야 한다.
- 저장소: `RUNS_DIR/04l/store`(판본·staging, 현행 서비스와 같은 구조), 렌더 기록 `RUNS_DIR/04l/renders`(24 h, 재기동 유지).
- 시드 판본: 트랙 A 생성 자산 `vhl-ta-moto`·`vhl-ta-snowman`·`vhl-ta-chair` v1(`seed.py`). 3090 은 같은 asset_id·v1 을 기준 판본으로 쓴다.
  v1 매니페스트: `GET /v2/trellis/assets/<id>/v1/manifest.json`, 청크: `GET /v2/trellis/assets/<id>/chunks/<key>.v1.cbin`.
- 편집: editcore(15/15, W34 입력 = 부모 판본 slat_tgt) → R4 `.cbin`(부기 ∪ 점유 변경 ∪ 신규, R11 승계, 경계 스냅) → staging.
  `POST /v2/trellis/assets/<id>/commit` 으로 판본 승격(현행과 같음).
- 조건 렌더: 포크 `utils/render_rgb_and_mask.py` 수정 없이(5시점 → 요청 시점 1장). `editcore.render.render_condition` 한 곳.
