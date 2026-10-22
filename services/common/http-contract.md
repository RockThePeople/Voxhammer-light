# 3090 ↔ A5000 HTTP 규약 초안 (05 R9 R-5, 정오표 E4 (가)안)

- 상태: **초안.** chief 검토 후 main 에 먼저 병합한다(CLAUDE.md: 공용 인터페이스는 정의 PR 먼저). ai-ar-v3 계약 변경 제안의 바탕이다.
- 범위:
  - 이 문서는 문서만이다.
  - A5000 렌더 엔드포인트와 편집 요청 변경은 4번 R8a 범위라 **여기서 구현하지 않는다.**
  - 3090 인페인팅 워커(§2)는 R9 에서 구현했다(`services/rtx3090_inpaint/`).
- 주소는 환경변수로만 쓴다:
  - `<EDIT_HOST>:<EDIT_PORT>` = A5000 편집·렌더 워커
  - `127.0.0.1:<INPAINT_PORT>` = 3090 인페인팅 워커(기본 8082, localhost 전용)
- 통신은 **3090 → A5000 한 방향**이다. A5000 은 3090 을 부르지 않는다.
- 오류 본문은 ai-ar-v3 `deltacontract.schemas.ErrorBody` 모양을 쓴다: `{"error_code": str, "message": str, "detail"?: {str: str}}`.

## 1. 흐름

```
Unity ──(자연어+라쏘 마스크)──▶ 3090 오케스트레이터
  ① POST <EDIT_HOST>/v2/trellis/render           A5000: 고정 시점(R10) 원본 렌더 + 2D 마스크
  ② POST 127.0.0.1:<INPAINT_PORT>/inpaint        3090: INT4 FLUX Fill, 마스크 밖 합성
  ③ POST <EDIT_HOST>/v2/trellis/edit (multipart) A5000: 렌더 해시·base_version 검증 → VoxHammer 편집
  ④ GET  <EDIT_HOST>/v2/trellis/jobs/<id>        (기존) chunks · to_version
```

- ③ 에서 A5000 은 받은 `2d_render`·`2d_mask` 의 sha256 과 `base_version` 이 ① 에서 자기가 만든 렌더와 같은지 확인한다. 다르면 4xx 로 거절한다(§3.3).
- 오케스트레이터에서는 t2i 조건 생성 부분(`server/voxhammer.py` 의 `render_rgba`)만 ①②로 바꾼다.

## 2. 3090 인페인팅 워커 (구현됨, R9)

`VHL_R9=1` 일 때만 뜬다. 모델(INT4 FLUX.1-Fill-dev)은 프로세스 시작 때 한 번 올린다. 큐는 한 번에 하나다.

### `GET /health`
`200 {"ok": bool, "vhl_r9": true, "busy": job_id|null, "queue": int, "calls": int, "load": {base, base_snapshot, transformer, transformer_snapshot, precision, seconds, spans, memory_allocated_after_load, torch, torch_cuda}}`

### `POST /inpaint` (multipart/form-data)

| 필드 | 형식 | 필수 | 뜻 |
|---|---|---|---|
| `image` | PNG 1024² | ✓ | 원본 렌더(①의 `2d_render`). RGBA 면 흰 배경에 합성해 쓴다(포크 `utils/inpaint.py` 와 같음) |
| `mask` | PNG 1024² | ✓ | 2D 마스크. 흰색(>127) = 다시 그릴 영역 |
| `prompt` | UTF-8 | ✓ | 인페인팅 프롬프트 |
| `seed` | 정수 | – | 기본 0 (정오표 E5). CPU generator |

- `?wait=1` 이 없으면 `202 {"job_id", "state": "queued", "ahead": int}` 를 돌려준다. 있으면 끝날 때까지 기다렸다가 `200`(완료) 또는 `500`(실패)으로 아래 잡 본문을 돌려준다.
- 고정 설정: 1024², 50스텝, guidance 30, max_sequence_length 512. 바꾸는 것은 6번 E4 범위다.

### `GET /jobs/<job_id>`
```
{"job_id", "state": "queued|running|done|failed", "prompt", "seed",
 "input_sha256": {"2d_render", "2d_mask"},
 "result": {"sha256": {"2d_render", "2d_mask", "2d_edit", "2d_edit_raw"},
            "outside_diff": {"before": {...}, "after": {...}},
            "spans": [{"name": "inpaint", "seconds", "max_memory_allocated", "nvidia_smi_mib_max", ...}],
            "settings": {...}},
 "error"?: ErrorBody}
```

### `GET /jobs/<job_id>/<name>.png`
`name` 은 `2d_render`, `2d_mask`, `2d_edit`(마스크 밖 합성 후), `2d_edit_raw`(합성 전) 중 하나다. 응답 헤더 `X-Sha256` 에 파일 해시가 담긴다.

**마스크 밖 합성:** `2d_edit` 의 마스크 밖 픽셀은 모델이 본 원본 렌더 RGB 와 바이트가 같다. 2진 마스크이고 페더는 없다. ③으로 보내는 것은 `2d_edit` 이다.

| 오류 | HTTP | 사유 |
|---|---|---|
| `MISSING_FIELD` | 400 | image·mask·prompt 중 하나가 없다 |
| `BAD_IMAGE` | 400 | PNG 를 읽을 수 없다 |
| `BAD_SIZE` | 400 | 1024² 가 아니다 |
| `BAD_SEED` | 400 | 정수가 아니다 |
| `BAD_MULTIPART` / `BAD_CONTENT_TYPE` | 400 / 415 | 본문 형식 |
| `INPAINT_FAILED` | 잡 `failed` | 모델 실행 실패(메시지에 예외 종류) |
| `NOT_FOUND` | 404 | 경로·잡·파일 없음 |

## 3. A5000 쪽 제안 (4번 R8a 에서 구현, 여기서는 규약만)

### 3.1 `POST /v2/trellis/render` (신규)

요청 JSON:
```
{"asset_id": str, "base_version": int,
 "mask": EditMask,            // 기존 BEditRequest.mask 와 같은 모양 (mode: voxels, grid_source: slat_coords, halo_margin_voxels)
 "view": int,                 // R10 고정 시점 번호. 생략하면 서버 기본값
 "idempotency_key"?: str}
```
응답 `200`:
```
{"render_id": str, "asset_id", "base_version", "view",
 "render_sha256": str,        // 2d_render.png 바이트
 "mask_sha256": str,          // 2d_mask.png 바이트
 "renderer": {"name", "version", "resolution": 1024, ...}}
```
파일: `GET /v2/trellis/renders/<render_id>/2d_render.png`, `.../2d_mask.png`

- 2D 마스크는 포크 `render_rgb_and_mask.py` 정의를 따른다: 원본 depth − 마스크 depth > 1e-4.
- 렌더가 결정적이지 않으면(R10a 측정) A5000 은 `render_id` 별로 자기가 낸 해시를 보관해 ③에서 대조한다.

### 3.2 `POST /v2/trellis/edit` 변경 (multipart)

기존 JSON 본문은 `meta` 필드로 옮긴다(generate 의 `meta`+`image` 관례와 같다). 새 필드:

| 필드 | 형식 | 뜻 |
|---|---|---|
| `meta` | JSON | 기존 `{asset_id, base_version, prompt, mask, seed, idempotency_key}` + **`render_id`, `render_sha256`, `mask_sha256`** |
| `2d_render` | PNG | ① 응답 파일 그대로 |
| `2d_mask` | PNG | ① 응답 파일 그대로 |
| `2d_edit` | PNG | ② 의 `2d_edit` (합성 후) |

- 응답은 기존과 같다: `{job_id}` → `GET /v2/trellis/jobs/<id>` → `chunks`, `to_version`.
- 기존의 `image`(t2i 전신 조건) 필드는 받지 않는다. 이전 기간에 함께 받을지는 결정이 필요하다(§4).
- 멱등 키: 기존 권장식(`asset_id|base_version|raw_prompt|mask_fp|seed`)에 `sha256(2d_edit)` 를 더할 것을 제안한다. 같은 마스크·프롬프트라도 조건 이미지가 다르면 다른 편집이기 때문이다.

### 3.3 A5000 거절 사유 (4xx)

| `error_code` | HTTP | 조건 |
|---|---|---|
| `CONDITION_IMAGES_REQUIRED` | 422 | `2d_render`·`2d_mask`·`2d_edit` 중 하나라도 없다. `detail.missing` 에 이름 |
| `RENDER_MISMATCH` | 409 | 받은 `2d_render`/`2d_mask` sha256 이 `render_id` 의 기록과 다르거나 `meta` 의 해시와 다르다. `detail` 에 기대값·받은 값 |
| `RENDER_UNKNOWN` | 404 | `render_id` 기록이 없다(만료 포함) |
| `VERSION_CONFLICT` | 409 | (기존) `base_version` 이 렌더 때와 다르거나 최신이 아니다. `detail.latest_version` |
| `BAD_CONDITION_IMAGE` | 400 | PNG 가 아니거나 1024² 가 아니다 |

## 4. 열린 질문 (chief·고객사)

1. `render_id` 기록 보관 기간과, A5000 재기동 뒤 ③이 `RENDER_UNKNOWN` 일 때 오케스트레이터가 ①부터 다시 할지.
2. 이전 기간에 기존 `image` 필드(t2i 조건)를 같이 받을지, 바로 끊을지.
3. 멱등 키에 `2d_edit` 해시를 넣는 것(계약 3.15.5 권장식 변경)을 ai-ar-v3 계약 변경 제안에 포함할지.
4. 렌더 시점(`view`)을 요청마다 받을지, R10 고정값만 허용할지.
