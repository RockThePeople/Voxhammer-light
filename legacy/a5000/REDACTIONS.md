# REDACTIONS — legacy/a5000

원본을 수정 없이 복사했고, 아래만 바꿨다(2026-10-23, a5000 세션).

| 바꾼 것 | 규칙 |
|---|---|
| 홈 경로 | 문자열 `/home/<사용자>` → `<HOME>` (경로 뒤쪽은 그대로) |
| 토큰·키 값 | 해당 없음 — 실제 값이 없었다(아래 스캔) |

## 홈 경로 치환 — 33행 / 19파일 (00a 스캔의 33행과 일치)

| 파일 | 행 |
|---|---|
| `edit_service/api/store.py` | 18 |
| `edit_service/lab/store.py` | 13 |
| `vh-track/chunk_compare.py` | 25 |
| `vh-track/debugview_standalone.py` | 22 |
| `vh-track/edit_slat_reuse.py` | 34, 37 |
| `vh-track/env.sh` | 5 |
| `vh-track/gen_asset.py` | 19, 34 |
| `vh-track/inpaint_offload.py` | 27 |
| `vh-track/make_mask_glb.py` | 27 |
| `vh-track/repeat_bench_analyze.py` | 19 |
| `vh-track/repeat_bench.py` | 23 |
| `vh-track/repeat_bench_run.sh` | 5, 6, 7 |
| `vh-track/restart_edit_service.sh` | 6, 8, 10, 11, 12 |
| `vh-track/run_voxhammer_checked.sh` | 19, 25 |
| `vh-track/start_debugview.sh` | 6, 7, 11 |
| `vh-track/w42_region_eval.py` | 25 |
| `vh-track/w42_run.sh` | 5, 6, 7 |
| `vh-track/w43_panels.py` | 28 |
| `vh-track/w43_run.sh` | 5, 6 |

## 비밀값 스캔 (치환 전 복사본, 실측)

- 패턴: `hf_…`, `sk-…`, `ghp_…` 토큰, `api_key|token|secret|password = "…"` 대입, IPv4 리터럴(0.0.0.0·127.0.0.1 제외) → **0건**.
- 검토 후 그대로 둔 것: `edit_service/api/routes.py:29` `API_KEY = os.environ.get("TRELLIS_API_KEY", "dev-key")` — 환경변수가 없을 때의 개발용 기본 문자열이며 실제 키가 아니다. `vh-track/restart_edit_service.sh:9` 는 키를 환경변수로만 받고 없으면 중단한다.
- 포트 기본값(8081·8082·8083)은 CLAUDE.md 「보안」(호스트·포트는 기본값으로 둬도 된다)에 따라 그대로 둔다.
