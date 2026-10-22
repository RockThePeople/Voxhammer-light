# 계측 JSON 스키마 `vhl-run/1`

경로: `$RUNS_DIR/<YYYY-MM-DD>/<asset>/<mode>-<mask_fingerprint 앞 12자>/<run>.json` (git 제외).
같은 디렉터리의 `<run>.work/` 에 `output.glb`·`slat_tgt.safetensors`(R2)·원저자 경로 `render/` 가 남는다.
이후 모든 항목이 이 스키마를 쓴다. 필드를 지우지 않고 추가만 한다.

| 필드 | 뜻 |
|---|---|
| `schema` | `"vhl-run/1"` |
| `mode` | `original`(원저자 경로, 0번 기준선) / `w34`(SLat 재사용, 참고값) |
| `asset`, `run`, `profile_run` | 자산 이름, 회차, 프로파일 회차 여부(평균 계산에서 제외) |
| `mask` | `file`, `fingerprint`(`deltacontract.mask_fingerprint`), `n_cells` |
| `inputs`, `image_sha256` | 입력 경로, 조건 이미지 3종 sha256 |
| `edit_params` | `skip_step`, `re_init`, `cfg` |
| `env` | 실제 attention 백엔드(dense/sparse), `SPCONV_ALGO`, `W39_ATTN_MASK`(env·실효), 포크 파일 경로 |
| `error` | 예외 문자열 또는 null |
| `checks` | 경로 실행 증거: `decode_slat_calls`, 원저자 경로 산출물 존재, `voxels_delete` 셀과 마스크 json 대조, `calls_match_expected`(기대 호출 횟수와 다른 것만, 빈 dict 이면 정상) |
| `counts` | `coords_tgt`(Stage 1 결과 셀), `coords_preserve` |
| `cache_bytes` | `stage1`·`stage2` inversion 이 돌려준 `latent`·`kv` 의 장치별 텐서 바이트와 항목 수 |
| `profile` | `--profile` 회차만: 구간 4·6 함수별 memcpy(DtoH/HtoD/DtoD)·매칭(aten::eq/all/argmax) 장치 시간(µs) |
| `segments` | 키 = 구간(`registration`, `3`–`8`). `seconds`(자기 시간 합), `parts`(span 이름별 자기 시간), `max_memory_allocated`(B), `nvidia_smi_mib_max`, `rss_bytes_max` |
| `edit_total_seconds` | `run_complete_pipeline`(또는 `run_edit`) 전체 |
| `unattributed_seconds_in_edit_total` | 전체 − 측정된 구간 합(감싸지 않은 코드: forward 교체, D51 isin, 파일 확인 등) |
| `process_wall_seconds`, `ru_maxrss_bytes` | 프로세스 시작부터의 벽시계, RSS 최대 |
| `calls` | span 이름별 호출 횟수 |
| `spans` | 모든 span: `name`, `parent`, `segment`, `seconds`, `self_seconds`, `t_start`/`t_end`(perf_counter), `max_memory_allocated`, `nvidia_smi_mib_max`, `rss_bytes_max` |
| `outputs` | GLB·`slat_tgt` 경로, `slat_tgt_sha256`, 포획 요약 |
| `host` | GPU, torch, CUDA, python, 저장소 커밋 |

구간 1·2 는 별도 JSON(`run_condition_render.py`, `run_inpaint.py` 의 `timing.json`)에 같은 `spans` 형식으로 남는다.

측정 규칙: span 경계마다 `torch.cuda.synchronize()`; VRAM 은 span 마다 `reset_peak_memory_stats()` →
`max_memory_allocated()`(중첩 시 바깥 피크 보존); `nvidia-smi`·RSS 는 0.1 s 샘플의 span 창 최대값.
