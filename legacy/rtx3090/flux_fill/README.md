# legacy/rtx3090/flux_fill — 3090 인페인팅 도구 보관본

- 출처: 3090 워크스테이션 `<HOME>/ai-ar-v3-assets/_tools/` (git 저장소 아님). 2026-10-22 복사.
- 용도: 5번 R9(인페인팅 3090 이전)의 출발점. **0번에서는 실행하지 않는다.**
- 가중치는 포함하지 않는다. 쓰는 가중치: `nunchaku-ai/nunchaku-flux.1-fill-dev` (`svdq-int4_r32`),
  `camenduru/FLUX.1-Fill-dev-ungated` (원본 `black-forest-labs/FLUX.1-Fill-dev` 의 비게이트 미러).
  미러 파일 sha256 은 `docs/reports/00-kickoff-3090.md` B-4.

## 파일과 원본 대조

| 파일 | 원본 sha256 | 보관본 sha256 | 변경 |
|---|---|---|---|
| `flux_fill.py` | `87c93d5e286dd2ce5d4d91be6a12c86e5e78e9a6dc2a8eb2784dc80a8365900c` | 동일 | 없음 |
| `flux_fill_bench.py` | `9f7e70e649643dbc091b9e047e1ba9dd2d423d8bd1a668d8bd6db060d8f82ebc` | 동일 | 없음 |
| `flux_fill_ab_bench.py` | `95ff43a5e9b51d9bd069dc59a2207e07025972ae4c3f035ca2e235c6a63a5d4c` | 동일 | 없음 |
| `flux_fill_probe.py` | `0c2e3d9262cab992a6b6d2f3687f6110742e8810c19a3b562abbcc9bc3126d30` | 동일 | 없음 |
| `README.flux-fill.md` | `80fc9dd81f7fc94864060f682b6052ab4db3dc41cd666d30215a8d8bfed8df4e` | `8d9b6a26a7b4b49da943e9b774aef43d1f7113c29b0bc8c69463a8d54a05ee2b` | 6·43행 `~/` → `<HOME>/` |
| `setup_fluxfill.sh` | `10f668f244cff9a38d644bd9a5754252d51101ad8b68aee2a3690c6cc2e9408f` | `a9e50c8d7fe1b9b39bf75a26c1b9d821cbc3812862d22d18c652e3f8fa8ee105` | 7·9행 `~/` → `<HOME>/` |

홈 경로 자리표시자 외에는 바꾸지 않았다. 키·호스트·포트는 원본에 없다.

## 원본 README 측정값의 지위

`README.flux-fill.md` 의 수치(INT4 1024²·30스텝 호출 15.62 s, peak 18.1 GiB; bf16+CPU offload 59.50 s,
peak 22.8 GiB; 품질 비교 표)는 이전 작업에서 기록된 것이고 이 저장소에서 재현하지 않았다 → **`미검증`**.
0번에서는 재실행하지 않는다(지시서 B-4). R9 에서 6장 계측 규칙으로 다시 잰다.

## 이 장비의 실행 환경 (00a 실측)

conda `flux-fill`: Python 3.11.15, torch 2.9.1+cu128, nunchaku 1.2.1+cu12.8torch2.9, diffusers 0.39.0,
transformers 4.57.6. pip freeze 는 `env/rtx3090-flux-fill.txt`. 0번에서 이 env 는 사용·변경하지 않는다.

## 라이선스

FLUX.1-Fill-dev 는 FLUX.1 [dev] Non-Commercial License 다(원본 README 참조).
