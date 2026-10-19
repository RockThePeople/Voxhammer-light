# 고객사 검증 스크립트 (수정 금지)

고객사가 제공한 원본이다. 내용을 고치지 않는다. 해시는 `SHA256SUMS`.
GPU·실제 규모 실행이 필요하면 **별도 사본**(`tests/customer/gpu/`)을 만들고, 원본 대비 diff 를
보고서에 그대로 붙인다.

| 스크립트 | 인자 | 검증 내용 |
|---|---|---|
| `v_stage1_cache.py` | 패치 적용 VoxHammer 경로 (`build/reference`) | Stage 1 캐시: 보존 행 K/V·hidden 저장, cross 1회 저장의 최종 z 동등성, 음성 대조 2종 |
| `v_stage2_misc.py` | 패치 적용 VoxHammer 경로 | Stage 2 매칭 1회화 동등성, `argmax(0)` 결함 재현, CFG 배치 동등성 |
| `v_cbin_direct.py` | ai-ar-v3 경로 (`third_party/ai-ar-v3`) | mesh → `.cbin` 직행 청크화·인코딩 시간, 변경 청크 수, 단위 법선 |

의존성(원본 그대로 실행 시): `torch`, `numpy`, `tqdm`, `easydict`, `trimesh`, `scipy`.
`trellis/__init__.py` 는 스텁으로 우회하므로 rembg·spconv 실연산은 필요 없다.
스크립트는 `torch.Tensor.cuda` 를 항등으로 바꿔 CPU fp32 로 돈다.

## chief 사전 실행 결과 (2026-10-19, CPU, torch 2.14.1, build/reference = f33cd96 + 패치 b886a283)

| 스크립트 | 결과 | 부록 대비 |
|---|---|---|
| v_stage1_cache | 원본 캐시 419.56 MB · cross 비트 동일 True · A 11.9% 비트 동일 · B 5.9% 비트 동일 · 음성대조 5.282e-06 / 4.130e-02 | 일치 |
| v_stage2_misc | 매칭 비트 동일 True · 원본 127 ms → 인덱스 0.9 ms · argmax 0번 덮어씀 True · 개선판 오류 중단 · CFG 배치 max\|Δ\| 1.42e-06, 비트 동일 False | 일치 |
| v_cbin_direct | 968 청크 · 9.4 MB · partition+encode 1.98 / 1.64 s · changed 68/972 · removed 12 · 단위 법선 100% | 일치 (시간은 CPU 차이) |

주의: `v_cbin_direct.py` 는 합성 메시에서 변경 청크를 **해시**로 정한다. 운영 경로(R11)는 점유 비교로 정하며 해시를 쓰지 않는다.
