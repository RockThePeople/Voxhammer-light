# Voxhammer-light

VoxHammer 기반 3D 국소 편집의 1회 대기 시간을 줄이는 구현.
요구사항: [ops/REQUIREMENTS.md](ops/REQUIREMENTS.md) · 진행 상태: [ops/STATE.md](ops/STATE.md) · 세션 규칙: [CLAUDE.md](CLAUDE.md)

## 구성

| 경로 | 내용 |
|---|---|
| `third_party/VoxHammer` | 기준 구현 submodule (f33cd96 고정, 수정 금지) |
| `third_party/ai-ar-v3` | 청크 계약·서버 submodule (읽기 전용) |
| `patches/` | 기준 패치 `edit_pipeline.f33cd96.patch` 와 sha256 |
| `tools/` | `check_pins.py` (고정 검사), `setup_reference.sh` (패치 적용본 빌드) |
| `reference/` | 기준 구현 계측 래퍼 (0번) |
| `editcore/` | 모듈별 이식 패키지 (0-1번부터) |
| `services/` | A5000 편집 워커, 3090 인페인팅 워커 |
| `tests/cpu`, `tests/gpu` | 동등성 테스트 |
| `ops/` | 요구사항, 진행 상태, 항목별 지시서 |
| `docs/reports/` | 항목별 보고서 |

## 시작

```bash
git clone --recurse-submodules https://github.com/RockThePeople/Voxhammer-light
cd Voxhammer-light
python3 tools/check_pins.py
bash tools/setup_reference.sh        # build/reference/ 에 패치 적용본
cp .env.example .env                 # 값은 로컬에서만
```
