"""editcore — VoxHammer 편집 경로의 모듈 이식본 (0-1, 결과 동일·개선 없음).

출처: VoxHammer `voxhammer/edit_pipeline.py` @ f33cd96 + 기준 패치(b886a283) — MIT License,
Copyright (c) the VoxHammer authors. TRELLIS 모델 코드(`trellis/`)는 VoxHammer 동봉본을 수정 없이
import 한다 — TRELLIS 는 MIT License, Copyright (c) Microsoft Corporation.
각 함수의 출처는 `# origin: voxhammer/edit_pipeline.py:<함수> @ f33cd96+patch` 주석으로 적는다.

모듈: models(모델 적재·상수 유도), preprocess, masks, cache, attention, blocks, sampler,
pipeline(run_edit 대응), export(디코딩·GLB·R2 저장), render(포크 렌더러 호출), service(입력 파일 읽기).
"""
