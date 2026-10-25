"""조건 렌더 — 0-1 에서는 포크 렌더러(`utils/render_rgb_and_mask.py`) 호출 래퍼만 둔다.

이 장비의 opencv 는 OpenEXR 없이 빌드돼 있어 0번에서 만든 계측 래퍼(`reference/run_condition_render.py`,
EXR 만 bpy 로 읽는 cv2 프록시)를 별도 프로세스로 부른다. R10a(3번)에서 교체한다.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .models import REPO, TRELLIS_ROOT


def render_condition(model_glb, mask_glb, out_dir, python=None) -> Path:
    py = python or os.environ.get("VHL_RENDER_PY", sys.executable)
    subprocess.run([py, str(REPO / "reference" / "run_condition_render.py"), "--model", str(model_glb),
                    "--mask", str(mask_glb), "--out", str(out_dir), "--reference", str(TRELLIS_ROOT)], check=True)
    return Path(out_dir) / "images"


# ── 2번 R4 렌더 입력 보정 (R-2, chief 승인 2026-10-26) ─────────────────────────────────────────
# glbbuild 정점색 GLB 는 COLOR_0 만 있고 재질이 없다 → Blender 4.0 glTF 가져오기가 정점색을 쓰지 않아 회색으로 렌더된다.
# 포크·glbbuild 는 고치지 않고, 렌더 입력 GLB 만 여기서 다듬는다:
#  (1) 재질 없는 프리미티브에 흰 비금속 기본 재질(baseColor 1, metallic 0, roughness 1) — glTF 규격상 COLOR_0 이 baseColor 에 곱해진다.
#  (2) COLOR_0 RGB 를 sRGB → 선형으로 바꿔 float32 로 쓴다. glTF 2.0 은 COLOR_0 을 **선형**으로 정의하고(규격 3.7.2.1),
#      TRELLIS 디코더 albedo(→ deltacontract.albedo_to_rgba8 → .cbin)는 텍스처 굽기(to_glb, sRGB 텍스처)와 같은 값을 쓰므로 sRGB 로 다룬다.
#      uint8 로 두면 어두운 색이 뭉개져(sRGB 0.04 → 선형 0.003 < 1/255) float32 로 쓴다.
def _srgb_to_linear(x):
    import numpy as np
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def vertex_color_render_glb(data: bytes) -> bytes:
    """정점색 GLB(바이트) → 렌더 입력용 GLB(바이트). 기하·면은 그대로다(2D 마스크가 바뀌지 않는다)."""
    import json
    import struct

    import numpy as np
    assert data[:4] == b"glTF", "GLB 가 아니다"
    jlen = struct.unpack("<I", data[12:16])[0]
    j = json.loads(data[20:20 + jlen])
    boff = 20 + jlen
    blen = struct.unpack("<I", data[boff:boff + 4])[0]
    bin_ = bytearray(data[boff + 8:boff + 8 + blen])
    j.setdefault("materials", [])
    mat_idx = len(j["materials"])
    j["materials"].append({"name": "vhl_vertex_color", "pbrMetallicRoughness": {
        "baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0}})
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
    for mesh in j["meshes"]:
        for pr in mesh["primitives"]:
            if pr.get("material") is None:
                pr["material"] = mat_idx
            ai = pr["attributes"].get("COLOR_0")
            if ai is None:
                continue
            acc = j["accessors"][ai]
            bv = j["bufferViews"][acc["bufferView"]]
            n, w = acc["count"], width[acc["type"]]
            off = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
            if acc["componentType"] == 5121:      # uint8 normalized
                c = np.frombuffer(bytes(bin_), np.uint8, n * w, off).reshape(n, w).astype(np.float64) / 255.0
            elif acc["componentType"] == 5126:    # float32
                c = np.frombuffer(bytes(bin_), np.float32, n * w, off).reshape(n, w).astype(np.float64)
            else:
                raise ValueError(f"COLOR_0 componentType {acc['componentType']} 미지원")
            if w == 3:
                c = np.concatenate([c, np.ones((n, 1))], 1)
            c[:, :3] = _srgb_to_linear(np.clip(c[:, :3], 0.0, 1.0))
            while len(bin_) % 4:
                bin_.append(0)
            start = len(bin_)
            bin_ += c.astype(np.float32).tobytes()
            j["bufferViews"].append({"buffer": 0, "byteOffset": start, "byteLength": n * 16})
            j["accessors"].append({"bufferView": len(j["bufferViews"]) - 1, "componentType": 5126, "count": n, "type": "VEC4"})
            pr["attributes"]["COLOR_0"] = len(j["accessors"]) - 1
    while len(bin_) % 4:
        bin_.append(0)
    j["buffers"][0]["byteLength"] = len(bin_)
    js = json.dumps(j, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    body = struct.pack("<I", len(js)) + b"JSON" + js + struct.pack("<I", len(bin_)) + b"BIN\x00" + bytes(bin_)
    return b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body
