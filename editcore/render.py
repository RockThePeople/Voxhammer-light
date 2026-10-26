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


def _fork_condition_render(model_glb, mask_glb, out_dir, python=None) -> dict:
    """포크 `utils/render_rgb_and_mask.process` 를 수정 없이 부르는 0번 계측 래퍼(EXR 은 bpy 프록시)를 별도 프로세스로."""
    py = python or os.environ.get("VHL_RENDER_PY", sys.executable)
    subprocess.run([py, str(REPO / "reference" / "run_condition_render.py"), "--model", str(model_glb),
                    "--mask", str(mask_glb), "--out", str(out_dir), "--reference", str(TRELLIS_ROOT)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import json
    return json.loads((Path(out_dir) / "timing.json").read_text())


# ── 04L 조건 렌더 (R8a A-2 규칙, R10a 는 나중에 render_condition 안에 끼운다) ───────────────────────
# 시점 정의: 포크 render_rgb_and_mask.py — 반지름 1.8, 고도 15°, 방위 [90,45,0,315,270][v] − 90, 원점 주시,
# bpyrenderer 기본 센서 32 mm·렌즈 35 mm(세로 화각 49.13°), 1024².
FORK_VIEWS = 5
_AZ = [90, 45, 0, 315, 270]


def _cells_boundary_mesh(cells):
    """마스크 셀 합집합의 노출면(닫힌 표면) — scenarios/s5_masks.py:boundary_mesh 와 같다. 좌표는 복셀 프레임 [-0.5, 0.5]."""
    # origin: scenarios/s5_masks.py:boundary_mesh (00s2 동결 마스크 GLB 생성)
    import numpy as np
    occ = {tuple(int(x) for x in c) for c in np.asarray(cells).reshape(-1, 3)}
    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    verts, faces = [], []
    for (i, j, k) in occ:
        for d in dirs:
            if (i + d[0], j + d[1], k + d[2]) in occ:
                continue
            ax = int(np.nonzero(d)[0][0])
            o = [i, j, k]
            o[ax] += 1 if d[ax] > 0 else 0
            u, v = [x for x in range(3) if x != ax]
            q = []
            for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                p = list(o); p[u] += du; p[v] += dv
                q.append(p)
            if d[ax] < 0:
                q = q[::-1]
            if ax == 1:
                q = q[::-1]
            b = len(verts)
            verts.extend(q)
            faces.extend([(b, b + 1, b + 2), (b, b + 2, b + 3)])
    return np.asarray(verts, float) / 64.0 - 0.5, np.asarray(faces, np.int64)


def _voxel_to_glb(v):
    import numpy as np
    v = np.asarray(v, float)
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)   # ai-ar-v3 frames.VOXEL_TO_GLB (D9)


def write_mask_glb(cells, xf, path):
    """조건 렌더용 mask.glb: 셀 노출면 → GLB 프레임 → 판본 정규화 (p − c) × s (s5_masks 와 같은 변환)."""
    import numpy as np
    import trimesh
    v, f = _cells_boundary_mesh(cells)
    vg = (_voxel_to_glb(v) - np.asarray(xf["center"])) * float(xf["scale"])
    trimesh.Trimesh(vg, f, process=True).export(str(path))
    return Path(path)


def mask_area_per_view(cells, xf, res=256):
    """마스크 셀(halo 전)을 5시점 카메라로 투영한 화면 면적(화소 수, res²). 렌더 없이 계산(가림 없음)."""
    import math
    import numpy as np
    from PIL import Image, ImageDraw
    v, f = _cells_boundary_mesh(cells)
    g = (_voxel_to_glb(v) - np.asarray(xf["center"])) * float(xf["scale"])
    b = np.stack([g[:, 0], -g[:, 2], g[:, 1]], 1)        # glTF(Y-up) → Blender(Z-up): (x, −z, y)
    t = 16.0 / 35.0
    out = []
    for view in range(FORK_VIEWS):
        el, az = math.radians(15), math.radians(_AZ[view] - 90)
        cam = 1.8 * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        fwd = -cam / np.linalg.norm(cam)
        r = np.cross(fwd, [0, 0, 1]); r /= np.linalg.norm(r)
        u = np.cross(r, fwd)
        d = b - cam
        z = d @ fwd
        px = (d @ r / z / t * 0.5 + 0.5) * res
        py = (0.5 - d @ u / z / t * 0.5) * res
        im = Image.new("1", (res, res), 0)
        dr = ImageDraw.Draw(im)
        for tri in f:
            dr.polygon([(px[i], py[i]) for i in tri], fill=1)
        out.append(int(np.asarray(im).sum()))
    return out


def choose_view(cells, xf):
    """R8a A-2: view 생략 시 투영 면적 최대 시점(동률이면 낮은 번호)."""
    a = mask_area_per_view(cells, xf)
    return int(max(range(FORK_VIEWS), key=lambda i: (a[i], -i))), a


def render_condition(mesh, mask_cells, view, *, xf, out_dir, python=None) -> dict:
    """조건 렌더 1회: 원본 메시(GLB) + 마스크 셀 → 요청 시점 1장(2d_render.png) + 2D 마스크(2d_mask.png).
    지금은 포크 렌더러(5시점, 수정 없음)를 그대로 쓰고 요청 시점만 고른다. R10a 는 이 함수 안에 끼운다."""
    import shutil
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mask_glb = write_mask_glb(mask_cells, xf, out / "mask.glb")
    t = _fork_condition_render(mesh, mask_glb, out / "fork", python=python)
    img = out / "fork" / "images"
    shutil.copyfile(img / f"render_{int(view):04d}.png", out / "2d_render.png")
    shutil.copyfile(img / f"mask_{int(view):04d}.png", out / "2d_mask.png")
    spans = {s["name"]: s["seconds"] for s in t.get("spans", [])}
    return {"render": out / "2d_render.png", "mask": out / "2d_mask.png", "view": int(view),
            "timing": {"total_s": spans.get("condition_render_total"), "source_5views_s": spans.get("render_source_5views"),
                       "mask_5views_s": spans.get("render_mask_5views"), "import_s": t.get("import_seconds")},
            "renderer": {"name": "fork utils/render_rgb_and_mask.py (Blender Cycles)", "views": FORK_VIEWS, "resolution": 1024,
                         "exr_reader": t.get("exr_reader")}}


# ── 2번 R4 렌더 입력 보정 (R-2, chief 승인 2026-10-26) ─────────────────────────────────────────
# glbbuild 정점색 GLB 는 COLOR_0 만 있고 재질이 없다 → Blender 4.0 glTF 가져오기가 정점색을 쓰지 않아 회색으로 렌더된다.
# 포크·glbbuild 는 고치지 않고, 렌더 입력 GLB 만 여기서 다듬는다:
#  (1) 재질 없는 프리미티브에 기본 재질(baseColor 1, roughness 1, metallicFactor 생략 = glTF 기본 1.0) — to_glb 의
#      PBRMaterial(roughnessFactor=1.0, metallicFactor 미지정)과 같은 재질이라 기존 텍스처 GLB 와 같은 모습으로 렌더된다
#      (3-8 색 원인 분리: 비금속은 밝고 대비가 낮음). glTF 규격상 COLOR_0 이 baseColor 에 곱해진다.
#  (2) COLOR_0 RGB 를 sRGB → 선형으로 바꿔 float32 로 쓴다. glTF 2.0 은 COLOR_0 을 **선형**으로 정의하고(규격 3.7.2.1),
#      TRELLIS 디코더 albedo(→ deltacontract.albedo_to_rgba8 → .cbin)는 텍스처 굽기(to_glb, sRGB 텍스처)와 같은 값을 쓰므로 sRGB 로 다룬다.
#      uint8 로 두면 어두운 색이 뭉개져(sRGB 0.04 → 선형 0.003 < 1/255) float32 로 쓴다.
def _srgb_to_linear(x):
    import numpy as np
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def vertex_color_render_glb(data: bytes, *, metallic: float | None = None, linearize: bool = True) -> bytes:
    """정점색 GLB(바이트) → 렌더 입력용 GLB(바이트). 기하·면은 그대로다(2D 마스크가 바뀌지 않는다).

    metallic: 기본 재질의 metallicFactor. None 이면 생략 → glTF 기본 1.0(to_glb 의 PBRMaterial 과 같은 재질).
    linearize: COLOR_0 RGB sRGB → 선형 변환 여부."""
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
    pbr = {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "roughnessFactor": 1.0}
    if metallic is not None:
        pbr["metallicFactor"] = float(metallic)
    j["materials"].append({"name": "vhl_vertex_color", "pbrMetallicRoughness": pbr})
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
            if linearize:
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
