"""청크 메시 직접 렌더. to_glb() 는 쓰지 않는다.

`MeshRenderer.render(return_types=["color"])` 가 `vertex_attrs[:, :3]` 를 보간한다
(renderers/mesh_renderer.py:118). 그래서 새 셰이더를 짤 필요 없이 **정점 색을
덮어쓰기만 하면** 마스크·델타 시각화를 기존 경로로 낼 수 있다.
"""
import os

import numpy as np
import torch
from PIL import Image

# 각도 키는 LAB_API.md §2 고정: front / left / back / top.
# (6차에서 쓰던 "side" 는 문서 어휘가 아니라 "left" 로 맞췄다.)
# z 가 상단축이다(선인장이 z 로 길다).
VIEWS = {
    "front": (0.0, 0.15),
    "left": (np.pi / 2, 0.15),
    "back": (np.pi, 0.15),
    "top": (0.0, 1.30),
}
DEFAULT_VIEWS = ["front", "left", "back", "top"]

GRAY = (0.55, 0.55, 0.55)
RED = (0.90, 0.12, 0.12)
ORANGE = (0.98, 0.55, 0.10)


def _renderer(resolution):
    from trellis1.renderers import MeshRenderer

    r = MeshRenderer()
    r.rendering_options.resolution = resolution
    r.rendering_options.near = 1
    r.rendering_options.far = 100
    r.rendering_options.ssaa = 4
    return r


def mask_view(centroid):
    """마스크 무게중심을 정면에서 바라보는 (yaw, pitch).

    편집이 뒤쪽이나 위쪽에 있으면 고정 4각도로는 안 보인다. 무게중심 방향에서
    카메라를 놓아 편집 영역이 반드시 화면에 오게 한다.
    """
    x, y, z = (float(v) for v in centroid)
    yaw = float(np.arctan2(x, y))          # render_utils 의 orig 정의와 같은 규약
    pitch = float(np.arctan2(z, np.hypot(x, y) + 1e-6))
    return yaw, float(np.clip(pitch, -1.2, 1.2))


def render_views(mesh, out_dir, prefix, colors=None, resolution=512, r=2.0, fov=40.0,
                 views=None, extra=None):
    """각도 이름 → 상대경로 dict 를 돌려준다 (LAB_API.md §2).

    colors: (V,3) float [0,1] 이면 정점 색을 대체한다.
    extra:  {이름: (yaw, pitch)} 추가 각도. edit 의 "mask" 각도용.
    배경은 검정이 아니라 흰색으로 깐다 — 회색 메시가 검정 배경에서 잘 안 보인다.
    """
    from trellis1.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics

    plan = {k: VIEWS[k] for k in (views or DEFAULT_VIEWS) if k in VIEWS}
    plan.update(extra or {})

    orig_attrs = mesh.vertex_attrs
    if colors is not None:
        c = torch.as_tensor(colors, dtype=torch.float32, device=mesh.vertices.device)
        mesh.vertex_attrs = torch.cat([c, orig_attrs[:, 3:]], dim=1)

    try:
        rend = _renderer(resolution)
        out_map = {}
        for name, (yaw, pitch) in plan.items():
            extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
            with torch.no_grad():
                out = rend.render(mesh, extr, intr, return_types=["color", "mask"])
            img = out["color"].detach().cpu().numpy().transpose(1, 2, 0)
            alpha = out["mask"].detach().cpu().numpy()
            if alpha.ndim == 3:
                alpha = alpha[0]
            alpha = alpha[..., None]
            img = np.clip(img, 0, 1) * alpha + (1.0 - alpha)   # 흰 배경 합성
            Image.fromarray((img * 255).astype(np.uint8)).save(f"{out_dir}/{prefix}_{name}.png")
            out_map[name] = f"render/{prefix}_{name}.png"
        return out_map
    finally:
        mesh.vertex_attrs = orig_attrs


def mask_colors(verts, core_codes, halo_codes):
    """마스크 시각화용 정점 색.

    ⚠️ 청크가 아니라 **복셀** 단위로 칠한다. 지시는 청크 단위였지만, 청크는 8³
    복셀이라 복셀 1,981개가 마스킹된 편집을 청크 12개(= 최대 6,144 복셀)로 칠하면
    편집 범위가 실제보다 훨씬 넓어 보인다. "여기가 편집된다"를 정확히 보여주는 게
    이 그림의 목적이라 복셀 단위를 택했다. 델타 그림은 청크 단위가 맞으므로
    그쪽은 청크로 칠한다.
    """
    from deltacontract import normalized_to_voxel, voxel_code

    codes = voxel_code(normalized_to_voxel(verts))
    col = np.tile(np.array(GRAY, np.float32), (len(verts), 1))
    if halo_codes:
        col[np.isin(codes, np.fromiter(halo_codes, np.int64, len(halo_codes)))] = ORANGE
    if core_codes:
        col[np.isin(codes, np.fromiter(core_codes, np.int64, len(core_codes)))] = RED
    return col


MAGENTA = (1.0, 0.0, 0.85)


def render_merged(chunk_dir, out_dir, prefix, views=None, extra=None,
                  seam_book=None, parent_present=None, resolution=512, r=2.0, fov=40.0):
    """실제 합성물(옛 몸통 + 새 머리)을 .cbin 에서 직접 조립해 렌더한다.

    지금까지 렌더한 건 전부 "새로 디코딩한 전체"였지 클라이언트가 받게 될
    합성물이 아니었다. 이건 승계된 청크의 옛 바이트를 그대로 써서 그린다.
    """
    import glob

    from deltacontract import decode

    from .representation import make_mesh

    pos, col, idx, off = [], [], [], 0
    for p in sorted(glob.glob(os.path.join(chunk_dir, "*.cbin"))):
        m = decode(open(p, "rb").read())
        if m.positions.shape[0] == 0:
            continue
        pos.append(m.positions)
        c = (m.colors[:, :3].astype(np.float32) / 255.0 if m.colors is not None
             else np.full((m.positions.shape[0], 3), 0.6, np.float32))
        col.append(c)
        idx.append(m.indices.reshape(-1, 3).astype(np.int64) + off)
        off += m.positions.shape[0]
    if not pos:
        return {}
    V = np.concatenate(pos).astype(np.float32)
    C = np.concatenate(col).astype(np.float32)
    F = np.concatenate(idx)

    mesh = make_mesh(V, F, C)
    return render_views(mesh, out_dir, prefix, colors=C, resolution=resolution,
                        r=r, fov=fov, views=views, extra=extra)


def delta_colors(verts, chunk_change, seam_mask=None):
    """변경 청크를 변화량에 따라 색조로. chunk_change: {chunk_key: 0.0~1.0}.

    0 = 회색(변화 없음) → 1 = 빨강. 정점이 속한 청크로 칠한다 — 델타는 청크
    단위로 전송되므로 여기서는 청크 입도가 맞다.
    """
    from deltacontract import chunk_key, normalized_to_chunk

    cids = normalized_to_chunk(verts)
    col = np.tile(np.array(GRAY, np.float32), (len(verts), 1))
    keys = np.array([chunk_key(c) for c in cids])
    for k, mag in chunk_change.items():
        m = keys == k
        if not m.any():
            continue
        t = float(np.clip(mag, 0.0, 1.0))
        col[m] = (
            GRAY[0] + (RED[0] - GRAY[0]) * t,
            GRAY[1] + (RED[1] - GRAY[1]) * t,
            GRAY[2] + (RED[2] - GRAY[2]) * t,
        )
    # 이음새 정점을 자홍색으로. 숫자가 작아도 눈에 보이면 문제고,
    # 커도 안 보이면 문제가 아니다.
    if seam_mask is not None and np.any(seam_mask):
        col[np.asarray(seam_mask, bool)] = MAGENTA
    return col
