"""렌더용 최소 MeshExtractResult 생성.

`MeshRenderer.render(return_types=["color"])` 가 요구하는 필드
(vertices / faces / vertex_attrs / face_normal)만 채운다. 합성물(.cbin 조립)을
기존 렌더 경로로 그리기 위한 얇은 어댑터다 — 벤더 트리는 건드리지 않는다.
"""
import numpy as np
import torch


def make_mesh(vertices, faces, colors):
    from trellis1.representations.mesh import MeshExtractResult

    v = torch.as_tensor(np.ascontiguousarray(vertices), dtype=torch.float32, device="cuda")
    f = torch.as_tensor(np.ascontiguousarray(faces), dtype=torch.long, device="cuda")
    c = torch.as_tensor(np.ascontiguousarray(colors), dtype=torch.float32, device="cuda")
    # vertex_attrs 는 (V,6) — 앞 3채널 albedo, 뒤 3채널 normal. 렌더는 앞 3채널만 쓴다.
    attrs = torch.cat([c, torch.full_like(c, 0.5)], dim=1)
    return MeshExtractResult(vertices=v, faces=f, vertex_attrs=attrs, res=256)
