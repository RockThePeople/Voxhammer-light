"""마스크·좌표 계산 — 포크와 같은 연산·같은 순서(비트 동일 대상). 파일 입출력 없음."""
from __future__ import annotations

import torch
import torchvision.transforms as transforms

from .models import Constants


# origin: voxhammer/edit_pipeline.py:coords_to_voxel @ f33cd96+patch
def coords_to_voxel(coords: torch.Tensor, C: Constants) -> torch.Tensor:
    r = C.voxel_res
    voxel = torch.zeros(1, 1, r, r, r, dtype=torch.float)
    voxel[:, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return voxel.cuda()


# origin: voxhammer/edit_pipeline.py:run_edit (D51 보존 판정, 612행) @ f33cd96+patch
def preserve_coords(coords_src: torch.Tensor, coords_delete: torch.Tensor, C: Constants) -> torch.Tensor:
    """(x,y,z) 를 정수 하나로 부호화해 행 단위로 판정한다(D51). 4096·64 = voxel_res², voxel_res."""
    r = C.voxel_res
    code = lambda c: c[:, 0] * (r * r) + c[:, 1] * r + c[:, 2]  # noqa: E731
    return coords_src[~torch.isin(code(coords_src), code(coords_delete))]


# origin: voxhammer/edit_pipeline.py:ply_to_ss_mask @ f33cd96+patch
def ss_masks(coords_mask: torch.Tensor, pre_mask, C: Constants):
    """Stage 1 마스크 4종. coords_mask = 보존 셀(run_edit 은 coords_preserve 를 넘긴다).
    반환: voxel_mask(1=편집), ss_latent_mask, ss_self_kv_mask(heads·head_dim), cross_kv_mask(+CLS·register)."""
    r, n, g, ps = C.voxel_res, C.ss_resolution, C.patch_grid, C.patch_size
    half = r // 2
    voxel_mask = torch.ones(1, 1, r, r, r, dtype=torch.float)
    voxel_mask[:, 0, coords_mask[:, 0], coords_mask[:, 1], coords_mask[:, 2]] = 0
    voxel_mask = voxel_mask.cuda()

    ss_latent_mask = voxel_mask.reshape(1, 1, half, 2, half, 2, half, 2)
    ss_latent_mask = ss_latent_mask.permute(0, 1, 3, 5, 7, 2, 4, 6)
    ss_latent_mask = ss_latent_mask.reshape(1, 8, half, half, half)
    ss_latent_mask = ss_latent_mask.reshape(1, 8, n, 2, n, 2, n, 2)
    ss_latent_mask = ss_latent_mask.permute(0, 1, 3, 5, 7, 2, 4, 6)
    ss_latent_mask = ss_latent_mask.reshape(1, 64, n, n, n)
    ss_latent_mask = ss_latent_mask.all(dim=1, keepdim=True)
    ss_latent_mask = ss_latent_mask.repeat(1, C.latent_channels, 1, 1, 1).contiguous().float().cuda()

    ss_self_kv_mask = ss_latent_mask.reshape(1, C.latent_channels, -1)
    ss_self_kv_mask = ss_self_kv_mask.permute(0, 2, 1)
    ss_self_kv_mask = ss_self_kv_mask.all(dim=2, keepdim=True)
    ss_self_kv_mask = ss_self_kv_mask.repeat(1, 1, C.ss_heads * C.ss_head_dim)
    ss_self_kv_mask = ss_self_kv_mask.reshape(1, C.ss_tokens, C.ss_heads, C.ss_head_dim)
    ss_self_kv_mask = ss_self_kv_mask.permute(0, 2, 1, 3).contiguous().float().cuda()

    if pre_mask is None:
        cross_kv_mask = None
    else:
        ch = C.ss_heads * C.ss_head_dim
        img_mask = transforms.ToTensor()(pre_mask)
        img_mask = (img_mask > 0).float()
        cross_kv_mask = img_mask.reshape(g, ps, g, ps)
        cross_kv_mask = cross_kv_mask.permute(1, 3, 0, 2)
        cross_kv_mask = cross_kv_mask.reshape(1, ps * ps, g, g)
        cross_kv_mask = cross_kv_mask.any(dim=1, keepdim=True)
        cross_kv_mask = cross_kv_mask.repeat(1, ch, 1, 1)
        cross_kv_mask = cross_kv_mask.reshape(1, ch, g * g)
        cross_kv_mask = cross_kv_mask.permute(0, 2, 1)
        # CLS·register 토큰은 항상 편집 쪽(1) — 이미지 패치가 아니다
        cross_kv_mask = torch.cat((torch.ones(1, C.cond_extra_tokens, ch), cross_kv_mask), dim=1)
        cross_kv_mask = cross_kv_mask.reshape(1, C.cond_tokens, C.ss_heads, C.ss_head_dim)
        cross_kv_mask = cross_kv_mask.permute(0, 2, 1, 3).contiguous().float().cuda()
    return voxel_mask, ss_latent_mask, ss_self_kv_mask, cross_kv_mask


# origin: voxhammer/edit_pipeline.py:ply_to_slat_mask @ f33cd96+patch
def slat_self_kv_mask(coords_tgt: torch.Tensor, coords_mask: torch.Tensor) -> torch.Tensor:
    """Stage 2 Transformer 단(2배 다운샘플) 보존 좌표 목록 (batch 열 포함)."""
    coords_mask = torch.cat([torch.zeros(coords_mask.shape[0], 1).int().cuda(), coords_mask], dim=1)
    factor = (2, 2, 2)
    coord_tgt = list(coords_tgt.unbind(dim=-1))
    coord_mask = list(coords_mask.unbind(dim=-1))
    for i, f in enumerate(factor):
        coord_tgt[i + 1] = coord_tgt[i + 1] // f
        coord_mask[i + 1] = coord_mask[i + 1] // f
    MAX = [coord_tgt[i + 1].max().item() + 1 for i in range(3)]
    OFFSET = torch.cumprod(torch.tensor(MAX[::-1]), 0).tolist()[::-1] + [1]
    code_tgt = sum([c * o for c, o in zip(coord_tgt, OFFSET)])
    code_mask = sum([c * o for c, o in zip(coord_mask, OFFSET)])
    code_tgt, idx_tgt = code_tgt.unique(return_inverse=True)
    code_mask, idx_mask = code_mask.unique(return_inverse=True)
    new_coords_mask = torch.stack([code_mask // OFFSET[0]] + [(code_mask // OFFSET[i + 1]) % MAX[i] for i in range(3)], dim=-1)
    return new_coords_mask.contiguous().cuda()
