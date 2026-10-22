"""편집 1회 — 포크 `run_edit` 의 W34 경로(입력 SLat 재사용 + 마스크 셀)를 이식.

입력은 메모리 객체(EditRequest)이고 파일 입출력은 service(읽기)·export(쓰기)가 한다.
출발 잡음(포크 512행 `torch.randn`)은 요청 seed 로 만든 CPU Generator 에서 뽑는다. 포크를
`torch.manual_seed(seed)` 직후 `run_edit` 로 부르면 그 사이 CPU 전역 RNG 소비가 없으므로(코드확인:
VAE 인코더는 sample_posterior=False, rembg 는 onnxruntime) 같은 텐서가 나온다 — tests/gpu 에서 비트 비교.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from . import masks
from .blocks import SlatFlow, SSFlow
from .models import Constants
from .preprocess import preprocess_images
from .sampler import InversionSampler


@dataclass
class EditRequest:
    slat_coords: torch.Tensor          # (N,4) 또는 (N,3) int
    slat_feats: torch.Tensor           # (N,8) float, denormalized
    mask_cells: np.ndarray             # (M,3) int, 64³ 편집 셀
    img_src: object                    # PIL 2d_render
    img_tgt: object                    # PIL 2d_edit
    img_mask: object                   # PIL 2d_mask
    seed: int = 0
    skip_step: int = 0
    re_init: bool = False
    cfg: tuple = (5.0, 6.0, 0.0, 0.0)  # stage1 inv, stage1 fwd, stage2 inv, stage2 fwd
    slat_meta: dict = field(default_factory=dict)


@dataclass
class EditResult:
    slat_tgt: object                   # sp.SparseTensor (denormalized)
    coords_tgt: torch.Tensor
    counts: dict
    caches: dict


class Editor:
    def __init__(self, pipeline, constants: Constants, rembg_session_factory=None):
        self.p = pipeline
        self.C = constants
        self.ss = SSFlow(pipeline.models["sparse_structure_flow_model"])
        self.slat = SlatFlow(pipeline.models["slat_flow_model"])
        self.sampler = InversionSampler()
        self._new_session = rembg_session_factory or (lambda: __import__("rembg").new_session("u2net"))

    # ── Stage 1 ──────────────────────────────────────────────────────────
    # origin: voxhammer/edit_pipeline.py:sample_sparse_structure_inverse @ f33cd96+patch
    def stage1_inverse(self, cond_src, voxel_src, cfg, skip_step, is_text=False):
        encoder = self.p.models["sparse_structure_encoder"]
        z_s = encoder(voxel_src)
        if cfg is None:
            cfg = self.p.sparse_structure_sampler_params["cfg_strength"]
        noise, latent, kv = self.sampler.sample(self.ss, 1, z_s, cond_src, cfg, skip_step=skip_step, is_text=is_text)
        return noise, latent, kv

    # origin: voxhammer/edit_pipeline.py:sample_sparse_structure_denoise @ f33cd96+patch
    def stage1_denoise(self, cond_tgt, noise, voxel_src, voxel_mask, latent, latent_mask, kv, self_kv_mask,
                       cross_kv_mask, cfg, skip_step, re_init, is_text=False):
        if cfg is None:
            cfg = self.p.sparse_structure_sampler_params["cfg_strength"]
        noise_init = self.p.models["sparse_structure_encoder"](voxel_src) if re_init else None
        z_s, latent, kv = self.sampler.sample(self.ss, 1, noise, cond_tgt, cfg, latent, latent_mask, kv,
                                              self_kv_mask, cross_kv_mask, skip_step, noise_init, is_text=is_text)
        voxel = self.p.models["sparse_structure_decoder"](z_s)
        # Stage 1 voxel 병합: 마스크 밖은 원본 점유를 그대로
        return voxel * voxel_mask + voxel_src * (1 - voxel_mask)

    # ── Stage 2 ──────────────────────────────────────────────────────────
    def _norm(self):
        d = self.p.device
        return (torch.tensor(self.p.slat_normalization["std"], device=d)[None],
                torch.tensor(self.p.slat_normalization["mean"], device=d)[None])

    # origin: voxhammer/edit_pipeline.py:sample_slat_inverse @ f33cd96+patch
    def stage2_inverse(self, cond_src, slat_src, coords_mask, cfg, is_text=False):
        """Stage 2 inversion 은 보존 좌표로 자른 SLat 만 돈다."""
        coords_mask = torch.cat([torch.zeros(coords_mask.shape[0], 1).int().cuda(), coords_mask], dim=1)
        sparse_tensor_mask = torch.zeros(slat_src.coords.shape[0], dtype=torch.bool, device="cuda")
        for coord in coords_mask:
            sparse_tensor_mask[torch.all(slat_src.coords == coord, dim=1)] = True
        slat_inverse = slat_src.replace(slat_src.feats[sparse_tensor_mask], slat_src.coords[sparse_tensor_mask])
        std, mean = self._norm()
        slat_inverse = (slat_inverse - mean) / std
        if cfg is None:
            cfg = self.p.sparse_structure_sampler_params["cfg_strength"]
        noise, latent, kv = self.sampler.sample(self.slat, 2, slat_inverse, cond_src, cfg, is_text=is_text)
        return latent, kv

    # origin: voxhammer/edit_pipeline.py:sample_slat_denoise @ f33cd96+patch
    def stage2_denoise(self, cond_tgt, coords_tgt, slat_src, coords_mask, latent, kv, self_kv_mask,
                       cross_kv_mask, cfg, generator: torch.Generator, is_text=False):
        import trellis.modules.sparse as sp
        coords_mask = torch.cat([torch.zeros(coords_mask.shape[0], 1).int().cuda(), coords_mask], dim=1)
        # 출발 잡음: 포크는 전역 RNG(CPU) → 여기서는 요청 seed 의 CPU Generator. 같은 shape·dtype·장치·순서
        noise_feats = torch.randn(coords_tgt.shape[0], self.slat.in_channels, generator=generator)
        noise = sp.SparseTensor(feats=noise_feats.to(self.p.device), coords=coords_tgt)
        if cfg is None:
            cfg = self.p.sparse_structure_sampler_params["cfg_strength"]
        slat, latent, kv = self.sampler.sample(self.slat, 2, noise, cond_tgt, cfg, latent, coords_mask, kv,
                                               self_kv_mask, cross_kv_mask, is_text=is_text)
        std, mean = self._norm()
        slat = slat * std + mean
        # Stage 2 보존 feats 최종 덮어쓰기
        match_1 = (coords_tgt.unsqueeze(1) == coords_mask.unsqueeze(0)).all(dim=-1)
        match_2 = (slat_src.coords.unsqueeze(1) == coords_mask.unsqueeze(0)).all(dim=-1)
        idx_1 = match_1.float().argmax(0)
        idx_2 = match_2.float().argmax(0)
        feats = slat.feats.clone()
        feats[idx_1] = slat_src.feats[idx_2]
        return slat.replace(feats)

    # ── 편집 1회 ─────────────────────────────────────────────────────────
    # origin: voxhammer/edit_pipeline.py:run_edit (slat_path·mask_cells 경로, is_text=False) @ f33cd96+patch
    @torch.no_grad()  # 포크 실행에서도 grad 는 꺼져 있다(extract_feature import 가 전역으로 끈다)
    def edit(self, req: EditRequest, hooks=None) -> EditResult:
        import trellis.modules.sparse as sp
        hk = hooks or {}
        span = hk.get("span") or (lambda name, seg: _null())
        C = self.C
        gen = torch.Generator("cpu").manual_seed(int(req.seed))
        cfg1i, cfg1f, cfg2i, cfg2f = req.cfg

        with span("load_slat", 3):
            if req.slat_meta.get("slat_space") not in (None, "denormalized"):
                raise ValueError(f"slat_space={req.slat_meta.get('slat_space')} — denormalized 여야 한다")
            c = torch.as_tensor(req.slat_coords)
            if c.shape[1] == 3:
                c = torch.cat([torch.zeros(c.shape[0], 1, dtype=c.dtype), c], dim=1)
            c = c.int().cuda()
            slat_src = sp.SparseTensor(feats=torch.as_tensor(req.slat_feats).float().cuda(), coords=c)
            coords_src = c[:, 1:].contiguous()
            voxel_src = masks.coords_to_voxel(coords_src, C)
        with span("preprocess_image", 3):
            pre_src, pre_tgt, pre_mask = preprocess_images(req.img_src, req.img_tgt, req.img_mask,
                                                           self._new_session(), self._new_session())
        with span("get_cond", 3):
            cond_src = self.p.get_cond([pre_src])
            cond_tgt = self.p.get_cond([pre_tgt])
        with span("masks", 3):
            coords_delete = torch.as_tensor(np.asarray(req.mask_cells), dtype=torch.int32).cuda()
            coords_preserve = masks.preserve_coords(coords_src, coords_delete, C)
            voxel_mask, ss_latent_mask, ss_self_kv_mask, cross_kv_mask = masks.ss_masks(coords_preserve, pre_mask, C)
        with span("stage1_inverse", 4):
            noise, ss_latent, ss_kv = self.stage1_inverse(cond_src, voxel_src, cfg1i, req.skip_step)
        with span("stage1_denoise", 4):
            voxel_tgt = self.stage1_denoise(cond_tgt, noise, voxel_src, voxel_mask, ss_latent, ss_latent_mask, ss_kv,
                                            ss_self_kv_mask, cross_kv_mask, cfg1f, req.skip_step, req.re_init)
            coords_tgt = torch.argwhere(voxel_tgt > 0)[:, [0, 2, 3, 4]].int()
        with span("slat_mask", 3):
            slat_kv_mask = masks.slat_self_kv_mask(coords_tgt, coords_preserve)
        with span("stage2_inverse", 6):
            slat_latent, slat_kv = self.stage2_inverse(cond_src, slat_src, coords_preserve, cfg2i)
        with span("stage2_denoise", 6):
            slat_tgt = self.stage2_denoise(cond_tgt, coords_tgt, slat_src, coords_preserve, slat_latent, slat_kv,
                                           slat_kv_mask, cross_kv_mask, cfg2f, gen)
        counts = {"coords_src": int(coords_src.shape[0]), "coords_delete": int(coords_delete.shape[0]),
                  "coords_preserve": int(coords_preserve.shape[0]), "coords_tgt": int(coords_tgt.shape[0])}
        return EditResult(slat_tgt=slat_tgt, coords_tgt=coords_tgt, counts=counts,
                          caches={"stage1": (ss_latent, ss_kv), "stage2": (slat_latent, slat_kv)})


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
