"""Transformer 블록과 flow 모델 forward — 명시적 래퍼 클래스(원본 모듈의 forward 를 바꾸지 않는다)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .attention import SlatAttention, SSAttention
from .cache import EditContext


class SSBlock:
    def __init__(self, block):
        self.b = block
        self.self_attn = SSAttention(block.self_attn)
        self.cross_attn = SSAttention(block.cross_attn)

    # origin: voxhammer/edit_pipeline.py:ss_trsfmr_forward @ f33cd96+patch
    def __call__(self, x, mod, context, ctx: EditContext):
        b = self.b
        if b.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = b.adaLN_modulation(mod).chunk(6, dim=1)
        h = b.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.self_attn(h, ctx)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = b.norm2(x)
        h = self.cross_attn(h, ctx, context)
        x = x + h
        h = b.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = b.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        return x + h


class SSFlow:
    """Stage 1 SparseStructureFlowModel forward (캐시·마스크 주입)."""

    def __init__(self, model):
        from trellis.modules.spatial import patchify, unpatchify
        self.m = model
        self.blocks = [SSBlock(b) for b in model.blocks]
        self._patchify, self._unpatchify = patchify, unpatchify

    # origin: voxhammer/edit_pipeline.py:ss_flow_forward @ f33cd96+patch
    def __call__(self, x, t, cond, ctx: EditContext):
        m = self.m
        assert [*x.shape] == [x.shape[0], m.in_channels, *[m.resolution] * 3], \
            f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], m.in_channels, *[m.resolution] * 3]}"
        h = self._patchify(x, m.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
        h = m.input_layer(h)
        h = h + m.pos_emb[None]
        t_emb = m.t_embedder(t)
        if m.share_mod:
            t_emb = m.adaLN_modulation(t_emb)
        t_emb = t_emb.type(m.dtype)
        h = h.type(m.dtype)
        cond = cond.type(m.dtype)
        for layer, block in enumerate(self.blocks):
            ctx.layer = layer
            h = block(h, t_emb, cond, ctx)
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = m.out_layer(h)
        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[m.resolution // m.patch_size] * 3)
        return self._unpatchify(h, m.patch_size).contiguous()


class SlatBlock:
    def __init__(self, block):
        self.b = block
        self.self_attn = SlatAttention(block.self_attn)
        self.cross_attn = SlatAttention(block.cross_attn)

    # origin: voxhammer/edit_pipeline.py:slat_trsfmr_forward @ f33cd96+patch
    def __call__(self, x, mod, context, ctx: EditContext):
        b = self.b
        if b.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = b.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(b.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h, ctx)
        h = h * gate_msa
        x = x + h
        h = x.replace(b.norm2(x.feats))
        h = self.cross_attn(h, ctx, context)
        x = x + h
        h = x.replace(b.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = b.mlp(h)
        h = h * gate_mlp
        return x + h


class SlatFlow:
    """Stage 2 SLatFlowModel forward (캐시·마스크 주입)."""

    def __init__(self, model):
        self.m = model
        self.blocks = [SlatBlock(b) for b in model.blocks]

    @property
    def in_channels(self):
        return self.m.in_channels

    # origin: voxhammer/edit_pipeline.py:slat_flow_forward @ f33cd96+patch
    def __call__(self, x, t, cond, ctx: EditContext):
        m = self.m
        h = m.input_layer(x).type(m.dtype)
        t_emb = m.t_embedder(t)
        if m.share_mod:
            t_emb = m.adaLN_modulation(t_emb)
        t_emb = t_emb.type(m.dtype)
        cond = cond.type(m.dtype)
        skips = []
        for block in m.input_blocks:
            h = block(h, t_emb)
            skips.append(h.feats)
        if m.pe_mode == "ape":
            h = h + m.pos_embedder(h.coords[:, 1:]).type(m.dtype)
        for layer, block in enumerate(self.blocks):
            ctx.layer = layer
            h = block(h, t_emb, cond, ctx)
        for block, skip in zip(m.out_blocks, reversed(skips)):
            if m.use_skip_connection:
                h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb)
            else:
                h = block(h, t_emb)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        return m.out_layer(h.type(x.dtype))
