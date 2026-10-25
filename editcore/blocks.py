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


def fixed_order_segment_mean(feats, idx, n):
    """`torch.scatter_reduce(zeros, 0, idx, feats, reduce='mean')`(include_self=True)와 같은 평균을 원자적 덧셈 없이 계산한다.

    목표 칸 idx 로 안정 정렬 → 칸마다 슬롯 번호(원래 순서) → [n, K, C] 에 배치 → 0 에서 시작해 슬롯 순서대로 더함
    → (개수 + 1) 로 나눔(include_self 의 0 이 개수에 들어간다). 누적 dtype 은 입력 dtype 그대로라, 결과는 원본의
    원자적 덧셈이 낼 수 있는 덧셈 순서 중 하나와 같다.
    """
    C = feats.shape[1]
    order = torch.sort(idx, stable=True).indices
    seg = idx[order]
    cnt = torch.bincount(idx, minlength=n)
    start = torch.cumsum(cnt, 0) - cnt
    slot = torch.arange(idx.shape[0], device=idx.device) - start[seg]
    dense = feats.new_zeros(n, int(cnt.max()), C)
    dense[seg, slot] = feats[order]
    acc = feats.new_zeros(n, C)
    for k in range(dense.shape[1]):
        acc = acc + dense[:, k]
    return acc / (cnt + 1).to(feats.dtype)[:, None]


class DeterministicSparseDownsample(torch.nn.Module):
    """정오표 E16 검증 전용: TRELLIS `SparseDownsample`(평균 풀링)을 바꿔 끼우는 결정적 모듈.

    `scatter_reduce(mean)` 만 `fixed_order_segment_mean` 으로 바꾸고 나머지(좌표 부호화·공간 캐시 등록)는 원본과 같다.
    `calls` 는 경로가 실제로 돌았다는 증거로 로그에 남긴다.
    """
    calls = 0

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    # origin: trellis/modules/sparse/spatial.py:SparseDownsample.forward @ f33cd96+patch
    def forward(self, input):
        DeterministicSparseDownsample.calls += 1
        DIM = input.coords.shape[-1] - 1
        factor = self.factor if isinstance(self.factor, tuple) else (self.factor,) * DIM
        assert DIM == len(factor), 'Input coordinates must have the same dimension as the downsample factor.'

        coord = list(input.coords.unbind(dim=-1))
        for i, f in enumerate(factor):
            coord[i + 1] = coord[i + 1] // f

        MAX = [coord[i + 1].max().item() + 1 for i in range(DIM)]
        OFFSET = torch.cumprod(torch.tensor(MAX[::-1]), 0).tolist()[::-1] + [1]
        code = sum([c * o for c, o in zip(coord, OFFSET)])
        code, idx = code.unique(return_inverse=True)

        new_feats = fixed_order_segment_mean(input.feats, idx, code.shape[0])
        new_coords = torch.stack(
            [code // OFFSET[0]] +
            [(code // OFFSET[i + 1]) % MAX[i] for i in range(DIM)],
            dim=-1
        )
        out = type(input)(new_feats, new_coords, input.shape)
        out._scale = tuple([s // f for s, f in zip(input._scale, factor)])
        out._spatial_cache = input._spatial_cache

        out.register_spatial_cache(f'upsample_{factor}_coords', input.coords)
        out.register_spatial_cache(f'upsample_{factor}_layout', input.layout)
        out.register_spatial_cache(f'upsample_{factor}_idx', idx)
        return out
