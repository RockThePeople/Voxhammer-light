"""Stage 1·2 attention — 포크의 MethodType forward 를 명시적 클래스로. 모듈 가중치는 원본 그대로 쓴다.

W39(편집 Query → 보존 Key 차단)는 이식하지 않는다(STATE 결정). 포크 기준 실행에서도 꺼져 있다.
Stage 2 의 self·cross attention 은 포크와 같이 `F.scaled_dot_product_attention` 을 직접 부른다
(sparse attention 백엔드를 거치지 않는다).
"""
from __future__ import annotations

import torch.nn.functional as F

from .cache import EditContext


class SSAttention:
    """Stage 1 (dense) self/cross attention."""

    def __init__(self, module):
        self.m = module

    # origin: voxhammer/edit_pipeline.py:ss_attn_forward @ f33cd96+patch
    def __call__(self, x, ctx: EditContext, context=None, indices=None):
        m = self.m
        kv_mask = ctx.self_kv_mask if m._type == "self" else ctx.cross_kv_mask
        B, L, C = x.shape
        if m._type == "self":
            qkv = m.to_qkv(x)
            qkv = qkv.reshape(B, L, 3, m.num_heads, -1)
            q, k, v = qkv.unbind(dim=2)
            if m.use_rope:
                q, k = m.rope(q, k, indices)
        else:
            Lkv = context.shape[1]
            q = m.to_q(x)
            kv = m.to_kv(context)
            q = q.reshape(B, L, m.num_heads, -1)
            kv = kv.reshape(B, Lkv, 2, m.num_heads, -1)
            k, v = kv.unbind(dim=2)
        if m.qk_rms_norm:
            q = m.q_rms_norm(q)
            k = m.k_rms_norm(k)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if not (ctx.is_text and (m._type != "self")):
            kk, vk = ctx.key(m._type, "k"), ctx.key(m._type, "v")
            if kv_mask is None:
                ctx.kv[kk] = k.cpu()
                ctx.kv[vk] = v.cpu()
            else:
                k = k * kv_mask + ctx.kv[kk].cuda() * (1 - kv_mask)
                v = v * kv_mask + ctx.kv[vk].cuda() * (1 - kv_mask)
                k = k.type(q.dtype)
                v = v.type(q.dtype)
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=None)
        h = h.permute(0, 2, 1, 3)
        h = h.reshape(B, L, -1)
        return m.to_out(h)


class SlatAttention:
    """Stage 2 (sparse 텐서, 전 토큰 SDPA) self/cross attention."""

    def __init__(self, module):
        self.m = module

    @staticmethod
    # origin: voxhammer/edit_pipeline.py:slat_attn_forward (보존 토큰 K/V 교체, argmax(0) 매칭) @ f33cd96+patch
    def _replace_preserved(cur, cached, kv_mask):
        match_1 = (cur.coords.unsqueeze(1) == kv_mask.unsqueeze(0)).all(dim=-1)
        match_2 = (cached.coords.cuda().unsqueeze(1) == kv_mask.unsqueeze(0)).all(dim=-1)
        idx_1 = match_1.float().argmax(0)
        idx_2 = match_2.float().argmax(0)
        feats = cur.feats.clone()
        feats[idx_1] = cached.feats.cuda()[idx_2]
        return cur.replace(feats)

    # origin: voxhammer/edit_pipeline.py:slat_attn_forward @ f33cd96+patch
    def __call__(self, x, ctx: EditContext, context=None):
        m = self.m
        if m._type == "self":
            kv_mask = ctx.self_kv_mask
            qkv = m._linear(m.to_qkv, x)
            qkv = m._fused_pre(qkv, num_fused=3)
            if m.use_rope:
                qkv = m._rope(qkv)
            q, k, v = qkv.unbind(dim=1)
            if m.qk_rms_norm:
                q = m.q_rms_norm(q)
                k = m.k_rms_norm(k)
            kk, vk = ctx.key(m._type, "k"), ctx.key(m._type, "v")
            if kv_mask is None:
                ctx.kv[kk] = k.cpu()
                ctx.kv[vk] = v.cpu()
            else:
                k = self._replace_preserved(k, ctx.kv[kk], kv_mask)
                k = k.type(q.dtype)
                v = self._replace_preserved(v, ctx.kv[vk], kv_mask)
                v = v.type(q.dtype)
            h = q
            q = q.feats.unsqueeze(0).permute(0, 2, 1, 3)
            k = k.feats.unsqueeze(0).permute(0, 2, 1, 3)
            v = v.feats.unsqueeze(0).permute(0, 2, 1, 3)
        else:
            kv_mask = ctx.cross_kv_mask
            q = m._linear(m.to_q, x)
            q = m._reshape_chs(q, (m.num_heads, -1))
            kv = m._linear(m.to_kv, context)
            kv = m._fused_pre(kv, num_fused=2)
            k, v = kv.unbind(dim=2)
            h = q
            q = q.feats.unsqueeze(0).permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            if not ctx.is_text:
                kk, vk = ctx.key(m._type, "k"), ctx.key(m._type, "v")
                if kv_mask is None:
                    ctx.kv[kk] = k.cpu()
                    ctx.kv[vk] = v.cpu()
                else:
                    k = k * kv_mask + ctx.kv[kk].cuda() * (1 - kv_mask)
                    v = v * kv_mask + ctx.kv[vk].cuda() * (1 - kv_mask)
                    k = k.type(q.dtype)
                    v = v.type(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None)
        out = out.permute(0, 2, 1, 3)[0]
        h = h.replace(out)
        h = m._reshape_chs(h, (-1,))
        return m._linear(m.to_out, h)
