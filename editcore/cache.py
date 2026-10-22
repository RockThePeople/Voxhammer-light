"""K/V·잠재 캐시와 편집 컨텍스트.

캐시 키는 포크와 같은 문자열이다: f"{t_latent}_{order}_{pos}_{layer}_{type}_{k|v}".
- inversion 은 t_prev, denoising 은 t_curr 를 t_latent 로 쓴다(같은 시각 → 같은 키, sampler 가 보장).
- 2차 solver 의 order 1·2 는 다른 키(분리 캐시).
- pos 1 = cond, 0 = neg_cond (CFG 구간 밖이거나 cfg=0 이면 pos 1 만).
캐시 텐서는 포크와 같이 CPU 에 둔다(`k.cpu()`), 쓸 때 `.cuda()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# origin: voxhammer/edit_pipeline.py:ss_attn_forward / slat_attn_forward (캐시 키) @ f33cd96+patch
def kv_key(t_latent, order, pos, layer, attn_type: str, which: str) -> str:
    return f"{t_latent}_{order}_{pos}_{layer}_{attn_type}_{which}"


@dataclass
class EditContext:
    """attention 이 쓰는 상태를 한 객체로 전달한다(포크는 forward 인자 9개로 넘겼다)."""
    kv: dict = field(default_factory=dict)
    self_kv_mask: Optional[Any] = None
    cross_kv_mask: Optional[Any] = None
    t_latent: Any = None
    order: Optional[int] = None
    pos: Optional[int] = None
    layer: Optional[int] = None
    is_text: bool = False

    def key(self, attn_type: str, which: str) -> str:
        return kv_key(self.t_latent, self.order, self.pos, self.layer, attn_type, which)
