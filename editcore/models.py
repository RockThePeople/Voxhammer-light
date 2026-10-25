"""모델 적재와 상수 유도.

포크는 import 시 `ATTN_BACKEND=sdpa`, `SPCONV_ALGO=native` 를 강제한다(`edit_pipeline.py:1-3`).
이식본도 trellis import 전에 같은 값으로 맞춘다(dense=sdpa; sparse 모듈은 sdpa 를 받지 않아 flash_attn).
TRELLIS 코드는 기준 사본(`build/reference-v2`, VoxHammer 동봉본과 바이트 동일)에서 수정 없이 import 한다.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# origin: voxhammer/edit_pipeline.py:1-3 @ f33cd96+patch
os.environ["ATTN_BACKEND"] = "sdpa"
os.environ["SPCONV_ALGO"] = "native"

REPO = Path(__file__).resolve().parents[1]
TRELLIS_ROOT = Path(os.environ.get("VHL_TRELLIS_ROOT", REPO / "build" / "reference-v2")).resolve()
if str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))

MODEL_ID = "microsoft/TRELLIS-image-large"
# TRELLIS `TrellisImageTo3DPipeline.preprocess_image`·포크 `preprocess_image` 의 고정 입력 크기
COND_IMAGE_SIZE = 518


@dataclass(frozen=True)
class Constants:
    """포크에 숫자로 박힌 상수를 모델 설정에서 유도한다(테스트가 원본 값과 비교)."""
    ss_resolution: int        # 16
    ss_tokens: int            # 4096 = 16³
    ss_heads: int             # 16
    ss_head_dim: int          # 64
    voxel_res: int            # 64
    latent_channels: int      # 8 (SS latent)
    patch_size: int           # 14 (DINOv2)
    patch_grid: int           # 37 = 518 / 14
    cond_extra_tokens: int    # 5 = CLS 1 + register 4
    cond_tokens: int          # 1374 = 37² + 5
    cond_channels: int        # 1024


def derive_constants(pipeline) -> Constants:
    ss = pipeline.models["sparse_structure_flow_model"]
    dino = pipeline.models["image_cond_model"]
    res = int(ss.resolution) // int(ss.patch_size)
    heads = int(ss.num_heads)
    ps = int(dino.patch_embed.patch_size[0]) if hasattr(dino.patch_embed.patch_size, "__len__") else int(dino.patch_embed.patch_size)
    grid = COND_IMAGE_SIZE // ps
    extra = 1 + int(getattr(dino, "num_register_tokens", 0))
    dec = pipeline.models["sparse_structure_decoder"]
    up = 2 ** sum(1 for b in dec.blocks if type(b).__name__.startswith("UpsampleBlock")) if hasattr(dec, "blocks") else 4
    return Constants(
        ss_resolution=res, ss_tokens=res ** 3, ss_heads=heads, ss_head_dim=int(ss.model_channels) // heads,
        voxel_res=int(ss.resolution) * up, latent_channels=int(ss.in_channels), patch_size=ps, patch_grid=grid,
        cond_extra_tokens=extra, cond_tokens=grid * grid + extra, cond_channels=int(ss.cond_channels))


def load_pipeline(model_id: str = MODEL_ID, device: str = "cuda"):
    from trellis.pipelines import TrellisImageTo3DPipeline
    pipe = TrellisImageTo3DPipeline.from_pretrained(model_id)
    if device == "cuda":
        pipe.cuda()
    if ds_enabled():
        names = apply_ds(pipe)[0]
        print(f"[ds] VHL_DS_DETERMINISTIC=1: slat_flow_model SparseDownsample {len(names)}개 교체 {names}", flush=True)
    return pipe


def r4_enabled():
    """2번 R4 토글. `VHL_R4=1` 일 때만 mesh 만 디코딩 → `.cbin` 직행(기본 꺼짐 = 기존 GLB 출력)."""
    return os.environ.get("VHL_R4", "") == "1"


def ds_enabled():
    """정오표 E16 검증 전용 토글. `VHL_DS_DETERMINISTIC=1` 일 때만 켜진다(기본 꺼짐 = 원본 비결정 연산)."""
    return os.environ.get("VHL_DS_DETERMINISTIC", "") == "1"


def apply_ds(pipe):
    """Stage 2 flow 모델의 `SparseDownsample` 을 `DeterministicSparseDownsample` 로 모듈 교체한다.
    (교체한 모듈 이름 목록, 원래대로 되돌리는 함수)를 돌려준다."""
    import trellis.modules.sparse as sp
    from .blocks import DeterministicSparseDownsample
    model = pipe.models["slat_flow_model"]
    done = []
    for name, m in list(model.named_modules()):
        if type(m) is sp.SparseDownsample:
            parent, _, attr = name.rpartition(".")
            owner = model.get_submodule(parent) if parent else model
            setattr(owner, attr, DeterministicSparseDownsample(m.factor))
            done.append((name, owner, attr, m))

    def restore():
        for _, owner, attr, m in done:
            setattr(owner, attr, m)
    return [d[0] for d in done], restore


E2A_DEFAULT_STEPS = 15  # 정오표 E20 (포크 25 는 sampler.STEPS)


def e2a_steps(cli_s1=None, cli_s2=None):
    """1번 E2a 스텝 수. CLI 인자 > 환경변수 `VHL_E2A_STEPS_S1/S2` > 기본 15(고객사 채택 15/15, 정오표 E20).
    포크와 같은 25 스텝은 환경변수·CLI 로 줄 때만(포크 대조)."""
    def one(cli, env):
        v = cli if cli is not None else os.environ.get(env)
        return E2A_DEFAULT_STEPS if v in (None, "") else int(v)
    return one(cli_s1, "VHL_E2A_STEPS_S1"), one(cli_s2, "VHL_E2A_STEPS_S2")
