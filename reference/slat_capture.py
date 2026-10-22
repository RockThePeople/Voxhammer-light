"""R2 SLat 포획 — 기준 구현 `run_edit` 가 디코더에 넘기는 편집 SLat(`slat_tgt`)을 저장한다.

포크(`build/reference-v2/voxhammer/edit_pipeline.py`)는 고치지 않는다. `run_edit` 는
`slat_tgt = sample_slat_denoise(...)`(:623) 직후 `pipeline.decode_slat(slat_tgt, ...)`(:624)
를 한 번 부르고 GLB 만 남긴다. 여기서는 그 `decode_slat` 를 파이프라인 인스턴스에서만
잠시 바꿔 끼워 입력 SLat 을 붙잡고, 블록을 나가면 원래 메서드로 되돌린 뒤 저장한다.

저장 형식 (생성 산출물 `slat.safetensors` 와 같다):
    coords  int32   (N, 4)   [batch, x, y, z], 64³ 격자
    feats   float32 (N, 8)
    메타    slat_space=denormalized, slat_channels, voxel_res=64, coord_order=canonical,
            norm_mean, norm_std (JSON 문자열)
`slat_tgt` 는 `sample_slat_denoise` 끝에서 `slat * std + mean`(:522) 을 거친 값이라
이미 denormalized 공간이다. W34 경로(`slat_path=`)가 다음 입력으로 그대로 읽는다.

사용:
    with SlatCapture(pipeline, out_path) as cap:
        run_edit(pipeline, ...)
    cap.n_calls  # 1 이어야 한다. 아니면 마지막 호출이 저장된다
"""
from __future__ import annotations

import json

__all__ = ["SlatCapture", "save_slat"]

SLAT_META_FIXED = {
    "slat_space": "denormalized",
    "voxel_res": "64",
    "coord_order": "canonical",
}


def save_slat(slat, path, normalization=None, extra_meta=None):
    """SparseTensor(또는 .coords/.feats 를 가진 객체)를 R2 형식으로 저장하고 요약을 돌려준다."""
    import torch
    from safetensors.torch import save_file

    coords = slat.coords.detach().to("cpu", torch.int32).contiguous()
    feats = slat.feats.detach().to("cpu", torch.float32).contiguous()
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"coords shape {tuple(coords.shape)} — (N,4) 여야 한다")
    if feats.shape[0] != coords.shape[0]:
        raise ValueError(f"feats {tuple(feats.shape)} 와 coords {tuple(coords.shape)} 행 수가 다르다")

    meta = dict(SLAT_META_FIXED, slat_channels=str(feats.shape[1]))
    if normalization is not None:
        meta["norm_mean"] = json.dumps(list(map(float, normalization["mean"])))
        meta["norm_std"] = json.dumps(list(map(float, normalization["std"])))
    for k, v in (extra_meta or {}).items():
        meta[k] = v if isinstance(v, str) else json.dumps(v)

    save_file({"coords": coords, "feats": feats}, str(path), metadata=meta)
    return {"coords": list(coords.shape), "feats": list(feats.shape), "meta_keys": sorted(meta)}


class SlatCapture:
    """`pipeline.decode_slat` 호출을 가로채 입력 SLat 을 저장하는 컨텍스트 매니저."""

    def __init__(self, pipeline, out_path, extra_meta=None, save=True):
        self.pipeline = pipeline
        self.out_path = out_path
        self.extra_meta = extra_meta
        self.save = save
        self.n_calls = 0
        self.slat = None
        self.summary = None
        self._orig = None

    def __enter__(self):
        self._orig = self.pipeline.decode_slat

        def decode_slat(slat, *args, **kwargs):
            self.n_calls += 1
            self.slat = slat
            return self._orig(slat, *args, **kwargs)

        # 인스턴스 속성으로만 덮는다 — 클래스·포크 모듈은 그대로다.
        self.pipeline.decode_slat = decode_slat
        return self

    def __exit__(self, exc_type, exc, tb):
        # 인스턴스 속성을 지우면 클래스 메서드가 다시 보인다.
        try:
            del self.pipeline.decode_slat
        except AttributeError:
            self.pipeline.decode_slat = self._orig
        if exc_type is None and self.save and self.slat is not None:
            self.summary = save_slat(
                self.slat, self.out_path,
                normalization=getattr(self.pipeline, "slat_normalization", None),
                extra_meta=self.extra_meta)
        return False
