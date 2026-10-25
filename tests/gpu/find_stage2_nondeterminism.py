"""Stage 2 비결정성 원인 연산 찾기 — slat_flow_model 에 같은 입력을 두 번 넣고, 모든 하위 모듈 출력의
해시를 순서대로 기록해 처음으로 갈리는 모듈(과 그 직전까지 같은 모듈)을 찾는다. Stage 1(ss_flow)도 대조로 같이 본다.

입력: moto-b feats SLat(정규화 공간으로), t=500, cond = 무작위(고정 seed) 1374×1024. 원본 forward(교체 없음)로 충분하다 —
편집 경로의 교체 forward 는 attention 계산만 바꾸고 io 블록(sparse conv·down/upsample)은 원본 모듈을 그대로 쓴다.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import editcore.models as em  # noqa: E402

import torch  # noqa: E402


def h(t):
    if hasattr(t, "feats"):
        t = t.feats
    if isinstance(t, (tuple, list)):
        t = t[0]
    if not torch.is_tensor(t):
        return None
    t = t.detach().contiguous().cpu()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def trace(model, args):
    rec = []
    hooks = []
    for name, m in model.named_modules():
        if name == "":
            continue
        hooks.append(m.register_forward_hook(lambda mod, i, o, n=name: rec.append((n, type(mod).__name__, h(o)))))
    with torch.no_grad():
        out = model(*args)
    for hk in hooks:
        hk.remove()
    return rec, h(out)


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x[2] != y[2]:
            return i, x, y
    return None


def main():
    out_path = Path(sys.argv[1])
    pipe = em.load_pipeline()
    import trellis.modules.sparse as sp
    from safetensors import safe_open
    with safe_open(str(Path.home() / "ai-ar-data/assets/v3-moto-b/v1/slat.safetensors"), "pt") as f:
        c, ft = f.get_tensor("coords").int().cuda(), f.get_tensor("feats").float().cuda()
    std = torch.tensor(pipe.slat_normalization["std"], device="cuda")[None]
    mean = torch.tensor(pipe.slat_normalization["mean"], device="cuda")[None]
    x = sp.SparseTensor(feats=(ft - mean) / std, coords=c)
    g = torch.Generator().manual_seed(0)
    cond = torch.randn(1, 1374, 1024, generator=g).cuda()
    t = torch.tensor([500.0], device="cuda")
    res = {}
    slat = pipe.models["slat_flow_model"]
    r1, o1 = trace(slat, (x, t, cond))
    r2, o2 = trace(slat, (x, t, cond))
    fd = first_diff(r1, r2)
    ndiff = sum(1 for a, b in zip(r1, r2) if a[2] != b[2])
    res["stage2_slat_flow"] = {"modules_traced": len(r1), "output_equal": o1 == o2, "modules_differing": ndiff,
                               "first_diff": None if fd is None else {"index": fd[0], "module": fd[1][0], "type": fd[1][1]},
                               "equal_before_first_diff": None if fd is None else [x[0] for x in r1[max(0, fd[0] - 3):fd[0]]],
                               "differing_types": sorted({a[1] for a, b in zip(r1, r2) if a[2] != b[2]})}
    ss = pipe.models["sparse_structure_flow_model"]
    xs = torch.randn(1, 8, 16, 16, 16, generator=g).cuda()
    s1, so1 = trace(ss, (xs, t, cond))
    s2, so2 = trace(ss, (xs, t, cond))
    res["stage1_ss_flow"] = {"modules_traced": len(s1), "output_equal": so1 == so2,
                             "modules_differing": sum(1 for a, b in zip(s1, s2) if a[2] != b[2])}
    # 단독 확인: 첫 sparse conv 한 개를 같은 입력으로 두 번
    conv = next(m for n, m in slat.named_modules() if type(m).__name__ == "SparseConv3d")
    blk_in = slat.input_layer(x).type(slat.dtype)
    a1, a2 = h(conv(blk_in)), h(conv(blk_in))
    res["single_SparseConv3d_twice_equal"] = a1 == a2
    import os
    res["SPCONV_ALGO"] = os.environ.get("SPCONV_ALGO")
    out_path.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
