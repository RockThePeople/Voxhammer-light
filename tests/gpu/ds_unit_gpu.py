"""보조 DS D-3 1: GPU 단위 — 무작위 sparse 입력에서 원본 SparseDownsample 2회·결정적 구현 2회 비트 동일 여부와 원본 대비 |Δ|.

사용: ds_unit_gpu.py <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import editcore.models  # noqa: E402,F401  (trellis 경로)

import torch  # noqa: E402


def main():
    import trellis.modules.sparse as sp
    from editcore.blocks import DeterministicSparseDownsample
    rows = []
    for n, C, res in [(2000, 64, 32), (20000, 1024, 64), (60000, 1024, 64), (120000, 1024, 64)]:
        for dtype in (torch.float32, torch.float16):
            g = torch.Generator().manual_seed(n + C)
            c = torch.unique(torch.randint(0, res, (n, 3), generator=g), dim=0)
            c = torch.cat([torch.zeros(len(c), 1, dtype=c.dtype), c], 1).int().cuda()
            f = torch.randn(len(c), C, generator=g).to(dtype).cuda()
            x = sp.SparseTensor(feats=f, coords=c)
            o1, o2 = sp.SparseDownsample(2)(x).feats, sp.SparseDownsample(2)(x).feats
            d1, d2 = DeterministicSparseDownsample(2)(x).feats, DeterministicSparseDownsample(2)(x).feats
            ref = sp.SparseDownsample(2)(sp.SparseTensor(feats=f.double(), coords=c)).feats
            dd = (d1.double() - o1.double()).abs()
            oo = (o1.double() - o2.double()).abs()
            rows.append({"cells": len(c), "channels": C, "dtype": str(dtype).split(".")[-1], "out_cells": int(d1.shape[0]),
                         "orig_twice_bit_equal": bool(torch.equal(o1, o2)), "orig_twice_max": float(oo.max()),
                         "det_twice_bit_equal": bool(torch.equal(d1, d2)),
                         "det_vs_orig_max": float(dd.max()), "det_vs_orig_mean": float(dd.mean()),
                         "err_vs_fp64_orig_max": float((o1.double() - ref).abs().max()),
                         "err_vs_fp64_det_max": float((d1.double() - ref).abs().max())})
            print(rows[-1], flush=True)
    Path(sys.argv[1]).write_text(json.dumps({"rows": rows, "torch": torch.__version__,
                                             "device": torch.cuda.get_device_name(0)}, indent=1))


if __name__ == "__main__":
    main()
