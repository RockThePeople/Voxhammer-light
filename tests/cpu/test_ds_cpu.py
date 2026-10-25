"""보조 DS(정오표 E16) CPU 테스트: 결정적 SparseDownsample 이 원본 평균과 같고, 토글이 기본 꺼짐인지."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
REPO = Path(__file__).resolve().parents[2]
if not (REPO / "build" / "reference-v2" / "voxhammer" / "edit_pipeline.py").exists():
    pytest.skip("build/reference-v2 가 없다", allow_module_level=True)
sys.path.insert(0, str(REPO))


@pytest.fixture(scope="module", autouse=True)
def _cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import editcore.models  # noqa: F401


def _sparse(n, C, res, dtype, seed):
    import trellis.modules.sparse as sp
    g = torch.Generator().manual_seed(seed)
    c = torch.unique(torch.randint(0, res, (n, 3), generator=g), dim=0)
    c = torch.cat([torch.zeros(len(c), 1, dtype=c.dtype), c], 1).int()
    return sp.SparseTensor(feats=torch.randn(len(c), C, generator=g).to(dtype), coords=c)


def test_toggle_default_off(monkeypatch):
    from editcore.models import ds_enabled
    monkeypatch.delenv("VHL_DS_DETERMINISTIC", raising=False)
    assert not ds_enabled()
    for v in ("", "0", "true"):
        monkeypatch.setenv("VHL_DS_DETERMINISTIC", v)
        assert not ds_enabled()
    monkeypatch.setenv("VHL_DS_DETERMINISTIC", "1")
    assert ds_enabled()


def test_include_self_mean():
    """원본 scatter_reduce(mean) 은 include_self=True 라 0 이 개수에 들어간다: (0+2+4)/3, (0+6)/2."""
    from editcore.blocks import fixed_order_segment_mean
    f = torch.tensor([[2.0], [4.0], [6.0]])
    idx = torch.tensor([0, 0, 1])
    ref = torch.scatter_reduce(torch.zeros(2, 1), 0, idx[:, None], f, reduce="mean")
    assert torch.equal(fixed_order_segment_mean(f, idx, 2), ref)
    assert ref.ravel().tolist() == [2.0, 3.0]


@pytest.mark.parametrize("n,C,res", [(500, 8, 16), (5000, 64, 64), (20000, 1024, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_matches_original_and_repeats(n, C, res, dtype):
    import trellis.modules.sparse as sp
    from editcore.blocks import DeterministicSparseDownsample
    x = _sparse(n, C, res, dtype, seed=n + C)
    a = sp.SparseDownsample(2)(x)
    b1 = DeterministicSparseDownsample(2)(x)
    b2 = DeterministicSparseDownsample(2)(x)
    assert torch.equal(a.coords, b1.coords)
    assert b1.feats.dtype == a.feats.dtype
    assert torch.equal(b1.feats, b2.feats)  # 같은 입력 → 같은 비트
    if dtype == torch.float32:
        # CPU scatter_reduce 는 원래 인덱스 순서로 더하므로 fp32 에서는 슬롯 순서 합과 비트까지 같다
        assert torch.equal(a.feats, b1.feats)
    else:
        # fp16: CPU 원본은 더 넓은 dtype 으로 누적해 반올림이 달라질 수 있다 → 부분합 기준 몇 ulp 이내, fp32 기준과의 오차가 원본 수준
        fa, fb = a.feats.float(), b1.feats.float()
        assert ((fa - fb).abs() <= fa.abs() * 2 ** -9 + 2 ** -9).all()  # 상쇄로 결과가 작아도 부분합 크기의 ulp
        ref = sp.SparseDownsample(2)(sp.SparseTensor(feats=x.feats.float(), coords=x.coords)).feats
        assert (fb - ref).abs().max() <= (fa - ref).abs().max() + fa.abs().max() * 2 ** -10
    for k in ("coords", "layout", "idx"):
        ka = a.get_spatial_cache(f"upsample_{(2, 2, 2)}_{k}")
        kb = b1.get_spatial_cache(f"upsample_{(2, 2, 2)}_{k}")
        assert (ka == kb) if isinstance(ka, list) else torch.equal(ka, kb)


def test_apply_ds_replaces_and_restores():
    import trellis.modules.sparse as sp
    from editcore.blocks import DeterministicSparseDownsample
    from editcore.models import apply_ds
    blk = torch.nn.Module()
    blk.updown = sp.SparseDownsample(2)
    model = torch.nn.Module()
    model.input_blocks = torch.nn.ModuleList([torch.nn.Identity(), blk])
    model.other = sp.SparseUpsample(2)
    pipe = SimpleNamespace(models={"slat_flow_model": model})
    names, restore = apply_ds(pipe)
    assert names == ["input_blocks.1.updown"]
    assert type(model.input_blocks[1].updown) is DeterministicSparseDownsample
    assert model.input_blocks[1].updown.factor == 2
    assert type(model.other) is sp.SparseUpsample
    restore()
    assert type(model.input_blocks[1].updown) is sp.SparseDownsample
