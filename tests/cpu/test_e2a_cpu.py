"""1번 E2a 스텝 수 토글 CPU 테스트. 기본(25)은 test_editcore_cpu.py 의 비트 동일 테스트가 그대로 지킨다."""
import copy
import os
import sys
from pathlib import Path

import numpy as np
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
    orig = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *a, **k: self
    yield
    torch.Tensor.cuda = orig


def test_toggle_precedence(monkeypatch):
    from editcore.models import e2a_steps
    monkeypatch.delenv("VHL_E2A_STEPS_S1", raising=False)
    monkeypatch.delenv("VHL_E2A_STEPS_S2", raising=False)
    assert e2a_steps() == (15, 15)                      # 기본 = 고객사 채택 15/15 (E20)
    monkeypatch.setenv("VHL_E2A_STEPS_S1", "25")
    monkeypatch.setenv("VHL_E2A_STEPS_S2", "25")
    assert e2a_steps() == (25, 25)                      # 포크 대조(0-1 과 같음)는 명시할 때만
    monkeypatch.setenv("VHL_E2A_STEPS_S1", "12")
    monkeypatch.setenv("VHL_E2A_STEPS_S2", "15")
    assert e2a_steps() == (12, 15)
    assert e2a_steps(20, None) == (20, 15)              # CLI 우선


def _small():
    from trellis.models.sparse_structure_flow import SparseStructureFlowModel
    torch.manual_seed(0)
    m = SparseStructureFlowModel(resolution=8, in_channels=8, model_channels=64, cond_channels=64, out_channels=8,
                                 num_blocks=2, num_heads=2, patch_size=1, pe_mode="ape", qk_rms_norm=True)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 1.0 / p.shape[1] ** 0.5) if p.dim() > 1 else p.normal_(0, 0.3)
    return m.eval()


@pytest.mark.parametrize("steps", [12, 15, 25])
@pytest.mark.parametrize("cfg", [0.0, 5.0])
def test_keys_match_and_count(steps, cfg):
    """inversion 키 집합 = denoising 이 찾는 키 집합, K/V 항목 수 = 스텝 × order 2 × pos 수 × 층 × (self,cross) × (k,v)."""
    from editcore.blocks import SSFlow
    from editcore.sampler import CFG_INTERVAL, InversionSampler
    S = InversionSampler(steps)
    runner = SSFlow(_small())
    g = torch.Generator().manual_seed(1)
    z = torch.randn(1, 8, 8, 8, 8, generator=g)
    cond = {"cond": torch.randn(1, 10, 64, generator=g), "neg_cond": torch.zeros(1, 10, 64)}
    noise, lat, kv = S.sample(runner, 1, z, cond, cfg)
    t, n = S.t_seq(0)
    assert n == steps and len(lat) == steps
    den_keys = {f"{t[i]}" for i in range(steps)}          # denoising 은 t_curr 로 찾는다
    assert set(lat) == den_keys
    layers = 2
    pos = [2 if (cfg != 0 and CFG_INTERVAL[0] <= float(k) <= CFG_INTERVAL[1]) else 1 for k in den_keys]
    assert len(kv) == sum(2 * p * layers * 2 * 2 for p in pos)
    # denoising 이 키를 전부 찾아 끝까지 돈다(마스크 전부 편집)
    mask = torch.ones(1, 8, 8, 8, 8)
    kvm = torch.ones(1, 2, 512, 32)
    out, _, _ = S.sample(runner, 1, noise, cond, cfg, lat, mask, kv, kvm, None)
    assert out.shape == z.shape


def test_mismatched_steps_raise_keyerror():
    """inversion 12 스텝 캐시를 denoising 15 스텝이 찾으면 KeyError 로 멈춘다(조용히 넘어가지 않음)."""
    from editcore.blocks import SSFlow
    from editcore.sampler import InversionSampler
    runner = SSFlow(_small())
    g = torch.Generator().manual_seed(2)
    z = torch.randn(1, 8, 8, 8, 8, generator=g)
    cond = {"cond": torch.randn(1, 10, 64, generator=g), "neg_cond": torch.zeros(1, 10, 64)}
    noise, lat, kv = InversionSampler(12).sample(runner, 1, z, cond, 0.0)
    with pytest.raises(KeyError):
        InversionSampler(15).sample(runner, 1, noise, cond, 0.0, lat, torch.ones(1, 8, 8, 8, 8), kv,
                                    torch.ones(1, 2, 512, 32), None)
