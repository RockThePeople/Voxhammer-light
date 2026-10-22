"""0-1 함수 단위 동등성 (CPU, GPU 없이). 포크(build/reference-v2/voxhammer/edit_pipeline.py) vs editcore.

포크 함수는 `.cuda()` 를 부르므로 이 테스트 안에서만 `torch.Tensor.cuda` 를 항등으로 바꾼다(고객사 검증 스크립트와 같은 방식).
Stage 1 은 축소 SparseStructureFlowModel(무작위 가중치)로 inversion → denoising 전체를 양쪽에서 돌려 비트 비교한다.
"""
import copy
import os
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
REPO = Path(__file__).resolve().parents[2]
REF = REPO / "build" / "reference-v2"
if not (REF / "voxhammer" / "edit_pipeline.py").exists():
    pytest.skip("build/reference-v2 가 없다 — tools/setup_reference.sh", allow_module_level=True)
sys.path.insert(0, str(REPO))


@pytest.fixture(scope="module")
def ep():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import editcore.models  # noqa: F401  (trellis 경로·환경변수)
    orig = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *a, **k: self
    import voxhammer.edit_pipeline as m
    yield m
    torch.Tensor.cuda = orig


@pytest.fixture(scope="module")
def C():
    from editcore.models import Constants
    return Constants(ss_resolution=16, ss_tokens=4096, ss_heads=16, ss_head_dim=64, voxel_res=64, latent_channels=8,
                     patch_size=14, patch_grid=37, cond_extra_tokens=5, cond_tokens=1374, cond_channels=1024)


def test_constants_derivation():
    """모델 설정(ckpt json 의 값) → 원본 상수 4096·16·64·37·14·1374."""
    from editcore.models import derive_constants
    ss = SimpleNamespace(resolution=16, patch_size=1, num_heads=16, model_channels=1024, in_channels=8, cond_channels=1024)
    dino = SimpleNamespace(patch_embed=SimpleNamespace(patch_size=(14, 14)), num_register_tokens=4)
    Up = type("UpsampleBlock3d", (), {})
    Res = type("ResBlock3d", (), {})
    dec = SimpleNamespace(blocks=[Res(), Res(), Up(), Res(), Res(), Up(), Res(), Res()])
    c = derive_constants(SimpleNamespace(models={"sparse_structure_flow_model": ss, "image_cond_model": dino,
                                                 "sparse_structure_decoder": dec}))
    assert (c.ss_tokens, c.ss_resolution, c.voxel_res, c.ss_head_dim, c.patch_grid, c.patch_size, c.cond_tokens) == \
           (4096, 16, 64, 64, 37, 14, 1374)
    assert c.ss_heads * c.ss_head_dim == 1024 and c.cond_extra_tokens == 5


def _cells(n, seed):
    g = np.random.default_rng(seed)
    return torch.as_tensor(np.unique(g.integers(0, 64, (n, 3)), axis=0), dtype=torch.int32)


def test_masks_bit_equal(ep, C):
    from PIL import Image
    from editcore import masks
    src = _cells(9000, 0)
    delete = src[torch.randperm(len(src), generator=torch.Generator().manual_seed(1))[:900]]
    # D51 보존 판정
    ref_pres = src[~torch.isin(src[:, 0] * 4096 + src[:, 1] * 64 + src[:, 2], delete[:, 0] * 4096 + delete[:, 1] * 64 + delete[:, 2])]
    pres = masks.preserve_coords(src, delete, C)
    assert torch.equal(pres, ref_pres) and len(pres) == len(src) - len(delete)
    assert torch.equal(masks.coords_to_voxel(src, C), ep.coords_to_voxel(src))
    pre_mask = Image.fromarray((np.random.default_rng(2).random((518, 518)) > 0.7).astype(np.uint8) * 255)
    for a, b in zip(masks.ss_masks(pres, pre_mask, C), ep.ply_to_ss_mask(pres, pre_mask)):
        assert a.dtype == b.dtype and torch.equal(a, b)
    assert masks.ss_masks(pres, None, C)[3] is None and ep.ply_to_ss_mask(pres, None)[3] is None
    tgt = torch.cat([torch.zeros(len(src), 1, dtype=torch.int32), src], 1)
    assert torch.equal(masks.slat_self_kv_mask(tgt, pres), ep.ply_to_slat_mask(tgt, pres))


def test_cache_keys_and_schedule(ep):
    """inversion 키(t_prev) 집합 == denoising 키(t_curr) 집합, 스텝 25, rescale_t 3.0."""
    from editcore.cache import kv_key
    from editcore.sampler import InversionSampler, STEPS, RESCALE_T, CFG_INTERVAL
    t, steps = InversionSampler.t_seq(0)
    ref = np.linspace(1, 0, 26)
    ref = 3.0 * ref / (1 + 2.0 * ref)
    assert steps == 25 == STEPS and RESCALE_T == 3.0 and CFG_INTERVAL == (0.5, 1.0) and np.array_equal(t, ref)
    inv = t[::-1]
    inv_keys = {f"{inv[i + 1]}" for i in range(steps)}
    den_keys = {f"{t[i]}" for i in range(steps)}
    assert inv_keys == den_keys
    assert kv_key(t[3], 2, 0, 7, "self", "k") == f"{t[3]}_2_0_7_self_k"


def _small_ss(seed=0):
    from trellis.models.sparse_structure_flow import SparseStructureFlowModel
    torch.manual_seed(seed)
    m = SparseStructureFlowModel(resolution=8, in_channels=8, model_channels=128, cond_channels=128, out_channels=8,
                                 num_blocks=3, num_heads=4, patch_size=1, pe_mode="ape", qk_rms_norm=True)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 1.0 / p.shape[1] ** 0.5) if p.dim() > 1 else p.normal_(0, 0.3)
    return m.eval()


def test_stage1_sampler_bit_equal(ep):
    """포크 InversionFlowEulerGuidanceIntervalSampler + MethodType forward vs editcore InversionSampler + SSFlow.
    inversion(cfg 5) → denoising(cfg 6, 보존 마스크·K/V 주입) 의 최종 z, 잠재·K/V 캐시까지 비트 동일."""
    from editcore.blocks import SSFlow
    from editcore.sampler import InversionSampler
    base = _small_ss()
    fork, port = copy.deepcopy(base), copy.deepcopy(base)
    fork.forward = MethodType(ep.ss_flow_forward, fork)
    for b in fork.blocks:
        b.forward = MethodType(ep.ss_trsfmr_forward, b)
        b.self_attn.forward = MethodType(ep.ss_attn_forward, b.self_attn)
        b.cross_attn.forward = MethodType(ep.ss_attn_forward, b.cross_attn)
    g = torch.Generator().manual_seed(3)
    R, L, H, D, LC = 8, 512, 4, 32, 30
    z = torch.randn(1, 8, R, R, R, generator=g)
    cs = {"cond": torch.randn(1, LC, 128, generator=g), "neg_cond": torch.zeros(1, LC, 128)}
    ct = {"cond": cs["cond"] + 0.5 * torch.randn(1, LC, 128, generator=g), "neg_cond": torch.zeros(1, LC, 128)}
    keep = torch.rand(L, generator=g) < 0.13
    edit = (~keep).float()
    lat_mask = edit.view(1, 1, R, R, R).repeat(1, 8, 1, 1, 1)
    self_kv = edit.view(1, 1, L, 1).expand(1, H, L, D).contiguous()
    cross = torch.ones(LC)
    cross[5 + torch.randperm(LC - 5, generator=g)[:10]] = 0
    cross_kv = cross.view(1, 1, LC, 1).expand(1, H, LC, D).contiguous()
    S = ep.InversionFlowEulerGuidanceIntervalSampler(1e-5)
    with torch.no_grad():
        n_f, lat_f, kv_f = S.sample(fork, 1, z, cs, 5.0)
        out_f, _, _ = S.sample(fork, 1, n_f, ct, 6.0, lat_f, lat_mask, kv_f, self_kv, cross_kv)
    P = InversionSampler()
    runner = SSFlow(port)
    n_p, lat_p, kv_p = P.sample(runner, 1, z, cs, 5.0)
    out_p, _, _ = P.sample(runner, 1, n_p, ct, 6.0, lat_p, lat_mask, kv_p, self_kv, cross_kv)
    assert torch.equal(n_f, n_p)
    assert lat_f.keys() == lat_p.keys() and all(torch.equal(lat_f[k], lat_p[k]) for k in lat_f)
    assert kv_f.keys() == kv_p.keys() and all(torch.equal(kv_f[k], kv_p[k]) for k in kv_f)
    assert torch.equal(out_f, out_p)
    # W38: cfg 0 이면 pos 0 키가 없다 / 2차 solver order 1·2 키가 모두 있다
    S0 = ep.InversionFlowEulerGuidanceIntervalSampler(1e-5)
    with torch.no_grad():
        _, _, kv0 = S0.sample(fork, 1, z, cs, 0.0)
    assert kv0 and all(k.split("_")[2] == "1" for k in kv0)   # pos 0(neg) 키 없음
    assert {k.split("_")[1] for k in kv_f} == {"1", "2"}


def test_preprocess_bit_equal(ep, tmp_path):
    """공동 bbox crop·518 리사이즈·rembg. 같은 이미지에서 출력 3장 비트 동일(세션은 이식본이 밖에서 만든다)."""
    rembg = pytest.importorskip("rembg")
    from PIL import Image
    from editcore.preprocess import preprocess_images
    rng = np.random.default_rng(4)
    img = np.full((640, 640, 3), 255, np.uint8)
    img[160:480, 200:420] = rng.integers(0, 120, (320, 220, 3), dtype=np.uint8)
    edit = img.copy()
    edit[120:200, 260:360] = rng.integers(0, 120, (80, 100, 3), dtype=np.uint8)
    mask = np.zeros((640, 640), np.uint8)
    mask[110:210, 250:370] = 255
    for n, a in (("2d_render.png", img), ("2d_edit.png", edit), ("2d_mask.png", mask)):
        Image.fromarray(a).save(tmp_path / n)
    ref = ep.preprocess_image(str(tmp_path / "2d_render.png"), str(tmp_path / "2d_edit.png"), str(tmp_path / "2d_mask.png"))
    got = preprocess_images(Image.open(tmp_path / "2d_render.png"), Image.open(tmp_path / "2d_edit.png"),
                            Image.open(tmp_path / "2d_mask.png"), rembg.new_session("u2net"), rembg.new_session("u2net"))
    for a, b in zip(ref, got):
        assert a.size == b.size and a.mode == b.mode and np.array_equal(np.asarray(a), np.asarray(b))
