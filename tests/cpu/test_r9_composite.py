"""services/rtx3090_inpaint/composite.py CPU 테스트 (05 R9 R-2 마스크 밖 합성).

pytest 로도, `python3 tests/cpu/test_r9_composite.py` 로도 돈다.
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from services.rtx3090_inpaint.composite import composite, mask_bool, outside_diff, render_rgb  # noqa: E402

N = 64
RNG = np.random.default_rng(0)
RENDER = Image.fromarray(RNG.integers(0, 256, (N, N, 3), dtype=np.uint8))
M = np.zeros((N, N), np.uint8)
M[16:40, 20:44] = 255
MASK = Image.fromarray(M)
# 모델 출력 흉내: 전체가 조금 흔들리고(VAE 왕복) 마스크 안은 완전히 다르다
_e = np.asarray(RENDER).astype(np.int16) + RNG.integers(-3, 4, (N, N, 3))
_e[16:40, 20:44] = 7
EDIT = Image.fromarray(np.clip(_e, 0, 255).astype(np.uint8))


def test_outside_restored_bitwise():
    c = np.asarray(composite(EDIT, RENDER, MASK))
    out = ~mask_bool(MASK)
    assert (c[out] == np.asarray(RENDER)[out]).all()


def test_inside_kept_from_edit():
    c = np.asarray(composite(EDIT, RENDER, MASK))
    assert (c[16:40, 20:44] == 7).all()


def test_diff_before_after():
    before = outside_diff(EDIT, RENDER, MASK)
    after = outside_diff(composite(EDIT, RENDER, MASK), RENDER, MASK)
    assert before["outside_identical_frac"] < 0.5 and before["outside_max_abs"] > 0
    assert after["outside_identical_frac"] == 1.0 and after["outside_max_abs"] == 0
    assert after["mask_pixels"] == 24 * 24 and after["outside_pixels"] == N * N - 24 * 24


def test_binary_threshold_no_feather():
    m = M.copy()
    m[0, 0] = 127   # 경계값: 보존 쪽
    m[0, 1] = 128   # 다시 그릴 쪽
    mb = mask_bool(Image.fromarray(m))
    assert not mb[0, 0] and mb[0, 1]


def test_rgba_render_uses_white_background():
    a = np.zeros((N, N, 4), np.uint8)
    a[..., 0] = 200
    a[:, : N // 2, 3] = 255          # 왼쪽 불투명, 오른쪽 투명
    rgb = np.asarray(render_rgb(Image.fromarray(a, "RGBA")))
    assert (rgb[:, 0] == [200, 0, 0]).all() and (rgb[:, -1] == 255).all()


def test_size_mismatch_raises():
    try:
        composite(EDIT, RENDER.resize((32, 32)), MASK)
    except ValueError:
        return
    raise AssertionError("크기 불일치를 통과시켰다")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok", f.__name__)
    print(f"{len(fns)} passed")
