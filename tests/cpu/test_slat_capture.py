"""reference/slat_capture.py CPU 테스트 — 가짜 파이프라인으로 형식·복원·호출 횟수를 본다."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors")
from safetensors import safe_open  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from reference.slat_capture import SlatCapture  # noqa: E402


class FakeSparse:
    def __init__(self, n):
        self.coords = torch.zeros(n, 4, dtype=torch.int64)
        self.coords[:, 1:] = torch.randint(0, 64, (n, 3))
        self.feats = torch.randn(n, 8, dtype=torch.float16)


class FakePipe:
    slat_normalization = {"mean": [0.0] * 8, "std": [1.0] * 8}

    def __init__(self):
        self.inner_calls = 0

    def decode_slat(self, slat, formats=("mesh",)):
        self.inner_calls += 1
        return {"mesh": [object()]}


def test_capture_format_and_restore(tmp_path):
    pipe = FakePipe()
    out = tmp_path / "slat.safetensors"
    sl = FakeSparse(123)
    with SlatCapture(pipe, out, extra_meta={"edit_seed": 0}) as cap:
        pipe.decode_slat(sl, ["gaussian", "mesh"])
    assert pipe.inner_calls == 1
    assert cap.n_calls == 1
    assert "decode_slat" not in vars(pipe), "인스턴스 속성이 남으면 안 된다"
    with safe_open(str(out), "pt") as f:
        c, ft, md = f.get_tensor("coords"), f.get_tensor("feats"), f.metadata()
    assert c.dtype == torch.int32 and tuple(c.shape) == (123, 4)
    assert ft.dtype == torch.float32 and tuple(ft.shape) == (123, 8)
    assert md["slat_space"] == "denormalized" and md["slat_channels"] == "8"
    assert "norm_mean" in md and "norm_std" in md and md["edit_seed"] == "0"
    assert torch.equal(c, sl.coords.int())


def test_no_save_on_exception(tmp_path):
    pipe = FakePipe()
    out = tmp_path / "x.safetensors"
    with pytest.raises(RuntimeError):
        with SlatCapture(pipe, out):
            pipe.decode_slat(FakeSparse(4))
            raise RuntimeError("boom")
    assert not out.exists()
    assert "decode_slat" not in vars(pipe)
