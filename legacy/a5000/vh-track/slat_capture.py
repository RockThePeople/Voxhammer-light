"""
Capture the EDITED SLAT from a VoxHammer edit — VoxHammer source unchanged (0 lines).

edit_pipeline.py:
  553:  coords_tgt = torch.argwhere(voxel_tgt > 0)[:, [0,2,3,4]].int()   # 64^3 SS grid = D28 canon
  557:  slat_tgt   = sample_slat_denoise(...)                            # the edited SLAT
  558:  assets_tgt = pipeline.decode_slat(slat_tgt, ["gaussian","mesh"]) # <-- we wrap this
  560:  glb_tgt.export(output_path)                                      # only the GLB is kept

Emitted format matches the generation artefact exactly:
    coords (N,4) int32   [batch, x, y, z]
    feats  (N,8) float32 (slat_channels=8)

Usage
-----
    from slat_capture import capture_slat

    with capture_slat(pipe, "/path/out.slat.safetensors") as cap:
        run_edit(pipe, RD, OUT, image_dir=IMG, is_text=False, ...)
    print(cap.path, cap.shape)      # written on exit

Caveat (D55): the final GLB comes from a GLOBAL decode of this SLAT. Emitting it lets a
delta be computed on the D28 canonical grid, but it does NOT remove the out-of-mask
change that the global decode produces.
"""
from contextlib import contextmanager


class _Cap:
    def __init__(self, path):
        self.path = path
        self.slat = None
        self.shape = None
        self.n_calls = 0


@contextmanager
def capture_slat(pipeline, out_path, *, save=True, meta=None):
    """Wrap pipeline.decode_slat so the SLAT it is handed is captured and saved.

    decode_slat is called exactly once on the edit path (line 558). If it fires more
    than once the LAST call wins and n_calls records it -- check that in the caller.
    """
    cap = _Cap(out_path)
    orig = pipeline.decode_slat

    def _wrapped(slat, formats, *a, **kw):
        cap.slat = slat
        cap.n_calls += 1
        return orig(slat, formats, *a, **kw)

    pipeline.decode_slat = _wrapped
    try:
        yield cap
    finally:
        pipeline.decode_slat = orig
        if save and cap.slat is not None:
            m = dict(meta or {})
            norm = getattr(pipeline, "slat_normalization", None)
            if norm and "norm_mean" not in m:
                m["norm_mean"], m["norm_std"] = norm["mean"], norm["std"]
            cap.shape = write_slat(cap.slat, out_path, m)


def write_slat(slat, out_path, meta=None):
    """Serialise a SparseTensor to the generation artefact's schema.

    🔴 METADATA IS NOT OPTIONAL. `ops.op_edit` reads `meta["norm_mean"]` /
    `meta["norm_std"]` from the PARENT slat (ops.py:265-266). A slat written
    without them is a **dead-end version** — the next edit dies on KeyError.
    Measured in W28: the first real-path capture wrote tensors only and came
    out with `slat_space=None`, metadata `{}`.
    """
    import json, torch
    from safetensors.torch import save_file
    coords = slat.coords.detach().cpu().contiguous().to(torch.int32)
    feats = slat.feats.detach().cpu().contiguous().float()
    md = {"contract_version": "4", "slat_channels": str(feats.shape[1]),
          "voxel_res": "64", "coord_order": "canonical",
          # slat_tgt is ALREADY denormalized: edit_pipeline.py:489 does slat*std+mean
          "slat_space": "denormalized"}
    if meta:
        md.update({k: (v if isinstance(v, str) else json.dumps(v)) for k, v in meta.items()})
    missing = [k for k in ("norm_mean", "norm_std") if k not in md]
    if missing:
        raise ValueError(
            f"slat 메타 {missing} 이 없다 — 이대로 저장하면 다음 op_edit 이 죽는다. "
            f"pipeline.slat_normalization 을 meta 로 넘겨라")
    save_file({"coords": coords, "feats": feats}, out_path, metadata=md)
    return {"coords": tuple(coords.shape), "feats": tuple(feats.shape), "metadata": sorted(md)}


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    import sys, tempfile, os
    import torch
    from safetensors import safe_open

    class _FakeSparse:                      # stands in for sp.SparseTensor
        def __init__(self, n):
            self.coords = torch.zeros(n, 4, dtype=torch.int32)
            self.coords[:, 1] = torch.arange(n) % 64
            self.feats = torch.randn(n, 8)

    class _FakePipe:
        def __init__(self): self.calls = 0
        def decode_slat(self, slat, formats):
            self.calls += 1
            return {"mesh": ["<mesh>"], "gaussian": ["<gs>"]}

    pipe = _FakePipe()
    out = os.path.join(tempfile.mkdtemp(), "probe.slat.safetensors")
    sl = _FakeSparse(9591)

    with capture_slat(pipe, out) as cap:
        pipe.decode_slat(sl, ["gaussian", "mesh"])          # what run_edit does at :558

    assert pipe.calls == 1, "inner decode must still run exactly once"
    assert cap.n_calls == 1, f"capture saw {cap.n_calls} calls"
    assert hasattr(pipe.decode_slat, "__self__"), "decode_slat must be restored"

    with safe_open(out, "pt") as f:
        c, ft = f.get_tensor("coords"), f.get_tensor("feats")
    print(f"  captured  coords {tuple(c.shape)} {c.dtype}   feats {tuple(ft.shape)} {ft.dtype}")
    assert c.shape == (9591, 4) and c.dtype == torch.int32, "coords schema mismatch"
    assert ft.shape == (9591, 8) and ft.dtype == torch.float32, "feats schema mismatch"
    print(f"  file      {os.path.getsize(out)} bytes")
    print("  decode_slat restored after the block:", pipe.decode_slat.__func__ is _FakePipe.decode_slat)
    print("SELFTEST OK — schema matches the generation artefact (coords N,4 I32 / feats N,8 F32)")
