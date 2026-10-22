"""워커 스크립트를 수정 없이 같은 프로세스에서 실행하며 적재·생성 시각과 CUDA 피크를 기록한다.

`measure_call.py` 가 subprocess 로 띄운다. 쓰임:
    python _inner.py <kind> <record.json> <worker.py> [worker args...]

kind: zimage | birefnet
워커 코드는 바꾸지 않는다. 라이브러리 진입점(from_pretrained, __call__/forward)을 감싸는 훅만 건다.
"""
import json
import os
import runpy
import sys
import time

T0 = time.perf_counter()
kind, record_path, worker = sys.argv[1], sys.argv[2], sys.argv[3]
worker_args = sys.argv[4:]

import torch  # noqa: E402

rec = {"kind": kind, "pid": os.getpid(), "events": {}}


def mark(name):
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
    rec["events"][name] = time.perf_counter() - T0


def peak(name):
    if torch.cuda.is_initialized():
        rec[name] = torch.cuda.max_memory_allocated()


if kind == "zimage":
    from diffusers import ZImagePipeline
    from diffusers import DiffusionPipeline

    _fp = ZImagePipeline.from_pretrained.__func__
    _call = ZImagePipeline.__call__
    _to = DiffusionPipeline.to

    @classmethod
    def from_pretrained(cls, *a, **k):
        mark("load_start")
        return _fp(cls, *a, **k)

    def to(self, *a, **k):
        out = _to(self, *a, **k)
        mark("load_end")
        peak("max_alloc_after_load")
        return out

    def call(self, *a, **k):
        torch.cuda.reset_peak_memory_stats()
        mark("gen_start")
        out = _call(self, *a, **k)
        mark("gen_end")
        peak("max_alloc_gen")
        return out

    ZImagePipeline.from_pretrained = from_pretrained
    ZImagePipeline.__call__ = call
    DiffusionPipeline.to = to

elif kind == "birefnet":
    from transformers import AutoModelForImageSegmentation

    _fp = AutoModelForImageSegmentation.from_pretrained.__func__

    def pre(module, args):
        mark("load_end")  # 첫 forward 직전 = 적재·GPU 전송·half 완료 후
        peak("max_alloc_after_load")
        torch.cuda.reset_peak_memory_stats()
        mark("gen_start")

    def post(module, args, out):
        mark("gen_end")
        peak("max_alloc_gen")

    @classmethod
    def from_pretrained(cls, *a, **k):
        mark("load_start")
        m = _fp(cls, *a, **k)
        m.register_forward_pre_hook(pre)
        m.register_forward_hook(post)
        return m

    AutoModelForImageSegmentation.from_pretrained = from_pretrained
else:
    raise SystemExit(f"unknown kind {kind}")

rec["events"]["imports_done"] = time.perf_counter() - T0
sys.argv = [worker] + worker_args
sys.path.insert(0, os.path.dirname(os.path.abspath(worker)))  # python worker.py 와 같은 sys.path[0]
try:
    runpy.run_path(worker, run_name="__main__")
    rec["ok"] = True
finally:
    rec["events"]["worker_end"] = time.perf_counter() - T0
    if torch.cuda.is_initialized():
        rec["max_alloc_process"] = torch.cuda.max_memory_allocated()  # 마지막 reset 이후
    rec["torch"] = torch.__version__
    rec["torch_cuda"] = torch.version.cuda
    with open(record_path, "w") as f:
        json.dump(rec, f, indent=1)
