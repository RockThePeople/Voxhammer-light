"""계측 도구 — 구간 시간·VRAM·nvidia-smi·RSS (CLAUDE.md 계측 규칙, 요구사항서 6장).

- 구간 경계마다 `torch.cuda.synchronize()` 후 `time.perf_counter()`.
- VRAM: 구간 시작에 `reset_peak_memory_stats()`, 끝에 `max_memory_allocated()`.
  구간이 중첩되면(예: Stage 1 denoise 안의 디코더) 바깥 구간의 피크를 잃지 않도록
  안쪽 진입 전 바깥의 현재 피크를 적어 두고, 안쪽이 끝나면 합친다.
- nvidia-smi `memory.used` 와 프로세스 RSS 는 별도 스레드가 0.1 s 간격으로 샘플링하고,
  구간의 시간 창으로 최대값을 뽑는다.
"""
from __future__ import annotations

import os
import resource
import subprocess
import threading
import time
from contextlib import contextmanager

import torch

_PAGE = os.sysconf("SC_PAGE_SIZE")


def _rss_bytes() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * _PAGE


class Sampler:
    """nvidia-smi(메모리 사용량, MiB)와 RSS 를 시간 도장과 함께 모은다."""

    def __init__(self, interval_ms: int = 100):
        self.interval = interval_ms / 1000
        self.smi: list[tuple[float, int]] = []
        self.rss: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._proc = None
        self._threads = []

    def start(self):
        try:
            self._proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                 f"-lms={int(self.interval * 1000)}"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
            t = threading.Thread(target=self._read_smi, daemon=True)
            t.start()
            self._threads.append(t)
        except FileNotFoundError:
            self._proc = None
        t = threading.Thread(target=self._read_rss, daemon=True)
        t.start()
        self._threads.append(t)
        return self

    def _read_smi(self):
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            try:
                self.smi.append((time.perf_counter(), int(line.strip().split()[0])))
            except (ValueError, IndexError):
                pass

    def _read_rss(self):
        while not self._stop.is_set():
            self.rss.append((time.perf_counter(), _rss_bytes()))
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()
        if self._proc:
            self._proc.terminate()

    @staticmethod
    def _max_in(series, t0, t1):
        vals = [v for t, v in series if t0 <= t <= t1]
        return max(vals) if vals else None

    def window(self, t0, t1):
        return {"nvidia_smi_mib_max": self._max_in(self.smi, t0, t1),
                "rss_bytes_max": self._max_in(self.rss, t0, t1)}


class Meter:
    """이름 붙은 구간들을 잰다. `spans` 는 진입 순서대로 남고 `parent` 로 중첩을 기록한다."""

    def __init__(self, sampler: Sampler | None = None, cuda: bool = True):
        self.sampler = sampler
        self.cuda = cuda and torch.cuda.is_available()
        self.spans: list[dict] = []
        self._stack: list[dict] = []
        self.calls: dict[str, int] = {}

    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    @contextmanager
    def span(self, name: str, **info):
        self._sync()
        if self.cuda:
            if self._stack:
                outer = self._stack[-1]
                outer["_peak"] = max(outer["_peak"], torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
        rec = {"name": name, "parent": self._stack[-1]["name"] if self._stack else None,
               "_peak": 0, **info}
        self.calls[name] = self.calls.get(name, 0) + 1
        self._stack.append(rec)
        t0 = time.perf_counter()
        try:
            yield rec
        finally:
            self._sync()
            t1 = time.perf_counter()
            self._stack.pop()
            peak = max(rec.pop("_peak"), torch.cuda.max_memory_allocated()) if self.cuda else None
            if self.cuda and self._stack:
                self._stack[-1]["_peak"] = max(self._stack[-1]["_peak"], peak)
            rec.update(t_start=t0, t_end=t1, seconds=t1 - t0, max_memory_allocated=peak)
            if self.sampler:
                rec.update(self.sampler.window(t0, t1))
            self.spans.append(rec)

    def wrap(self, fn, name: str, **info):
        """함수를 감싸 호출마다 span 을 남긴다. 원본은 `__wrapped__` 로 둔다."""
        meter = self

        def wrapped(*a, **kw):
            with meter.span(name, **info):
                return fn(*a, **kw)

        wrapped.__wrapped__ = fn
        wrapped.__name__ = getattr(fn, "__name__", name)
        return wrapped


def tensor_bytes(obj, _seen=None) -> dict:
    """dict/list/텐서/SparseTensor 안의 텐서 바이트 합을 장치별로 센다."""
    if _seen is None:
        _seen = set()
    out = {"cpu": 0, "cuda": 0}

    def add(t):
        if id(t) in _seen:
            return
        _seen.add(id(t))
        out["cuda" if t.is_cuda else "cpu"] += t.numel() * t.element_size()

    def walk(o):
        if isinstance(o, torch.Tensor):
            add(o)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif hasattr(o, "feats") and hasattr(o, "coords"):
            walk(o.feats)
            walk(o.coords)

    walk(obj)
    return out


def ru_maxrss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
