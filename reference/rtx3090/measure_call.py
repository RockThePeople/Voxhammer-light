"""B-3: Z-Image-Turbo · BiRefNet 워커 호출 1회 비용 측정 (00-kickoff B-3).

워커 스크립트(`zimage_gen.py`, `birefnet_seg.py`)는 수정하지 않는다. 각 호출을 subprocess 로 띄우고
- 벽시계: subprocess 기동 ~ 종료
- 적재·생성: `_inner.py` 훅이 기록한 시각 (구간 경계에서 torch.cuda.synchronize)
- VRAM: 훅 안의 torch.cuda.max_memory_allocated, 그리고 nvidia-smi 0.1 s 샘플 최대값
          (GPU 전체 memory.used, 워커 pid 의 used_memory 두 열)
을 JSON 으로 남긴다. 실행 순서는 실제 t2i 경로(ai-ar-v3 server/t2i.py)처럼 Z-Image → BiRefNet 순차.

환경변수로만 받는다 (ai-ar-v3 서버와 같은 이름):
    T2I_TOOLS_DIR, ZIMAGE_PYTHON, BIREFNET_PYTHON, RUNS_DIR(기본 ./runs)

    python3 reference/rtx3090/measure_call.py --repeat 3 --prompt "..."
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def smi(args):
    return subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=5).stdout.strip()


def gpu_busy():
    return [l for l in smi(["--query-compute-apps=pid,process_name,used_memory"]).splitlines() if l.strip()]


class Sampler(threading.Thread):
    """nvidia-smi 0.1 s 간격. GPU 전체 memory.used 와 지정 pid 의 used_memory."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.pid = None
        self.total, self.proc = [], []

    def run(self):
        while not self.stop.is_set():
            t = time.perf_counter()
            try:
                self.total.append(int(smi(["--query-gpu=memory.used"]).splitlines()[0]))
                if self.pid:
                    for l in smi(["--query-compute-apps=pid,used_memory"]).splitlines():
                        pid, mb = [x.strip() for x in l.split(",")]
                        if int(pid) == self.pid:
                            self.proc.append(int(mb))
            except Exception:
                pass
            time.sleep(max(0.0, 0.1 - (time.perf_counter() - t)))


def call(kind, py, worker, args, rec_path, log_path):
    s = Sampler()
    base = int(smi(["--query-gpu=memory.used"]).splitlines()[0])
    s.start()
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        p = subprocess.Popen([py, str(HERE / "_inner.py"), kind, str(rec_path), str(worker), *args],
                             stdout=log, stderr=subprocess.STDOUT)
        s.pid = p.pid
        rc = p.wait()
    wall = time.perf_counter() - t0
    s.stop.set()
    s.join()
    rec = json.loads(Path(rec_path).read_text()) if Path(rec_path).is_file() else {}
    ev = rec.get("events", {})
    out = {
        "kind": kind, "rc": rc, "wall_s": round(wall, 3),
        "startup_to_load_start_s": ev.get("load_start"),
        "load_s": (ev["load_end"] - ev["load_start"]) if {"load_end", "load_start"} <= ev.keys() else None,
        "gen_s": (ev["gen_end"] - ev["gen_start"]) if {"gen_end", "gen_start"} <= ev.keys() else None,
        "after_gen_to_exit_s": (wall - ev["gen_end"]) if "gen_end" in ev else None,
        "max_alloc_after_load_GiB": rec.get("max_alloc_after_load", 0) / 2**30 or None,
        "max_alloc_gen_GiB": rec.get("max_alloc_gen", 0) / 2**30 or None,
        "smi_gpu_used_baseline_MiB": base,
        "smi_gpu_used_max_MiB": max(s.total) if s.total else None,
        "smi_proc_used_max_MiB": max(s.proc) if s.proc else None,
        "smi_samples": len(s.total),
        "inner": rec,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    env = {n: os.environ.get(n, "").strip() for n in ("T2I_TOOLS_DIR", "ZIMAGE_PYTHON", "BIREFNET_PYTHON")}
    miss = [n for n, v in env.items() if not v]
    if miss:
        sys.exit(f"환경변수가 없다: {', '.join(miss)}")
    tools = Path(env["T2I_TOOLS_DIR"])
    zw, bw = tools / "zimage_gen.py", tools / "birefnet_seg.py"

    busy = gpu_busy()
    if busy:
        sys.exit(f"다른 GPU 작업이 있다 — 측정하지 않는다: {busy}")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(os.environ.get("RUNS_DIR") or "runs") / f"b3-call-cost-{stamp}"
    out_dir.mkdir(parents=True)

    runs = []
    for i in range(a.repeat):
        rgb, rgba = out_dir / f"z{i}.png", out_dir / f"b{i}.png"
        z = call("zimage", env["ZIMAGE_PYTHON"], zw,
                 ["--prompt", a.prompt, "--out", str(rgb), "--seed", str(a.seed)],
                 out_dir / f"z{i}.inner.json", out_dir / f"z{i}.log")
        z["out_sha256"] = sha256(rgb) if rgb.is_file() else None
        runs.append(z)
        print(f"[{i}] zimage rc={z['rc']} wall={z['wall_s']:.1f}s load={z['load_s']} gen={z['gen_s']}", flush=True)
        b = call("birefnet", env["BIREFNET_PYTHON"], bw, ["--in", str(rgb), "--out", str(rgba)],
                 out_dir / f"b{i}.inner.json", out_dir / f"b{i}.log")
        b["out_sha256"] = sha256(rgba) if rgba.is_file() else None
        runs.append(b)
        print(f"[{i}] birefnet rc={b['rc']} wall={b['wall_s']:.1f}s load={b['load_s']} gen={b['gen_s']}", flush=True)

    result = {
        "item": "00-kickoff B-3", "started": stamp, "gpu": smi(["--query-gpu=name,driver_version"]),
        "prompt": a.prompt, "seed": a.seed, "repeat": a.repeat,
        "worker_sha256": {"zimage_gen.py": sha256(zw), "birefnet_seg.py": sha256(bw)},
        "runs": runs,
    }
    path = out_dir / "b3_call_cost.json"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print("RESULT", path, sha256(path))


if __name__ == "__main__":
    main()
