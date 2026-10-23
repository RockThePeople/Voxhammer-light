"""05 R9 R-1·R-2 계측: 상주 워커 기동~ready, 호출 1회차(첫 호출)·2회차 시간·VRAM. N 주기 반복.

워커를 `VHL_R9=1` 로 띄우고 클라이언트(services/rtx3090_inpaint/client.py)로 부른 뒤 종료한다.
입력: --image(1024² PNG) --mask(1024² PNG) --prompt. 결과 JSON 은 RUNS_DIR/r9-worker-<시각>/.

    python3 reference/rtx3090/measure_r9_worker.py --image ... --mask ... --prompt ... --cycles 3 --calls 2
"""
import argparse, datetime, hashlib, json, os, re, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from services.rtx3090_inpaint import client  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True); ap.add_argument("--mask", required=True)
ap.add_argument("--prompt", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cycles", type=int, default=3); ap.add_argument("--calls", type=int, default=2)
ap.add_argument("--python", default=os.environ.get("RTX3090_PYTHON") or sys.executable)
a = ap.parse_args()


def smi(q):
    return subprocess.run(["nvidia-smi", f"--query-{q}", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()


if smi("compute-apps=pid"):
    sys.exit("다른 GPU 작업이 있다 — 측정하지 않는다")
out = Path(os.environ.get("RUNS_DIR") or ROOT / "runs") / f"r9-worker-{datetime.datetime.now():%Y%m%d-%H%M%S}"
out.mkdir(parents=True)
img, msk = Path(a.image).read_bytes(), Path(a.mask).read_bytes()
env = dict(os.environ, VHL_R9="1", RUNS_DIR=str(out))
res = {"item": "05 R9 R-1/R-2", "prompt": a.prompt, "seed": a.seed,
       "input_sha256": {"image": hashlib.sha256(img).hexdigest(), "mask": hashlib.sha256(msk).hexdigest()},
       "gpu": smi("gpu=name,driver_version"), "cycles": []}
for c in range(a.cycles):
    logp = out / f"worker{c}.log"
    t0 = time.perf_counter()
    p = subprocess.Popen([a.python, "-u", str(ROOT / "services/rtx3090_inpaint/worker.py")], env=env,
                         stdout=open(logp, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    while "[ready]" not in logp.read_text():
        if p.poll() is not None:
            sys.exit(f"워커 종료: {logp}")
        time.sleep(0.05)
    cyc = {"start_to_ready_s": time.perf_counter() - t0, "resident_smi_mib": int(smi("gpu=memory.used")),
           "load": client.health()["load"], "calls": []}
    for k in range(a.calls):
        t = time.perf_counter()
        r = client.inpaint(img, msk, a.prompt, seed=a.seed, out_dir=out / f"c{c}_call{k}")
        sp = {s["name"]: s for s in r["result"]["spans"]}
        cyc["calls"].append({"client_s": time.perf_counter() - t, "inpaint_s": sp["inpaint"]["seconds"],
                             "composite_s": sp["composite"]["seconds"],
                             "max_alloc_GiB": sp["inpaint"]["max_memory_allocated"] / 2**30,
                             "smi_mib_max": sp["inpaint"].get("nvidia_smi_mib_max"),
                             "sha256": r["result"]["sha256"], "outside_diff": r["result"]["outside_diff"]})
    cyc["after_calls_smi_mib"] = int(smi("gpu=memory.used"))
    os.killpg(p.pid, signal.SIGTERM); p.wait()
    res["cycles"].append(cyc)
    print(f"[{c}] ready {cyc['start_to_ready_s']:.2f}s load {cyc['load']['seconds']:.2f}s resident {cyc['resident_smi_mib']} MiB | "
          + " | ".join(f"call{k} {x['inpaint_s']:.2f}s peak {x['max_alloc_GiB']:.2f}GiB smi {x['smi_mib_max']}"
                       for k, x in enumerate(cyc["calls"])), flush=True)
path = out / "r9_worker.json"
path.write_text(json.dumps(res, indent=1, default=str))
print("RESULT", path, hashlib.sha256(path.read_bytes()).hexdigest())
