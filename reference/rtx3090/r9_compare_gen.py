"""05 R9 R-4 생성: 트랙 A 조건(2d_render·2d_mask·프롬프트, A5000 동결본)으로 INT4 워커를 N회 돌린다.

워커는 한 번만 띄운다(적재 1회). 시나리오마다 같은 입력·seed 0 으로 --repeat 회 호출한다
(INT4 가 같은 시드에서도 비결정적이라 자기 반복 차이를 바닥값으로 쓴다 — 보고서 R-2).

    python3 reference/rtx3090/r9_compare_gen.py --cond $ASSETS_DIR/conditions/track_a --repeat 3
산출: RUNS_DIR/r9-r4-<시각>/<ID>/run<k>/{2d_render,2d_mask,2d_edit,2d_edit_raw}.png, gen.json
"""
import argparse, datetime, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from services.rtx3090_inpaint import client  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cond", required=True)
ap.add_argument("--repeat", type=int, default=3)
ap.add_argument("--python", default=os.environ.get("RTX3090_PYTHON") or sys.executable)
a = ap.parse_args()


def smi(q):
    return subprocess.run(["nvidia-smi", f"--query-{q}", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()


if smi("compute-apps=pid"):
    sys.exit("다른 GPU 작업이 있다 — 측정하지 않는다")
cond = Path(a.cond)
man = json.loads((cond / "manifest.json").read_text())["scenarios"]
out = Path(os.environ.get("RUNS_DIR") or ROOT / "runs") / f"r9-r4-{datetime.datetime.now():%Y%m%d-%H%M%S}"
out.mkdir(parents=True)
env = dict(os.environ, VHL_R9="1", RUNS_DIR=str(out / "_worker"))
logp = out / "worker.log"
t0 = time.perf_counter()
p = subprocess.Popen([a.python, "-u", str(ROOT / "services/rtx3090_inpaint/worker.py")], env=env,
                     stdout=open(logp, "w"), stderr=subprocess.STDOUT, start_new_session=True)
while "[ready]" not in logp.read_text():
    if p.poll() is not None:
        sys.exit(f"워커 종료: {logp}")
    time.sleep(0.05)
res = {"item": "05 R9 R-4", "start_to_ready_s": time.perf_counter() - t0, "load": client.health()["load"],
       "resident_smi_mib": int(smi("gpu=memory.used")), "repeat": a.repeat, "scenarios": {}}
try:
    for sid, v in sorted(man.items()):
        img, msk = (cond / sid / "2d_render.png").read_bytes(), (cond / sid / "2d_mask.png").read_bytes()
        runs = []
        for k in range(a.repeat):
            r = client.inpaint(img, msk, v["prompt"], seed=v["seed"], out_dir=out / sid / f"run{k}")
            sp = {s["name"]: s for s in r["result"]["spans"]}
            runs.append({"inpaint_s": sp["inpaint"]["seconds"], "max_alloc_GiB": sp["inpaint"]["max_memory_allocated"] / 2**30,
                         "smi_mib_max": sp["inpaint"].get("nvidia_smi_mib_max"), "sha256": r["result"]["sha256"],
                         "outside_diff": r["result"]["outside_diff"]})
            print(f"{sid} run{k} {runs[-1]['inpaint_s']:.2f}s peak {runs[-1]['max_alloc_GiB']:.2f}GiB "
                  f"after_identical {r['result']['outside_diff']['after']['outside_identical_frac']}", flush=True)
        res["scenarios"][sid] = {"prompt": v["prompt"], "seed": v["seed"], "edit_view": v["edit_view"],
                                 "input_sha256": v["sha256"], "runs": runs}
finally:
    os.killpg(p.pid, signal.SIGTERM)
    p.wait()
path = out / "gen.json"
path.write_text(json.dumps(res, indent=1, default=str))
print("RESULT", path, hashlib.sha256(path.read_bytes()).hexdigest())
