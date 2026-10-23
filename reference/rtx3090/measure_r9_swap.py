"""05 R9 R-3 (a) 교대 로드: 워커 종료 → Z-Image 1회 → 워커 재기동(ready 까지), 3주기. 실행 중인 워커(:8082)가 있어야 한다."""
import json, os, signal, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, 'reference/rtx3090')
import measure_call as mc
P = ("a man riding a black motorcycle, full body, 3/4 view, product photography, studio lighting, "
     "plain white background, centered, entire subject fully visible with margin, no text, no watermark")
tools = Path(os.environ['T2I_TOOLS_DIR']); out = Path('runs/r9-r3'); res = []
env = dict(os.environ, VHL_R9='1', RUNS_DIR='runs', FLUX_FILL_BASE=os.environ.get('FLUX_FILL_BASE', 'camenduru/FLUX.1-Fill-dev-ungated'))
def worker_pid():
    o = subprocess.run(['ss', '-ltnpH', 'sport = :8082'], capture_output=True, text=True).stdout
    import re; m = re.search(r'pid=(\d+)', o); return int(m.group(1)) if m else None
def start_worker(i):
    t = time.perf_counter()
    log = open(out / f'worker_restart{i}.log', 'w')
    p = subprocess.Popen([os.path.expanduser('~/anaconda3/envs/flux-fill/bin/python'), '-u', 'services/rtx3090_inpaint/worker.py'],
                         env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    while '[ready]' not in (out / f'worker_restart{i}.log').read_text():
        if p.poll() is not None: raise SystemExit('worker died')
        time.sleep(0.1)
    return time.perf_counter() - t
for i in range(3):
    cyc = {}
    t0 = time.perf_counter()
    pid = worker_pid(); os.kill(pid, signal.SIGTERM)
    while worker_pid() or subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True).stdout.strip():
        time.sleep(0.1)
    cyc['stop_s'] = time.perf_counter() - t0
    z = mc.call('zimage', os.environ['ZIMAGE_PYTHON'], tools/'zimage_gen.py', ['--prompt', P, '--out', str(out/f'z_swap{i}.png'), '--seed', '42'],
                out/f'z_swap{i}.inner.json', out/f'z_swap{i}.log'); z.pop('inner', None)
    cyc['zimage'] = z
    cyc['worker_restart_to_ready_s'] = start_worker(i)
    cyc['cycle_total_s'] = time.perf_counter() - t0
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in cyc.items() if k != 'zimage'}),
          'z wall', z['wall_s'], 'rc', z['rc'], 'sha', mc.sha256(out/f'z_swap{i}.png')[:12] if (out/f'z_swap{i}.png').is_file() else None, flush=True)
    res.append(cyc)
(out/'r3_swap.json').write_text(json.dumps(res, indent=1, default=str))
