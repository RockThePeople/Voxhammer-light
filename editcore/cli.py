"""진입점: python -m editcore.cli --slat S --mask_json M --image_dir D --out OUT.glb [--seed 0] [--timing T.json]

포크 `run_edit`(W34 경로)와 같은 입력·출력(GLB). `--slat_out` 으로 편집 SLat(R2) 도 저장한다.
`--timing` 이면 reference/meter.py 로 구간 3–8 과 메모리 4종을 JSON 에 남긴다(스키마 vhl-run/1 의 부분집합).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--slat", required=True)
    ap.add_argument("--mask_json", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slat_out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timing")
    ap.add_argument("--parent_cbin", help="2번 R4: 부모 .cbin 디렉터리(chunks/ 또는 *.cbin). 있으면 마스크∪halo 부기 밖 청크를 승계")
    ap.add_argument("--cbin_out", help="2번 R4: .cbin 출력 디렉터리(기본 <--out 폴더>/cbin)")
    ap.add_argument("--steps_s1", type=int, help="1번 E2a: Stage 1 스텝(기본 VHL_E2A_STEPS_S1 또는 15, E20)")
    ap.add_argument("--steps_s2", type=int, help="1번 E2a: Stage 2 스텝(기본 VHL_E2A_STEPS_S2 또는 15, E20)")
    a = ap.parse_args(argv)
    t_proc = time.perf_counter()

    meter = sampler = None
    if a.timing:
        from reference.meter import Meter, Sampler, tensor_bytes
        sampler = Sampler().start()
        meter = Meter(sampler)
    span = meter.span if meter else None
    hooks = {"span": (lambda n, s: meter.span(n, segment=s))} if meter else None

    from .models import derive_constants, load_pipeline
    ctx = span("model_load", segment=8) if span else None
    if ctx:
        ctx.__enter__()
    pipe = load_pipeline()
    if ctx:
        ctx.__exit__(None, None, None)
    from .export import decode_to_glb, save_slat
    from .pipeline import Editor
    from .service import load_request

    C = derive_constants(pipe)
    req = load_request(a.slat, a.mask_json, a.image_dir, seed=a.seed)
    editor = Editor(pipe, C, steps_s1=a.steps_s1, steps_s2=a.steps_s2)
    print(f"[e2a] steps s1={editor.steps['s1']} s2={editor.steps['s2']}", flush=True)
    res = editor.edit(req, hooks=hooks)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.slat_out:
        save_slat(res.slat_tgt, a.slat_out, pipe.slat_normalization, {"seed": a.seed, "source": "editcore", "steps": editor.steps})
    from .models import r4_enabled
    dec_calls = {n: 0 for n in ("slat_decoder_gs", "slat_decoder_mesh")}
    handles = [pipe.models[n].register_forward_hook(lambda *_, n=n: dec_calls.__setitem__(n, dec_calls[n] + 1))
               for n in dec_calls if n in pipe.models]
    r4 = None
    if r4_enabled():
        from .export import decode_to_cbin
        mask_cells = json.loads(Path(a.mask_json).read_text())["mask"]["voxels"]
        cdir = Path(a.cbin_out) if a.cbin_out else Path(a.out).parent / "cbin"
        r4 = decode_to_cbin(pipe, res.slat_tgt, cdir, mask_cells=mask_cells, parent_dir=a.parent_cbin, glb_path=a.out, hooks=hooks)
        print(f"[r4] VHL_R4=1 mesh-only → .cbin {cdir}: chunks {r4['chunks']} (새 {len(r4['sent'])}, 부모 승계 {len(r4['reused'])}, "
              f"removed {len(r4['removed'])}, 부기 {len(r4['book'])}), 면 {r4['mesh']['faces']}, 바이트 {r4['total_bytes']}", flush=True)
    else:
        decode_to_glb(pipe, res.slat_tgt, a.out, hooks=hooks)
    for h in handles:
        h.remove()
    print(f"[r4] decoder calls {dec_calls}", flush=True)
    cache_entries = {k: {"latent": len(v[0]), "kv": len(v[1])} for k, v in res.caches.items()}
    print(f"[e2a] cache entries {cache_entries}", flush=True)
    out = {"counts": res.counts, "out": a.out, "seed": a.seed, "steps": editor.steps, "cache_entries": cache_entries,
           "decoder_calls": dec_calls, "r4": r4}
    from .models import ds_enabled
    if ds_enabled():
        from .blocks import DeterministicSparseDownsample
        out["ds"] = {"enabled": True, "deterministic_downsample_calls": DeterministicSparseDownsample.calls}
        print(f"[ds] deterministic downsample calls {DeterministicSparseDownsample.calls}", flush=True)
    if meter:
        sampler.stop()
        segs = {}
        for s in meter.spans:
            d = segs.setdefault(str(s["segment"]), {"seconds": 0.0, "max_memory_allocated": 0,
                                                    "nvidia_smi_mib_max": 0, "rss_bytes_max": 0, "parts": {}})
            d["seconds"] += s["seconds"]
            d["parts"][s["name"]] = d["parts"].get(s["name"], 0.0) + s["seconds"]
            for k in ("max_memory_allocated", "nvidia_smi_mib_max", "rss_bytes_max"):
                if s.get(k) is not None:
                    d[k] = max(d[k], s[k])
        out.update(schema="vhl-run/1", impl="editcore", segments=segs, spans=meter.spans, calls=meter.calls,
                   cache_bytes={k: {"latent": tensor_bytes(v[0]), "kv": tensor_bytes(v[1])} for k, v in res.caches.items()},
                   process_wall_seconds=time.perf_counter() - t_proc)
        Path(a.timing).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("counts", "out", "seed", "steps", "cache_entries")}))


if __name__ == "__main__":
    main()
