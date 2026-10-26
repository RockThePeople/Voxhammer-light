"""청크 경계 봉합 띠 오프라인 적용(chief 21:4x 결정 (d)). 편집을 다시 하지 않는다 — 저장된 R4 .cbin 만 다시 조립한다.

회차 k 의 메시 = 승계 청크(그 회차 r4.json 의 reused)는 **직전 회차 봉합본**에서, 새 청크(sent)는 원래 회차의 스냅 후 바이트에서 가져와
`editcore.export.stitch_seams` 로 새 청크에만 봉합 띠를 더한다. 승계 청크 바이트는 직전 봉합본 그대로다(R11).
결과 <out>/cbin/chunks/*.cbin · output.glb · view.glb((iv)) · stitch.json(열린 모서리 전·후, 띠 면 수·면적 비율, 승계 바이트 해시 대조).

    python tools/paper/stitch_offline.py x3 <p1|p2> <자산>      # $RUNS_DIR/paper/x3/stitched/<경로>/<자산>/r{1..5}/
    python tools/paper/stitch_offline.py x0 <시나리오> [r1 r2 r3] # $RUNS_DIR/paper/x0/cbin_stitched/<시나리오>/<rK>/
    python tools/paper/stitch_offline.py x2a                       # X2-A P1-cbin = X0 c15-cbin 과 같은 출처라 x0 결과를 쓴다
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests" / "gpu"), str(ROOT / "third_party" / "ai-ar-v3"),
                str(ROOT / "third_party" / "ai-ar-v3" / "contract" / "python")]
RUNS = Path(os.environ["RUNS_DIR"])


def chunk_files(d):
    d = Path(d)
    d = d / "chunks" if (d / "chunks").is_dir() else d
    return {p.stem.split(".")[0]: p for p in sorted(d.glob("*.cbin"))}


def stitch_round(orig_cbin, prev_chunks, out, stitch=True, skirt=False):
    """orig_cbin: 원래 회차 cbin 폴더(r4.json, chunks/), prev_chunks: 직전 회차(봉합본) 청크 폴더.
    stitch: 봉합 띠, skirt: 치마 벽(chief 22:2x 시험). 둘 다 새 청크에만 더한다."""
    import deltacontract as dc
    from editcore.export import assembled_open_edges, skirt_seams, stitch_seams
    from editcore.render import vertex_color_render_glb
    from server.glbbuild import build_glb
    out = Path(out)
    r4 = json.loads((Path(orig_cbin) / "r4.json").read_text())
    sent, reused = set(r4["sent"]), set(r4["reused"])
    orig, prev = chunk_files(orig_cbin), chunk_files(prev_chunks)
    final = {}
    for k in sent:
        final[k] = orig[k].read_bytes()
    for k in reused:
        final[k] = (prev[k] if k in prev else orig[k]).read_bytes()
    meshes = {k: dc.decode(b) for k, b in final.items()}
    before = int(len(assembled_open_edges(meshes)[1]))
    st = {"open_edges_before": before, "strip_faces": 0, "strip_area_ratio": 0.0}
    if stitch:
        changed, st = stitch_seams(meshes, sent)
        for k, cm in changed.items():
            final[k] = dc.encode(cm)
            meshes[k] = cm
    if skirt:
        changed, kst = skirt_seams(meshes, sent)
        for k, cm in changed.items():
            final[k] = dc.encode(cm)
            meshes[k] = cm
        st["skirt"] = kst
    st.update(stitch=stitch, skirt_on=skirt)
    st["open_edges_after"] = int(len(assembled_open_edges(meshes)[1]))
    st["open_edges_before_check"] = before
    (out / "cbin" / "chunks").mkdir(parents=True, exist_ok=True)
    for k, b in final.items():
        (out / "cbin" / "chunks" / f"{k}.cbin").write_bytes(b)
    inh_same = all(final[k] == (prev[k] if k in prev else orig[k]).read_bytes() for k in reused)
    st.update(sent=len(sent), reused=len(reused), inherited_bytes_unchanged=bool(inh_same),
              orig_cbin=str(orig_cbin), prev_chunks=str(prev_chunks),
              sha_chunks=hashlib.sha256(b"".join(final[k] for k in sorted(final))).hexdigest())
    build_glb(out / "cbin" / "chunks", out / "output.glb")
    (out / "view.glb").write_bytes(vertex_color_render_glb((out / "output.glb").read_bytes()))
    (out / "stitch.json").write_text(json.dumps(st, indent=1))
    return st


def x3(path, asset):
    from tools.paper.x3 import OUT, gen
    g, _ = gen(asset)
    prev = g / "chunks"
    for k in range(1, 6):
        o = OUT / "stitched" / path / asset / f"r{k}"
        if not (o / "stitch.json").exists():
            st = stitch_round(OUT / path / asset / f"r{k}" / "cbin", prev, o)
            print(f"stitch x3 {path} {asset} r{k} open {st['open_edges_before']} → {st['open_edges_after']} strip {st['strip_faces']} "
                  f"({st['strip_area_ratio'] * 100:.4f}% area) inh_same {st['inherited_bytes_unchanged']}", flush=True)
        prev = o / "cbin" / "chunks"


def x3_variant(variant, path, asset, rounds=5):
    """chief 22:2x 2-1: 조합별 체인(회차 순서, 직전 회차 같은 조합본 승계). variant = skirt | both."""
    from tools.paper.x3 import OUT, gen
    flags = {"skirt": (False, True), "both": (True, True), "stitch": (True, False)}[variant]
    g, _ = gen(asset)
    prev = g / "chunks"
    for k in range(1, rounds + 1):
        o = OUT / "seamvar" / variant / path / asset / f"r{k}"
        if not (o / "stitch.json").exists():
            st = stitch_round(OUT / path / asset / f"r{k}" / "cbin", prev, o, *flags)
            print(f"seamvar {variant} {path} {asset} r{k} open {st['open_edges_before']} → {st['open_edges_after']} "
                  f"strip {st['strip_faces']} skirt {st.get('skirt', {}).get('skirt_faces', 0)} inh_same {st['inherited_bytes_unchanged']}", flush=True)
        prev = o / "cbin" / "chunks"


def x0(sid, runs=("r1", "r2", "r3")):
    from r4_checks import scenarios
    S = {s["id"]: s for s in scenarios()}
    base = RUNS / "paper" / "x0" / "cbin" / sid
    for r in runs:
        src = base if r == "r1" else base / r
        o = RUNS / "paper" / "x0" / "cbin_stitched" / sid / r
        if (o / "stitch.json").exists() or not (src / "cbin" / "r4.json").exists():
            continue
        st = stitch_round(src / "cbin", S[sid]["parent"], o)
        print(f"stitch x0 {sid} {r} open {st['open_edges_before']} → {st['open_edges_after']} strip {st['strip_faces']} "
              f"inh_same {st['inherited_bytes_unchanged']}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "x3":
        x3(sys.argv[2], sys.argv[3])
    elif cmd == "x3v":
        x3_variant(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "x0":
        x0(sys.argv[2], tuple(sys.argv[3:]) or ("r1", "r2", "r3"))
