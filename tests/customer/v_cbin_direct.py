"""mesh 디코더 native 출력 → .cbin 직행 경로 검증 (to_glb 없이).
TRELLIS MeshExtractResult 모양을 흉내: vertices (V,3) in [-0.5,0.5], faces, vertex_attrs (V,6) in [0,1]."""
import sys, time, hashlib
import numpy as np, trimesh
REPO = sys.argv[1]
sys.path.insert(0, REPO + "/contract/python")
from deltacontract.partition import split_trellis_vertex_attrs, albedo_to_rgba8, partition_mesh
from deltacontract.chunkbin import encode, decode

def fake_decoder_output(seed, bump=False):
    m = trimesh.creation.icosphere(subdivisions=7, radius=0.45)       # 163,842 verts
    v = m.vertices.astype(np.float32).copy()
    if bump:  # 국소 편집: +x 쪽 작은 영역만 변형
        sel = v[:, 0] > 0.40
        v[sel] *= 1.05
        v = np.clip(v, -0.499, 0.499)
    rng = np.random.default_rng(seed)
    n = (m.vertex_normals.astype(np.float32) + 1) / 2                  # [0,1] 인코딩 법선
    albedo = np.full((len(v), 3), 0.6, np.float32)
    return v, m.faces.astype(np.int64), np.concatenate([albedo, n], 1)

def to_chunks(v, f, attrs):
    albedo, normals = split_trellis_vertex_attrs(attrs)               # 기본 "0_1"
    meshes = partition_mesh(v, f, normals=normals, colors_rgba8=albedo_to_rgba8(albedo))
    return {k: encode(m) for k, m in meshes.items()}

t0 = time.time(); base = to_chunks(*fake_decoder_output(0)); t1 = time.time()
edit = to_chunks(*fake_decoder_output(0, bump=True)); t2 = time.time()
h = lambda d: {k: hashlib.sha256(b).hexdigest() for k, b in d.items()}
hb, he = h(base), h(edit)
changed = [k for k in he if hb.get(k) != he[k]]; removed = [k for k in hb if k not in he]
nbytes = sum(len(b) for b in base.values())
d = decode(next(iter(base.values())))
print(f"verts 163842 · chunks {len(base)} · bytes {nbytes/1e6:.1f} MB")
print(f"partition+encode: base {t1-t0:.2f}s · edit {t2-t1:.2f}s (CPU)")
print(f"changed {len(changed)} / {len(he)} · removed {len(removed)} · 변경 바이트 {sum(len(edit[k]) for k in changed)/1e6:.2f} MB")
print(f"decode ok: V={len(d.positions)} normals={'Y' if d.normals is not None else 'N'} colors={'Y' if d.colors is not None else 'N'}")
un = np.linalg.norm(d.normals, axis=1); print(f"단위법선 비율 {(abs(un-1)<1e-3).mean()*100:.1f}%")
