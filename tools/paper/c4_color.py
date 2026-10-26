"""C4 색 표류 (지시서 main 88d2ab0 5절, 사용자 발견 2026-10-26 18:02). CPU 만.

sheet  : 0·1·3·5회차 × P0 | P1 | P2, 같은 시점(포크 1·3 = 양쪽 3/4)·같은 렌더러(Bench render.py, Blender).
         렌더는 x3_score 와 같은 캐시(같은 키)를 쓴다. P1·P2 정점색은 (iv) 보정, P0 to_glb 는 그대로 — 재질은 셋 다 metallic 기본 1.0.
albedo : 보존 영역 표면 점(1절과 같은 100k 점·seed 0·누적 마스크 ∪ halo 2 밖)에서 색을 직접 읽는다(렌더러 무관).
         P0 = 텍스처 UV 조회(최근접 텍셀), P1·P2 = 정점색 무게중심 보간(sRGB 값). A_origin(생성 텍스처 GLB) 표면 점마다 결과의
         최근접 점(2셀 이내) 색을 읽어 평균 ΔL*·ΔC*·ΔE76(CIELAB, D65), 절대 평균 L*·C*(같은 점들).
render : Bench 16시점 렌더에서 보존 영역 화소(결과 알파 − 누적 마스크 ∪ halo 2 실루엣) 평균 L*·C*.
plot   : 회차–평균 L* 곡선(알베도·렌더, P0·P1·P2).

    python tools/paper/c4_color.py sheet|metrics|plot [자산 …]   # 결과 $RUNS_DIR/paper/x3/c4/
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party" / "ai-ar-v3"), str(ROOT / "third_party" / "ai-ar-v3" / "contract" / "python")]
from tools.paper.x3 import OUT  # noqa: E402
from tools.paper.x3_score import CACHE, cum_mask, frame, mesh_spec  # noqa: E402

C4 = OUT / "c4"
PATHS = ("p0", "p1", "p2")


# ── 색 공간 ────────────────────────────────────────────────────────────────────
def srgb_to_lab(rgb):
    """sRGB [0,1] (N,3) → CIELAB (D65)."""
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def lc(lab):
    return lab[:, 0], np.hypot(lab[:, 1], lab[:, 2])


# ── 표면 점과 색 ───────────────────────────────────────────────────────────────
def sample_colored(spec, fr, n=100_000, seed=0):
    """paper_metrics.sample 과 같은 RNG 순서로 N 좌표 점 + 그 점의 sRGB 색."""
    import trimesh
    m = trimesh.load(spec["mesh"], force="mesh", process=False)
    V = fr.to_n(np.asarray(m.vertices, float), spec["kind"])
    F = np.asarray(m.faces)
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    fi = rng.choice(len(area), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[fi]
    pts = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
    w = np.stack([1 - u - v, u, v], 1)
    if spec.get("vertex_color"):
        vc = np.asarray(m.visual.vertex_colors, float)[:, :3] / 255.0
        col = (vc[F[fi]] * w[:, :, None]).sum(1)
    else:
        uv = np.asarray(m.visual.uv, float)
        uvp = (uv[F[fi]] * w[:, :, None]).sum(1)
        img = np.asarray(m.visual.material.baseColorTexture.convert("RGB"), float) / 255.0
        H, W = img.shape[:2]
        x = np.clip(np.round((uvp[:, 0] % 1.0) * (W - 1)).astype(int), 0, W - 1)
        y = np.clip(np.round((1.0 - (uvp[:, 1] % 1.0)) * (H - 1)).astype(int), 0, H - 1)   # trimesh 는 glTF v 를 뒤집어 읽는다
        fac = np.asarray(m.visual.material.baseColorFactor if m.visual.material.baseColorFactor is not None else [255] * 4, float)[:3]
        col = img[y, x] * (fac / 255.0 if fac.max() > 1 else fac)
    return pts, np.clip(col, 0, 1)


def preserved(pts, fr, asset, k):
    from tools.paper_metrics import excluded_set, preserved_mask
    return preserved_mask(pts, fr, excluded_set(cum_mask(asset, k)) if k else np.zeros(0, np.int64))


def albedo(asset, path, k, ref):
    from scipy.spatial import cKDTree
    fr = frame(asset)
    spec = mesh_spec(path, asset, k)
    pts, col = sample_colored(spec, fr)
    keep = preserved(pts, fr, asset, k)
    rp, rlab = ref
    rk = preserved(rp, fr, asset, k)
    # 방향: A_origin 바깥 표면 점 → 결과의 최근접 점 색. R4 메시는 to_glb 가 지우던 가려진 내부 면을 갖고 있어
    # 결과 → 원본 방향(면적 비례 표본)이면 어두운 내부 면이 평균에 섞인다(오두막 P1 r0 L* 23.5 대 텍스처 35.9 — 렌더 명도는 같음, 실측).
    dist, idx = cKDTree(pts[keep]).query(rp[rk])
    ok = dist < 2 * fr.cell_n                                     # 2셀 안에 결과 표면이 있는 점만(형상이 바뀐 곳 제외)
    lab = srgb_to_lab(col[keep][idx[ok]])
    rl = rlab[rk][ok]
    L, C = lc(lab)
    L0, C0 = lc(rl)
    return {"n": int(ok.sum()), "matched_frac": float(ok.mean()), "L": float(L.mean()), "C": float(C.mean()),
            "L_ref": float(L0.mean()), "dL": float((L - L0).mean()), "dC": float((C - C0).mean()),
            "dE": float(np.linalg.norm(lab - rl, axis=1).mean()), "mesh": spec["mesh"]}


def render_dir(asset, path, k):
    from tools.paper_metrics import bench_render, n_frame_glb
    fr = frame(asset)
    spec = mesh_spec(path, asset, k)
    key = Path(spec["mesh"]).resolve().as_posix() + spec["kind"] + str(spec.get("vertex_color")) + json.dumps(fr.as_dict())
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    glb = n_frame_glb(spec["mesh"], fr, spec["kind"], spec.get("vertex_color", False), CACHE / "glb" / f"{h}.glb")
    return bench_render(glb, CACHE / "renders" / h)


def render_lightness(asset, path, k):
    from PIL import Image
    from deltacontract.coords import dilate_cells
    from tools.paper_metrics import BENCH_VIEWS, HALO, silhouette
    fr = frame(asset)
    d = render_dir(asset, path, k)
    cm = cum_mask(asset, k)
    dil = np.asarray(dilate_cells(cm, HALO)) if len(cm) else None
    Ls, Cs = [], []
    for i, vw in enumerate(BENCH_VIEWS):
        a = np.asarray(Image.open(d / f"render_{i:04d}.png").convert("RGBA"), float) / 255.0
        m = a[..., 3] > 0.5
        if dil is not None:
            m &= ~silhouette(dil, fr, vw, res=a.shape[0])
        L, C = lc(srgb_to_lab(a[..., :3][m]))
        Ls.append(L)
        Cs.append(C)
    L, C = np.concatenate(Ls), np.concatenate(Cs)
    return {"L": float(L.mean()), "C": float(C.mean()), "pixels": int(L.size)}


def metrics(asset):
    fr = frame(asset)
    rp, rcol = sample_colored(mesh_spec("common", asset, 0), fr)
    ref = (rp, srgb_to_lab(rcol))
    out_p = C4 / f"{asset}_color.json"
    res = json.loads(out_p.read_text()) if out_p.exists() else {}
    for path in PATHS:
        for k in range(0, 6):
            key = f"{path}_r{k}"
            if key in res:
                continue
            if k and not (OUT / path / asset / f"r{k}" / "round.json").exists():
                continue
            r = {"albedo": albedo(asset, path, k, ref), "render": render_lightness(asset, path, k)}
            res[key] = r
            a, b = r["albedo"], r["render"]
            print(f"c4 {asset} {key} albedo L {a['L']:.1f} C {a['C']:.1f} dL {a['dL']:+.2f} dC {a['dC']:+.2f} dE {a['dE']:.2f} | render L {b['L']:.1f} C {b['C']:.1f}", flush=True)
            C4.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(res, indent=1))


def sheet(asset, rounds=(0, 1, 3, 5), views=(1, 3)):
    from PIL import Image, ImageDraw
    from tools.paper_metrics import FORK_TO_BENCH
    S = 300
    W, H = S * 3 * len(views), (S + 4) * len(rounds) + 40
    sh = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(sh)
    for j, v in enumerate(views):
        for i, p in enumerate(PATHS):
            dr.text((j * 3 * S + i * S + 6, 8), f"{p.upper()}  (fork view {v})", fill="black")
    for r, k in enumerate(rounds):
        for j, v in enumerate(views):
            for i, p in enumerate(PATHS):
                im = Image.open(render_dir(asset, p, k) / f"render_{FORK_TO_BENCH[v]:04d}.png").convert("RGBA")
                bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                bg.alpha_composite(im)
                sh.paste(bg.convert("RGB").crop((112, 112, 912, 912)).resize((S, S)), (j * 3 * S + i * S, 40 + r * (S + 4)))
        dr.text((4, 40 + r * (S + 4) + 4), f"r{k}", fill="red")
    C4.mkdir(parents=True, exist_ok=True)
    sh.save(C4 / f"{asset}_sheet.png")
    print("sheet", C4 / f"{asset}_sheet.png", flush=True)


def plot(assets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(assets), figsize=(4.2 * len(assets), 6.4), squeeze=False)
    col = {"p0": "#c0392b", "p1": "#2f6fdb", "p2": "#2e8b57"}
    for j, a in enumerate(assets):
        res = json.loads((C4 / f"{a}_color.json").read_text())
        for i, kind in enumerate(("albedo", "render")):
            ax = axes[i][j]
            for p in PATHS:
                ks = [k for k in range(6) if f"{p}_r{k}" in res]
                ax.plot(ks, [res[f"{p}_r{k}"][kind]["L"] for k in ks], "o-", color=col[p], label=p.upper())
            ax.set_title(f"{a} — {'albedo (surface color)' if kind == 'albedo' else 'Bench render'} mean L*", fontsize=10)
            ax.set_xlabel("round")
            ax.set_xticks(range(6))
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.tight_layout()
    C4.mkdir(parents=True, exist_ok=True)
    fig.savefig(C4 / "c4_L_curves.png", dpi=110)
    print("plot", C4 / "c4_L_curves.png", flush=True)


if __name__ == "__main__":
    cmd, names = sys.argv[1], sys.argv[2:] or ["jeep", "cottage"]
    if cmd == "plot":
        plot(names)
    else:
        for n in names:
            {"sheet": sheet, "metrics": metrics}[cmd](n)
