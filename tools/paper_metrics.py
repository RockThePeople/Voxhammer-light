"""논문 실험 채점 도구 (X0·X2·X3·X4 공통, ops/tasks/X-paper.md 1절). 정의는 여기와 보고서에 같은 문장으로 적는다.

좌표계 (자산마다 하나)
- **N 좌표계** = 그 자산 v0 메시의 bbox 로 정한 변환 하나: p_N = (p_v0 − bbox 중심) / bbox 최대 변. 자산의 모든 결과에 같은 변환을
  쓴다(결과마다 다시 정규화하지 않는다 — 편집으로 bbox 가 바뀌어도 보존 영역이 움직이지 않게). 이 식은 포크
  `voxhammer/bpy_render.py:normalize_scene`(1/최대 변, bbox 중심)과 같다.
- 메시 종류: `v0` = v0 가 저장된 좌표(트랙 A 생성 model.glb, 트랙 B 데이터셋 GLB) → T0 를 건다.
  `n` = 이미 N(P0 출력: 원저자 경로는 입력 GLB 를 normalize_scene 해서 렌더·인코딩하므로 출력 to_glb 가 그 좌표).
  `raw` = editcore SLat 을 디코딩한 프레임(to_glb·glbbuild 출력, 셀 중심 (i,j,k)+0.5)/64−0.5 → glTF (x, z, −y)).
  raw → v0 저장 좌표는 트랙 A (p − c)·s (gen.json glb_normalize), 트랙 B 항등(00s2 S-8). 그 뒤 T0.
- 64³ 셀은 raw 프레임 기준이다. 셀 → N 과 N 점 → 셀 은 위 식과 역식(T1 뷰어 `common.js:voxelToWorld` 와 같은 식).

지표
- 표면 샘플: 면적 비례 균일 100,000점(면 법선 포함), RNG seed 0.
- 보존 영역 점: 점이 속한 64³ 셀이 `dilate_cells(제외 셀, 2)`(E2) 밖인 점. 제외 셀이 없으면(X4) 전체.
- CD = ½·(mean_{a∈R} min_{b∈G} ‖a−b‖ + mean_{b∈G} min_{a∈R} ‖a−b‖)  (양방향 평균, 합이 아님). R = 결과 보존 점, G = 참조 보존 점.
- NC = ½·(mean_{a∈R} |n_a·n_{nn(a)}| + mean_{b∈G} |n_b·n_{nn(b)}|).
- F1@τ(τ=0.01) = 2PR/(P+R), P = mean_{a∈R}[d(a,G) ≤ τ], R = mean_{b∈G}[d(b,R) ≤ τ].
- 점유: eval/rkeep.py(E18 정의 그대로) — R_keep IoU, 소실·신규 셀. 결과 SLat coords 가 있을 때만.
- PSNR·SSIM·LPIPS(M): Edit3D-Bench `render.py` 16시점(고도 15·30 × 방위 8, 반지름 1.8, 화각 49.13°, 1024²) 로 N 좌표 메시를 렌더.
  제외 마스크 = 제외 셀(∪ halo 2)을 같은 카메라로 투영한 실루엣(셀을 정육면체로 래스터, 가림 없음).
  비교 화소 = (참조 알파 ∪ 결과 알파) − 제외 마스크. 계산은 Bench `eval_modules/image_metrics.py` 메서드 그대로
  (회색 127 배경 합성, 512² 리사이즈, 마스크 NEAREST, LPIPS alex spatial). 16시점 평균.
- DINO-I = 편집 시점 결과 렌더 vs 그 편집의 2d_edit(Bench DINOv2-base 평균 토큰 코사인). CLIP-T = 편집 시점 결과 렌더 vs 프롬프트
  (Bench CLIP ViT-B/32). 편집 시점은 포크 시점 번호 v → Bench 16시점 번호 {0:2, 1:1, 2:0, 3:7, 4:6}(고도 15°, 같은 방위).
  E2a(tools/e2a_metrics.py → Bench eval_main) 정의도 같이 낸다: clip_t_front6 = Bench 정면 6시점(0,1,7,8,9,15) 평균,
  dino_i_ref16 = 결과 16시점 vs 참조 16시점 평균. 같은 렌더에서 E2a 값과 4자리 일치(X0 A-M1 실측). 단일 시점 값은 시점 하나라 흔들림이 크다.
- 정점색 GLB(R4 `.cbin` → glbbuild)는 렌더 전에 `editcore/render.py:vertex_color_render_glb` (iv) 보정을 건다(chief 보정 2).
- 정렬 확인 값: v0 SLat 셀 중심(N)에서 v0 메시(N) 표면까지 1셀(N 단위 셀 크기) 이내 비율.

사용 (VoxHammer venv; 영상 지표는 내부에서 EVAL_PY(edit3dbench venv)를 부른다):
    python tools/paper_metrics.py score job.json out.json
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "third_party" / "ai-ar-v3" / "contract" / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

N_SAMPLES = 100_000
SEED = 0
F1_TAU = 0.01
HALO = 2
BENCH_SRC = ROOT / "third_party" / "VoxHammer" / "Edit3D-Bench"
EVAL_PY = os.environ.get("EVAL_PY", str(Path.home() / "vhl-venvs/edit3dbench/bin/python"))
FORK_TO_BENCH = {0: 2, 1: 1, 2: 0, 3: 7, 4: 6}
CAM_R, FOV_T = 1.8, 16.0 / 35.0          # tan(fov/2), 센서 32 mm · 렌즈 35 mm
BENCH_VIEWS = [(e, a) for e in (15, 30) for a in (0, 45, 90, 135, 180, 225, 270, 315)]
FORK_VIEWS = [(15, a) for a in (90, 45, 0, 315, 270)]


# ── 좌표 ────────────────────────────────────────────────────────────────────────
def voxel_to_glb(v):
    v = np.asarray(v, float)
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)


def glb_to_voxel(g):
    g = np.asarray(g, float)
    return np.stack([g[:, 0], -g[:, 2], g[:, 1]], 1)


def load_mesh(path):
    import trimesh
    m = trimesh.load(str(path), force="mesh", process=False)
    return trimesh.Trimesh(np.asarray(m.vertices, float), np.asarray(m.faces), process=False)


@dataclass
class AssetFrame:
    """자산 하나의 좌표 변환. raw_c·raw_s: raw → v0 저장 좌표 (p − c)·s. t0_c·t0_s: v0 저장 → N (p − c)·s."""
    raw_c: np.ndarray = field(default_factory=lambda: np.zeros(3))
    raw_s: float = 1.0
    t0_c: np.ndarray = field(default_factory=lambda: np.zeros(3))
    t0_s: float = 1.0

    @classmethod
    def from_v0(cls, v0_mesh, raw_c=(0, 0, 0), raw_s=1.0):
        m = load_mesh(v0_mesh) if not hasattr(v0_mesh, "vertices") else v0_mesh
        lo, hi = m.vertices.min(0), m.vertices.max(0)
        return cls(np.asarray(raw_c, float), float(raw_s), (lo + hi) / 2, 1.0 / float((hi - lo).max()))

    def to_n(self, p, kind):
        p = np.asarray(p, float)
        if kind == "n":
            return p
        if kind == "raw":
            p = (p - self.raw_c) * self.raw_s
        elif kind != "v0":
            raise ValueError(kind)
        return (p - self.t0_c) * self.t0_s

    def n_to_raw(self, p):
        return (np.asarray(p, float) / self.t0_s + self.t0_c) / self.raw_s + self.raw_c

    @property
    def cell_n(self):
        return self.raw_s * self.t0_s / 64.0

    def cells_to_n(self, cells):
        return self.to_n(voxel_to_glb((np.asarray(cells, float) + 0.5) / 64.0 - 0.5), "raw")

    def points_to_cells(self, p_n):
        v = glb_to_voxel(self.n_to_raw(p_n))
        return np.floor((v + 0.5) * 64.0).astype(np.int64)

    def as_dict(self):
        return {"raw_c": self.raw_c.tolist(), "raw_s": self.raw_s, "t0_c": self.t0_c.tolist(), "t0_s": self.t0_s}


def mesh_in_n(path, frame: AssetFrame, kind):
    import trimesh
    m = load_mesh(path)
    return trimesh.Trimesh(frame.to_n(m.vertices, kind), m.faces, process=False)


# ── 표면 지표 ───────────────────────────────────────────────────────────────────
def sample(mesh, n=N_SAMPLES, seed=SEED):
    """면적 비례 균일 샘플(고정 RNG)과 면 법선."""
    tri = mesh.vertices[mesh.faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    fi = rng.choice(len(area), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[fi]
    pts = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
    nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-20)
    return pts, nrm


def excluded_set(cells, halo=HALO):
    """E2: dilate_cells(제외 셀, halo) 의 정수 코드 집합."""
    if cells is None or len(cells) == 0:
        return np.zeros(0, np.int64)
    from deltacontract.coords import dilate_cells
    d = np.asarray(dilate_cells(np.asarray(cells, np.int64).reshape(-1, 3), halo), np.int64)
    return np.unique(d[:, 0] * 4096 + d[:, 1] * 64 + d[:, 2])


def preserved_mask(pts_n, frame, excl_codes):
    if excl_codes.size == 0:
        return np.ones(len(pts_n), bool)
    c = frame.points_to_cells(pts_n)
    code = c[:, 0] * 4096 + c[:, 1] * 64 + c[:, 2]
    inside = (c >= 0).all(1) & (c < 64).all(1)
    return ~(inside & np.isin(code, excl_codes))


def surface_metrics(res_pts, res_n, ref_pts, ref_n, tau=F1_TAU):
    from scipy.spatial import cKDTree
    if len(res_pts) == 0 or len(ref_pts) == 0:
        return {"cd": None, "nc": None, "f1": None, "precision": None, "recall": None, "n_res": len(res_pts), "n_ref": len(ref_pts)}
    d_rg, i_rg = cKDTree(ref_pts).query(res_pts)
    d_gr, i_gr = cKDTree(res_pts).query(ref_pts)
    nc = 0.5 * (np.abs((res_n * ref_n[i_rg]).sum(1)).mean() + np.abs((ref_n * res_n[i_gr]).sum(1)).mean())
    p, r = float((d_rg <= tau).mean()), float((d_gr <= tau).mean())
    return {"cd": float(0.5 * (d_rg.mean() + d_gr.mean())), "nc": float(nc), "f1": float(2 * p * r / (p + r)) if p + r > 0 else 0.0,
            "precision": p, "recall": r, "n_res": int(len(res_pts)), "n_ref": int(len(ref_pts))}


# ── 보이는 표면(chief 00:5x 2): R4 .cbin 의 가려진 내부 면이 전체 표면 CD·F1 을 불리하게 만든다 → Bench 16시점에서 보이는 점만 ──
def view_rays(view, res):
    """Bench/포크 카메라(project 와 같은 정의)의 화소 중심 광선. 반환 (원점 glTF, 방향 glTF (res², 3))."""
    el, az = math.radians(view[0]), math.radians(view[1] - 90)
    cam = CAM_R * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    f = -cam / np.linalg.norm(cam)
    r = np.cross(f, [0, 0, 1]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    px = (np.arange(res) + 0.5) / res * 2 - 1
    X, Y = np.meshgrid(px, px)                                   # Y: 화면 아래쪽이 +
    d = f[None] + (X.ravel()[:, None] * FOV_T) * r[None] - (Y.ravel()[:, None] * FOV_T) * u[None]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    to_glb = lambda b: np.stack([b[..., 0], b[..., 2], -b[..., 1]], -1)   # noqa: E731  Blender (x, −z, y) 의 역
    return to_glb(cam), to_glb(d)


def visible_points(mesh, n=N_SAMPLES, res=256, seed=SEED, views=None):
    """mesh(N 좌표) 를 Bench 16시점에서 레이캐스팅한 첫 교차점과 면 법선, 합쳐서 n 점(seed 고정)으로 고른다."""
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)), o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
    P, N = [], []
    for v in views or BENCH_VIEWS:
        o, d = view_rays(v, res)
        rays = np.concatenate([np.broadcast_to(o, d.shape), d], 1).astype(np.float32)
        h = sc.cast_rays(o3d.core.Tensor(rays))
        t = h["t_hit"].numpy()
        ok = np.isfinite(t)
        P.append(o[None] + t[ok, None] * d[ok])
        N.append(h["primitive_normals"].numpy()[ok])
    P, N = np.concatenate(P), np.concatenate(N)
    if len(P) > n:
        idx = np.random.default_rng(seed).choice(len(P), n, replace=False)
        P, N = P[idx], N[idx]
    return P, N


def visible_metrics(job, excl_cells=None):
    """job(score 와 같은 형식)의 보존 영역 보이는 표면 CD·NC·F1. excl_cells 를 주면 그 영역(예: 고정 영역)으로."""
    a = job["asset"]
    frame = AssetFrame.from_v0(a["v0_mesh"], a.get("raw_c", (0, 0, 0)), a.get("raw_s", 1.0))
    rp, rn = visible_points(mesh_in_n(job["ref"]["mesh"], frame, job["ref"]["kind"]))
    qp, qn = visible_points(mesh_in_n(job["res"]["mesh"], frame, job["res"]["kind"]))
    excl = excluded_set(job.get("excl_cells") if excl_cells is None else excl_cells)
    rk, qk = preserved_mask(rp, frame, excl), preserved_mask(qp, frame, excl)
    return {**surface_metrics(qp[qk], qn[qk], rp[rk], rn[rk]), "visible_points": {"ref": int(len(rp)), "res": int(len(qp))}}


def alignment_ratio(cells, mesh_n, frame, n=2_000_000):
    """정렬 확인: SLat 셀 중심(N) → 메시 표면(N, 균일 n점 근사) 거리가 1셀 이내인 비율."""
    from scipy.spatial import cKDTree
    pts, _ = sample(mesh_n, n=n, seed=SEED)
    d, _ = cKDTree(pts).query(frame.cells_to_n(cells))
    return float((d <= frame.cell_n).mean())


# ── 카메라·실루엣 ─────────────────────────────────────────────────────────────
def cells_boundary_mesh(cells):
    """셀 합집합의 노출면(닫힌 표면), 복셀 프레임 [-0.5, 0.5]. origin: scenarios/s5_masks.py:boundary_mesh (00s2 동결 마스크 GLB)."""
    occ = {tuple(int(x) for x in c) for c in np.asarray(cells).reshape(-1, 3)}
    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    verts, faces = [], []
    for (i, j, k) in occ:
        for d in dirs:
            if (i + d[0], j + d[1], k + d[2]) in occ:
                continue
            ax = int(np.nonzero(d)[0][0])
            o = [i, j, k]
            o[ax] += 1 if d[ax] > 0 else 0
            u, v = [x for x in range(3) if x != ax]
            q = []
            for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                pp = list(o); pp[u] += du; pp[v] += dv
                q.append(pp)
            if d[ax] < 0:
                q = q[::-1]
            if ax == 1:
                q = q[::-1]
            b = len(verts)
            verts.extend(q)
            faces.extend([(b, b + 1, b + 2), (b, b + 2, b + 3)])
    return np.asarray(verts, float) / 64.0 - 0.5, np.asarray(faces, np.int64)


def project(p_glb, elev_deg, az_deg, res):
    """glTF(Y-up) 점 → 화면 (x, y). Blender (x, −z, y), 카메라 반지름 1.8, 방위 az − 90, 원점 주시."""
    b = np.stack([p_glb[:, 0], -p_glb[:, 2], p_glb[:, 1]], 1)
    el, az = math.radians(elev_deg), math.radians(az_deg - 90)
    cam = CAM_R * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    f = -cam / np.linalg.norm(cam)
    r = np.cross(f, [0, 0, 1]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    d = b - cam
    z = d @ f
    return np.stack([(d @ r / z / FOV_T * 0.5 + 0.5) * res, (0.5 - d @ u / z / FOV_T * 0.5) * res], 1)


def silhouette(cells, frame, view, res=1024, kind_frame="n"):
    """셀(정육면체 합집합의 노출면)을 카메라 view=(고도, 방위)로 래스터한 bool 실루엣. kind_frame: 'n' 이면 N 좌표, 'v0' 이면 v0 저장 좌표."""
    from PIL import Image, ImageDraw
    if cells is None or len(cells) == 0:
        return np.zeros((res, res), bool)
    v, f = cells_boundary_mesh(np.asarray(cells))
    g = voxel_to_glb(v)
    g = frame.to_n(g, "raw") if kind_frame == "n" else (g - frame.raw_c) * frame.raw_s
    xy = project(g, view[0], view[1], res)
    im = Image.new("1", (res, res), 0)
    dr = ImageDraw.Draw(im)
    for t in f:
        dr.polygon([tuple(xy[i]) for i in t], fill=1)
    return np.asarray(im, bool)


# ── 렌더 (Bench render.py, N 좌표 메시) ─────────────────────────────────────────
def n_frame_glb(src, frame, kind, vertex_color, dst):
    """N 좌표 GLB(장면 구조·재질 유지) → 필요하면 (iv) 보정. 같은 입력이면 다시 만들지 않는다."""
    import trimesh
    dst = Path(dst)
    if dst.is_file():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    sc = trimesh.load(str(src))
    M = np.eye(4)
    if kind == "raw":
        M = np.diag([frame.raw_s] * 3 + [1.0]) @ np.block([[np.eye(3), -frame.raw_c[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])
    if kind in ("raw", "v0"):
        T = np.diag([frame.t0_s] * 3 + [1.0]) @ np.block([[np.eye(3), -frame.t0_c[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])
        M = T @ M
    sc.apply_transform(M)
    tmp = dst.with_suffix(".tmp.glb")
    sc.export(str(tmp))
    if vertex_color:
        from editcore.render import vertex_color_render_glb
        dst.write_bytes(vertex_color_render_glb(tmp.read_bytes()))
        tmp.unlink()
    else:
        tmp.replace(dst)
    return dst


def bench_render(glb, out_dir):
    out_dir = Path(out_dir)
    if (out_dir / "render_0015.png").is_file():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([EVAL_PY, "-c", "import sys; sys.path.insert(0,'.'); import render; render.render_images(sys.argv[1], sys.argv[2])",
                    str(glb), str(out_dir)], cwd=BENCH_SRC, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_dir


# ── 영상 지표 (EVAL_PY 안에서 실행: `paper_metrics.py image spec.json out.json`) ─────────────
def _image_metrics_main(spec_path, out_path):
    import torch
    from PIL import Image
    from torchvision import transforms
    sys.path.insert(0, str(BENCH_SRC))
    from eval_modules.image_metrics import ImageMetricsEvaluator
    from eval_modules.text_alignment_metrics import TextAlignmentMetricsEvaluator
    spec = json.loads(Path(spec_path).read_text())
    dev = "cuda:0" if torch.cuda.is_available() and spec.get("device", os.environ.get("PAPER_IMG_DEVICE", "cpu")) == "cuda" else "cpu"
    ime = ImageMetricsEvaluator(device=dev)
    to_t = transforms.ToTensor()

    def prep(p):
        im = Image.open(p)
        a = np.asarray(im.convert("RGBA"))[..., 3] > 0
        if im.mode != "RGB":
            bg = Image.new("RGBA", im.size, (127, 127, 127, 255))
            bg.paste(im, (0, 0), im)
            im = bg.convert("RGB")
        return im.resize((512, 512), Image.BILINEAR), a

    per = []
    for v, (rp, qp, ex) in enumerate(zip(spec["ref_views"], spec["res_views"], spec["excl_masks"])):
        ri, ra = prep(rp)
        qi, qa = prep(qp)
        excl = np.asarray(Image.open(ex).convert("L")) > 127
        inc = (ra | qa) & ~excl
        inc = np.asarray(Image.fromarray((inc * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST)) > 127
        m = torch.from_numpy(inc.astype(np.float32))
        a, b = to_t(ri).unsqueeze(0), to_t(qi).unsqueeze(0)
        psnr = ime._calculate_psnr_torch(a[0], b[0], m)
        ssim = ime._calculate_ssim_torch(a, b, m)
        # Bench 와 같게 [0,1] 텐서를 그대로 넣는다(eval_modules/image_metrics.py 의 LPIPS 호출과 같은 전처리)
        lp = ime._calculate_lpips_masked(a.to(ime.device), b.to(ime.device), m.to(ime.device)[None, None])
        per.append({"view": v, "psnr": psnr, "ssim": ssim, "lpips": lp, "pixels": int(inc.sum())})
    out = {"per_view": per}
    for k in ("psnr", "ssim", "lpips"):
        vals = [x[k] for x in per if x[k] is not None]
        out[k] = float(np.mean(vals)) if vals else None
    if spec.get("edit_render") and spec.get("edit_image"):
        e = Image.open(spec["edit_render"])
        if e.mode != "RGB":
            bg = Image.new("RGBA", e.size, (127, 127, 127, 255)); bg.paste(e, (0, 0), e); e = bg.convert("RGB")
        e = e.resize((512, 512), Image.BILINEAR)
        cond = Image.open(spec["edit_image"]).convert("RGB").resize((512, 512), Image.BILINEAR)
        out["dino_i"] = float(ime._calculate_dino_i([e], [cond])[0])
        if spec.get("prompt"):
            te = TextAlignmentMetricsEvaluator(device=dev)
            out["clip_t"] = te._compute_clip_similarity(e, spec["prompt"])
            from eval_modules.data_loader import ImageTextPair
            front = [spec["res_views"][i] for i in (0, 1, 7, 8, 9, 15)]       # Bench data_loader 정면 −45~45°
            out["clip_t_front6"] = te.compute_clip_t([ImageTextPair(image_path=q, text=spec["prompt"], id=str(i))
                                                      for i, q in enumerate(front)])["mean"]
    if spec.get("ref_views") and spec.get("res_views"):
        def gray(p):
            e = Image.open(p)
            bg = Image.new("RGBA", e.size, (127, 127, 127, 255)); bg.paste(e, (0, 0), e)
            return bg.convert("RGB").resize((512, 512), Image.BILINEAR)
        out["dino_i_ref16"] = float(np.mean(ime._calculate_dino_i([gray(q) for q in spec["res_views"]],
                                                                  [gray(r) for r in spec["ref_views"]])))
    Path(out_path).write_text(json.dumps(out, indent=1))


# ── 채점 1건 ────────────────────────────────────────────────────────────────────
def score(job: dict) -> dict:
    """job: asset{v0_mesh, raw_c, raw_s, v0_slat?}, ref{mesh, kind, vertex_color}, res{mesh, kind, vertex_color, slat?},
    excl_cells[], rkeep{parent_slat, mask_cells}?, edit_view?, edit_image?, prompt?, work_dir, images(bool)."""
    from safetensors import safe_open
    a = job["asset"]
    frame = AssetFrame.from_v0(a["v0_mesh"], a.get("raw_c", (0, 0, 0)), a.get("raw_s", 1.0))
    work = Path(job["work_dir"]); work.mkdir(parents=True, exist_ok=True)
    ref = mesh_in_n(job["ref"]["mesh"], frame, job["ref"]["kind"])
    res = mesh_in_n(job["res"]["mesh"], frame, job["res"]["kind"])
    rp, rn = sample(ref); qp, qn = sample(res)
    excl = excluded_set(job.get("excl_cells"))
    rk, qk = preserved_mask(rp, frame, excl), preserved_mask(qp, frame, excl)
    out = {"frame": frame.as_dict(), "preserved": surface_metrics(qp[qk], qn[qk], rp[rk], rn[rk]),
           "whole": surface_metrics(qp, qn, rp, rn) if job.get("whole", False) else None,
           "preserved_point_ratio": {"ref": float(rk.mean()), "res": float(qk.mean())}}
    coords = lambda p: safe_open(str(p), "np").get_tensor("coords")
    if a.get("v0_slat"):
        out["alignment_v0"] = alignment_ratio(coords(a["v0_slat"])[:, 1:], mesh_in_n(a["v0_mesh"], frame, "v0"), frame)
    if job.get("res", {}).get("slat") and job.get("rkeep"):
        from eval.rkeep import rkeep_metrics
        out["occupancy"] = rkeep_metrics(coords(job["rkeep"]["parent_slat"]), coords(job["res"]["slat"]),
                                         np.asarray(job["rkeep"]["mask_cells"], np.int64).reshape(-1, 3))
    if job.get("images", True):
        cache = Path(job.get("cache_dir") or os.environ.get("PAPER_CACHE") or work)   # 같은 메시·변환이면 렌더를 다시 쓴다

        def views(spec, tag):
            key = Path(spec["mesh"]).resolve().as_posix() + spec["kind"] + str(spec.get("vertex_color")) + json.dumps(frame.as_dict())
            h = hashlib.sha256(key.encode()).hexdigest()[:16]
            g = n_frame_glb(spec["mesh"], frame, spec["kind"], spec.get("vertex_color", False), cache / "glb" / f"{h}.glb")
            d = bench_render(g, cache / "renders" / h)
            return [str(d / f"render_{i:04d}.png") for i in range(16)]
        rv, qv = views(job["ref"], "ref"), views(job["res"], "res")
        from PIL import Image
        mdir = work / "excl"; mdir.mkdir(exist_ok=True)
        masks = []
        dil = None
        if len(excl):
            from deltacontract.coords import dilate_cells
            dil = np.asarray(dilate_cells(np.asarray(job["excl_cells"], np.int64).reshape(-1, 3), HALO))
        for i, vw in enumerate(BENCH_VIEWS):
            s = silhouette(dil, frame, vw) if dil is not None else np.zeros((1024, 1024), bool)
            p = mdir / f"excl_{i:04d}.png"
            Image.fromarray((s * 255).astype(np.uint8)).save(p)
            masks.append(str(p))
        spec = {"ref_views": rv, "res_views": qv, "excl_masks": masks}
        if job.get("edit_view") is not None:
            spec["edit_render"] = qv[FORK_TO_BENCH[int(job["edit_view"])]]
            spec["edit_image"], spec["prompt"] = job.get("edit_image"), job.get("prompt")
        sp, op = work / "image_spec.json", work / "image_out.json"
        sp.write_text(json.dumps(spec))
        subprocess.run([EVAL_PY, str(Path(__file__).resolve()), "image", str(sp), str(op)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out["image"] = json.loads(op.read_text())
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "image":
        _image_metrics_main(sys.argv[2], sys.argv[3])
    elif cmd == "score":
        Path(sys.argv[3]).write_text(json.dumps(score(json.loads(Path(sys.argv[2]).read_text())), indent=1))
    else:
        raise SystemExit(f"모르는 명령: {cmd}")
