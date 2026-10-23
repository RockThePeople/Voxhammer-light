"""GLB 의 **정면 텍스처 캡쳐**를 PNG 로 굽는다 — DebugView 목록 썸네일용.

왜 정지 캡쳐인가:
    목록에 `<model-viewer>` 를 카드마다 두면 WebGL 컨텍스트를 카드 수만큼 잡는다.
    브라우저 상한이 대략 16이라 17개짜리 목록에서 오래된 것부터 화면이 죽었다
    (실측: 서버는 HTML 18ms · GLB 1.7MB 17ms 로 병목이 아니었다).
    정지 캡쳐면 컨텍스트가 0이고, 무엇이 나왔는지는 오히려 더 빨리 읽힌다.

렌더 방식:
    직교 투영 + z-버퍼. 색은 baseColorTexture 를 UV 로 찍어 오고, 여기에
    면 법선 기반 램버트 음영을 곱한다 (텍스처만 쓰면 굴곡이 안 보이고,
    음영만 쓰면 호박이 주황인지 눈이 흰지 구분이 안 된다 — 둘 다 필요하다).

시점 고르기:
    🔴 "정면"을 축 하나로 못 박으면 자산마다 엉뚱한 데를 본다 — 실측: -Z 로 고정했더니
    눈사람은 앞얼굴이 나왔는데 오토바이는 **뒤꽁무니**가 나왔다 (입력 이미지가 측면이었다).
    그래서 45도 간격 8방향 중 **실루엣 면적이 가장 큰 방향**을 고른다. 물체가 가장 크게
    펼쳐 보이는 각도이고, 자산 종류와 무관하게 결정적이다.
    track.json 의 `thumb.azimuth_deg` 로 작업별 고정도 가능하다.
"""
import argparse
import os
import sys

import numpy as np
import trimesh

# 🔴 카메라는 z=-inf 에서 +z 를 본다 (z-버퍼가 **작은 z** 를 남긴다).
#    그러니 보는 사람 쪽을 향한 면의 법선은 **-Z** 다. 빛의 z 성분도 음수여야 한다 —
#    처음에 +0.76 로 두었더니 뒷면만 밝아져서 썸네일이 통째로 어두웠다.
LIGHT = np.array([-0.35, 0.55, -0.76])          # 좌상 앞(카메라 쪽)에서 오는 빛
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def face_colors(g, fi):
    """면별 색(0..255). 텍스처 → 정점색 → baseColorFactor 순으로 떨어진다."""
    n = len(fi)
    try:
        tex = g.visual.material.baseColorTexture
        uv = g.visual.uv[g.faces[fi]].mean(1)
        img = np.asarray(tex.convert("RGB"))
        h, w, _ = img.shape
        px = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
        py = np.clip(((1 - uv[:, 1]) * h).astype(int), 0, h - 1)
        return img[py, px].astype(float)
    except Exception:                            # noqa: BLE001
        pass
    try:
        return np.asarray(g.visual.to_color().face_colors)[fi][:, :3].astype(float)
    except Exception:                            # noqa: BLE001
        pass
    try:
        f = np.asarray(g.visual.material.baseColorFactor)[:3].astype(float)
        f = f * 255 if f.max() <= 1.0 else f
        return np.repeat(f[None], n, 0)
    except Exception:                            # noqa: BLE001
        return np.full((n, 3), 190.0)


def best_azimuth(P, size=96, steps=8):
    """실루엣 면적이 최대인 방위각(라디안). 물체가 가장 크게 보이는 각도다."""
    sub = P[:: max(1, len(P) // 120_000)]
    best, ba = -1, 0.0
    for i in range(steps):
        a = 2 * np.pi * i / steps
        x = sub[:, 0] * np.cos(a) + sub[:, 2] * np.sin(a)
        y = sub[:, 1]
        span = max(x.max() - x.min(), y.max() - y.min(), 1e-6)
        u = ((x - x.min()) / span * (size - 1)).astype(int)
        v = ((y - y.min()) / span * (size - 1)).astype(int)
        area = len(set(zip(u.tolist(), v.tolist())))
        if area > best:
            best, ba = area, a
    return ba


def render(path, size=512, samples=2_500_000, azimuth=None, elevation=0.0, margin=0.06):
    g = trimesh.load(path)
    g = g.to_geometry() if hasattr(g, "to_geometry") else g
    P, fi = trimesh.sample.sample_surface(g, samples)
    if azimuth is None:
        azimuth = best_azimuth(P)
    C = face_colors(g, fi)
    N = np.asarray(g.face_normals)[fi]

    ca, sa = np.cos(azimuth), np.sin(azimuth)
    ce, se = np.cos(elevation), np.sin(elevation)

    def rot(V):
        x = V[:, 0] * ca + V[:, 2] * sa
        z = -V[:, 0] * sa + V[:, 2] * ca
        y = V[:, 1] * ce - z * se
        z = V[:, 1] * se + z * ce
        return x, y, z

    x, y, z = rot(P)
    nx, ny, nz = rot(N)
    lam = np.abs(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2])   # 법선 방향이 뒤집힌
    lam = np.clip(lam, 0, 1)                     # 면이 섞여 있어 절댓값으로 받는다
    shade = 0.62 + 0.42 * lam                    # 바탕이 흰색이라 너무 어두우면 안 읽힌다

    lo = np.array([x.min(), y.min()])
    hi = np.array([x.max(), y.max()])
    span = max((hi - lo).max(), 1e-6) * (1 + 2 * margin)
    cx, cy = (lo + hi) / 2

    u = ((x - cx) / span + 0.5) * size
    v = (0.5 - (y - cy) / span) * size
    u = u.astype(int)
    v = v.astype(int)
    k = (u >= 0) & (u < size) & (v >= 0) & (v < size)

    zbuf = np.full((size, size), np.inf)
    out = np.full((size, size, 3), 255, np.uint8)
    order = np.argsort(z[k])                     # 앞(작은 z)이 나중에 덮도록 뒤에서부터
    uu, vv, zz = u[k][order][::-1], v[k][order][::-1], z[k][order][::-1]
    cc = (C[k][order][::-1] * shade[k][order][::-1][:, None])
    for a, b, d, col in zip(vv, uu, zz, cc):
        if d <= zbuf[a, b]:
            zbuf[a, b] = d
            out[a, b] = np.clip(col, 0, 255)

    # 표본 구멍 메우기: 안 찍힌 화소인데 8-이웃 중 5개 이상이 찍혔으면 그 평균으로 채운다.
    # (안 하면 물체 위에 흰 점이 뿌려져 노이즈처럼 보인다)
    filled = np.isfinite(zbuf)
    nb = np.zeros_like(filled, dtype=np.int16)
    acc = np.zeros((size, size, 3), np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            sh = np.roll(np.roll(filled, dy, 0), dx, 1)
            nb += sh
            acc += np.roll(np.roll(out.astype(np.float32) * filled[..., None], dy, 0), dx, 1)
    hole = (~filled) & (nb >= 5)
    out[hole] = (acc[hole] / nb[hole][:, None]).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb")
    ap.add_argument("out")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--azimuth", type=float, default=None, help="라디안. 생략하면 실루엣 최대 방향을 자동 선택")
    ap.add_argument("--elevation", type=float, default=0.0)
    a = ap.parse_args()
    from PIL import Image
    img = render(a.glb, size=a.size, azimuth=a.azimuth, elevation=a.elevation)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    Image.fromarray(img).save(a.out)
    print(f"→ {a.out}  ({os.path.getsize(a.out)} B)")


if __name__ == "__main__":
    sys.exit(main())
