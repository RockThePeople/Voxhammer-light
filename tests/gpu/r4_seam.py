"""2번 R4 3-8 이음새: 경계 스냅 전·후 측정(14건 r1) + 렌더 시트(B8·B6·B2). 계측 구간 밖, 한 프로세스.

저장된 r4-on r1 slat_tgt 를 mesh 만 다시 디코딩 → 정정 규칙(부기 ∪ 점유 변경 ∪ 신규)으로 조립하며 `write_cbin(snap=True)` 가
내는 measure_seams 전·후를 기록한다. 결과는 `<회차>/cbin_snap/`, `output_snap.glb`.
시트: 같은 slat_tgt 의 (스냅 없음 = cbin_r11) | (스냅) 를 포크 Cycles 로 렌더해 편집 시점 틈 화소(스냅 없음이 40 이상 어두운 화소)를 센다.
결과: $RUNS_DIR/02-r4/metrics/seam.json, docs/reports/img/02-R4-seam-snap.png
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tests" / "gpu"), str(REPO / "third_party" / "ai-ar-v3")]
import editcore.models as em  # noqa: E402

RUNS = Path(os.environ["RUNS_DIR"])


def main():
    import torch
    import trellis.modules.sparse as sp
    from safetensors import safe_open
    from editcore.export import decode_to_cbin
    from server.glbbuild import build_glb
    from r4_checks import scenarios
    from tools.r4_metrics import scenarios as mscen
    masks = {m["id"]: m["mask"] for m in mscen()}
    from r4_quality import prepared, PY, REF, RENDER
    from PIL import Image, ImageDraw
    pipe = em.load_pipeline()
    res = {}
    S = {s["id"]: s for s in scenarios()}
    for s in scenarios():
        d = RUNS / "02-r4" / s["name"] / "r4-on" / "r1"
        with safe_open(str(d / "slat_tgt.safetensors"), "pt") as f:
            slat = sp.SparseTensor(feats=f.get_tensor("feats").float().cuda(), coords=f.get_tensor("coords").int().cuda())
        with safe_open(str(s["slat"]), "np") as f:
            pc = f.get_tensor("coords")[:, 1:]
        mask = json.loads(Path(masks[s["id"]]).read_text())["mask"]["voxels"]
        w = decode_to_cbin(pipe, slat, d / "cbin_snap", mask_cells=mask, parent_dir=s["parent"], parent_cells=pc,
                           glb_path=d / "output_snap.glb")
        res[s["id"]] = w["seam"]
        b, a = w["seam"]["before"], w["seam"]["after"]
        print(s["id"], "faces", b["seam_faces"], "max %.2f→%.2f median %.3f→%.3f over_cell %d→%d snapped %d" % (
            b["cells_max"] or 0, a["cells_max"] or 0, b["cells_median"] or 0, a["cells_median"] or 0,
            b["over_cell"], a["over_cell"], w["seam"]["snapped"]), flush=True)
        del slat
        torch.cuda.empty_cache()   # 검증 스크립트(계측 구간 밖)
    rows = []
    for sid, box in (("B8", (360, 200, 720, 880)), ("B6", (420, 230, 640, 450)), ("B2", (250, 600, 710, 800))):
        s = S[sid]
        v = s["edit_view"]
        q = RUNS / "02-r4" / "checks" / "seam" / sid
        imgs = {}
        for tag, glb in (("nosnap", RUNS / "02-r4" / s["name"] / "r4-on" / "r1" / "output_r11.glb"),
                         ("snap", RUNS / "02-r4" / s["name"] / "r4-on" / "r1" / "output_snap.glb")):
            (q / tag).mkdir(parents=True, exist_ok=True)
            if not (q / tag / f"render_{v:04d}.png").exists():
                g = prepared(s, glb, "iv", q / tag / "render_input.glb")
                subprocess.run([PY, "-c", RENDER, str(REF), str(g), str(q / tag)], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            imgs[tag] = np.asarray(Image.open(q / tag / f"render_{v:04d}.png").convert("RGBA")).astype(int)
        # 틈 화소: 승계 없는 새 디코딩(02-R4-seam 시트 기준)보다 40 이상 어두운 화소 — 여기서는 스냅 전·후 각각을 승계 없음과 비교
        fresh = np.asarray(Image.open(f"/tmp/claude-1000/-home-dweb-voxhammer-light/a7791ca2-943c-4fea-b74c-e86bdeb4f329/scratchpad/seam/{sid}/fresh_render/render_{v:04d}.png").convert("RGBA")).astype(int) \
            if Path(f"/tmp/claude-1000/-home-dweb-voxhammer-light/a7791ca2-943c-4fea-b74c-e86bdeb4f329/scratchpad/seam/{sid}/fresh_render/render_{v:04d}.png").exists() else None
        cnt = {}
        for tag, im in imgs.items():
            if fresh is not None:
                obj = (im[..., 3] > 0) & (fresh[..., 3] > 0)
                cnt[tag] = int((obj & (fresh[..., :3].mean(-1) - im[..., :3].mean(-1) > 40)).sum())
        res[sid]["crack_px_vs_no_inherit"] = cnt
        print(sid, "crack px", cnt, flush=True)
        tiles = []
        for tag in ("nosnap", "snap"):
            im = Image.fromarray(imgs[tag].astype(np.uint8))
            t = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im).convert("RGB").crop(box)
            sc = 360 / max(t.size)
            t = t.resize((int(t.width * sc), int(t.height * sc)), Image.LANCZOS)
            ImageDraw.Draw(t).text((4, 4), f"{sid} {'no snap' if tag == 'nosnap' else 'boundary snap'} crack px {cnt.get(tag)}", fill=(200, 0, 0))
            tiles.append(t)
        r = Image.new("RGB", (tiles[0].width * 2 + 8, tiles[0].height), "white")
        r.paste(tiles[0], (0, 0)); r.paste(tiles[1], (tiles[0].width + 8, 0))
        rows.append(r)
    W = max(r.width for r in rows)
    sh = Image.new("RGB", (W, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        sh.paste(r, (0, y)); y += r.height
    sh.save(REPO / "docs" / "reports" / "img" / "02-R4-seam-snap.png", optimize=True)
    (RUNS / "02-r4" / "metrics" / "seam.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
