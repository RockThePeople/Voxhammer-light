"""X3 회차–지표 표(보고서용 markdown). 입력 $RUNS_DIR/paper/x3/score/x3_score.json(+ C4 색 지표 c4/<자산>_color.json 이 있으면 ΔL*).

    python tools/paper/x3_table.py 자산 …     # markdown 을 표준출력으로
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RUNS = Path(os.environ["RUNS_DIR"])
SC = RUNS / "paper" / "x3" / "score" / "x3_score.json"
C4 = RUNS / "paper" / "x3" / "c4"


def table(asset):
    j = json.loads(SC.read_text())[asset]
    c4p = C4 / f"{asset}_color.json"
    c4 = json.loads(c4p.read_text()) if c4p.exists() else {}
    sp = SC.parent / "surface_iou.json"
    surf = (json.loads(sp.read_text()) if sp.exists() else {}).get(asset, {})
    out = [f"**{asset}** — 자기 출발점 대비(own; P0 = 생성 텍스처 GLB, P1·P2 = A_origin R4 .cbin), 누적 마스크 ∪ halo 2 밖. "
           f"공통 기준(생성 텍스처 GLB) PSNR 을 함께 적는다. 점유·신규 셀은 A_origin SLat 대비, 마스크 밖.",
           "",
           "| 경로 | 회차 | CD | F1@0.01 | PSNR(M) | LPIPS(M) | PSNR(M) 공통 | R_keep IoU | IoU 표면 셀 | 마스크 밖 신규 셀 | DINO-I | CLIP-T | ΔL* (C4) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in ("p0", "p1", "p2"):
        for k in range(1, 6):
            o = j.get(f"{p}_r{k}_own")
            if not o:
                continue
            c = j.get(f"{p}_r{k}_common") or (o if p == "p0" else None)
            occ = o.get("occupancy_vs_origin") or {}
            im = o["image"]
            dl = c4.get(f"{p}_r{k}", {}).get("albedo", {}).get("dL")
            out.append(f"| {p.upper()} | {k} | {o['preserved']['cd']:.4f} | {o['preserved']['f1']:.3f} | {im['psnr']:.2f} | {im['lpips']:.4f} | "
                       f"{c['image']['psnr']:.2f} | {occ.get('rkeep_iou', float('nan')):.4f} | "
                       f"{surf.get(f'{p}_r{k}', {}).get('iou_surf', float('nan')):.4f} | {occ.get('rkeep_new', '–')} | "
                       f"{(im.get('dino_i') or float('nan')):.3f} | {(im.get('clip_t') or float('nan')):.3f} | "
                       f"{'–' if dl is None else f'{dl:+.1f}'} |")
    r0 = j.get("p1_r0_common")
    if r0:
        out += ["", f"0회차(P1·P2 출발점 = A_origin R4 .cbin vs 생성 텍스처 GLB): CD {r0['preserved']['cd']:.4f}, F1 {r0['preserved']['f1']:.3f}, "
                    f"PSNR(M) {r0['image']['psnr']:.2f}, LPIPS(M) {r0['image']['lpips']:.4f}. P0 0회차는 정의상 0."]
    return "\n".join(out)


if __name__ == "__main__":
    print("\n\n".join(table(a) for a in sys.argv[1:]))
