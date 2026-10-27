"""논문 캡처(chief 10-27 paper-results). 새 편집·새 채점 없음 — 기존 출력 메시를 같은 시점·같은 렌더러로 다시 찍는다.

좌표: 자산마다 v0 메시 bbox 로 정한 N 좌표(paper_metrics.AssetFrame, 1절) 하나에 모든 결과를 옮겨 찍는다
(P0 = 포크 정규화 좌표 → v0 → N, R4 = raw → v0 → N; paper_metrics.n_frame_glb). 카메라는 Bench 정의(반지름 1.8, 원점 주시).
렌더 = bpy_render.py(Bench 설정 + GPU). 흰 배경 = 투명 RGBA 를 알파로 흰색 위에 합성(가장자리 검은 테두리 없음).

    python tools/paper/capture/capture.py fronts        # 정면 찾기용 8방향 시트(고도 10°)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party" / "ai-ar-v3" / "contract" / "python")]
from tools.paper_metrics import AssetFrame, BENCH_SRC, EVAL_PY, n_frame_glb  # noqa: E402

RUNS = Path(os.environ["RUNS_DIR"])
ASSETS = Path(os.environ["ASSETS_DIR"])
CACHE = RUNS / "paper" / "capture_cache"
HERE = Path(__file__).resolve().parent


def registry():
    """자산 id → {v0, raw_c, raw_s, title}. 트랙 A·X3 = 생성 GLB(gen.json 정규화), 트랙 B·X2-B = 데이터셋 원본(raw = v0)."""
    reg = {}
    for a in ("moto", "snowman", "chair"):
        g = json.loads((ASSETS / "track_a" / a / "gen" / "gen.json").read_text())["glb_normalize"]
        reg[a] = {"v0": ASSETS / "track_a" / a / "gen" / "model.glb", "raw_c": g["center_glb_frame"], "raw_s": g["scale"], "title": a}
    for a in ("jeep", "cottage", "adventurer"):
        g = json.loads((ASSETS / "x3" / "track_a" / a / "gen" / "gen.json").read_text())["glb_normalize"]
        reg[a] = {"v0": ASSETS / "x3" / "track_a" / a / "gen" / "model.glb", "raw_c": g["center_glb_frame"], "raw_s": g["scale"], "title": a}
    tb = json.loads((ROOT / "scenarios" / "track_b.json").read_text())["scenarios"]
    for b, s in tb.items():
        reg[b] = {"v0": ASSETS / "Edit3D-Bench" / "data" / s["dataset"] / s["object"] / "source_model" / "model.glb",
                  "raw_c": [0, 0, 0], "raw_s": 1.0, "title": f"{b} {s['object'][:28]}"}
    for s in json.loads((ROOT / "scenarios" / "x2b_list.json").read_text())["scenarios"]:
        reg[s["id"]] = {"v0": ASSETS / "Edit3D-Bench" / "data" / s["dataset"] / s["object"] / "source_model" / "model.glb",
                        "raw_c": [0, 0, 0], "raw_s": 1.0, "title": f"{s['id']} {s['object'][:28]}"}
    return reg


def frame(reg, a):
    r = reg[a]
    return AssetFrame.from_v0(r["v0"], r["raw_c"], r["raw_s"])


def n_glb(reg, a, mesh, kind, vertex_color=False):
    """메시를 자산 N 좌표 GLB 로(같은 입력이면 캐시)."""
    fr = frame(reg, a)
    key = hashlib.sha256((str(Path(mesh).resolve()) + kind + str(vertex_color) + json.dumps(fr.as_dict())).encode()).hexdigest()[:20]
    return n_frame_glb(mesh, fr, kind, vertex_color, CACHE / "glb" / f"{key}.glb")


def render(jobs, tag):
    """jobs = [{"glb", "res", "views": [{"elev","az","shaded","albedo"}]}] — 투명 PNG 로 찍고 흰 배경으로 바꾼다(이미 있는 파일은 건너뜀)."""
    from PIL import Image
    todo, pairs = [], []
    for j in jobs:
        vv = []
        for v in j["views"]:
            w = {"elev": v["elev"], "az": v["az"]}
            for m in ("shaded", "albedo"):
                if v.get(m) and not Path(v[m]).exists():
                    w[m] = str(Path(v[m]).with_suffix(".rgba.png"))
                    Path(v[m]).parent.mkdir(parents=True, exist_ok=True)
                    pairs.append((w[m], v[m]))
            if len(w) > 2:
                vv.append(w)
        if vv:
            todo.append({**j, "glb": str(j["glb"]), "views": vv})
    if not todo:
        return
    jp = CACHE / "jobs" / f"{tag}.json"
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(todo))
    with open(CACHE / "jobs" / f"{tag}.log", "w") as f:
        subprocess.run([EVAL_PY, str(HERE / "bpy_render.py"), str(jp)], cwd=BENCH_SRC, stdout=f, stderr=subprocess.STDOUT, check=True)
    for src, dst in pairs:
        a = np.asarray(Image.open(src).convert("RGBA"), np.float32) / 255.0
        rgb = a[..., :3] * a[..., 3:] + (1.0 - a[..., 3:])               # 직선 알파 → 흰 배경 위 합성(= 프리멀티 합성)
        Image.fromarray(np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8), "RGB").save(dst)
        if str(dst).endswith("shaded.png"):                                # 실루엣 확인용 알파(캐시에만)
            Image.fromarray((a[..., 3] * 255 + 0.5).astype(np.uint8), "L").save(str(dst).replace(".png", "_alpha.png"))
        Path(src).unlink()


def fronts():
    """정면 찾기: 원본마다 고도 10°, Bench 방위 0·45·…·315 8장(384²) 시트."""
    from PIL import Image, ImageDraw
    reg = registry()
    out = CACHE / "fronts"
    jobs = []
    for a in reg:
        g = n_glb(reg, a, reg[a]["v0"], "v0")
        jobs.append({"glb": g, "res": 384, "views": [{"elev": 10, "az": az, "shaded": str(out / a / f"az{az:03d}.png")} for az in range(0, 360, 45)]})
    render(jobs, "fronts")
    S = 192
    sh = Image.new("RGB", (S * 8 + 150, S * len(reg) + 24), "white")
    d = ImageDraw.Draw(sh)
    for i, az in enumerate(range(0, 360, 45)):
        d.text((150 + i * S + 4, 4), f"az {az}", fill="black")
    for r, a in enumerate(reg):
        d.text((4, 24 + r * S + S // 2), reg[a]["title"][:22], fill="black")
        for i, az in enumerate(range(0, 360, 45)):
            sh.paste(Image.open(out / a / f"az{az:03d}.png").resize((S, S)), (150 + i * S, 24 + r * S))
    sh.save(CACHE / "fronts_sheet.png")
    print("fronts", CACHE / "fronts_sheet.png")


# ── 캡처 계획: 같은 메시·같은 시점은 한 번만 찍고(캐시) 결과 폴더로 복사 ─────────────────────────────────────────
CAMS = CACHE / "cameras.json"


def cams():
    return json.loads(CAMS.read_text())["assets"]


class Plan:
    def __init__(self):
        self.reg = registry()
        self.cam = cams()
        self.items = {}                                    # 캐시 키 → (glb, view, [목적 폴더])

    def add(self, asset, mesh, kind, vertex_color, dest, albedo=True):
        g = n_glb(self.reg, asset, mesh, kind, vertex_color)
        c = self.cam[asset]
        key = hashlib.sha256((str(g) + f"{c['elev']}/{c['az']}").encode()).hexdigest()[:20]
        it = self.items.setdefault(key, {"glb": g, "elev": c["elev"], "az": c["az"], "albedo": albedo, "dest": []})
        it["dest"].append(Path(dest))
        return key

    def run(self, tag):
        import shutil
        jobs = []
        for key, it in self.items.items():
            d = CACHE / "renders" / key
            v = {"elev": it["elev"], "az": it["az"], "shaded": str(d / "shaded.png")}
            if it["albedo"]:
                v["albedo"] = str(d / "albedo.png")
            jobs.append({"glb": it["glb"], "res": 1024, "views": [v]})
        render(jobs, tag)
        for key, it in self.items.items():
            d = CACHE / "renders" / key
            for dest in it["dest"]:
                dest.mkdir(parents=True, exist_ok=True)
                for m in ("shaded", "albedo") if it["albedo"] else ("shaded",):
                    shutil.copyfile(d / f"{m}.png", dest / f"{m}.png")


def view_check(out_root):
    """커밋 1: 자산마다 정해진 시점의 원본 r0 한 장 → sheets/view_check.png, cameras.json."""
    import shutil
    from PIL import Image, ImageDraw
    out_root = Path(out_root)
    pl = Plan()
    for a, r in pl.reg.items():
        pl.add(a, r["v0"], "v0", False, CACHE / "viewcheck" / a, albedo=False)
    pl.run("viewcheck")
    S = 300
    names = list(pl.reg)
    cols = 6
    rows = (len(names) + cols - 1) // cols
    sh = Image.new("RGB", (cols * S, rows * (S + 34)), "white")
    d = ImageDraw.Draw(sh)
    for i, a in enumerate(names):
        x, y = (i % cols) * S, (i // cols) * (S + 34)
        sh.paste(Image.open(CACHE / "viewcheck" / a / "shaded.png").resize((S, S), Image.LANCZOS), (x, y + 34))
        c = pl.cam[a]
        d.text((x + 4, y + 2), f"{a} - {c['class']}", fill="black")
        d.text((x + 4, y + 16), f"front {c['front_az']}°, cam az {c['az']}° el {c['elev']}°", fill="black")
    (out_root / "sheets").mkdir(parents=True, exist_ok=True)
    sh.save(out_root / "sheets" / "view_check.png")
    shutil.copyfile(CAMS, out_root / "cameras.json")
    print("view_check", out_root / "sheets" / "view_check.png")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fronts":
        fronts()
    elif cmd == "view_check":
        view_check(sys.argv[2])
