"""/debug — 사람이 눈으로 보는 읽기 전용 관측 화면.

lab/api.py 가 JSON 을 낸다면 여기는 **그 JSON 을 눈으로 판정할 수 있게** 만든다.
새 프레임워크를 들이지 않는다 — FastAPI + 인라인 HTML 뿐이고, 파일 서빙은
이미 있는 `/lab/runs/{id}/files/{path}` 를 그대로 쓴다(경로 탈출 검증이 거기
한 곳에만 있게 유지한다).

이 화면이 존재하는 이유는 하나다 — **항진명제를 눈으로 잡기 위해서다.**
지난번 "마스크 안 신규 0" 은 마스크 전체가 점유라 신규가 정의상 불가능했던
경우였다. 모델이 무엇을 하든 0 이 나온다. 그래서 이 화면은 어떤 수치를 낼 때
**그 수치가 나올 수 있었던 모집단(빈 공간)** 을 항상 옆에 같이 띄운다.

⚠️ 읽기 전용이다. lab 라우터가 인증 없음을 전제로 하므로(main.py 주석) 여기에
   쓰기·삭제·실행 엔드포인트를 추가하면 그 전제가 깨진다.
"""
import html
import json
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from .store import LAB_ROOT, RUN_ID_RE, list_run_ids, read_meta, run_dir

router = APIRouter()

# VoxHammer 단독 트랙의 산출물 루트. lab run 형식이 아니라 그냥 디렉터리다
# (이 트랙엔 청크·델타·계약이 없다). 홈 경로를 코드에 박지 않으려고 LAB_ROOT 에서 파생시킨다.
TRACK_ROOT = os.environ.get(
    "VH_TRACK_ROOT", os.path.join(os.path.dirname(LAB_ROOT), "ai-ar-data", "vh-track"))
TRACK_NAME_OK = __import__("re").compile(r"^[A-Za-z0-9_.-]{1,64}$")

# 클라이언트 3D 뷰어. **로컬에 받아 둔다** — CDN 을 물면 그 기계 브라우저가 외부망을
# 못 쓸 때 화면이 죽는다. 서버 의존성은 늘지 않는다(그냥 정적 파일 하나다).
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
STATIC_OK = {"model-viewer.min.js"}

VIEWS = ["front", "left", "back", "top", "mask"]

# 관문. stage_detail 이 이 순서로 올라간다(store.STAGE_MAP). 실패한 런에서
# "어디까지 갔는가" 를 이 목록의 인덱스로 판정한다.
GEN_GATES = [
    ("image_load", "이미지 로드"),
    ("model_load", "모델 로드"),
    ("sparse_structure", "sparse structure"),
    ("slat_sample", "slat 샘플링"),
    ("slat_save", "slat 저장"),
    ("decode", "디코딩"),
    ("chunk", "청크 분할"),
    ("render", "렌더"),
]
EDIT_GATES = [
    ("mask", "마스크"),
    ("model_load", "모델 로드"),
    ("repaint", "RePaint (편집)"),
    ("decode", "디코딩"),
    ("chunk", "청크·부기 합성"),
    ("render", "렌더"),
]

_space_cache = {}


# ─────────────────────────────────────────────────────────── 계산
def mask_space(run_id, meta):
    """마스크의 **빈 공간 비율**. meta 에 없는 수라 여기서 실제로 계산한다.

    meta 의 `after_halo` 는 점유 복셀만 센다(pipeline.py 에서 `& active` 로
    걸러진다). 그래서 그 값만 보면 마스크가 꽉 차 있는지 비어 있는지 알 수 없고,
    바로 그것 때문에 "신규 0" 이 항진명제인 줄 몰랐다.

    여기서는 mask.json 의 core 복셀을 halo 만큼 실제로 팽창시켜 **부피**를 얻고,
    부모 slat 의 활성 복셀과 교집합을 내어 점유/빈칸을 가른다.

    실패해도 화면을 죽이지 않는다 — None 을 돌려 "미계산" 으로 표시한다.
    """
    if run_id in _space_cache:
        return _space_cache[run_id]
    out = None
    try:
        import numpy as np
        from safetensors import safe_open

        from deltacontract import dilate_cells, voxel_code

        d = run_dir(run_id)
        with open(os.path.join(d, "mask.json")) as f:
            mj = json.load(f)
        core = np.array(mj["spec"]["voxels"], dtype=np.int64)
        halo = int(mj.get("halo", 0))
        parent = meta.get("parent_run_id")
        sp = os.path.join(run_dir(parent), "slat.safetensors") if parent else None
        if not (sp and os.path.exists(sp)):
            raise FileNotFoundError("부모 slat 없음")
        with safe_open(sp, "pt") as f:
            xyz = f.get_tensor("coords").numpy()[:, 1:].astype(np.int64)

        cells = dilate_cells(core, halo) if halo else core
        active = set(voxel_code(xyz).tolist())
        vol = set(voxel_code(cells).tolist())
        occ = vol & active
        empty = len(vol) - len(occ)
        out = {
            "core": int(core.shape[0]),
            "volume": len(vol),
            "occupied": len(occ),
            "empty": empty,
            "empty_ratio": (empty / len(vol)) if vol else None,
            "halo": halo,
        }
    except Exception as e:                      # noqa: BLE001 — 화면은 절대 안 죽인다
        out = {"error": f"{type(e).__name__}: {e}"}
    _space_cache[run_id] = out
    return out


def gates_of(meta):
    """(라벨, 상태) 목록. 상태 = passed | died | pending | unknown."""
    kind = meta.get("kind")
    gates = EDIT_GATES if kind == "edit" else GEN_GATES
    keys = [k for k, _ in gates]
    detail = meta.get("stage_detail")
    state = meta.get("state")

    if state == "succeeded" or detail == "done":
        return [(lbl, "passed") for _, lbl in gates], None
    if detail not in keys:
        # T1_coord_transform_only 같은 계약 밖 단계. 지어내지 않는다.
        return [(lbl, "unknown") for _, lbl in gates], detail

    i = keys.index(detail)
    rows, died = [], None
    for j, (_, lbl) in enumerate(gates):
        if j < i:
            rows.append((lbl, "passed"))
        elif j == i:
            if state == "failed":
                rows.append((lbl, "died"))
                died = lbl
            else:
                rows.append((lbl, "running"))
        else:
            rows.append((lbl, "pending"))
    return rows, died


def list_files(run_id):
    root = run_dir(run_id)
    out = []
    for base, _dirs, files in os.walk(root):
        for fn in sorted(files):
            p = os.path.join(base, fn)
            rel = os.path.relpath(p, root)
            try:
                out.append((rel, os.path.getsize(p)))
            except OSError:
                pass
    return sorted(out)


def tail(run_id, name="log.txt", n=4000):
    p = os.path.join(run_dir(run_id), name)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, errors="replace") as f:
            t = f.read()
        return t[-n:]
    except OSError:
        return None


# ─────────────────────────────────────────────────────────── HTML 조각
# 🔴 3D 뷰어 바탕만 흰색 + 격자. 페이지 테마는 어두운 그대로 둔다.
#   산출물 GLB 가 대부분 **밝은 회색**이라 검은 바탕에서는 실루엣만 보이고 표면 굴곡이
#   안 읽혔다. 흰 바탕이면 굴곡이 보이고, 격자는 눈금 노릇을 해 크기·이동을 가늠하게 한다.
VIEWER_BG = ("background-color:#ffffff;"
             "background-image:linear-gradient(#dde3ea 1px,transparent 1px),"
             "linear-gradient(90deg,#dde3ea 1px,transparent 1px);"
             "background-size:24px 24px")

CSS = """
*{box-sizing:border-box}
body{margin:0;background:#0e1013;color:#d7dbe0;
     font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}
a{color:#6cb6ff;text-decoration:none} a:hover{text-decoration:underline}
header{position:sticky;top:0;z-index:9;background:#15181d;border-bottom:1px solid #262b33;
       padding:10px 18px;display:flex;gap:16px;align-items:baseline}
h1{font-size:15px;margin:0;font-weight:600}
h2{font-size:13px;margin:26px 0 8px;color:#9aa4b2;text-transform:uppercase;
   letter-spacing:.09em;border-bottom:1px solid #262b33;padding-bottom:5px}
main{padding:16px 18px 70px;max-width:1500px}
.muted{color:#79828f}
.badge{display:inline-block;padding:1px 7px;border-radius:3px;font-size:11px;font-weight:600}
.ok{background:#12351f;color:#6ee7a0} .bad{background:#3a1518;color:#ff8f9a}
.run{background:#2a2410;color:#f5c451} .idle{background:#22262d;color:#98a2b0}
table{border-collapse:collapse;width:100%}
td,th{padding:5px 9px;border-bottom:1px solid #1e222a;text-align:left;vertical-align:top}
th{color:#79828f;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.06em}
tr:hover td{background:#161a20}
.grid{display:grid;gap:10px;grid-template-columns:repeat(auto-fill,minmax(230px,1fr))}
.card{background:#161a20;border:1px solid #242932;border-radius:6px;padding:10px}
.card img{width:100%;display:block;border-radius:4px;background:#0a0c0f}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 14px}
.kv div:nth-child(odd){color:#79828f}
pre{background:#0a0c0f;border:1px solid #1e222a;border-radius:5px;padding:10px;
    overflow:auto;max-height:440px;font-size:12px;white-space:pre-wrap;word-break:break-word}
.gates{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.gate{padding:3px 10px;border-radius:12px;font-size:11px;border:1px solid #2c323c}
.g-passed{background:#12351f;color:#6ee7a0;border-color:#1d5233}
.g-died{background:#3a1518;color:#ff8f9a;border-color:#6b2027;font-weight:700}
.g-pending{background:#14171c;color:#5b6472} .g-running{background:#2a2410;color:#f5c451}
.g-unknown{background:#14171c;color:#79828f;border-style:dashed}
.arrow{color:#3a424e}
.cmp{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:12px}
.cmp figure{margin:0} .cmp figcaption{font-size:11px;color:#79828f;padding:3px 1px}
.cmp img{width:100%;display:block;border-radius:4px;background:#0a0c0f;cursor:zoom-in}
.bar{height:7px;border-radius:4px;background:#2a1518;overflow:hidden;margin-top:4px}
.bar>i{display:block;height:100%;background:#6ee7a0}
.warn{background:#2a1e10;border:1px solid #5c4218;color:#f5c451;
      padding:8px 11px;border-radius:5px;margin:8px 0}
.note{background:#101820;border:1px solid #1e3040;color:#8fb8d6;
      padding:8px 11px;border-radius:5px;margin:8px 0;font-size:12px}
#lb{position:fixed;inset:0;background:#000000e8;display:none;z-index:99;
    align-items:center;justify-content:center;cursor:zoom-out}
#lb img{max-width:96vw;max-height:96vh;image-rendering:pixelated}
.viewers{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
.viewer{background:#161a20;border:1px solid #242932;border-radius:6px;overflow:hidden}
.viewer model-viewer{width:100%;height:440px;display:block;--poster-color:transparent;""" + VIEWER_BG + """}
.viewer h3{margin:0;padding:8px 11px;font-size:12px;font-weight:600;
  border-bottom:1px solid #242932;color:#d7dbe0}
.viewer .cap{padding:6px 11px;font-size:11px;color:#79828f;border-top:1px solid #242932}
/* 목록 카드의 작은 3D 뷰도 같은 바탕을 쓴다 — 목록과 상세가 달라 보이면 안 된다.
   자리표시(div)와 실제 뷰어(model-viewer)가 같은 규칙을 공유해 눌러도 안 흔들린다. */
.mv-thumb{width:100%;height:190px;border-radius:4px;display:flex;
  align-items:center;justify-content:center;cursor:pointer;object-fit:contain;""" + VIEWER_BG + """}
.mv-thumb>span{font-size:12px;color:#5b6472;background:#ffffffcc;
  padding:4px 10px;border-radius:12px;border:1px solid #cfd6de}
.mv-thumb:hover>span{color:#0d1117;border-color:#8a94a2}
"""

JS = """
function lb(src){var d=document.getElementById('lb');
 d.querySelector('img').src=src;d.style.display='flex';}
// 목록 카드: 누른 것만 뷰어로 바꾼다. WebGL 컨텍스트는 브라우저 상한(대략 16)이 있어
// 카드마다 미리 만들면 상한을 넘기고, 넘긴 순간 오래된 것부터 화면이 죽는다.
// 열어 둔 개수가 상한에 닿지 않도록 8개를 넘으면 가장 먼저 연 것을 자리표시로 되돌린다.
var MVOPEN=[];
function mv(el){
 var src=el.getAttribute('data-src'); if(!src) return;
 var v=document.createElement('model-viewer');
 v.className='mv-thumb'; v.setAttribute('src',src);
 v.setAttribute('camera-controls',''); v.setAttribute('touch-action','pan-y');
 v.setAttribute('interaction-prompt','none'); v.setAttribute('alt','3D');
 el.replaceWith(v); MVOPEN.push({v:v,src:src});
 while(MVOPEN.length>8){
  var o=MVOPEN.shift();
  if(!o.v.isConnected) continue;
  var d=document.createElement('div');
  d.className='mv-thumb'; d.setAttribute('data-src',o.src);
  d.setAttribute('onclick','mv(this)'); d.title='눌러서 3D 로 보기';
  d.innerHTML='<span>\u25B6 3D 로 보기</span>';
  o.v.replaceWith(d);                 // 컨텍스트를 실제로 놓아준다
 }}
document.addEventListener('click',function(e){
 if(e.target.id==='lb'||e.target.parentElement.id==='lb')
   document.getElementById('lb').style.display='none';});
document.addEventListener('keydown',function(e){
 if(e.key==='Escape')document.getElementById('lb').style.display='none';});
"""


def page(title, body):
    return HTMLResponse(
        f"<!doctype html><meta charset=utf-8>"
        f"<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style>"
        f"<header><h1>VoxHammer / lab debug</h1>"
        f"<a href='/debug'>lab runs</a><a href='/debug/vh'>VoxHammer 트랙</a>"
        f"<span class=muted>읽기 전용</span></header>"
        f"<main>{body}</main>"
        f"<div id=lb><img alt=zoom></div><script>{JS}</script>"
        f"<script type='module' src='/debug/static/model-viewer.min.js'></script>")


def esc(v):
    return html.escape("" if v is None else str(v))


def badge(state):
    cls = {"succeeded": "ok", "failed": "bad", "running": "run"}.get(state, "idle")
    return f"<span class='badge {cls}'>{esc(state or '?')}</span>"


def fileurl(run_id, rel):
    return f"/lab/runs/{run_id}/files/{rel}"


def human(n):
    if n is None:
        return "—"
    for u in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or u == "GiB":
            return f"{n:,.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024.0


def kvtable(d, keys=None):
    rows = []
    for k in (keys or d.keys()):
        if k not in d or d[k] is None:
            continue
        v = d[k]
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
            if len(v) > 300:
                v = v[:300] + " …"
        rows.append(f"<div>{esc(k)}</div><div>{esc(v)}</div>")
    return f"<div class=kv>{''.join(rows)}</div>" if rows else "<p class=muted>없음</p>"


# ═══════════════════════════════════ VoxHammer 단독 트랙 ═══════════════
#
# lab run 형식이 아니다. 각 작업 디렉터리에 `track.json` 을 두면 그걸 그대로
# 보여준다 — 이 화면은 계산하지 않고 **계측 스크립트가 남긴 것만** 띄운다.
# (지표를 두 곳에서 계산하면 둘이 갈라지고, 갈라진 걸 아무도 모른다)
def track_dir(name):
    if not TRACK_NAME_OK.match(name):
        raise HTTPException(400, "이름 형식이 아니다")
    d = os.path.realpath(os.path.join(TRACK_ROOT, name))
    root = os.path.realpath(TRACK_ROOT)
    if d != root and not d.startswith(root + os.sep):
        raise HTTPException(403, "트랙 디렉터리 밖")
    if not os.path.isdir(d):
        raise HTTPException(404, "없는 작업")
    return d


def track_json(name):
    p = os.path.join(track_dir(name), "track.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:                            # noqa: BLE001
        return None


@router.get("/debug/static/{name}")
def static_file(name: str):
    """뷰어 자산만 내보낸다. 화이트리스트라 경로 조작이 성립하지 않는다."""
    if name not in STATIC_OK:
        raise HTTPException(404, "없는 파일")
    p = os.path.join(STATIC_DIR, name)
    if not os.path.isfile(p):
        raise HTTPException(404, "뷰어 자산이 없다")
    from fastapi.responses import FileResponse
    return FileResponse(p, media_type="text/javascript")


def models_present(name, t):
    """track.json 이 선언한 모델 중 **파일이 실제로 있는 것만** 돌려준다.

    선언은 있는데 파일이 없는 경우가 실제로 생긴다 (런이 중간에 죽거나, 경로를 고쳤거나).
    그때 목록에 빈 뷰어가 뜨면 "3D 산출물이 있다" 는 오해가 생긴다 —
    **없는 것은 아예 안 띄운다.**
    """
    out = []
    try:
        d = track_dir(name)
    except HTTPException:
        return out
    models = (t or {}).get("models") or []
    if not models:
        # 🔴 track.json 이 없거나 models 를 안 적었어도 **GLB 가 실제로 있으면 보여준다.**
        #    목록의 취지는 "3D 산출물이 있는 것만" 인데, 구현이 "track.json 이 선언한
        #    모델이 있는 것만" 이라 실제 산출물이 통째로 숨는 일이 있었다 (t1v2·t1v3 실측).
        #    선언은 없어도 산출은 산출이다.
        import glob as _glob
        for p_ in sorted(_glob.glob(os.path.join(d, "*.glb"))):
            models.append({"path": os.path.basename(p_),
                           "label": os.path.basename(p_) + " (track.json 없음 — 자동 발견)"})
    for m in models:
        rel = m.get("path")
        if not rel:
            continue
        p = os.path.realpath(os.path.join(d, rel))
        troot = os.path.realpath(TRACK_ROOT)
        if (p == troot or p.startswith(troot + os.sep)) and os.path.isfile(p):
            out.append(m)
    return out


# 이 오브젝트가 **거쳐온 과정**. 원저자 경로는 언제나 이 세 장에서 시작한다.
PROCESS_STEPS = [
    ("render", "① 원본 렌더 (2d_render)", "편집 전 자산을 그 카메라로 렌더한 것"),
    ("mask", "② 마스크 (2d_mask)", "같은 카메라에서 mask.glb 를 투영한 것 — 흰색이 편집 영역"),
    ("edit", "③ 인페인트 (2d_edit)", "①의 마스크 안쪽만 다시 그린 것 — 이것이 편집의 조건이다"),
]


def process_html(name, t):
    """3D 산출물 바로 아래에 **무엇을 조건으로 받아 이렇게 됐는지**를 붙인다.

    수치만 보면 "바뀌었다" 는 알아도 "무엇을 시켰길래 그렇게 됐나" 는 모른다.
    작업 1 에서 마스크가 모자를 덮은 탓에 '모자→호박' 이 나온 것을 이 세 장이 바로 보여준다.
    """
    proc = (t or {}).get("process") or {}
    cells = []
    for key, label, hint in PROCESS_STEPS:
        rel = proc.get(key)
        if not rel:
            continue
        # 과정 이미지는 다른 작업 디렉터리에 있다 — 그쪽을 지우면 파일만 없어진다.
        # 없는 것을 <img> 로 내보내면 깨진 아이콘이 남으므로 아예 건너뛴다.
        try:
            if not os.path.isfile(os.path.join(track_dir(name), rel)):
                continue
        except HTTPException:
            continue
        cells.append(
            f"<div class=viewer><h3>{esc(label)}</h3>"
            f"<img src='/debug/vhf/{name}/{rel}' loading=lazy "
            f"onclick=\"lb('/debug/vhf/{name}/{rel}')\" alt='{esc(label)}' "
            f"style='width:100%;border-radius:5px;cursor:zoom-in;background:#0a0c0f'>"
            f"<div class=cap>{esc(hint)}</div></div>")
    if not cells:
        return ""
    note = proc.get("note")
    return ("<h2>이 오브젝트가 거쳐온 과정 — 원본 → 마스크 → 인페인트</h2>"
            "<div class=note>왼쪽에서 오른쪽으로 읽는다. <b>③이 곧 편집의 조건</b>이고, "
            "위의 3D 산출물은 ③을 3D 로 옮긴 결과다. 눌러서 크게 볼 수 있다.</div>"
            + (f"<div class=note>{esc(note)}</div>" if note else "")
            + f"<div class=viewers>{''.join(cells)}</div>")


def viewers_html(name, models):
    """원본 / 수정후를 나란히 놓고 **직접 돌려 볼 수 있게** 한다.

    각도별 캡쳐를 늘어놓는 대신 실물 GLB 를 띄운다 — 어느 각도가 필요한지는
    보는 사람이 그때 정한다.
    """
    if not models:
        return ""
    cards = []
    for m in models:
        rel, label = m.get("path"), m.get("label") or m.get("path")
        if not rel:
            continue
        src = f"/debug/vhf/{name}/{rel}"
        note = f"<div class=cap>{esc(m['note'])}</div>" if m.get("note") else ""
        cards.append(
            f"<div class=viewer><h3>{esc(label)}</h3>"
            f"<model-viewer src='{src}' alt='{esc(label)}' camera-controls "
            f"touch-action='pan-y' shadow-intensity='0.6' exposure='1.0' "
            f"interaction-prompt='none' loading='eager'></model-viewer>{note}</div>")
    if not cards:
        return ""
    return ("<h2>3D — 직접 돌려 보기</h2>"
            "<div class=note>드래그로 회전 · 휠로 확대 · 우클릭 드래그로 이동. "
            "모든 뷰어는 같은 초기 시점에서 시작하므로 나란히 두고 비교할 수 있다.</div>"
            f"<div class=viewers>{''.join(cards)}</div>")


@router.get("/debug/vh", response_class=HTMLResponse)
def vh_index():
    if not os.path.isdir(TRACK_ROOT):
        return page("VoxHammer 트랙", "<h2>VoxHammer 트랙</h2>"
                    "<p class=muted>산출물 루트가 아직 없다.</p>")
    names = [d for d in os.listdir(TRACK_ROOT)
             if os.path.isdir(os.path.join(TRACK_ROOT, d)) and TRACK_NAME_OK.match(d)]
    # 🔴 **최근 생성한 순**으로 세운다. 기준은 그 작업의 산출 GLB mtime 이다 —
    #    디렉터리 mtime 은 나중에 track.json 을 고치기만 해도 올라가서 순서를 속인다.
    def newest(n):
        t = track_json(n) or {}
        ts = []
        for m in models_present(n, t):
            try:
                ts.append(os.path.getmtime(os.path.join(track_dir(n), m["path"])))
            except OSError:
                pass
        return max(ts) if ts else 0.0
    names.sort(key=lambda n: (-newest(n), n))
    cards, hidden = [], 0
    for n in names:
        t = track_json(n) or {}
        models = models_present(n, t)
        # 🔴 3D 산출물이 **실제로 있는 것만** 띄운다. 없는 것은 목록에서 뺀다 —
        #    빈 자리를 늘어놓으면 무엇이 실제 결과인지 한눈에 안 들어온다.
        if not models:
            hidden += 1
            continue
        last = models[-1]
        # 🔴 목록 썸네일은 **새로 생성된 오브젝트의 정면 정지 캡쳐**다 (tools/make_thumb.py).
        #    model-viewer 를 카드마다 두면 WebGL 컨텍스트를 카드 수만큼 잡는데 브라우저
        #    상한이 대략 16이라, 17개짜리 목록에서 오래된 것부터 화면이 죽었다.
        #    (서버는 병목이 아니다: 실측 HTML 18ms · GLB 1.7MB 17ms)
        #    캡쳐면 컨텍스트가 0이고 무엇이 나왔는지도 더 빨리 읽힌다.
        if os.path.isfile(os.path.join(track_dir(n), "thumb.png")):
            view = (f"<img class=mv-thumb src='/debug/vhf/{n}/thumb.png' "
                    f"alt='{esc(n)} 정면' loading=lazy>")
        else:
            # 캡쳐가 아직 없는 작업만 클릭-생성으로 떨어진다. 자리를 비워 두지 않는다.
            view = (f"<div class=mv-thumb data-src='/debug/vhf/{n}/{last['path']}' "
                    f"onclick='mv(this)' title='캡쳐 없음 — 눌러서 3D 로 보기'>"
                    f"<span>&#9654; 3D 로 보기</span></div>")
        st = t.get("verdict") or "—"
        proc = sum(1 for k, _, _ in PROCESS_STEPS if (t.get("process") or {}).get(k))
        tag = (f" · <span class=muted style='font-size:11px'>과정 {proc}/3</span>"
               if proc else "")
        cards.append(f"<div class=card>{view}"
                     f"<div style='margin-top:7px'><a href='/debug/vh/{n}'><b>{esc(n)}</b></a> "
                     f"<span class=muted style='font-size:11px'>({len(models)} 모델)</span>{tag}</div>"
                     f"<div class=muted style='font-size:11px'>{esc(t.get('title') or '')}</div>"
                     f"<div class=muted style='font-size:11px'>{esc(st)}</div></div>")
    return page("VoxHammer 트랙",
                "<h2>VoxHammer 단독 트랙</h2>"
                "<div class=note>상위 프로젝트와 분리된 트랙이다 — 청크·델타·계약이 여기 없다. "
                "각 작업의 수치는 계측 스크립트가 <code>track.json</code> 에 남긴 것을 "
                "그대로 띄운다.</div>"
                "<div class=note>🔴 <b>3D 산출물이 있는 작업만</b> 나온다. 각 작업 안에서 "
                "그 오브젝트가 거쳐온 <b>원본 → 마스크 → 인페인트</b> 를 함께 볼 수 있다. "
                "<b>최근 생성한 순</b>으로 세우고, 썸네일은 <b>새로 만들어진 오브젝트의 정면 캡쳐</b>다. "
                "3D 로 돌려 보려면 카드를 눌러 상세로 들어간다."
                + (f" (산출물 없어 숨긴 디렉터리 {hidden}개)" if hidden else "") + "</div>"
                + (f"<div class=grid>{''.join(cards)}</div>" if cards
                   else "<p class=muted>3D 산출물이 있는 작업이 아직 없다</p>"))


@router.get("/debug/vh/{name}", response_class=HTMLResponse)
def vh_detail(name: str):
    d = track_dir(name)
    t = track_json(name)
    S = [f"<h2>{esc(name)}</h2>"]
    if not t:
        S.append("<div class=warn>track.json 이 없다 — 파일만 나열한다.</div>")
    else:
        S.append(f"<p><b>{esc(t.get('title'))}</b></p>")
        if t.get("verdict"):
            S.append(f"<div class=note><b>판정:</b> {esc(t['verdict'])}</div>")
        if t.get("settings"):
            S.append("<h2>설정</h2>" + kvtable(t["settings"]))

        sp = t.get("space") or {}
        if sp:
            r = sp.get("empty_ratio") or 0.0
            S.append("<h2>편집 볼륨 빈 공간 — 신규 기하가 자랄 자리</h2>"
                     + kvtable(sp) +
                     f"<div class=bar><i style='width:{r * 100:.1f}%'></i></div>")
            if sp.get("empty") == 0:
                S.append("<div class=warn>🔴 빈 칸 0% — '마스크 안 신규' 는 정의상 0 이다 "
                         "(항진명제). 이 런의 신규 수치를 실패의 증거로 읽지 마라.</div>")

        vx = t.get("voxels") or {}
        if vx:
            def n(k):
                """없는 값은 '—'. 숫자에만 자릿점을 찍는다 (문자열에 ',' 포맷은 ValueError)."""
                v = vx.get(k)
                return f"{v:,}" if isinstance(v, (int, float)) else "—"

            has_std = any(k in vx for k in
                          ("inside_new", "inside_removed", "outside_new", "outside_removed"))
            if has_std:
                S.append("<h2>복셀 계측 — 마스크 안/밖 신규·제거</h2>"
                         "<table><tr><th></th><th>신규</th><th>제거</th></tr>"
                         f"<tr><td>마스크 안</td><td>{n('inside_new')}</td>"
                         f"<td>{n('inside_removed')}</td></tr>"
                         f"<tr><td>마스크 밖</td><td>{n('outside_new')}</td>"
                         f"<td>{n('outside_removed')}</td></tr></table>")
            else:
                S.append("<h2>복셀 계측</h2>")
            rest = [k for k in vx if k not in
                    ("inside_new", "inside_removed", "outside_new", "outside_removed")]
            S.append(kvtable(vx, rest))
            if t.get("voxels_note"):
                S.append(f"<div class=note>{esc(t['voxels_note'])}</div>")

    S.append(viewers_html(name, models_present(name, t)))
    S.append(process_html(name, t))

    # 3D 로 못 보여주는 근거 이미지 (전처리 덤프 · diff 등)만 여기 붙는다.
    # 각도별 정지 캡쳐를 되살리는 자리가 아니다.
    for im in (t or {}).get("images") or []:
        rel = im.get("path")
        if not rel:
            continue
        u = f"/debug/vhf/{name}/{rel}"
        S.append(f"<h2>{esc(im.get('label') or rel)}</h2>"
                 + (f"<div class=note>{esc(im['note'])}</div>" if im.get("note") else "")
                 + f"<img src='{u}' loading=lazy onclick=\"lb('{u}')\" alt='' "
                   f"style='width:100%;border-radius:5px;cursor:zoom-in;background:#0a0c0f'>")

    rows = []
    for base, _dirs, files in os.walk(d):
        for fn in sorted(files):
            fp = os.path.join(base, fn)
            rel = os.path.relpath(fp, d)
            try:
                sz = os.path.getsize(fp)
            except OSError:
                continue
            rows.append(f"<tr><td><a href='/debug/vhf/{name}/{rel}'>{esc(rel)}</a></td>"
                        f"<td class=muted>{human(sz)}</td></tr>")
    S.append(f"<h2>산출 파일 ({len(rows)}개)</h2>"
             f"<table><tr><th>경로</th><th>크기</th></tr>{''.join(sorted(rows))}</table>")
    return page(f"{name} — VoxHammer 트랙", "".join(S))


@router.get("/debug/vh/{name}/mask", response_class=HTMLResponse)
def vh_mask(name: str, mode: str = "neck", z_min: int = None, halo: int = 0,
            active_only: int = 1, pad_xy: int = 0, largest: int = 0):
    """마스크 규칙을 **URL 로 조정**하고 3D 오버레이로 확인한다. GPU 0런.

    라쏘 대신 숫자로 맞추기 위한 화면이다. 그린 마스크는 매번 파일로 남는다.
    """
    d = track_dir(name)
    ply = os.path.join(d, "render", "voxels.ply")
    if not os.path.isfile(ply):
        raise HTTPException(404, "render/voxels.ply 가 없다 — 먼저 생성·렌더가 필요하다")
    from . import vhmask

    xyz = vhmask.read_voxels(ply)
    try:
        m = vhmask.build(xyz, mode=mode, z_min=z_min, halo=halo,
                         active_only=bool(active_only), pad_xy=pad_xy,
                         largest=largest)
    except ValueError as e:
        raise HTTPException(400, str(e))
    st, pr = m["stats"], m["params"]

    slug = (f"{pr['mode']}_z{pr['z_min']}_h{pr['halo']}"
            f"_a{int(pr['active_only'])}_p{pr['pad_xy']}_L{pr['largest']}")
    outdir = os.path.join(d, "masks", slug)
    vhmask.save(outdir, m, name)                       # 🔴 항상 보존
    rel_glb = os.path.join("masks", slug, "overlay.glb")
    glb = os.path.join(d, rel_glb)
    asset = os.path.join(d, "model.glb")
    if not os.path.isfile(glb) and os.path.isfile(asset):
        try:
            vhmask.overlay_glb(asset, m["cells"], glb)
        except Exception as e:                          # noqa: BLE001
            st = dict(st, overlay_error=f"{type(e).__name__}: {e}")

    def link(**kw):
        p = {"mode": pr["mode"], "z_min": pr["z_min"], "halo": pr["halo"],
             "active_only": int(pr["active_only"]), "pad_xy": pr["pad_xy"],
             "largest": pr["largest"]}
        p.update(kw)
        return "?" + "&".join(f"{k}={v}" for k, v in p.items())

    S = [f"<h2>{esc(name)} — 마스크 오버레이</h2>",
         "<div class=note>URL 파라미터로 규칙을 바꾼다 (라쏘 없이 숫자로 조정). "
         "GPU 를 쓰지 않는다. 그린 마스크는 <code>masks/&lt;슬러그&gt;/</code> 에 "
         "<b>voxels_delete.ply + mask.json 으로 항상 저장</b>된다.</div>"]

    rows = []
    for k, opts in (("z_min", [pr["z_min"] - 4, pr["z_min"] - 2, pr["z_min"] - 1,
                               pr["z_min"] + 1, pr["z_min"] + 2, pr["z_min"] + 4]),
                    ("halo", [0, 1, 2, 3]),
                    ("active_only", [1, 0]),
                    ("pad_xy", [0, 1, 2, 4]),
                    ("largest", [0, 1]),
                    ("mode", ["neck", "z"])):
        cur = pr[k] if k != "active_only" else int(pr["active_only"])
        cells = " ".join(
            (f"<b>{o}</b>" if str(o) == str(cur)
             else f"<a href='/debug/vh/{name}/mask{link(**{k: o})}'>{o}</a>")
            for o in opts)
        rows.append(f"<tr><td>{k}</td><td>{cells}</td></tr>")
    S.append("<h2>파라미터</h2><table><tr><th>이름</th><th>값 (눌러서 변경)</th></tr>"
             + "".join(rows) + "</table>"
             + f"<div class=muted style='margin-top:6px'>자동 검출한 목 z = "
               f"{esc(m['auto_neck'])} · 현재 z_min = <b>{pr['z_min']}</b> "
               f"(z 가 수직축이다 — 실측 확인)</div>")

    S.append("<h2>마스크 수치</h2>" + kvtable({
        "선택 셀 수": st["cells"],
        "활성 복셀 대비": f"{st['cells_pct_of_active']}%  (전체 {st['active_total']:,})",
        "bbox": st["bbox"], "치수(x,y,z)": st["dims"],
        "선택 볼륨(팽창 전 상자)": st["volume_cells"],
        "  └ 점유": st["volume_occupied"],
        "  └ 빈 칸": f"{st['volume_empty']} ({st['empty_pct']}%)",
        "★ 번짐(볼륨 중 물체가 아닌 칸)": f"{st['spill_pct']}%",
        "largest 로 떨어진 셀": st.get("dropped_by_largest"),
        "지문 mask_fingerprint": st["fingerprint"],
        "저장 위치": f"masks/{slug}/",
        "오버레이 오류": st.get("overlay_error"),
    }))
    if st["empty_pct"] == 0:
        S.append("<div class=warn>빈 칸 0% — 이 마스크에서 '마스크 안 신규' 는 "
                 "정의상 0 이다. 효능 판정에 그 수를 쓰지 마라 (2단계).</div>")

    if os.path.isfile(glb):
        S.append(viewers_html(name, [
            {"label": "마스크 오버레이 (빨간 큐브 = 선택된 셀)", "path": rel_glb,
             "note": f"{st['cells']:,}셀 · 지문 {st['fingerprint']}"},
            {"label": "원본 자산", "path": "model.glb"}]))
    else:
        S.append("<p class=muted>오버레이 GLB 를 못 만들었다.</p>")

    S.append(f"<h2>산출</h2><table>"
             f"<tr><td><a href='/debug/vhf/{name}/masks/{slug}/voxels_delete.ply'>"
             f"voxels_delete.ply</a></td><td class=muted>편집에 그대로 넣는 파일</td></tr>"
             f"<tr><td><a href='/debug/vhf/{name}/masks/{slug}/mask.json'>mask.json</a>"
             f"</td><td class=muted>규칙·수치 기록</td></tr></table>"
             f"<p><a href='/debug/vh/{name}'>← {esc(name)}</a></p>")
    return page(f"{name} 마스크 — lab debug", "".join(S))


@router.get("/debug/vhf/{name}/{path:path}")
def vh_file(name: str, path: str):
    # 과정 이미지(2d_render / 2d_mask / 2d_edit)는 **다른 작업 디렉터리**에 있다 —
    # 조건은 `<작업>-cond`, 인페인트는 `<작업>-inpaint`, 산출 GLB 는 `<작업>-edit` 다.
    # 그래서 경계를 작업 디렉터리가 아니라 **트랙 루트**로 잡는다. 루트 밖은 여전히 막힌다.
    root = track_dir(name)                      # 이름 형식·존재 검사는 그대로 탄다
    troot = os.path.realpath(TRACK_ROOT)
    target = os.path.realpath(os.path.join(root, path))
    if target != troot and not target.startswith(troot + os.sep):
        raise HTTPException(403, "트랙 루트 밖은 접근할 수 없다")
    if not os.path.isfile(target):
        raise HTTPException(404, "없는 파일")
    from fastapi.responses import FileResponse
    return FileResponse(target)


# ─────────────────────────────────────────────────────────── 목록
@router.get("/debug", response_class=HTMLResponse)
def index():
    ids = list_run_ids()[::-1]
    rows = []
    for rid in ids:
        m = read_meta(rid) or {}
        _g, died = gates_of(m)
        thumbs = m.get("thumbs") or {}
        th = thumbs.get("merged") or thumbs.get("base")
        img = (f"<img src='{fileurl(rid, th)}' loading=lazy alt=''>"
               if th else "<div class=muted style='padding:22px 0;text-align:center'>렌더 없음</div>")
        err = m.get("error")
        note = ""
        if m.get("state") == "failed":
            note = (f"<div class=warn style='margin:6px 0 0'>죽은 관문: "
                    f"<b>{esc(died or m.get('stage_detail'))}</b>"
                    + (f"<br>{esc(str(err)[:150])}" if err else "") + "</div>")
        parent = m.get("parent_run_id")
        rows.append(
            f"<div class=card><a href='/debug/run/{rid}'>{img}</a>"
            f"<div style='margin-top:7px'><a href='/debug/run/{rid}'><b>{rid}</b></a> "
            f"{badge(m.get('state'))}</div>"
            f"<div class=muted style='font-size:11px'>"
            f"{esc(m.get('kind'))}"
            + (f" ← <a href='/debug/run/{parent}'>{parent}</a>" if parent else "")
            + f" · {esc(m.get('stage_detail'))} · {esc(m.get('duration_s'))}s</div>"
            + note + "</div>")

    n_fail = sum(1 for r in ids if (read_meta(r) or {}).get("state") == "failed")
    head = (f"<h2>런 {len(ids)}개 · 실패 {n_fail}개</h2>"
            "<div class=note>실패한 런도 여기 전부 보인다. 카드의 <b>죽은 관문</b> 이 "
            "어디서 멈췄는지다. 아무 런이나 눌러 중간 산출물을 본다.</div>")
    return page("lab debug", head + f"<div class=grid>{''.join(rows)}</div>")


# ─────────────────────────────────────────────────────────── 상세
@router.get("/debug/run/{run_id}", response_class=HTMLResponse)
def detail(run_id: str):
    if not RUN_ID_RE.match(run_id):
        raise HTTPException(400, "run_id 형식이 아니다")
    m = read_meta(run_id)
    if m is None:
        raise HTTPException(404, "없는 run")

    parent = m.get("parent_run_id")
    pm = read_meta(parent) if parent else None
    S = []

    # ── 머리말 + 관문 ───────────────────────────────────────────
    grows, died = gates_of(m)
    chips = "<span class=arrow>→</span>".join(
        f"<span class='gate g-{st}'>{esc(lbl)}</span>" for lbl, st in grows)
    S.append(f"<h2>{esc(run_id)} {badge(m.get('state'))}</h2>")
    S.append(f"<div class=gates>{chips}</div>")
    if m.get("state") == "failed":
        S.append(f"<div class=warn><b>죽은 관문: {esc(died or m.get('stage_detail'))}</b>"
                 f"<br>error_code: {esc(m.get('error_code'))}"
                 f"<br>{esc(m.get('error'))}</div>")
    if any(st == "unknown" for _, st in grows):
        S.append("<div class=note>stage_detail 이 계약 어휘 밖이라 관문 진행을 "
                 "판정하지 않았다 — 지어내지 않는다.</div>")
    S.append(kvtable(m, ["kind", "parent_run_id", "region", "note", "halo", "seed",
                         "fraction", "deterministic", "redecode", "snap_cells",
                         "started_at", "finished_at", "duration_s", "voxels",
                         "chunk_count", "latent_preserved", "preserved_rows"]))

    # ── 🔴 빈 공간 — 항진명제 방지 ──────────────────────────────
    if m.get("kind") == "edit":
        sp = mask_space(run_id, m)
        S.append("<h2>마스크 빈 공간 — 신규 기하가 자랄 자리</h2>")
        if sp and "error" not in sp:
            r = sp["empty_ratio"] or 0.0
            S.append(
                f"<div class=kv>"
                f"<div>core 복셀</div><div>{sp['core']:,}</div>"
                f"<div>halo</div><div>{sp['halo']}</div>"
                f"<div>마스크 부피(팽창 후)</div><div>{sp['volume']:,}</div>"
                f"<div>점유</div><div>{sp['occupied']:,}</div>"
                f"<div><b>빈 칸</b></div><div><b>{sp['empty']:,} "
                f"({r * 100:.1f}%)</b></div></div>"
                f"<div class=bar><i style='width:{r * 100:.1f}%'></i></div>")
            if sp["empty"] == 0:
                S.append("<div class=warn>🔴 <b>빈 칸 0%.</b> 마스크가 전부 점유다 — "
                         "'마스크 안 신규 N' 은 <b>정의상 0</b> 이고, 모델이 무엇을 "
                         "하든 0 이 나온다. 이 런의 신규 수치를 실패의 증거로 읽지 "
                         "마라 (항진명제).</div>")
            S.append("<div class=note>meta 의 <code>after_halo</code> 는 점유 복셀만 "
                     "센다. 위 <b>부피</b>·<b>빈 칸</b>은 mask.json 의 core 를 halo 만큼 "
                     "실제로 팽창시켜 이 화면이 직접 계산한 값이다.</div>")
        else:
            S.append(f"<p class=muted>미계산 — {esc((sp or {}).get('error'))}</p>")

    # ── before / after ─────────────────────────────────────────
    rend = m.get("renders") or {}
    if rend:
        S.append("<h2>before / after · 마스크 영역</h2>")
        S.append("<div class=note>왼쪽 <b>before</b> 는 부모 런의 렌더, 가운데는 "
                 "<b>마스크 색칠</b>(core/halo), 오른쪽 <b>after</b> 는 이 런이 실제로 "
                 "내보내는 합성물(merged). 이미지를 누르면 원본 크기로 확대된다. "
                 "<code>mask</code> 뷰는 마스크 무게중심을 바라보는 자동 카메라다.</div>")
        prend = (pm or {}).get("renders") or {}
        pbefore = prend.get("merged") or prend.get("base") or {}
        after = rend.get("merged") or rend.get("base") or {}
        maskr = rend.get("mask") or {}
        for v in VIEWS:
            cells = []
            for lbl, rid_, rel in (
                    (f"before ({parent})" if parent else "before", parent, pbefore.get(v)),
                    ("mask 색칠", run_id, maskr.get(v)),
                    ("after (merged)", run_id, after.get(v))):
                if rel and rid_:
                    u = fileurl(rid_, rel)
                    cells.append(f"<figure><img src='{u}' loading=lazy onclick=\"lb('{u}')\" alt=''>"
                                 f"<figcaption>{esc(lbl)} · {esc(v)}</figcaption></figure>")
                else:
                    cells.append(f"<figure><div class=muted style='padding:30px 0;"
                                 f"text-align:center;background:#12161c;border-radius:4px'>없음</div>"
                                 f"<figcaption>{esc(lbl)} · {esc(v)}</figcaption></figure>")
            S.append(f"<div class=cmp>{''.join(cells)}</div>")

        other = {k: v for k, v in rend.items() if k not in ("merged", "mask")}
        if other:
            S.append("<h2>그 밖의 렌더 (base=전체 재디코딩 · delta=변경 청크/이음새)</h2>")
            cards = []
            for kind, views in other.items():
                for v, rel in (views or {}).items():
                    u = fileurl(run_id, rel)
                    cards.append(f"<div class=card><img src='{u}' loading=lazy "
                                 f"onclick=\"lb('{u}')\" alt=''>"
                                 f"<div class=muted style='font-size:11px;margin-top:5px'>"
                                 f"{esc(kind)} · {esc(v)}</div></div>")
            S.append(f"<div class=grid>{''.join(cards)}</div>")

    # ── 중간 산출물 ────────────────────────────────────────────
    S.append("<h2>중간 산출물</h2>")
    d = run_dir(run_id)
    inter = []

    ip = m.get("input_image")
    if ip and os.path.exists(os.path.join(d, ip)):
        u = fileurl(run_id, ip)
        inter.append(f"<div class=card><img src='{u}' onclick=\"lb('{u}')\" alt=''>"
                     f"<div class=muted style='font-size:11px;margin-top:5px'>"
                     f"조건 이미지 {esc(ip)}</div></div>")
    elif m.get("kind") == "edit" and parent:
        pip = (pm or {}).get("input_image")
        if pip and os.path.exists(os.path.join(run_dir(parent), pip)):
            u = fileurl(parent, pip)
            inter.append(f"<div class=card><img src='{u}' onclick=\"lb('{u}')\" alt=''>"
                         f"<div class=muted style='font-size:11px;margin-top:5px'>"
                         f"조건 이미지 (부모 {esc(parent)} 것을 재사용)</div></div>")
    if inter:
        S.append(f"<div class=grid>{''.join(inter)}</div>")

    # slat — 헤더만 읽는다 (텐서는 안 읽는다)
    sp_path = os.path.join(d, "slat.safetensors")
    if os.path.exists(sp_path):
        info = {}
        try:
            from safetensors import safe_open
            with safe_open(sp_path, "pt") as f:
                info["metadata"] = f.metadata() or {}
                for k in f.keys():
                    sl = f.get_slice(k)
                    info[k] = f"{tuple(sl.get_shape())} {sl.get_dtype()}"
        except Exception as e:                  # noqa: BLE001
            info["error"] = f"{type(e).__name__}: {e}"
        S.append("<h3 class=muted>slat.safetensors</h3>" + kvtable(info))
        md = info.get("metadata") or {}
        missing = [k for k in ("norm_mean", "norm_std", "coord_order", "slat_space")
                   if k not in md]
        if missing:
            S.append(f"<div class=warn>slat 메타에 {esc(missing)} 이 없다 — "
                     "다음 편집이 KeyError 로 막다른 판본이 된다.</div>")

    for name in ("mask.json", "manifest.json"):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        try:
            j = json.load(open(p))
        except Exception:                        # noqa: BLE001
            continue
        if name == "manifest.json":
            j = {"asset_id": j.get("asset_id"), "version": j.get("version"),
                 "chunks": f"{len(j.get('chunks') or {})}개"}
        else:
            sp_ = j.get("spec") or {}
            j = {k: v for k, v in j.items() if k != "spec"}
            j["spec.mode"] = sp_.get("mode")
            j["spec.voxels"] = f"{len(sp_.get('voxels') or [])}개"
        S.append(f"<h3 class=muted>{name} "
                 f"<a href='{fileurl(run_id, name)}'>원본</a></h3>" + kvtable(j))

    # ── 수정 사항 · 지표 ───────────────────────────────────────
    S.append("<h2>수정 사항 · 지표</h2>")
    for label, key in (("delta (전송)", "delta"), ("payload", "payload"),
                       ("seam (이음새)", "seam"), ("snap", "snap"),
                       ("mask (meta 기록분)", "mask")):
        if m.get(key):
            v = dict(m[key])
            if key == "mask":
                v["chunk_keys"] = f"{len(v.get('chunk_keys') or [])}개"
            S.append(f"<h3 class=muted>{label}</h3>" + kvtable(v))
    for lst in ("reused_chunks_list", "newly_active_chunks"):
        if m.get(lst):
            S.append(f"<h3 class=muted>{lst} ({len(m[lst])}개)</h3>"
                     f"<p class=muted style='word-break:break-all'>{esc(' '.join(m[lst]))}</p>")
    if m.get("comparisons"):
        rows = "".join(
            f"<tr><td>{esc(c.get('metric'))}</td>"
            f"<td>{esc(c.get('baseline_label'))}: {c.get('baseline'):,}</td>"
            f"<td>{esc(c.get('value_label'))}: {c.get('value'):,}</td>"
            f"<td>{esc(c.get('unit'))}</td><td class=muted>{esc(c.get('note'))}</td></tr>"
            for c in m["comparisons"])
        S.append("<h3 class=muted>comparisons</h3><table><tr><th>지표</th><th>기준</th>"
                 f"<th>값</th><th>단위</th><th>비고</th></tr>{rows}</table>")

    # ── 로그 · 파일 ────────────────────────────────────────────
    lg = tail(run_id)
    if lg:
        S.append(f"<h2>log.txt (끝부분) <a href='{fileurl(run_id, 'log.txt')}'>전체</a></h2>"
                 f"<pre>{esc(lg)}</pre>")

    files = list_files(run_id)
    nch = sum(1 for r, _ in files if r.startswith("chunks" + os.sep))
    shown = [(r, s) for r, s in files if not r.startswith("chunks" + os.sep)]
    rows = "".join(f"<tr><td><a href='{fileurl(run_id, r)}'>{esc(r)}</a></td>"
                   f"<td class=muted>{human(s)}</td></tr>" for r, s in shown)
    S.append(f"<h2>생성 파일 ({len(files)}개 · chunks/*.cbin {nch}개는 접었다)</h2>"
             f"<table><tr><th>경로</th><th>크기</th></tr>{rows}</table>")

    S.append(f"<h2>meta.json 전체 <a href='{fileurl(run_id, 'meta.json')}'>원본</a></h2>"
             f"<pre>{esc(json.dumps(m, ensure_ascii=False, indent=2))}</pre>")

    return page(f"{run_id} — lab debug", "".join(S))
