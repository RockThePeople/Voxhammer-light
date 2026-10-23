"""터미널 표시 — ASCII 청크 맵과 델타 표. SSH 로 이게 제일 빠르다."""
import json
import os

from deltacontract import chunk_key, parse_chunk_key
from deltacontract.coords import CHUNK_GRID_RES

from .store import read_meta, run_dir

EMPTY, SAME, CHANGED, MASKED, REMOVED = "·", "▪", "▲", "■", "×"


def load_chunks(run_id):
    p = os.path.join(run_dir(run_id), "manifest.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        man = json.load(f)
    return {k: v for k, v in man["chunks"].items()}


def chunk_map(present, masked=frozenset(), changed=frozenset(), removed=frozenset(),
              per_row=3):
    """청크 격자를 z 층별로 찍는다. 행=x, 열=y.

    ⚠️ 격자 크기는 CHUNK_GRID_RES 에서 유도한다 — 값으로 박으면 D75 같은 전환에서
    조용히 절반만 그린다.
    """
    out = [f"청크 맵 (z 층별,  {EMPTY} 비어있음  {SAME} 변화없음  "
           f"{CHANGED} 변경  {MASKED} 마스크  {REMOVED} 제거)", ""]

    # 완전히 빈 z 층은 접는다. 전 층을 다 찍으면 show 출력이 한 화면을
    # 넘는데, 그 중 절반이 점만 찍힌 층인 경우가 흔하다.
    def occupied(z):
        return any(chunk_key((x, y, z)) in present
                   for x in range(CHUNK_GRID_RES) for y in range(CHUNK_GRID_RES))

    layers = [z for z in range(CHUNK_GRID_RES) if occupied(z)]
    empty = [z for z in range(CHUNK_GRID_RES) if z not in layers]
    if empty:
        runs, cur = [], [empty[0]]
        for z in empty[1:]:
            (cur.append(z) if z == cur[-1] + 1 else (runs.append(cur), cur := [z]))
        runs.append(cur)
        desc = ", ".join(f"z={r[0]}" if len(r) == 1 else f"z={r[0]}~{r[-1]}" for r in runs)
        out.append(f"  {desc}: (비어있음)")
        out.append("")
    if not layers:
        return "\n".join(out)

    for start in range(0, len(layers), per_row):
        group = layers[start:start + per_row]
        out.append("   " + "   ".join(f"z={z}".ljust(CHUNK_GRID_RES + 2) for z in group).rstrip())
        _hdr = "".join(str(y % 10) for y in range(CHUNK_GRID_RES))
        out.append("   " + "   ".join(_hdr.ljust(CHUNK_GRID_RES + 4) for _ in group).rstrip())
        for x in range(CHUNK_GRID_RES):
            row = []
            for z in group:
                cells = []
                for y in range(CHUNK_GRID_RES):
                    k = chunk_key((x, y, z))
                    if k in removed:
                        cells.append(REMOVED)
                    elif k in masked:
                        cells.append(MASKED)
                    elif k in changed:
                        cells.append(CHANGED)
                    elif k in present:
                        cells.append(SAME)
                    else:
                        cells.append(EMPTY)
                row.append(f"{x} " + "".join(cells) + "   ")
            out.append(" " + "  ".join(row).rstrip())
        out.append("")
    return "\n".join(out)


def _mib(n):
    return f"{n / 1024 / 1024:.2f} MiB"


def diff_table(a_id, b_id, with_patch=None):
    """원본 대비 무엇이 어떻게 바뀌었나. 숫자를 미화하지 않는다."""
    ma, mb = read_meta(a_id), read_meta(b_id)
    ca, cb = load_chunks(a_id), load_chunks(b_id)
    ka, kb = set(ca), set(cb)

    added = sorted(kb - ka)
    removed = sorted(ka - kb)
    common = sorted(ka & kb)
    changed = [k for k in common if ca[k]["hash"] != cb[k]["hash"]]
    same_topo = [k for k in common
                 if ca[k]["vertex_count"] == cb[k]["vertex_count"]
                 and ca[k]["index_count"] == cb[k]["index_count"]]

    total_b = sum(v["byte_length"] for v in cb.values())
    changed_b = sum(cb[k]["byte_length"] for k in changed) + \
        sum(cb[k]["byte_length"] for k in added)

    book = set(((mb or {}).get("mask") or {}).get("chunk_keys", []))
    dl = (mb or {}).get("delta") or {}
    sm = (mb or {}).get("seam") or {}

    sent_n = dl.get("changed_chunks")
    sent_b = dl.get("changed_bytes")
    rows = [
        ("청크 수", f"{len(ca)} → {len(cb)} (신규 {len(added)}, 제거 {len(removed)})"),
        ("전체 바이트", _mib(total_b)),
    ]
    if sent_n is not None:
        # 계약 3.4.0 — 해시를 비교하지 않는다. 부기가 무엇이 바뀌었는지를 정의한다.
        rows += [
            ("부기 청크", f"{dl.get('bookkeeping_chunks')}개 (마스크+halo)"),
            ("전송 청크", f"{sent_n}/{dl.get('total_chunks')} "
                       f"(승계 {dl.get('reused_chunks')}, "
                       f"부기 밖 {dl.get('missed_by_bookkeeping')})"),
            ("전송 바이트", f"{_mib(sent_b)} / {_mib(total_b)} "
                        f"({sent_b / max(total_b, 1) * 100:.1f}%)"),
        ]
    if sm.get("max") is not None:
        rows += [
            ("이음새 면", f"{sm['seam_faces']}개, 경계 정점 {sm['boundary_vertices']:,}"),
            ("이음새 max", f"{sm['max']:.3e}  ({sm['cells_max']:.2f} 셀)"),
            ("이음새 median", f"{sm['median']:.3e}  ({sm['cells_median']:.2f} 셀)"),
            ("이음새 p99", f"{sm['p99']:.3e}"),
            ("1셀 초과 정점", f"{sm['over_cell']:,}/{sm['boundary_vertices']:,} "
                          f"({sm['over_cell_ratio'] * 100:.1f}%)"),
        ]
    lp = (mb or {}).get("latent_preserved")
    if lp is not None:
        rows.append(("latent 보존",
                     f"{(mb or {}).get('preserved_rows', '?')} 비트 동일 "
                     + ("✓" if lp else "✗ 실패")))
    if with_patch:
        rows.append(("패치 크기(zstd-19)", with_patch))

    w = max(len(r[0]) for r in rows)
    lines = [f"{a_id}  →  {b_id}", ""]
    lines += [f"  {k:<{w}}  {v}" for k, v in rows]
    if sent_n is not None:
        lines += [
            "",
            "  참고: 부기 밖 청크는 새 디코딩 결과를 버리고 부모 바이트를 승계한다 (계약 3.4.0).",
            "        해시를 비교하지 않으므로 디코딩 결정성은 불필요하다.",
            "        대가는 이음새 하나뿐이다 — 메시 셀 한 변 = 3.906e-03.",
        ]
    return "\n".join(lines), {
        "added": added, "removed": removed, "changed": changed,
        "same_topo": same_topo, "total_bytes": total_b,
        "changed_bytes": changed_b, "bookkeeping": sorted(book),
        "sent_chunks": sent_n, "sent_bytes": sent_b, "seam": sm,
    }
