"""제품 경로의 계산 코어.

**lab 과 같은 파이프라인을 부른다.** lab/pipeline.py 의 primitive(load_pipe,
decode_and_chunk)와 lab/seam.py 의 부기 합성·스냅·이음새 측정을 그대로 재사용하고,
여기서는 자산/버전 레이아웃에 맞춰 호출만 한다.

lab 은 건드리지 않는다. 대신 decode_and_chunk 가 요구하는 Run 인터페이스를
얇은 shim 으로 흉내낸다 — 로그·렌더 없이 타이밍만 걷는다.
"""
import json
import os
import time

import numpy as np
import safetensors
import torch
from PIL import Image
from safetensors.torch import load_file

from deltacontract import (
    COORD_ORDER,
    add_batch,
    CONTRACT_VERSION,
    MaskEmpty,
    affected_chunks,
    mask_fingerprint,
    bbox_to_voxel_cells,
    dilate_cells,
    strip_batch,
    voxel_code,
)
from deltacontract.coords import canonical_order
from deltacontract.assemble import (
    assemble_zones, crop_rows, fit_offset, place_cells, split_removed)

from lab import seam
from lab.pipeline import decode_and_chunk, load_pipe, save_slat, set_determinism

STAGES = ("load", "structure", "slat", "repaint", "decode", "chunk")

# 제품 경로는 렌더가 필요 없다(요청마다 GPU·디스크 낭비). 그런데 lab 의
# render_views 는 `views or DEFAULT_VIEWS` 라 빈 리스트가 falsy 로 떨어져 기본
# 4각도로 되돌아간다. lab 을 고치지 않기로 했으므로 VIEWS 에 없는 키를 넘겨
# plan 을 비운다.
NO_RENDER = ["__off__"]


class _Run:
    """decode_and_chunk 가 기대하는 최소 인터페이스. 렌더는 views=[] 로 끈다."""

    def __init__(self, out_dir, progress=None):
        self.dir = out_dir
        self.run_id = os.path.basename(out_dir)
        self.meta = {}
        self.timings = {}
        self._progress = progress
        os.makedirs(os.path.join(out_dir, "chunks"), exist_ok=True)

    def begin(self, key):
        if self._progress:
            self._progress(key)

    def stage(self, key, label, detail="", secs=None):
        if secs is not None:
            self.timings[key] = secs * 1000.0     # ms
        if self._progress:
            self._progress(key)

    def log(self, *a, **k):
        pass

    def warn(self, *a, **k):
        pass

    def set(self, **kw):
        self.meta.update(kw)

    def path(self, *p):
        return os.path.join(self.dir, *p)


def _load_slat(path):
    d = load_file(path)
    with safetensors.safe_open(path, "pt") as f:
        meta = f.metadata() or {}
    # 계약대로 로드 시점에 검증한다. native 순서로 디코딩하면 정점 수 자체가
    # 다른 메시가 나온다 (2차 실험 C).
    if meta.get("coord_order") != COORD_ORDER:
        raise ValueError(
            f"coord_order 불일치: 저장={meta.get('coord_order')!r} 계약={COORD_ORDER!r}")
    if meta.get("slat_space") != "denormalized":
        raise ValueError(f"slat_space 가 denormalized 가 아니다: {meta.get('slat_space')!r}")
    return d, meta


def mask_rows(spec, xyz, halo):
    """EditMask -> (행 마스크, 부기 청크 키, 지문). 부기가 '무엇이 바뀌었는가'의 정의다."""
    if spec.mode == "bbox":
        core = bbox_to_voxel_cells(spec.bbox_min, spec.bbox_max)
    else:
        core = np.asarray(spec.voxels, dtype=np.int64).reshape(-1, 3)
    cells = dilate_cells(core, halo)
    active = set(voxel_code(xyz).tolist())
    codes = set(voxel_code(cells).tolist()) & active
    if not codes:
        raise MaskEmpty("마스크가 활성 복셀을 하나도 덮지 않는다")
    rows = np.isin(voxel_code(xyz), np.fromiter(codes, np.int64, len(codes)))
    # 부기는 마스크+halo **전체**에서 계산한다. 그 청크에 실제로 삼각형이
    # 생기는지는 모르고, 안 생기면 removed_chunk_ids 로 간다 (계약 3.4.3 전칭 규칙).
    book = affected_chunks(None, None, cells)
    # 지문은 **halo 적용 전 원본 셀**로 낸다 — 요청이 같은지를 보려는 것이고,
    # halo 는 서버가 붙이는 값이라 요청 동일성과 무관하다.
    fp = {"raw": mask_fingerprint(core), "dilated": mask_fingerprint(cells),
          "raw_cells": int(np.asarray(core).reshape(-1, 3).shape[0]),
          "dilated_cells": int(cells.shape[0]), "halo": int(halo),
          "mode": spec.mode}
    return rows, book, fp



# ══════════════════════════════════════════════════ 승계 검증 · 매니페스트
MESH_CELL = 1.0 / 256.0


def _chunk_drift(fresh_path, parent_path):
    """신선 디코드 바이트 vs 부모 바이트의 기하 거리 (메시 셀). 없으면 None.

    ★ 왜 필요한가 — **승계가 자기 검증을 지운다.**
    부기 밖 청크는 부모 바이트를 그대로 복사하므로 "밖 바이트 100% 동일"은
    항상 참이다. 그건 재현성의 증거가 아니라 승계의 정의다. "이번 연산이 먼
    청크를 실제로 흔들었는가"는 **덮어쓰기 전 신선 바이트로만** 잴 수 있고,
    지금까지는 랩에서 따로 돌려야 알 수 있었다.

    ⚠️ 해시로는 못 잰다. 재디코딩은 프로세스가 바뀌면 바이트가 재현되지 않는다 —
    기증자 없이 base 만 다시 디코딩한 대조군에서 152/152 가 전부 다른 해시를
    냈고, 기하 차이는 중앙값 0.0002 셀이었다. 부동소수 잡음이다.
    """
    from scipy.spatial import cKDTree

    from deltacontract import decode
    try:
        a = decode(open(fresh_path, "rb").read()).positions.astype(np.float64)
        b = decode(open(parent_path, "rb").read()).positions.astype(np.float64)
    except (OSError, ValueError):
        return None
    if a.shape[0] == 0 or b.shape[0] == 0:
        return None
    d, _ = cKDTree(b).query(a)
    return float(np.median(d) / MESH_CELL)


def _drift_stats(vals):
    """임계값을 두지 않는다 — 어느 선을 넘으면 승계가 위험한지 **아직 모른다.**
    값을 쌓아야 알 수 있으므로 지금은 기록만 한다."""
    if not vals:
        return {}
    a = np.asarray(vals, dtype=np.float64)
    return {"inherited_chunks": float(a.size),
            "inherited_drift_median_cells": float(np.median(a)),
            "inherited_drift_p95_cells": float(np.percentile(a, 95)),
            "inherited_drift_max_cells": float(a.max())}


def write_manifest(out_dir, asset_id, version, entries, sent=()):
    """staging 에 **합성물의** 매니페스트를 쓴다. 정본이 되는 파일이다.

    🔴 decode_and_chunk 가 먼저 쓰는 manifest.json 은 **승계 합성 전 전체
    재디코딩**을 기술한다 (asset_id=staging 키, version=1, 승계 청크의 해시·
    byte_length 가 실제 바이트와 다르다 — 실측 102/147 불일치).

    그 파일이 그대로 남으면 조용히 틀리는 데 그치지 않는다:
    `store.commit` 이 staging 디렉터리를 **통째로 v{n} 으로 옮기므로**
    (`shutil.move`), 그 매니페스트가 곧 **버전 매니페스트**가 된다. 그리고
    다음 편집이 그걸 `parent_man` 으로 읽어 승계 청크의 해시를 거기서 베낀다
    ⇒ 3090 이 받는 해시가 실제 바이트와 어긋난다. 예외는 안 난다.

    지금까지 안 터진 이유는 검증한 편집이 전부 ephemeral 이라 commit 을 한 번도
    안 탔기 때문이다. **첫 커밋에서 터진다.**

    ⚠️ 엔트리별 `version` 을 여기서 확정한다. `decode_and_chunk` 는 전부 1 로 박고
    나가는데(자기가 몇 번째 버전인지 모른다), 승계 청크는 그게 맞고 **새로 쓴 청크는
    틀리다.** 그대로 커밋되면 v2 바이트가 v1 로 표기되어 3090 이 `.v1.cbin` 을
    되받으러 간다 — 매니페스트에 버전이 섞이는 건 정상이지만 **틀리게** 섞이면 안 된다.
    """
    from deltacontract.schemas import ChunkManifest

    sent = set(sent)
    man = ChunkManifest(
        asset_id=asset_id, version=int(version),
        chunks={k: (v.model_copy(update={"version": int(version)}) if k in sent else v)
                for k, v in entries.items()})
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        f.write(man.model_dump_json(indent=2))


# ══════════════════════════════════════════════════════════ generate
def op_generate(out_dir, image_path, seed, progress=None):
    run = _Run(out_dir, progress)
    set_determinism(False)
    pipe = load_pipe(run)

    img = Image.open(image_path)
    img_pre = pipe.preprocess_image(img)
    img_pre.save(os.path.join(out_dir, "input.png"))
    cond = pipe.get_cond([img_pre])
    torch.manual_seed(seed)

    run.begin("structure")
    t = time.time()
    coords = pipe.sample_sparse_structure(cond, 1, {})
    torch.cuda.synchronize()
    run.timings["structure"] = (time.time() - t) * 1000

    run.begin("slat")
    t = time.time()
    slat = pipe.sample_slat(cond, coords, {})
    torch.cuda.synchronize()
    run.timings["repaint"] = (time.time() - t) * 1000   # generate 는 slat 샘플링이 그 자리

    c = slat.coords.detach().cpu().to(torch.int32)
    f = slat.feats.detach().cpu().float()
    order = canonical_order(strip_batch(c.numpy()))
    c, f = c[order].contiguous(), f[order].contiguous()
    save_slat(os.path.join(out_dir, "slat.safetensors"), c, f, {
        "slat_space": "denormalized", "coord_order": COORD_ORDER,
        "contract_version": CONTRACT_VERSION, "voxel_res": 64, "slat_channels": 8,
        "seed": seed,
        "norm_mean": json.dumps(pipe.slat_normalization["mean"]),
        "norm_std": json.dumps(pipe.slat_normalization["std"]),
    })

    xyz = strip_batch(c.numpy())
    _, _, entries, total = decode_and_chunk(
        run, pipe, c.cuda().int(), f.cuda().float(), xyz, views=NO_RENDER)
    return {
        "entries": entries, "removed": [], "book": sorted(entries),
        "timings": run.timings, "snapped": 0, "seam": None,
        "voxels": int(xyz.shape[0]),
    }


# ══════════════════════════════════════════════════════════ edit
def op_edit(parent_dir, out_dir, spec, seed, snap_cells=2.0, progress=None):
    """부기 기반 편집. 부기 밖 청크는 부모 바이트를 승계한다 (계약 3.4.0)."""
    import sys

    run = _Run(out_dir, progress)
    set_determinism(False)
    pipe = load_pipe(run)

    d, meta = _load_slat(os.path.join(parent_dir, "slat.safetensors"))
    coords = d["coords"].cuda().int()
    feats = d["feats"].cuda().float()
    xyz = strip_batch(coords.cpu().numpy())

    rows, book, mask_fp = mask_rows(spec, xyz, spec.halo_margin_voxels)
    run.begin("repaint")

    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "repaint"))
    from masked_sampler import MaskedSlatSampler

    cond = pipe.get_cond([Image.open(os.path.join(parent_dir, "input.png"))])
    p = pipe.slat_sampler_params
    mean = torch.tensor(json.loads(meta["norm_mean"]), device="cuda")[None].float()
    std = torch.tensor(json.loads(meta["norm_std"]), device="cuda")[None].float()
    sampler = MaskedSlatSampler(pipe.slat_sampler, steps=p["steps"],
                                rescale_t=p["rescale_t"])
    t = time.time()
    out, _info = sampler.sample(
        pipe.models["slat_flow_model"], coords, (feats - mean) / std,
        torch.as_tensor(rows, device="cuda"), cond["cond"], cond["neg_cond"],
        cfg_strength=p["cfg_strength"], cfg_interval=p["cfg_interval"], seed=seed,
        mean=mean, std=std, x0_ref_denorm=feats, verbose=False)
    torch.cuda.synchronize()
    run.timings["repaint"] = (time.time() - t) * 1000

    # §9-2 — 마스크 밖 latent 는 비트 단위로 보존돼야 한다.
    keep = ~torch.as_tensor(rows, device="cuda")
    ndiff = int((out[keep] != feats[keep]).any(1).sum())
    if ndiff:
        raise RuntimeError(f"latent 보존 실패: 마스크 밖 {ndiff}/{int(keep.sum())} 행 상이")

    save_slat(os.path.join(out_dir, "slat.safetensors"), coords, out,
              {**meta, "edit_seed": seed})
    # 편집 체인이 이어지려면 다음 편집도 조건화 이미지가 필요하다.
    src = os.path.join(parent_dir, "input.png")
    if os.path.exists(src):
        with open(src, "rb") as fi, open(os.path.join(out_dir, "input.png"), "wb") as fo:
            fo.write(fi.read())

    _, _, entries, _ = decode_and_chunk(run, pipe, coords, out, xyz, views=NO_RENDER)

    # ── 부기 합성: 부기 밖은 부모 바이트 승계 ──────────────────────
    parent_man = json.load(open(os.path.join(parent_dir, "manifest.json")))["chunks"]
    bookset = set(book)
    final, reused, sent, drift = {}, [], [], []
    for k, e in entries.items():
        if k in bookset or k not in parent_man:
            final[k] = e
            sent.append(k)
        else:
            # ★ 덮어쓰기 **전에** 잰다. 이 값만이 "먼 청크가 흔들렸는가"를 답한다.
            fp = os.path.join(out_dir, "chunks", f"{k}.cbin")
            pp = os.path.join(parent_dir, "chunks", f"{k}.cbin")
            dv = _chunk_drift(fp, pp)
            if dv is not None:
                drift.append(dv)
            with open(pp, "rb") as fi, open(fp, "wb") as fo:
                fo.write(fi.read())
            final[k] = e.model_copy(update=parent_man[k])
            reused.append(k)

    snapped = 0
    if snap_cells > 0:
        from deltacontract import blob_hash

        st = seam.snap_boundary(out_dir, parent_dir, bookset, set(final),
                                set(parent_man), cells=snap_cells)
        snapped = st["snapped"]
        for k in sent:
            fp = os.path.join(out_dir, "chunks", f"{k}.cbin")
            if os.path.exists(fp):
                b = open(fp, "rb").read()
                final[k] = final[k].model_copy(
                    update={"hash": blob_hash(b), "byte_length": len(b)})

    seam_stats = seam.measure_seams(out_dir, parent_dir, bookset, set(final),
                                    set(parent_man))

    # ★ 부기 가지치기 (계약 3.6.0)
    #
    # 마스크+halo 를 청크로 접으면 오브젝트가 없는 공간까지 포함된다. bbox 마스크
    # 하나로 부기 256, 실제 청크 39, removed 217 이 나왔는데 그 217 중 대부분은
    # base 에도 새 디코딩에도 없는 **유령**이었다.
    #
    #     보고할 부기 = 원래 부기 ∩ (base 매니페스트 청크 ∪ 새 디코딩 청크)
    #
    # ⚠️ 디코딩 **후에** 자른다. 활성 복셀로 미리 자르면 안 된다 — partition_mesh 가
    # 삼각형을 **무게중심**으로 배정하므로 활성 복셀이 없는 청크에도 삼각형이 생긴다.
    existing = set(parent_man) | set(entries)
    book_reported = sorted(k for k in book if k in existing)

    # 계약 3.4.3 전칭 규칙: 부기 청크는 chunks 아니면 removed 에 반드시 들어간다.
    # 기하가 없으면 removed. "아무 데도 안 넣기"는 BOOKKEEPING_MISMATCH 로 거부된다.
    removed = sorted(k for k in book_reported if k not in final)
    return {
        "entries": final, "removed": removed, "book": book_reported,
        "book_raw": len(book), "sent": sent, "reused": reused,
        "timings": run.timings, "masked_rows": int(rows.sum()), "mask_fp": mask_fp,
        "snapped": snapped, "seam": seam_stats,
        "voxels": int(xyz.shape[0]), "drift": _drift_stats(drift),
    }


# ══════════════════════════════════════════════════════════ assemble
def op_assemble(parent_dir, donor_dir, out_dir, spec, crop, offset,
                snap_cells=2.0, progress=None):
    """조립 — 마스크 안을 **비우고** 기증자 조각을 끼워 넣는다 (계약 3.19.0).

    편집(op_edit)과 다른 점 셋. 전부 실측에서 나온 것이고 계약 함수가 강제한다:

      1. 샘플러를 안 탄다. `coords 고정` 제약(masked_sampler.py:15) 밖이라
         **복셀이 늘어난다** — 그게 이 연산의 존재 이유다.
      2. 부기를 **배치에서** 유도한다 (assemble_zones). diff 로 만들면 비워진
         청크가 목록에서 빠지고 옛 기하를 들고 남는다.
      3. 검증을 해시로 하지 않는다. 재디코딩은 바이트가 재현되지 않는다
         (대조군 152/152 불일치, 기하 차이는 0.0002 메시셀).

    ⚠️ OVERLAP(겹쳐 넣기)은 구현하지 않는다. 좌표 충돌 시 어느 latent 를 쓰는지가
       계약에 없다 (3.17.0 실측 806건). 정의되기 전에는 REMOVE 만이 결정적이다.
    """
    run = _Run(out_dir, progress)
    set_determinism(False)
    pipe = load_pipe(run)

    bd, bmeta = _load_slat(os.path.join(parent_dir, "slat.safetensors"))
    dd, _ = _load_slat(os.path.join(donor_dir, "slat.safetensors"))
    bxyz = strip_batch(bd["coords"].cpu().numpy())
    dxyz = strip_batch(dd["coords"].cpu().numpy())
    bfe, dfe = bd["feats"].float(), dd["feats"].float()

    run.begin("repaint")          # 어휘가 이 단계를 안 덮는다 — §stage 갭으로 보고됨
    t = time.time()

    # ── 기증자: 크롭 → 배치. 스케일은 없다 ────────────────────────────
    sel = crop_rows(dxyz, crop["fraction"], axis=crop.get("axis", 2),
                    keep=crop.get("keep", "top"))
    dsub, dsub_f = dxyz[sel], dfe[torch.as_tensor(np.flatnonzero(sel))]
    # 마스크 셀은 op_edit 과 **같은 함수**로 낸다 — 두 경로가 마스크를 다르게
    # 해석하면 3090 이 같은 요청에 다른 부기를 받는다.
    mrows, _, mask_fp = mask_rows(spec, bxyz, spec.halo_margin_voxels)
    seat = bxyz[mrows]
    off = offset if offset is not None else fit_offset(dsub, seat)
    placed = place_cells(dsub, off)         # 정수·범위·중복을 여기서 거부한다

    # ── base: 마스크 안을 비우고, 기증자와 겹치는 행도 뺀다 ───────────
    pcodes = set(voxel_code(placed).tolist())
    collide = np.isin(voxel_code(bxyz), np.fromiter(pcodes, np.int64, len(pcodes)))
    keep = ~mrows & ~collide
    emptied = bxyz[~keep]                   # ② 비워진 자리 (충돌분 포함)

    mxyz = np.concatenate([bxyz[keep], placed])
    mfe = torch.cat([bfe[torch.as_tensor(np.flatnonzero(keep))], dsub_f])
    if np.unique(mxyz, axis=0).shape[0] != mxyz.shape[0]:
        raise RuntimeError("병합 후 중복 좌표 — 배치 검증이 통과했는데 여기서 났다")
    order = canonical_order(mxyz)
    mxyz, mfe = mxyz[order], mfe[order]
    run.timings["repaint"] = (time.time() - t) * 1000

    coords = torch.as_tensor(add_batch(mxyz), device="cuda").int()
    feats = mfe.cuda().float()
    save_slat(os.path.join(out_dir, "slat.safetensors"), coords.cpu(), feats.cpu(), {
        **bmeta, "assembled_from": os.path.basename(donor_dir),
        "assemble_offset": json.dumps([int(v) for v in off]),
        "assemble_crop": json.dumps(crop),
    })
    src = os.path.join(parent_dir, "input.png")
    if os.path.exists(src):
        with open(src, "rb") as fi, open(os.path.join(out_dir, "input.png"), "wb") as fo:
            fo.write(fi.read())

    _, _, entries, _ = decode_and_chunk(run, pipe, coords, feats, mxyz, views=NO_RENDER)

    # ── 부기: **배치에서** 유도한다 (assemble_zones) ──────────────────
    zones = assemble_zones(placed, emptied)
    book = zones["book"]
    parent_man = json.load(open(os.path.join(parent_dir, "manifest.json")))["chunks"]

    # 가지치기: 부기 ∩ (부모에 있음 ∪ 새로 나옴). 3.6.0 과 같은 규칙.
    existing = set(parent_man) | set(entries)
    book_reported = [k for k in book if k in existing]
    bookset = set(book_reported)

    final, sent, reused, drift = {}, [], [], []
    for k, e in entries.items():
        if k in bookset or k not in parent_man:
            final[k] = e
            sent.append(k)
        else:
            # ⚠️ 승계. 새로 디코딩한 바이트를 쓰면 안 된다 — 재디코딩은 재현되지
            # 않아서 "안 바뀐 청크"가 전부 바뀐 것으로 나간다.
            # ★ 덮어쓰기 **전에** drift 를 잰다 (승계가 지우는 유일한 증거다).
            fp = os.path.join(out_dir, "chunks", f"{k}.cbin")
            pp = os.path.join(parent_dir, "chunks", f"{k}.cbin")
            dv = _chunk_drift(fp, pp)
            if dv is not None:
                drift.append(dv)
            with open(pp, "rb") as fi, open(fp, "wb") as fo:
                fo.write(fi.read())
            final[k] = e.model_copy(update=parent_man[k])
            reused.append(k)
    # 부기에 있는데 기하가 없는 청크 = 비워진 자리. 승계하면 옛 머리가 남는다.
    for k in bookset:
        if k not in entries:
            p = os.path.join(out_dir, "chunks", f"{k}.cbin")
            if os.path.exists(p):
                os.remove(p)
            final.pop(k, None)

    snapped = 0
    if snap_cells > 0:
        from deltacontract import blob_hash
        st = seam.snap_boundary(out_dir, parent_dir, bookset, set(final),
                                set(parent_man), cells=snap_cells)
        snapped = st["snapped"]
        for k in sent:
            fp = os.path.join(out_dir, "chunks", f"{k}.cbin")
            if os.path.exists(fp):
                b = open(fp, "rb").read()
                final[k] = final[k].model_copy(
                    update={"hash": blob_hash(b), "byte_length": len(b)})

    seam_stats = seam.measure_seams(out_dir, parent_dir, bookset, set(final),
                                    set(parent_man))
    _, removed = split_removed(book_reported, final)
    return {
        "entries": final, "removed": removed, "book": book_reported,
        "book_raw": len(book), "sent": sent, "reused": reused,
        "timings": run.timings, "masked_rows": int(mrows.sum()), "mask_fp": mask_fp,
        "snapped": snapped, "seam": seam_stats, "voxels": int(mxyz.shape[0]),
        "drift": _drift_stats(drift),
        "zones": zones, "offset": [int(v) for v in off],
        "donor_voxels": int(placed.shape[0]), "emptied_voxels": int(emptied.shape[0]),
    }
