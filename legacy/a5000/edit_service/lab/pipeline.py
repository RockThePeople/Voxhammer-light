"""gen / edit 실행 코어. 모델 로드·디코딩·청크 분할·렌더를 한 자리에 모은다."""
import json
import os
import sys
import time

import numpy as np
import safetensors
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from deltacontract import (
    CONTRACT_CONSTANTS,
    CONTRACT_VERSION,
    COORD_ORDER,
    ChunkEntry,
    ChunkManifest,
    EditMask,
    MaskEmpty,
    affected_chunks,
    albedo_to_rgba8,
    assert_in_normalized_bounds,
    blob_hash,
    chunk_key,
    dilate_cells,
    encode,
    partition_mesh,
    split_trellis_vertex_attrs,
    strip_batch,
    voxel_code,
    voxel_to_chunk,
)
from deltacontract.coords import canonical_order

from . import regions, render, seam
from .store import Run, run_dir

MODEL_ID = "microsoft/TRELLIS-image-large"
_PIPE = None


def set_determinism(on):
    """계약 3.4.0: 결정성 플래그는 더 이상 전제가 아니다. 기본 off.

    3.3.x 까지는 "안 바뀐 청크는 바이트 동일"을 **해시 비교**로 판정했기 때문에
    디코딩이 결정적이어야 했다. 3.4.0 은 무엇이 바뀌었는지를 **부기(마스크+halo)가
    정의**하므로 해시를 비교하지 않는다 — 결정성이 필요 없다.

    끄면 연산 -21%, 그리고 4차에서 발견한 "프로세스 내 반복 디코딩이 결정성을
    깬다" 문제가 소멸한다. --deterministic 은 진단용으로만 남긴다.
    """
    if on:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(bool(on), warn_only=True)
    torch.backends.cudnn.deterministic = bool(on)
    torch.backends.cudnn.benchmark = not on


def _silence_sampler_tqdm():
    """업스트림 샘플러의 진행바를 끈다.

    flow_euler.sample() 은 `disable=not verbose` 로 **False 를 명시**하므로
    TQDM_DISABLE 환경변수가 통하지 않는다. verbose 도 하드코딩(trellis_image_to_3d.py
    :216)이라 인자로 못 끈다. 벤더 트리를 수정하지 않는다는 원칙대로 모듈 심볼만
    런타임에 갈아끼운다. 단계별 소요시간은 우리가 따로 찍는다.
    """
    from trellis1.pipelines.samplers import flow_euler

    def _quiet(it, *a, **k):
        return it

    flow_euler.tqdm = _quiet


def load_pipe(run):
    global _PIPE
    if _PIPE is None:
        run.begin("model_load")
        t = time.time()
        from trellis1.pipelines import TrellisImageTo3DPipeline

        _silence_sampler_tqdm()

        _PIPE = TrellisImageTo3DPipeline.from_pretrained(MODEL_ID)
        _PIPE.cuda()
        run.stage("model_load", "모델 로드", MODEL_ID, time.time() - t)
    return _PIPE


# ══════════════════════════════════════════════════ 공통 뒷단
def decode_and_chunk(run, pipe, coords, feats, xyz, extra_renders=None,
                     views=None, extra_views=None):
    """디코딩 → 청크 분할 → manifest → 렌더. to_glb() 는 타지 않는다."""
    from trellis1.modules import sparse as sp

    run.begin("decode")
    t = time.time()
    with torch.no_grad():
        mesh = pipe.decode_slat(
            sp.SparseTensor(feats=feats, coords=coords), formats=["mesh"]
        )["mesh"][0]
    torch.cuda.synchronize()
    run.stage("decode", "디코딩", f"→ V={mesh.vertices.shape[0]:,} F={mesh.faces.shape[0]:,}",
              time.time() - t)

    verts = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    vattrs = mesh.vertex_attrs.detach().cpu().numpy()
    assert_in_normalized_bounds(verts)
    albedo, normal = split_trellis_vertex_attrs(vattrs)

    run.begin("chunk")
    t = time.time()
    chunks = partition_mesh(verts, faces, normals=normal,
                            colors_rgba8=albedo_to_rgba8(albedo), voxel_cells=xyz)
    entries, total = {}, 0
    for k in sorted(chunks):
        blob = encode(chunks[k])
        with open(run.path("chunks", f"{k}.cbin"), "wb") as f:
            f.write(blob)
        total += len(blob)
        entries[k] = ChunkEntry(
            uri=f"chunks/{k}.cbin", hash=blob_hash(blob), byte_length=len(blob),
            vertex_count=chunks[k].vertex_count, index_count=chunks[k].index_count,
            voxel_count=chunks[k].voxel_count, version=1,
        )
    man = ChunkManifest(asset_id=run.run_id, version=1, chunks=entries)
    with open(run.path("manifest.json"), "w") as f:
        f.write(man.model_dump_json(indent=2))
    run.stage("chunk", "청크 분할",
              f"→ {len(entries)} 청크 / 512 슬롯, {total / 1024 / 1024:.2f} MiB",
              time.time() - t)

    run.begin("render")
    t = time.time()
    rends = {"base": render.render_views(mesh, run.path("render"), "base",
                                         views=views, extra=extra_views)}
    for name, colors in (extra_renders or {}).items():
        rends[name] = render.render_views(mesh, run.path("render"), name,
                                          colors=colors(verts), views=views,
                                          extra=extra_views)
    run.stage("render", "렌더", f"{sum(len(v) for v in rends.values())}장", time.time() - t)

    # LAB_API.md §3 — 크기는 용도별로 분리하고 **미측정은 null** 로 둔다.
    # 0 으로 채우면 3090이 "재봤더니 0" 으로 찍는다.
    run.set(chunk_count=len(entries),
            payload={"encoding": "f32+normal+rgba8",
                     "raw_bytes": total,
                     "compressed_bytes": None,
                     "glb_bytes": None},
            renders=rends,
            thumbs={k: v.get("front") or next(iter(v.values()), None)
                    for k, v in rends.items() if v},
            mesh={"vertices": int(verts.shape[0]), "faces": int(faces.shape[0])})
    return mesh, verts, entries, total


def save_slat(path, coords, feats, meta):
    save_file({"coords": coords.cpu().contiguous(), "feats": feats.cpu().contiguous()},
              path, metadata={k: str(v) for k, v in meta.items()})


# ══════════════════════════════════════════════════ gen
def run_gen(image_path, seed=42, name=None, deterministic=False, views=None):
    run = Run("gen")
    try:
        set_determinism(deterministic)
        run.set(input_image="input.png", input_image_src=os.path.abspath(image_path),
                seed=seed, name=name, deterministic=deterministic,
                contract=CONTRACT_CONSTANTS, mask=None, parent_run_id=None)
        img = Image.open(image_path)
        a = np.array(img)
        alpha = ("알파 유효" if img.mode == "RGBA" and not (a[:, :, 3] == 255).all()
                 else "알파 없음 → rembg 사용")
        run.stage("image_load", "이미지 로드", f"{img.size[0]}x{img.size[1]} {img.mode}, {alpha}")

        pipe = load_pipe(run)
        img_pre = pipe.preprocess_image(img)
        img_pre.save(run.path("input.png"))

        cond = pipe.get_cond([img_pre])
        torch.manual_seed(seed)

        run.begin("sparse_structure")
        t = time.time()
        coords = pipe.sample_sparse_structure(cond, 1, {})
        torch.cuda.synchronize()
        run.stage("sparse_structure", "sparse structure", f"→ 활성 복셀 {coords.shape[0]:,}", time.time() - t)

        run.begin("slat_sample")
        t = time.time()
        slat = pipe.sample_slat(cond, coords, {})
        torch.cuda.synchronize()
        run.stage("slat_sample", "slat 샘플링", f"→ {tuple(slat.feats.shape)}", time.time() - t)

        # canonical 정렬 (계약 COORD_ORDER)
        c_native = slat.coords.detach().cpu().to(torch.int32)
        f_native = slat.feats.detach().cpu().float()
        order = canonical_order(strip_batch(c_native.numpy()))
        c_sorted = c_native[order].contiguous()
        f_sorted = f_native[order].contiguous()
        save_slat(run.path("slat.safetensors"), c_sorted, f_sorted, {
            "slat_space": "denormalized", "coord_order": COORD_ORDER,
            "contract_version": CONTRACT_VERSION, "voxel_res": 64, "slat_channels": 8,
            "seed": seed, "image": os.path.basename(image_path),
            "norm_mean": json.dumps(pipe.slat_normalization["mean"]),
            "norm_std": json.dumps(pipe.slat_normalization["std"]),
        })
        run.stage("slat_save", "slat 저장", f"{COORD_ORDER} 순서, denormalized")

        xyz = strip_batch(c_sorted.numpy())
        decode_and_chunk(run, pipe, c_sorted.cuda().int(), f_sorted.cuda().float(),
                         xyz, views=views)
        run.set(voxels=int(xyz.shape[0]))
        run.done()
        return run.run_id
    except Exception as e:
        run.fail_loud(f"{type(e).__name__}: {e}")
        run.failed(e)
        raise


# ══════════════════════════════════════════════════ edit
def run_edit(parent_id, region, note=None, halo=1, seed=42, fraction=0.25,
             deterministic=False, jump=0, views=None, redecode=False, snap=0.0):
    run = Run("edit")
    try:
        set_determinism(deterministic)
        pdir = run_dir(parent_id)
        run.set(parent_run_id=parent_id, region=region, note=note, halo=halo,
                seed=seed, fraction=fraction, deterministic=deterministic,
                redecode=redecode, snap_cells=snap,
                contract=CONTRACT_CONSTANTS, input_image=None)
        if note:
            run.log(f"note: {note}")

        sp_path = os.path.join(pdir, "slat.safetensors")
        d = load_file(sp_path)
        with safetensors.safe_open(sp_path, "pt") as f:
            meta = f.metadata() or {}
        if meta.get("coord_order") != COORD_ORDER:
            raise RuntimeError(
                f"coord_order 불일치: 저장={meta.get('coord_order')!r} "
                f"계약={COORD_ORDER!r}. 순서가 다르면 디코더가 다른 메시를 낸다.")
        if meta.get("slat_space") != "denormalized":
            raise RuntimeError(f"slat_space 가 denormalized 가 아니다: {meta.get('slat_space')!r}")

        coords = d["coords"].cuda().int()
        feats = d["feats"].cuda().float()
        xyz = strip_batch(coords.cpu().numpy())
        n = xyz.shape[0]

        # ── 마스크 ──────────────────────────────────────────────────
        why = regions.fraction_ignored(region)
        if why and fraction != 0.25:
            run.warn(f"--fraction={fraction} 은 무시된다 — {why}")
        sel = regions.select(xyz, region, fraction)
        core = xyz[sel]
        if core.shape[0] == 0:
            raise MaskEmpty(f"region={region!r} 가 활성 복셀을 하나도 고르지 못했다")
        spec = EditMask(mode="voxels",
                        voxels=[tuple(int(v) for v in c) for c in core],
                        halo_margin_voxels=halo)
        cells = dilate_cells(core, halo)
        active = set(voxel_code(xyz).tolist())
        mcodes = set(voxel_code(cells).tolist()) & active
        if not mcodes:
            raise MaskEmpty(f"halo={halo} 확장 후에도 활성 복셀을 덮지 못했다")

        mrows = np.isin(voxel_code(xyz),
                        np.fromiter(mcodes, np.int64, len(mcodes)))
        core_codes = set(voxel_code(core).tolist())
        halo_only = mcodes - core_codes
        mchunks = affected_chunks(None, None, xyz[mrows])

        run.stage("mask", "마스크", f"region={region} fraction={fraction}")
        run.log(f"활성 복셀 {n:,} 중 {core.shape[0]:,} "
                f"({core.shape[0] / n * 100:.1f}%) 선택", indent=True)
        run.log(f"halo={halo} 확장 → {int(mrows.sum()):,} "
                f"({mrows.mean() * 100:.1f}%)", indent=True)
        run.log(f"덮는 청크 {len(mchunks)}개: {' '.join(mchunks[:8])}"
                + (" ..." if len(mchunks) > 8 else ""), indent=True)

        with open(run.path("mask.json"), "w") as f:
            json.dump({"spec": json.loads(spec.model_dump_json()),
                       "halo": halo, "region": region, "fraction": fraction,
                       "core_voxels": int(core.shape[0]),
                       "after_halo_voxels": int(mrows.sum()),
                       "chunks": mchunks}, f, indent=2)
        run.set(mask={"region": region, "mode": "voxels", "fraction": fraction,
                      "voxel_count": int(core.shape[0]),
                      "halo_margin_voxels": halo,
                      "after_halo": int(mrows.sum()),
                      "active_voxels_total": int(n),
                      "chunk_keys": mchunks, "prompt": note})

        # ── RePaint ────────────────────────────────────────────────
        pipe = load_pipe(run)

        if redecode:
            # 대조군 Z: 편집 없이 **같은 latent 를 재디코딩만** 한다.
            # 그래도 부기 합성은 그대로 태우므로, 디코딩이 프로세스 경계를 넘어
            # 재현되면 이음새가 정확히 0 이어야 한다. 0 이 아니면 그 크기가 곧
            # "비결정성이 이음새에 기여하는 몫"이다.
            out = feats
            run.stage("repaint", "재디코딩", "편집 없음 (대조군 Z)")
        else:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "repaint"))
            from masked_sampler import MaskedSlatSampler

            img_p = os.path.join(pdir, "input.png")
            cond = pipe.get_cond([Image.open(img_p)])
            p = pipe.slat_sampler_params
            mean = torch.tensor(json.loads(meta["norm_mean"]), device="cuda")[None].float()
            std = torch.tensor(json.loads(meta["norm_std"]), device="cuda")[None].float()

            sampler = MaskedSlatSampler(pipe.slat_sampler, steps=p["steps"],
                                        rescale_t=p["rescale_t"])
            run.begin("repaint")
            t = time.time()
            out, info = sampler.sample(
                pipe.models["slat_flow_model"], coords, (feats - mean) / std,
                torch.as_tensor(mrows, device="cuda"), cond["cond"], cond["neg_cond"],
                cfg_strength=p["cfg_strength"], cfg_interval=p["cfg_interval"], seed=seed,
                jump_length=jump, jump_n_sample=2 if jump else 1,
                mean=mean, std=std, x0_ref_denorm=feats, verbose=False)
            torch.cuda.synchronize()
            run.stage("repaint", "RePaint", f"({info['model_calls']} step)", time.time() - t)

        # ── §9-2 자동 검증. 깨지면 크게 실패시킨다 ──────────────────
        # redecode 는 편집 자체가 없으므로 항상 통과한다 (out is feats).
        keep = ~torch.as_tensor(mrows, device="cuda")
        nkeep = int(keep.sum())
        ndiff = int((out[keep] != feats[keep]).any(1).sum())
        ok = ndiff == 0
        run.set(latent_preserved=ok, preserved_rows=f"{nkeep - ndiff}/{nkeep}")
        run.log(f"latent 보존  마스크 밖 {nkeep - ndiff:,}/{nkeep:,} 비트 동일 "
                + ("✓" if ok else "✗"), indent=True)
        if not ok:
            run.fail_loud(
                f"§9-2 위반: 마스크 밖 {ndiff}/{nkeep} 행이 달라졌다. "
                "하드 오버라이트가 동작하지 않았다.")
            raise RuntimeError(f"latent 보존 실패: {ndiff}/{nkeep} 행 상이")

        save_slat(run.path("slat.safetensors"), coords, out, {
            **meta, "parent_slat": sp_path, "edit_region": region,
            "edit_halo": halo, "edit_seed": seed,
        })

        # ── 디코딩 + 렌더 ──────────────────────────────────────────
        parent_ch = {}
        pman = os.path.join(pdir, "manifest.json")
        if os.path.exists(pman):
            with open(pman) as f:
                parent_ch = json.load(f)["chunks"]

        def mask_col(v):
            return render.mask_colors(v, core_codes, halo_only)

        run.set(voxels=int(n))
        # 편집이 뒤/위에 있으면 고정 4각도로 안 보인다. 마스크 무게중심을
        # 바라보는 각도를 "mask" 이름으로 자동 추가한다.
        centroid = (core.mean(axis=0) / 64.0) - 0.5
        extra_views = {"mask": render.mask_view(centroid)}
        mesh, verts, entries, total = decode_and_chunk(
            run, pipe, coords, out, xyz, extra_renders={"mask": mask_col},
            views=views, extra_views=extra_views)

        # ── ★ 부기 기반 합성 (계약 3.4.0) ─────────────────────────
        #
        # 해시를 비교하지 않는다. 부기(마스크+halo)가 무엇이 바뀌었는지를 정의하고,
        # 부기 밖 청크는 새로 디코딩한 결과를 **버리고** 부모 바이트를 승계한다.
        # 산출물은 "옛 몸통 + 새 머리" — 매니페스트에 서로 다른 버전이 섞이는 게 정상.
        run.begin("chunk")
        book = set(mchunks)
        parent_present = set(parent_ch)
        reused, rewritten, newly_active = [], [], []
        final, final_bytes = {}, 0
        for k, e in entries.items():
            if k in book:
                rewritten.append(k)
                final[k] = e
            elif k in parent_present:
                # 승계: 부모 .cbin 을 그대로 복사하고 version·hash·uri 도 승계한다.
                pe = parent_ch[k]
                src = os.path.join(pdir, "chunks", f"{k}.cbin")
                with open(src, "rb") as fi, open(run.path("chunks", f"{k}.cbin"), "wb") as fo:
                    fo.write(fi.read())
                reused.append(k)
                final[k] = ChunkEntry(**pe)
            else:
                # 새로 활성화된 청크 — 승계할 이전 바이트가 없으니 새 바이트를 쓴다.
                newly_active.append(k)
                final[k] = e
            final_bytes += final[k].byte_length

        sent = [k for k in final if k in book or k in newly_active]
        sent_bytes = sum(final[k].byte_length for k in sent)
        man = ChunkManifest(asset_id=run.run_id, version=1, chunks=final)
        with open(run.path("manifest.json"), "w") as f:
            f.write(man.model_dump_json(indent=2))
        run.log(f"부기 합성  전송 {len(sent)}/{len(final)} 청크 "
                f"({sent_bytes / 1024 / 1024:.2f} / {final_bytes / 1024 / 1024:.2f} MiB), "
                f"승계 {len(reused)}", indent=True)
        if newly_active:
            run.log(f"새로 활성화된 청크 {len(newly_active)}개 "
                    f"(승계할 옛 바이트 없음 → 새 바이트 전송): "
                    f"{' '.join(sorted(newly_active))}", indent=True)

        # ── 이음새 스냅 (선택) ─────────────────────────────────────
        snap_stats = None
        if snap > 0:
            snap_stats = seam.snap_boundary(run.dir, pdir, book, set(final),
                                            parent_present, cells=snap)
            run.log(f"이음새 스냅  {snap_stats['snapped']:,}/{snap_stats['candidates']:,} 정점 "
                    f"({snap_stats['snapped_ratio'] * 100:.1f}%), 평균 이동 "
                    f"{snap_stats['move_mean']:.3e} ({snap_stats['move_mean_cells']:.2f} 셀), "
                    f"최대 {snap_stats['move_max']:.3e}", indent=True)
            # 스냅으로 바이트가 바뀌었으니 매니페스트 엔트리를 갱신한다.
            for k in list(final):
                if k not in book:
                    continue
                fp = run.path("chunks", f"{k}.cbin")
                if not os.path.exists(fp):
                    continue
                b = open(fp, "rb").read()
                final[k] = final[k].model_copy(update={
                    "hash": blob_hash(b), "byte_length": len(b)})
            final_bytes = sum(e.byte_length for e in final.values())
            sent_bytes = sum(final[k].byte_length for k in sent)
            man = ChunkManifest(asset_id=run.run_id, version=1, chunks=final)
            with open(run.path("manifest.json"), "w") as f:
                f.write(man.model_dump_json(indent=2))

        # ── 이음새 측정 ────────────────────────────────────────────
        seam_stats = seam.measure_seams(run.dir, pdir, book, set(final), parent_present)
        run.log(f"이음새  면 {seam_stats['seam_faces']}개, 정점 "
                f"{seam_stats['boundary_vertices']:,}  "
                + (f"max {seam_stats['max']:.3e} ({seam_stats['cells_max']:.2f} 셀)  "
                   f"median {seam_stats['median']:.3e}  "
                   f">1셀 {seam_stats['over_cell']:,}"
                   if seam_stats["max"] is not None else "(없음)"), indent=True)

        # ── 렌더: delta(이음새 표시) + merged(실제 합성물) ─────────
        run.begin("render")
        seam_mask = seam.seam_vertices(verts, book, set(final), parent_present)
        change = {k: 1.0 for k in rewritten}
        dr = render.render_views(
            mesh, run.path("render"), "delta",
            colors=render.delta_colors(verts, change, seam_mask=seam_mask),
            views=views, extra=extra_views)

        # merged: 클라이언트가 실제로 받게 될 합성물. 지금까지 렌더한 건 전부
        # "새로 디코딩한 전체"였지 합성물이 아니었다.
        mg = render.render_merged(run.path("chunks"), run.path("render"), "merged",
                                  views=views, extra=extra_views, seam_book=book,
                                  parent_present=parent_present)

        rd = run.meta.get("renders", {})
        rd["delta"] = dr
        rd["merged"] = mg
        th = run.meta.get("thumbs", {})
        th["delta"] = dr.get("front") or next(iter(dr.values()), None)
        th["merged"] = mg.get("front") or next(iter(mg.values()), None)

        # LAB_API.md §4 — gain 을 계산해 보내지 않는다.
        comparisons = [
            {"metric": "편집당 전송량", "baseline_label": "전체 raw",
             "baseline": final_bytes, "value_label": "부기분만",
             "value": sent_bytes, "unit": "bytes", "lower_is_better": True,
             "note": "부기 밖은 부모 바이트를 승계한다 (계약 3.4.0)"},
            {"metric": "편집당 전송 청크", "baseline_label": "전체 청크",
             "baseline": len(final), "value_label": "전송 청크",
             "value": len(sent), "unit": "chunks", "lower_is_better": True,
             "note": f"승계 {len(reused)}, 새로 활성화 {len(newly_active)}"},
        ]
        run.set(renders=rd, thumbs=th, comparisons=comparisons,
                chunk_count=len(final), seam=seam_stats, snap=snap_stats,
                payload={"encoding": "f32+normal+rgba8", "raw_bytes": final_bytes,
                         "compressed_bytes": None, "glb_bytes": None},
                reused_chunks_list=sorted(reused),
                newly_active_chunks=sorted(newly_active),
                delta={"total_chunks": len(final),
                       "changed_chunks": len(sent),
                       "changed_bytes": sent_bytes,
                       "reused_chunks": len(reused),
                       "bookkeeping_chunks": len(book),
                       # 부기가 정의이므로 항상 0 이어야 한다.
                       "missed_by_bookkeeping": len(set(sent) - book - set(newly_active))})
        run.done()
        return run.run_id
    except Exception as e:
        if not isinstance(e, RuntimeError) or "latent 보존" not in str(e):
            run.fail_loud(f"{type(e).__name__}: {e}")
        run.failed(e)
        raise
