"""Stage 2 좌표 매칭(편집당 1회 계산) 동등성 · argmax(0) 결함 재현 · CFG 배치 동등성."""
import ast, os, sys, time, types
os.environ["ATTN_BACKEND"] = "sdpa"
VH = sys.argv[1]; sys.path.insert(0, VH)
for name, sub in [("trellis", "trellis"), ("trellis.pipelines", "trellis/pipelines")]:
    m = types.ModuleType(name); m.__path__ = [os.path.join(VH, sub)]; sys.modules[name] = m
import torch, numpy as np
torch.Tensor.cuda = lambda self, *a, **k: self

src = open(os.path.join(VH, "voxhammer/edit_pipeline.py"), encoding="utf-8").read()
ns = dict(torch=torch, np=np)
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name == "ply_to_slat_mask":
        exec(compile(ast.Module([node], []), "edit_pipeline.py", "exec"), ns)

# ── 자산 모양: 64³ 안의 껍질 형태 점유 (t1s 규모 ≈ 13k 셀) ───────────────────
g = torch.Generator().manual_seed(0)
grid = torch.stack(torch.meshgrid(*[torch.arange(64)] * 3, indexing="ij"), -1).reshape(-1, 3)
r = ((grid.float() - 31.5) / torch.tensor([28.0, 20.0, 24.0])).norm(dim=1)
coords_src = grid[(r > 0.93) & (r < 1.0)].int()
edit_region = coords_src[:, 2] > 44                                   # 위쪽을 편집
coords_preserve = coords_src[~edit_region]
new_cells = grid[(grid[:, 2] > 44) & (r < 0.9) & (r > 0.8)].int()      # 편집으로 생긴 셀
coords_tgt = torch.cat([torch.zeros(len(coords_preserve) + len(new_cells), 1, dtype=torch.int32),
                        torch.cat([coords_preserve, new_cells])], 1)
print(f"[자산] src {len(coords_src)} · 보존 {len(coords_preserve)} · 편집 후 {len(coords_tgt)}")

kv_mask = ns["ply_to_slat_mask"](coords_tgt, coords_preserve)        # 원본 함수 그대로
down = lambda c: torch.unique(torch.cat([c[:, :1], c[:, 1:] // 2], 1), dim=0)
k_coords = down(coords_tgt)                                           # Transformer 단 토큰 (32³)
cache_coords = down(torch.cat([torch.zeros(len(coords_preserve), 1, dtype=torch.int32), coords_preserve], 1))
N, M, D = len(k_coords), len(kv_mask), 64
k_feats = torch.randn(N, 16, D, generator=g); cache_feats = torch.randn(len(cache_coords), 16, D, generator=g)

def original(k_coords, k_feats, cache_coords, cache_feats, kv_mask):      # slat_attn_forward 의 해당 줄
    match_1 = (k_coords.unsqueeze(1) == kv_mask.unsqueeze(0)).all(dim=-1)
    match_2 = (cache_coords.unsqueeze(1) == kv_mask.unsqueeze(0)).all(dim=-1)
    idx_1 = match_1.float().argmax(0); idx_2 = match_2.float().argmax(0)
    f = k_feats.clone(); f[idx_1] = cache_feats[idx_2]
    return f, match_1

def code(c):                                                          # 32³ 좌표 → 정수 하나
    return ((c[:, 0].long() * 32 + c[:, 1]) * 32 + c[:, 2]) * 32 + c[:, 3]
def precompute(k_coords, cache_coords, kv_mask):
    ks, ko = torch.sort(code(k_coords)); cs, co = torch.sort(code(cache_coords)); m = code(kv_mask)
    i1 = torch.searchsorted(ks, m); i2 = torch.searchsorted(cs, m)
    ok1 = (i1 < len(ks)) & (ks[i1.clamp(max=len(ks) - 1)] == m)
    ok2 = (i2 < len(cs)) & (cs[i2.clamp(max=len(cs) - 1)] == m)
    if not bool(ok1.all() and ok2.all()):
        raise ValueError(f"보존 좌표 불일치: 현재토큰 {int((~ok1).sum())}, 캐시 {int((~ok2).sum())}")
    return ko[i1], co[i2]

t0 = time.time(); ref, match = original(k_coords, k_feats, cache_coords, cache_feats, kv_mask); t_orig = time.time() - t0
t0 = time.time(); i1, i2 = precompute(k_coords, cache_coords, kv_mask); t_pre = time.time() - t0
t0 = time.time(); out = k_feats.clone(); out[i1] = cache_feats[i2]; t_apply = time.time() - t0
print(f"[매칭] 토큰 N={N} · 보존 M={M} · 매칭 텐서 {N*M*4/1e6:.0f} MB(bool,4열) + {N*M*4/1e6:.0f} MB(float)")
print(f"[매칭] 원본 1회 {t_orig*1e3:.0f} ms (CPU) → 편집당 2,400회 · 개선: 인덱스 {t_pre*1e3:.1f} ms 1회 + 적용 {t_apply*1e3:.2f} ms/회")
print(f"[매칭] 결과 비트 동일: {torch.equal(out, ref)}")

# ── argmax(0) 결함: 현재 토큰에 없는 보존 좌표가 하나라도 있으면 ───────────────
bad_mask = torch.cat([kv_mask, torch.tensor([[0, 0, 0, 0]], dtype=kv_mask.dtype)])   # 존재하지 않는 좌표
cache_bad = torch.cat([cache_coords, torch.tensor([[0, 0, 0, 0]], dtype=cache_coords.dtype)])
cf_bad = torch.cat([cache_feats, torch.randn(1, 16, D, generator=g)])
o_bad, _ = original(k_coords, k_feats, cache_bad, cf_bad, bad_mask)
row0_changed = not torch.equal(o_bad[0], ref[0])
print(f"[argmax] 없는 좌표 1개 추가 → 0번 토큰 덮어씀: {row0_changed} (0번 토큰은 보존 토큰인가: {bool(match[0].any())})")
try:
    precompute(k_coords, cache_bad, bad_mask); print("[argmax] 개선판: 통과(문제)")
except ValueError as e:
    print(f"[argmax] 개선판: 오류로 중단 → {e}")
print(f"[전제] 정상 흐름에서 보존 좌표 ⊂ 현재 토큰: {bool(match.any(0).all())}")

# ── CFG 배치: 조건/무조건 두 번 호출 vs 배치 하나 ─────────────────────────
from trellis.models.sparse_structure_flow import SparseStructureFlowModel
torch.manual_seed(0)
mdl = SparseStructureFlowModel(resolution=8, in_channels=8, model_channels=256, cond_channels=256, out_channels=8,
                               num_blocks=4, num_heads=4, patch_size=1, pe_mode="ape", qk_rms_norm=True).eval()
with torch.no_grad():
    for p in mdl.parameters():
        p.normal_(0, 1.0 / p.shape[1] ** 0.5) if p.dim() > 1 else p.normal_(0, 0.3)
    x = torch.randn(1, 8, 8, 8, 8, generator=g); t = torch.tensor([700.0])
    c = torch.randn(1, 70, 256, generator=g); n = torch.zeros(1, 70, 256)
    a = mdl(x, t, c); b = mdl(x, t, n)
    ab = mdl(x.repeat(2, 1, 1, 1, 1), t.repeat(2), torch.cat([c, n]))
    print(f"[CFG 배치] max|Δ| cond {(ab[0]-a[0]).abs().max().item():.2e} · neg {(ab[1]-b[0]).abs().max().item():.2e} · "
          f"비트 동일 {torch.equal(ab[:1], a) and torch.equal(ab[1:], b)}")
