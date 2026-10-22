"""Stage 1 캐시 개선안 동등성 실험 (CPU, fp32).

패치 적용된 VoxHammer edit_pipeline.py 의 함수(ss_attn_forward, ss_trsfmr_forward,
ss_flow_forward, InversionFlowEulerGuidanceIntervalSampler)를 AST 로 떼어 그대로 실행하고,
같은 inversion → denoising 을 개선안 attention 으로 다시 돌려 최종 z 를 비교한다.

모델: TRELLIS SparseStructureFlowModel 축소판 (해상도 8³ → 512 토큰, 256ch, 4블록, 4헤드,
      qk_rms_norm=True, APE).  가중치는 무작위 (adaLN zero-init 해제).
"""
import ast, os, sys, types, time
os.environ["ATTN_BACKEND"] = "sdpa"
VH = sys.argv[1]
sys.path.insert(0, VH)
# trellis/__init__.py 가 rembg 등을 끌고 오므로 패키지만 스텁으로 등록
for name, sub in [("trellis", "trellis"), ("trellis.pipelines", "trellis/pipelines")]:
    m = types.ModuleType(name); m.__path__ = [os.path.join(VH, sub)]; sys.modules[name] = m

import numpy as np, torch, torch.nn.functional as F
from types import MethodType
from tqdm import tqdm
from trellis.modules.spatial import patchify, unpatchify
from trellis.models.sparse_structure_flow import SparseStructureFlowModel
from trellis.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler

src = open(os.path.join(VH, "voxhammer/edit_pipeline.py"), encoding="utf-8").read()
tree = ast.parse(src)
want = {"ss_attn_forward", "ss_trsfmr_forward", "ss_flow_forward", "InversionFlowEulerGuidanceIntervalSampler",
        "_w39_allow"}
ns = dict(torch=torch, F=F, np=np, tqdm=tqdm, patchify=patchify, unpatchify=unpatchify, os=os,
          FlowEulerGuidanceIntervalSampler=FlowEulerGuidanceIntervalSampler, MethodType=MethodType,
          W39_ATTN_MASK=False, W39_STATS={})
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in want:
        exec(compile(ast.Module([node], []), "edit_pipeline.py", "exec"), ns)
# [GPU 사본] .cuda() 항등 치환 제거 — 실제 GPU 로 돈다. 순전파는 TRELLIS 와 같은 fp16(convert_to_fp16).
DEV = "cuda"
# [GPU 사본] 규모는 환경변수로만 바꾼다. 기본값 = 원본. 실제 규모: V_R=16 V_C=1024 V_NB=24 V_H=16 V_LCTX=1374
#            V_CKPT=<ss_flow_img_dit_L_16l8_fp16.safetensors> 이면 무작위 대신 실제 체크포인트 가중치.
_e = lambda k, d: int(os.environ.get(k, d))
torch.manual_seed(0)
R, C, H, NB, LCTX = _e("V_R", 8), _e("V_C", 256), _e("V_H", 4), _e("V_NB", 4), _e("V_LCTX", 70)
CKPT = os.environ.get("V_CKPT")
L = R ** 3
def build():
    torch.manual_seed(0)
    m = SparseStructureFlowModel(resolution=R, in_channels=8, model_channels=C, cond_channels=C, out_channels=8,
                                 num_blocks=NB, num_heads=H, patch_size=1, pe_mode="ape", qk_rms_norm=True,
                                 use_fp16=True)
    with torch.no_grad():
        if CKPT:
            from safetensors.torch import load_file
            m.load_state_dict(load_file(CKPT), strict=True)
        else:
            for p in m.parameters():                      # zero-init 해제, fan-in 규모
                if p.dim() > 1: p.normal_(0, 1.0 / p.shape[1] ** 0.5)
                else: p.normal_(0, 0.3)
    m.convert_to_fp16()
    m.to(DEV)
    m.eval()
    m.forward = MethodType(ns["ss_flow_forward"], m)
    for b in m.blocks:
        b.forward = MethodType(ns["ss_trsfmr_forward"], b)
    return m

def set_attn(model, fn):
    for b in model.blocks:
        b.self_attn.forward = MethodType(fn, b.self_attn)
        b.cross_attn.forward = MethodType(fn, b.cross_attn)

# ── 입력: 원본 잠재, 조건(원본/편집), 마스크 ────────────────────────────
g = torch.Generator().manual_seed(1)
z_src = torch.randn(1, 8, R, R, R, generator=g)
cond_src = {"cond": torch.randn(1, LCTX, C, generator=g), "neg_cond": torch.zeros(1, LCTX, C)}
cond_tgt = {"cond": cond_src["cond"] + 0.5 * torch.randn(1, LCTX, C, generator=g), "neg_cond": torch.zeros(1, LCTX, C)}
# 토큰 마스크: 1 = 편집, 0 = 보존.  moto 자산 비율(546/4096 보존)에 맞춰 약 13% 보존
pres = torch.zeros(L, dtype=torch.bool); pres[torch.randperm(L, generator=g)[: int(L * 546 / 4096)]] = True
tok_edit = (~pres).float()
ss_latent_mask = tok_edit.view(1, 1, R, R, R).repeat(1, 8, 1, 1, 1)       # (1,8,R,R,R)
self_kv_mask = tok_edit.view(1, 1, L, 1).expand(1, H, L, C // H).contiguous()
img_edit = torch.zeros(LCTX); img_edit[:5] = 1; img_edit[5 + torch.randperm(LCTX - 5, generator=g)[:20]] = 1
cross_kv_mask = img_edit.view(1, 1, LCTX, 1).expand(1, H, LCTX, C // H).contiguous()
# [GPU 사본] 같은 CPU 생성기로 만든 값을 장치로 옮긴다 (값은 원본과 같다)
z_src = z_src.to(DEV)
cond_src = {k_: v_.to(DEV) for k_, v_ in cond_src.items()}
cond_tgt = {k_: v_.to(DEV) for k_, v_ in cond_tgt.items()}
ss_latent_mask, self_kv_mask, cross_kv_mask = (t_.to(DEV) for t_ in (ss_latent_mask, self_kv_mask, cross_kv_mask))

def run(model, attn_fn):
    set_attn(model, attn_fn)
    S = ns["InversionFlowEulerGuidanceIntervalSampler"](1e-5)
    with torch.no_grad():
        noise, lat, kv = S.sample(model, 1, z_src, cond_src, 5.0)
        out, _, _ = S.sample(model, 1, noise, cond_tgt, 6.0, lat, ss_latent_mask, kv, self_kv_mask, cross_kv_mask)
    torch.cuda.synchronize()  # [GPU 사본] 시간 측정 경계
    return out, kv

def nbytes(d):
    return sum(v.numel() * v.element_size() for v in d.values() if torch.is_tensor(v))

# ── 개선안: 보존 토큰만 · hidden 입력 저장 · cross 1회 저장 ─────────────────
PRES_IDX = torch.nonzero(pres).squeeze(1).to(DEV)
LOAD_SHIFT = lambda i: i
XPRES_IDX = torch.nonzero(img_edit == 0).squeeze(1).to(DEV)
def make_attn(mode):
    def f(self, x, context=None, indices=None, ss_kv=None, kv_mask=None, t_latent=None, order=None,
          pos=None, layer=None, is_text=False):
        B, L_, C_ = x.shape
        if self._type == "self":
            qkv = self.to_qkv(x).reshape(B, L_, 3, self.num_heads, -1)
            q, k, v = qkv.unbind(dim=2)
            src_rows, idx = x, PRES_IDX
        else:
            q = self.to_q(x).reshape(B, L_, self.num_heads, -1)
            kv_ = self.to_kv(context).reshape(B, context.shape[1], 2, self.num_heads, -1)
            k, v = kv_.unbind(dim=2)
            src_rows, idx = context, XPRES_IDX
        if self.qk_rms_norm:
            q = self.q_rms_norm(q); k = self.k_rms_norm(k)
        q, k, v = (t_.permute(0, 2, 1, 3) for t_ in (q, k, v))
        if self._type == "self":
            key = f"{t_latent}_{order}_{pos}_{layer}_self"
        else:
            key = f"{pos}_{layer}_cross"                    # t·order 와 무관 → 1회만
        if kv_mask is None:
            if key in ss_kv:
                pass
            elif mode == "kv":                              # 보존 행의 K,V 만
                ss_kv[key + "_k"] = k[:, :, idx].cpu(); ss_kv[key + "_v"] = v[:, :, idx].cpu()
            else:                                           # 보존 행의 projection 입력만
                ss_kv[key] = src_rows[:, idx].cpu()
        else:
            if mode == "kv":
                kc, vc = ss_kv[key + "_k"], ss_kv[key + "_v"]
            else:
                rows = ss_kv[key].to(DEV)  # [GPU 사본] CPU 캐시 → 장치
                if self._type == "self":
                    kc, vc = self.to_qkv(rows).reshape(B, rows.shape[1], 3, self.num_heads, -1).unbind(2)[1:]
                else:
                    kc, vc = self.to_kv(rows).reshape(B, rows.shape[1], 2, self.num_heads, -1).unbind(2)
                if self.qk_rms_norm:
                    kc = self.k_rms_norm(kc)
                kc, vc = kc.permute(0, 2, 1, 3), vc.permute(0, 2, 1, 3)
            k = k.clone(); v = v.clone()
            li = LOAD_SHIFT(idx) if self._type == "self" else idx
            k[:, :, li] = kc.to(q.device).type(q.dtype); v[:, :, li] = vc.to(q.device).type(q.dtype)  # [GPU 사본] CPU 캐시 → 장치
        h = F.scaled_dot_product_attention(q, k, v)
        return self.to_out(h.permute(0, 2, 1, 3).reshape(B, L_, -1))
    return f

model = build()
torch.cuda.reset_peak_memory_stats()  # [GPU 사본] 실행별 시간·VRAM 피크
t0 = time.time(); ref, kv_ref = run(model, ns["ss_attn_forward"]); t_ref = time.time() - t0
peak_ref = torch.cuda.max_memory_allocated(); torch.cuda.reset_peak_memory_stats()
t0 = time.time(); outA, kvA = run(model, make_attn("kv")); t_A = time.time() - t0
peak_A = torch.cuda.max_memory_allocated(); torch.cuda.reset_peak_memory_stats()
t0 = time.time(); outB, kvB = run(model, make_attn("hidden")); t_B = time.time() - t0
peak_B = torch.cuda.max_memory_allocated()
print(f"[GPU] {torch.cuda.get_device_name(0)} · fp16 · ckpt {'실제' if CKPT else '무작위'} · 실행 원본 {t_ref:.1f}s A {t_A:.1f}s B {t_B:.1f}s · "
      f"VRAM 피크 원본 {peak_ref/2**30:.2f} A {peak_A/2**30:.2f} B {peak_B/2**30:.2f} GiB")

# 원본 캐시에서 cross K/V 가 t·order 와 무관한지 (비트 동일)
cross_keys = {}
for k_, v_ in kv_ref.items():
    if "_cross_" in k_:
        _, _, pos, layer, _, kind = k_.split("_")
        cross_keys.setdefault((pos, layer, kind), []).append(v_)
cross_same = all(all(torch.equal(a[0], a_) for a_ in a) for a in cross_keys.values())

print(f"[모델] 토큰 {L} · 채널 {C} · 블록 {NB} · 보존 토큰 {int(pres.sum())} ({pres.float().mean()*100:.1f}%)")
print(f"[원본] 캐시 {nbytes(kv_ref)/1e6:.2f} MB · 항목 {len(kv_ref)} · 실행 {t_ref:.1f}s")
print(f"[cross] 같은 (pos,layer) 의 K/V 가 모든 t·order 에서 비트 동일: {cross_same}")
for name, out, kv in [("A 보존행 K/V + cross 1회", outA, kvA), ("B 보존행 hidden + cross 1회", outB, kvB)]:
    d = (out - ref).abs().max().item()
    print(f"[{name}] 캐시 {nbytes(kv)/1e6:.2f} MB ({nbytes(kv)/nbytes(kv_ref)*100:.1f}%) · "
          f"최종 z max|Δ| = {d:.3e} · 비트 동일 {torch.equal(out, ref)}")
print(f"[참고] 최종 z 크기 max|z| = {ref.abs().max().item():.3f}")

# ── 음성 대조: 비교가 민감한가 ───────────────────────────────────────────
LOAD_SHIFT = lambda i: torch.roll(i, 1)             # 저장한 행을 다른 보존 행에 넣는다
outN, _ = run(model, make_attn("kv"))
LOAD_SHIFT = lambda i: i
print(f"[음성대조 1: 캐시 행 어긋나게 주입] max|Δ| = {(outN - ref).abs().max().item():.3e}")
def no_self_inject(self, *a, **k):
    if self._type == "self" and k.get("kv_mask") is not None:
        k = {**k, "kv_mask": None, "ss_kv": {}}
    return ns["ss_attn_forward"](self, *a, **k)
outC, _ = run(model, no_self_inject)
print(f"[음성대조 2: self 캐시 주입 끔] max|Δ| = {(outC - ref).abs().max().item():.3e}")
