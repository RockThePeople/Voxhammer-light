"""SLat RePaint — 마스크 안만 재생성하고 밖은 원본을 강제 보존한다.

업스트림 `FlowEulerGuidanceIntervalSampler` 를 **상속하지 않고 감싼다**(composition).
vendor/trellis1 은 수정하지 않는다는 원칙과 일관되게, 그리고 상속하면
`sample()` 시그니처를 오버라이드하게 되어 업스트림이 바뀔 때 조용히 어긋난다.

이 클래스가 책임지는 것:
  1. t_seq 를 업스트림과 **정확히 동일하게** 재현 (스텝 스케줄이 다르면 비교 불가)
  2. 정규화 공간 왕복을 **두 지점에서만** 수행
  3. 매 스텝 마스크 밖을 forward-noise 로 재주입
  4. 루프 종료 후 마스크 밖 **하드 오버라이트** — §9-2 를 근사가 아닌 보장으로

Phase 1 제약: coords 는 고정이다. sparse structure 를 건드리지 않으므로 복셀을
추가/제거할 수 없다. added/removed 는 항상 공집합이고 modified 만 나온다.
"""
from typing import Optional

import numpy as np
import torch


class MaskedSlatSampler:
    """FlowEuler 계열 샘플러를 감싸 RePaint 마스킹을 얹는다.

    Args:
        sampler: `FlowEulerGuidanceIntervalSampler` 인스턴스. `sample_once()` 와
            `sigma_min` 만 쓴다. `sample()` 은 쓰지 않는다 — 루프를 우리가 돈다.
        steps, rescale_t: pipeline.json 의 slat_sampler.params 와 일치해야 한다.
    """

    def __init__(self, sampler, steps: int = 25, rescale_t: float = 3.0):
        self.sampler = sampler
        self.steps = steps
        self.rescale_t = rescale_t
        # ⚠️ 하드코딩 금지. pipeline.json 은 1e-05 지만 정본은 샘플러 인스턴스다.
        self.sigma_min = float(sampler.sigma_min)

    # ── 스텝 스케줄 ────────────────────────────────────────────────────────
    def t_pairs(self):
        """업스트림 FlowEulerSampler.sample() 과 **바이트 단위로 같은** 수식.

        vendor/trellis1/pipelines/samplers/flow_euler.py:109-111 참조.
        여기가 어긋나면 base 와 편집본을 비교하는 것 자체가 무의미해진다.
        """
        t_seq = np.linspace(1, 0, self.steps + 1)
        t_seq = self.rescale_t * t_seq / (1 + (self.rescale_t - 1) * t_seq)
        return [(t_seq[i], t_seq[i + 1]) for i in range(self.steps)]

    # ── flow matching forward noising ─────────────────────────────────────
    def forward_noise(self, x0, t, eps):
        """x_t = (1-t)*x_0 + (sigma_min + (1-sigma_min)*t)*eps

        FlowEulerSampler._xstart_to_eps 의 역변환이다 (flow_euler.py:28-30).
        t=0 에서도 계수가 sigma_min(1e-5) 이라 0 이 아니다 — 이게 하드
        오버라이트가 필요한 이유다.
        """
        return (1.0 - t) * x0 + (self.sigma_min + (1.0 - self.sigma_min) * t) * eps

    @torch.no_grad()
    def sample(
        self,
        model,
        coords: torch.Tensor,
        x0_ref_norm: torch.Tensor,
        mask_rows: torch.Tensor,
        cond,
        neg_cond,
        cfg_strength: float = 5.0,
        cfg_interval=(0.5, 1.0),
        seed: int = 0,
        jump_length: int = 0,
        jump_n_sample: int = 1,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        x0_ref_denorm: Optional[torch.Tensor] = None,
        verbose: bool = True,
    ):
        """마스크 안만 재생성한다.

        Args:
            coords: (N,4) int32 [b,x,y,z]. **canonical 순서 전제** (계약 v3 COORD_ORDER).
            x0_ref_norm: (N,8) float32, **정규화 공간**의 원본 latent.
            mask_rows: (N,) bool. True = 재생성 대상.
            mean/std: (1,8). 주면 반환값을 denormalize 한다 (왕복 지점 2/2).
            x0_ref_denorm: (N,8) 저장된 **원본 그대로의** denormalized latent.
                주면 마스크 밖을 이 값으로 한 번 더 하드 오버라이트한다. 아래 참조.
            jump_length/jump_n_sample: RePaint jump-back resampling. 0 이면 off.

        Returns:
            (x_out, info) — x_out 은 mean/std 를 주면 denormalized, 아니면 정규화 공간.

        ⚠️ 왜 x0_ref_denorm 이 따로 필요한가 (A5000 실측 2026-07-29):
        `(x - mean)/std * std + mean` 은 float32 에서 **비트 정확하지 않다**.
        실측 422/66352 원소(0.636%)가 max|Δ|=9.54e-7 만큼 어긋난다. 정규화 공간에서
        아무리 완벽하게 보존해도 denormalize 하는 순간 이 오차가 다시 붙는다.
        디코더가 먹는 것은 denormalized 값이므로, §9-2 를 보장하려면 **출력 공간에서**
        원본 바이트를 그대로 되돌려놓아야 한다.
        """
        from trellis1.modules import sparse as sp

        device = x0_ref_norm.device
        n = x0_ref_norm.shape[0]
        assert mask_rows.shape == (n,), f"mask_rows shape {mask_rows.shape} != ({n},)"
        assert coords.shape[0] == n, f"coords {coords.shape[0]} != feats {n}"

        keep = ~mask_rows
        # 열 방향 브로드캐스트용. (N,1) 이어야 feats (N,8) 에 행 단위로 걸린다.
        keep_col = keep[:, None]

        # 스텝마다 새 노이즈를 뽑되 seed 로 재현 가능해야 한다.
        # generator 를 하나 만들어 재사용 → 스텝 순서까지 결정적.
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

        def randn_like_feats():
            return torch.randn(
                x0_ref_norm.shape, generator=gen, device=device, dtype=x0_ref_norm.dtype
            )

        # ── 초기값: 마스크 안은 순수 노이즈(t=1), 밖은 원본을 t=1 로 forward-noise.
        pairs = self.t_pairs()
        t_start = pairs[0][0]
        x = self.forward_noise(x0_ref_norm, t_start, randn_like_feats())
        x = torch.where(keep_col, x, randn_like_feats())

        def step(x, t, t_prev):
            """1 스텝. 마스크 밖 재주입 → sample_once → 마스크 밖 재주입."""
            # 모델에 넣기 전에 마스크 밖을 t 시점 원본으로 되돌린다.
            eps = randn_like_feats()
            x = torch.where(keep_col, self.forward_noise(x0_ref_norm, t, eps), x)

            st = sp.SparseTensor(feats=x, coords=coords)
            out = self.sampler.sample_once(
                model,
                st,
                float(t),
                float(t_prev),
                cond,
                neg_cond=neg_cond,
                cfg_strength=cfg_strength,
                cfg_interval=cfg_interval,
            )
            return out.pred_x_prev.feats

        n_model_calls = 0
        if jump_length > 0 and jump_n_sample > 1:
            # ── jump-back resampling (FINAL §4.1-3 / RePaint 논문 Alg.1)
            #    jump_length 스텝마다 되돌아가 다시 내려온다. 경계 조화를 위해.
            i = 0
            while i < len(pairs):
                block = pairs[i : i + jump_length]
                for r in range(jump_n_sample):
                    xb = x
                    for t, t_prev in block:
                        xb = step(xb, t, t_prev)
                        n_model_calls += 1
                    if r < jump_n_sample - 1:
                        # 블록 시작 t 로 되돌린다 (forward diffusion).
                        t_back = block[0][0]
                        xb = torch.where(
                            keep_col,
                            self.forward_noise(x0_ref_norm, t_back, randn_like_feats()),
                            self.forward_noise(xb, t_back, randn_like_feats()),
                        )
                    x = xb
                i += jump_length
        else:
            it = pairs
            if verbose:
                from tqdm import tqdm

                it = tqdm(pairs, desc="RePaint")
            for t, t_prev in it:
                x = step(x, t, t_prev)
                n_model_calls += 1

        # ── ★ 하드 오버라이트. 이 한 줄이 §9-2 를 구조적 보장으로 만든다.
        #
        # 마지막 스텝의 t_prev 는 0 이지만, forward_noise 의 노이즈 계수는
        # sigma_min + (1-sigma_min)*0 = sigma_min = 1e-5 로 0 이 아니다.
        # 1e-5 잔차는 position 20bit 양자화 1LSB(9.54e-7)의 10배라 canonicalize
        # 의 정렬 순서를 뒤집기 충분하다 → "거의 같음"이 "청크 해시 불일치"가 된다.
        x_out = torch.where(keep_col, x0_ref_norm, x)

        info = {
            "n_rows": int(n),
            "n_masked": int(mask_rows.sum()),
            "n_kept": int(keep.sum()),
            "steps": self.steps,
            "rescale_t": self.rescale_t,
            "sigma_min": self.sigma_min,
            "cfg_strength": cfg_strength,
            "cfg_interval": list(cfg_interval),
            "seed": int(seed),
            "jump_length": int(jump_length),
            "jump_n_sample": int(jump_n_sample),
            "model_calls": int(n_model_calls),
        }

        if mean is not None and std is not None:
            # 왕복 지점 2/2. 진입은 edit_probe 가, 복귀는 여기가 한다.
            x_out = x_out * std + mean
            info["returned_space"] = "denormalized"
            if x0_ref_denorm is not None:
                # 출력 공간에서의 2차 하드 오버라이트. 위 docstring 의 이유.
                x_out = torch.where(keep_col, x0_ref_denorm, x_out)
                info["denorm_hard_overwrite"] = True
            else:
                info["denorm_hard_overwrite"] = False
        else:
            info["returned_space"] = "normalized"

        return x_out, info
