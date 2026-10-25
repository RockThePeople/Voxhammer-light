"""Inversion/denoising 샘플러 — 2차(Heun 류) solver, CFG 구간 [0.5, 1.0], rescale_t 3.0, W38 cfg=0 게이트.

스텝 수는 1번 E2a 토글(`VHL_E2A_STEPS_S1`/`VHL_E2A_STEPS_S2`, editcore 기본 15 = 정오표 E20)로 샘플러마다 정한다. 아래 STEPS 25 는 포크 값(포크 대조용 기본).
같은 샘플러(같은 stage)의 inversion 과 denoising 은 같은 스텝 수를 쓰므로 캐시 키(t_latent)가 일치한다 —
어긋나면 denoising 의 `latent[key]` 가 KeyError 로 멈춘다(조용히 넘어가지 않는다).

`runner` 는 blocks.SSFlow / blocks.SlatFlow (호출 규약 runner(x, t, cond, ctx)).
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from .cache import EditContext

STEPS = 25  # 기본값(포크 edit_pipeline.py:409)
RESCALE_T = 3.0
CFG_INTERVAL = (0.5, 1.0)


class InversionSampler:
    def __init__(self, steps: int = STEPS):
        if int(steps) < 1:
            raise ValueError(f"steps={steps}")
        self.steps = int(steps)

    # origin: voxhammer/edit_pipeline.py:InversionFlowEulerGuidanceIntervalSampler._inference_model @ f33cd96+patch
    @staticmethod
    def _inference_model(runner, sample, t, cond, ctx: EditContext):
        t = torch.tensor([1000 * t] * sample.shape[0], device=sample.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and sample.shape[0] > 1:
            cond = cond.repeat(sample.shape[0], *([1] * (len(cond.shape) - 1)))
        return runner(sample, t, cond, ctx)

    # origin: voxhammer/edit_pipeline.py:InversionFlowEulerGuidanceIntervalSampler.inference_model @ f33cd96+patch
    def inference_model(self, runner, sample, t, cond, cfg_strength, ctx: EditContext):
        # W38: cfg_strength == 0 이면 음 분기를 계산하지 않는다(inversion·forward 같은 판정 → pos 0 키 없음)
        if cfg_strength != 0 and CFG_INTERVAL[0] <= ctx.t_latent <= CFG_INTERVAL[1]:
            ctx.pos = 1
            pred = self._inference_model(runner, sample, t, cond["cond"], ctx)
            ctx.pos = 0
            neg_pred = self._inference_model(runner, sample, t, cond["neg_cond"], ctx)
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        ctx.pos = 1
        return self._inference_model(runner, sample, t, cond["cond"], ctx)

    # origin: voxhammer/edit_pipeline.py:InversionFlowEulerGuidanceIntervalSampler.sample_once @ f33cd96+patch
    def sample_once(self, runner, sample, t_curr, t_prev, cond, cfg_strength, ctx: EditContext):
        ctx.order = 1
        pred = self.inference_model(runner, sample, t_curr, cond, cfg_strength, ctx)
        sample_mid = sample + (t_prev - t_curr) / 2 * pred
        t_mid = t_curr + (t_prev - t_curr) / 2
        ctx.order = 2
        pred_mid = self.inference_model(runner, sample_mid, t_mid, cond, cfg_strength, ctx)
        first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
        return sample + (t_prev - t_curr) * pred - 0.5 * (t_prev - t_curr) ** 2 * first_order

    def t_seq(self, skip_step=None):
        steps = self.steps
        t = np.linspace(1, 0, steps + 1)
        t = RESCALE_T * t / (1 + (RESCALE_T - 1) * t)
        if skip_step is not None:
            t = t[skip_step:]
            steps = steps - skip_step
        return t, steps

    # origin: voxhammer/edit_pipeline.py:InversionFlowEulerGuidanceIntervalSampler.sample @ f33cd96+patch
    @torch.no_grad()
    def sample(self, runner, stage, noise, cond, cfg_strength, latent=None, latent_mask=None, kv=None,
               self_kv_mask=None, cross_kv_mask=None, skip_step=None, noise_init=None, is_text=False):
        sample = noise
        t_seq, steps = self.t_seq(skip_step)
        inverse_bool = latent_mask is None
        if noise_init is not None:
            noise_randn = torch.randn_like(noise)
            t_init = t_seq[0]
            sample = noise_init * (1 - t_init) + noise_randn * t_init
        if inverse_bool:
            t_seq = t_seq[::-1]
            desc = "Inversing"
            latent = {}
            kv = {}
        else:
            desc = "Sampling"
        ctx = EditContext(kv=kv, self_kv_mask=self_kv_mask, cross_kv_mask=cross_kv_mask, is_text=is_text)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        for t_curr, t_prev in tqdm(t_pairs, desc=desc, disable=False):
            # 캐시 키: inversion 은 t_prev, denoising 은 t_curr → 같은 시각이 같은 키
            if inverse_bool:
                ctx.t_latent = t_prev
            else:
                ctx.t_latent = t_curr
                key = f"{ctx.t_latent}"
                if stage == 1:
                    sample = sample * latent_mask + latent[key].cuda() * (1 - latent_mask)
                elif stage == 2:
                    match_1 = (sample.coords.unsqueeze(1) == latent_mask.unsqueeze(0)).all(dim=-1)
                    match_2 = (latent[key].coords.cuda().unsqueeze(1) == latent_mask.unsqueeze(0)).all(dim=-1)
                    idx_1 = match_1.float().argmax(0)
                    idx_2 = match_2.float().argmax(0)
                    feats = sample.feats.clone()
                    feats[idx_1] = latent[key].feats.cuda()[idx_2]
                    sample = sample.replace(feats)
            sample = self.sample_once(runner, sample, t_curr, t_prev, cond, cfg_strength, ctx)
            if inverse_bool:
                latent[f"{ctx.t_latent}"] = sample.cpu()
        return sample, latent, kv
