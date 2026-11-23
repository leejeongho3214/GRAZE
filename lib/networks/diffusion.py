import random
import time
import tqdm
import math

import torch
import torch.nn as nn
from lib.utils.rot import rot6d_to_rotmat

from lib.utils.loss import get_l2_loss, get_object_pose_losses


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)


class Diffusion(nn.Module):
    def __init__(
        self, lhand_layer, rhand_layer,
        beta_1=1e-4, beta_T=0.02, T=1000,
        obj_diffusion_lag=0,
        replace_sampling="none",
        contact_consistency="none",   # none | both | object_only | hand_only
        schedule_name="cosine",
        cfg_scale=1.0,
        sampler="ddpm",
        sample_steps=0,
    ):
        super().__init__()

        self.lhand_layer = lhand_layer
        self.rhand_layer = rhand_layer
        self.obj_diffusion_lag = int(obj_diffusion_lag)
        self.replace_sampling = str(replace_sampling).lower()
        self.cfg_scale = float(cfg_scale)
        self.sampler = str(sampler).lower()
        self.sample_steps = int(sample_steps)
        assert self.sampler in ("ddpm", "ddim"), self.sampler

        self.contact_consistency = str(contact_consistency).lower()
        assert self.contact_consistency in ("none", "both", "object_only", "hand_only")
        self._loss_alpha_bar = None
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        if schedule_name == "linear":
            betas = torch.linspace(start=beta_1, end=beta_T, steps=T)
        elif schedule_name == "cosine":
            betas = betas_for_alpha_bar(
                T,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )

        alphas = 1 - betas
        alpha_bars = torch.cumprod(
            alphas,
            dim=0
        )

        alpha_prev_bars = torch.cat([torch.Tensor([1]), alpha_bars[:-1]])
        sigmas = torch.sqrt((1 - alpha_prev_bars) / (1 - alpha_bars)) * \
            torch.sqrt(1 - (alpha_bars / alpha_prev_bars))

        posterior_variance = (
            betas * (1.0 - alpha_prev_bars) / (1.0 - alpha_bars)
        )
        posterior_log_variance_clipped = torch.log(
            torch.hstack([posterior_variance[1], posterior_variance[1:]])
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_prev_bars", alpha_prev_bars)
        self.register_buffer("sigmas", sigmas)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped",
                             posterior_log_variance_clipped)

    def lag_timesteps(self, k):
        """Global counter k -> (hand timestep, object timestep). k runs from T-1 down to -|lag|."""
        L, T = self.obj_diffusion_lag, len(self.alpha_bars)
        t_hand = (k + max(-L, 0)).clamp(0, T - 1)
        t_obj = (k + max(L, 0)).clamp(0, T - 1)
        return t_hand, t_obj

    def forward(
        self, model, x_lhand, x_rhand,
        x_obj, obj_feat,
        timesteps=None, obj_timesteps=None, enc_text=None,
        force_mask=False,
        get_losses=False,
        valid_mask_lhand=None,
        valid_mask_rhand=None,
        valid_mask_obj=None,
        ldist_map=None,
        rdist_map=None,
        obj_verts_org=None,
        obj_pc_top_idx=None,
        motion_normalizer=None,
    ):
        return_list = []
        timesteps_pair = None
        used_alpha_bars_obj = None
        if timesteps == None:
            if self.obj_diffusion_lag != 0:
                L = abs(self.obj_diffusion_lag)
                k = torch.randint(-L, len(self.alpha_bars),
                                  (x_obj.size(0), )).to(x_obj.device)
                t_hand, timesteps = self.lag_timesteps(k)
                timesteps_pair = (t_hand, timesteps)
                used_alpha_bars = self.alpha_bars[t_hand][:, None, None]
                used_alpha_bars_obj = self.alpha_bars[timesteps][:, None, None]
            else:
                timesteps = torch.randint(
                    0, len(self.alpha_bars), (x_obj.size(0), )).to(x_obj.device)
                used_alpha_bars = self.alpha_bars[timesteps][:, None, None]
            epsilon_lhand = torch.randn_like(x_lhand)
            x_tilde_lhand = torch.sqrt(
                used_alpha_bars) * x_lhand + torch.sqrt(1 - used_alpha_bars) * epsilon_lhand
            epsilon_rhand = torch.randn_like(x_rhand)
            x_tilde_rhand = torch.sqrt(
                used_alpha_bars) * x_rhand + torch.sqrt(1 - used_alpha_bars) * epsilon_rhand
            epsilon_obj = torch.randn_like(x_obj)
            ab_obj = (used_alpha_bars if used_alpha_bars_obj is None
                      else used_alpha_bars_obj)
            x_tilde_obj = torch.sqrt(
                ab_obj) * x_obj + torch.sqrt(1 - ab_obj) * epsilon_obj

        else:
            timesteps = torch.Tensor(
                [timesteps for _ in range(x_obj.size(0))]).to(x_obj.device).long()
            x_tilde_lhand = x_lhand
            x_tilde_rhand = x_rhand
            x_tilde_obj = x_obj
            if obj_timesteps is not None:
                obj_timesteps = torch.Tensor(
                    [obj_timesteps for _ in range(x_obj.size(0))]
                ).to(x_obj.device).long()

        if timesteps_pair is not None:
            model_timesteps, timesteps_obj = timesteps_pair
        else:
            model_timesteps, timesteps_obj = timesteps, obj_timesteps
        pred_X0_lhand, pred_X0_rhand, pred_X0_obj, pred_lift_gate = model(
                x_tilde_lhand, x_tilde_rhand,
                x_tilde_obj, obj_feat,
                model_timesteps,
                force_mask,

                valid_mask_lhand,
                valid_mask_rhand,
                valid_mask_obj,
                timesteps_obj,
            )  # pred_X0_Hand (B x 60 x 99), pred_X0_obj (B x 60 x 9)

        return_list.append(pred_X0_lhand)
        return_list.append(pred_X0_rhand)
        return_list.append(pred_X0_obj)

        if get_losses:
            ab = used_alpha_bars if used_alpha_bars_obj is None else used_alpha_bars_obj
            self._loss_alpha_bar = ab.reshape(x_obj.shape[0]).detach() if ab is not None else None
            target_lhand = x_lhand.clone().detach()
            target_rhand = x_rhand.clone().detach()
            target_obj = x_obj.clone().detach()
            total_loss = self.get_loss(
                pred_X0_lhand, pred_X0_rhand, pred_X0_obj,
                target_lhand, target_rhand, target_obj,
                valid_mask_lhand, valid_mask_rhand, valid_mask_obj,
                motion_normalizer=motion_normalizer,
            )
            return_list.append(total_loss)

        return return_list

    def _contact_consistency_loss(
        self, pred_lhand, pred_rhand, pred_obj, targ_lhand, targ_rhand, targ_obj,
        valid_mask_lhand, valid_mask_rhand, valid_mask_obj, alpha_bar=None, tau=0.02,
    ):
        """(Unused ablation) disagreement between the contact surface point implied by the
        hand branch and by the object branch (m^2).

        Hand token = 100 points x [position p, displacement d to the nearest object point]
        (initial object frame); object token = relative pose (R, t). With the constant
        c = R_gt^-1 ((p_gt + d_gt) - t_gt) from GT, the penalty is
            || (p_hat + d_hat) - (R_hat c + t_hat) ||^2
        weighted by the GT contact proximity exp(-|d_gt|/tau) x valid mask x alpha_bar.
        """
        if pred_lhand.shape[-1] % 6 != 0:
            return pred_lhand.new_zeros(())
        B, T = pred_obj.shape[:2]
        Rg = rot6d_to_rotmat(targ_obj[..., 3:9].reshape(-1, 6)).reshape(B, T, 3, 3)
        tg = targ_obj[..., :3]
        Rp = rot6d_to_rotmat(pred_obj[..., 3:9].reshape(-1, 6)).reshape(B, T, 3, 3)
        tp = pred_obj[..., :3]
        if self.contact_consistency == "object_only":
            pass
        num, den = pred_lhand.new_zeros(()), pred_lhand.new_zeros(())
        for pred_h, targ_h, vm in ((pred_lhand, targ_lhand, valid_mask_lhand),
                                   (pred_rhand, targ_rhand, valid_mask_rhand)):
            ph = pred_h.reshape(B, T, -1, 6); th = targ_h.reshape(B, T, -1, 6)
            s_gt = th[..., :3] + th[..., 3:6]
            c = torch.einsum("btji,btnj->btni", Rg, s_gt - tg[:, :, None, :]).detach()
            s_pred = ph[..., :3] + ph[..., 3:6]
            q_pred = torch.einsum("btij,btnj->btni", Rp, c) + tp[:, :, None, :]
            if self.contact_consistency == "object_only":
                s_pred = s_pred.detach()
            elif self.contact_consistency == "hand_only":
                q_pred = q_pred.detach()
            w = torch.exp(-th[..., 3:6].norm(dim=-1) / tau)
            m = (vm.to(dtype=w.dtype) * valid_mask_obj.to(dtype=w.dtype))[:, :, None]
            if alpha_bar is not None:
                m = m * alpha_bar.reshape(B, 1, 1).to(dtype=w.dtype)
            w = w * m
            num = num + (w * (s_pred - q_pred).pow(2).sum(-1)).sum()
            den = den + w.sum()
        return num / den.clamp_min(1e-8)

    def get_loss(
        self,
        pred_X0_lhand, pred_X0_rhand, pred_X0_obj,
        targ_lhand, targ_rhand, targ_obj,
        valid_mask_lhand, valid_mask_rhand, valid_mask_obj,
        motion_normalizer=None,
    ):
        assert (
            valid_mask_lhand is not None and
            valid_mask_rhand is not None and
            valid_mask_obj is not None
        )

        simple_loss = get_l2_loss(
            pred_X0_lhand, pred_X0_rhand, pred_X0_obj,
            targ_lhand, targ_rhand, targ_obj,
            valid_mask_lhand, valid_mask_rhand, valid_mask_obj,
        )

        if motion_normalizer is not None:
            physical_pred_lhand, physical_pred_rhand, physical_pred_obj = (
                motion_normalizer.denormalize(
                    pred_X0_lhand, pred_X0_rhand, pred_X0_obj
                )
            )
            physical_targ_lhand, physical_targ_rhand, physical_targ_obj = (
                motion_normalizer.denormalize(targ_lhand, targ_rhand, targ_obj)
            )
        else:
            physical_pred_lhand = pred_X0_lhand
            physical_pred_rhand = pred_X0_rhand
            physical_pred_obj = pred_X0_obj
            physical_targ_lhand = targ_lhand
            physical_targ_rhand = targ_rhand
            physical_targ_obj = targ_obj

        object_translation_loss, object_rotation_loss = get_object_pose_losses(
            physical_pred_obj, physical_targ_obj, valid_mask_obj
        )

        contact_loss = physical_pred_obj.new_zeros(())
        if self.contact_consistency != "none":
            contact_loss = self._contact_consistency_loss(
                physical_pred_lhand, physical_pred_rhand, physical_pred_obj,
                physical_targ_lhand, physical_targ_rhand, physical_targ_obj,
                valid_mask_lhand, valid_mask_rhand, valid_mask_obj,
                alpha_bar=self._loss_alpha_bar)
        total_loss = {
            "simple_loss": simple_loss,
            "object_translation_loss": object_translation_loss,
            "object_rotation_loss": object_rotation_loss,
            "contact_consistency_loss": contact_loss,
        }
        return total_loss

    @torch.no_grad()
    def sampling(
        self,
        model, obj_feat,
        max_nframes,
        hand_nfeats, obj_nfeats,
        valid_mask_lhand,
        valid_mask_rhand,
        valid_mask_obj,
        device,
        enc_text=None,
        shared_noise_across_batch=False,
        gt_x0=None,
    ):
        sampling_number = len(valid_mask_lhand)
        if self.replace_sampling in ("gt_hand_fixed", "gt_obj_fixed"):
            assert gt_x0 is not None, "gt_* modes need sampling(gt_x0=...)"
            gl, gr, go = gt_x0
            assert gl.shape[-1] == hand_nfeats and go.shape[-1] == obj_nfeats, (gl.shape, go.shape)
            return self.ddpm_loop_replace(
                gl, gr, go, model, obj_feat,
                valid_mask_lhand, valid_mask_rhand, valid_mask_obj,
                fixed=self.replace_sampling[3:], shared_noise_across_batch=shared_noise_across_batch)

        noise_batch = 1 if shared_noise_across_batch else sampling_number
        sample_lhand = torch.randn(
            [noise_batch, max_nframes, hand_nfeats], device=device
        ).expand(sampling_number, -1, -1).clone()
        sample_rhand = torch.randn(
            [noise_batch, max_nframes, hand_nfeats], device=device
        ).expand(sampling_number, -1, -1).clone()
        sample_obj = torch.randn(
            [noise_batch, max_nframes, obj_nfeats], device=device
        ).expand(sampling_number, -1, -1).clone()

        loop = self.ddim_loop if self.sampler == "ddim" else self.ddpm_loop
        sample_lhand, sample_rhand, sample_obj = loop(
            sample_lhand, sample_rhand, sample_obj,
            model, obj_feat,
            valid_mask_lhand, valid_mask_rhand, valid_mask_obj,
            shared_noise_across_batch=shared_noise_across_batch,
        )
        if self.replace_sampling in ("hand_fixed", "obj_fixed"):
            sample_lhand, sample_rhand, sample_obj = self.ddpm_loop_replace(
                sample_lhand, sample_rhand, sample_obj, model, obj_feat,
                valid_mask_lhand, valid_mask_rhand, valid_mask_obj,
                fixed=self.replace_sampling, shared_noise_across_batch=shared_noise_across_batch)
        return sample_lhand, sample_rhand, sample_obj

    def ddpm_loop(
        self, sample_lhand, sample_rhand, sample_obj,
        model, obj_feat,
        valid_mask_lhand, valid_mask_rhand, valid_mask_obj,
        shared_noise_across_batch=False,
    ):
        """Reverse process where hand and object may use different diffusion timesteps
        (obj_diffusion_lag).

        lag > 0 delays the object so the hands finish first; lag < 0 does the opposite.
        With lag = 0 the steps are range(T-1, -1, -1) and both branches update together,
        identical to the plain loop including the order of random draws.
        """
        T = len(self.alpha_bars)
        L = int(self.obj_diffusion_lag)
        off_h, off_o = max(-L, 0), max(L, 0)
        steps = list(range(T - 1, -abs(L) - 1, -1))

        def posterior(x0, xt, t, noise):
            beta = self.betas[t]
            alpha = self.alphas[t]
            alpha_prev_bar = self.alpha_prev_bars[t]
            alpha_bar = self.alpha_bars[t]
            log_variance = self.posterior_log_variance_clipped[t]
            coefficient_X0 = (beta * torch.sqrt(alpha_prev_bar) / (1 - alpha_bar))
            coefficient_noise = ((1 - alpha_prev_bar) *
                                 torch.sqrt(alpha) / (1 - alpha_bar))
            mu = x0 * coefficient_X0 + xt * coefficient_noise
            return mu + torch.exp(0.5 * log_variance) * noise

        def draw(x, zero):
            if zero:
                return torch.zeros_like(x)
            if shared_noise_across_batch:
                return torch.randn_like(x[:1]).expand_as(x)
            return torch.randn_like(x)

        for k in tqdm.tqdm(steps, desc="sampling", total=len(steps)):
            t_hand = min(max(k + off_h, 0), T - 1)
            t_obj = min(max(k + off_o, 0), T - 1)
            hand_on = 0 <= k + off_h <= T - 1
            obj_on = 0 <= k + off_o <= T - 1

            noise_lhand = draw(sample_lhand, t_hand == 0)
            noise_rhand = draw(sample_rhand, t_hand == 0)
            noise_obj = draw(sample_obj, t_obj == 0)

            pred_X0_lhand, pred_X0_rhand, pred_X0_obj \
                = self.forward(
                    model, sample_lhand, sample_rhand,
                    sample_obj, obj_feat,
                    timesteps=t_hand,
                    obj_timesteps=(t_obj if L != 0 else None),
                    valid_mask_lhand=valid_mask_lhand,
                    valid_mask_rhand=valid_mask_rhand,
                    valid_mask_obj=valid_mask_obj
                )
            if self.cfg_scale != 1.0:
                u_lhand, u_rhand, u_obj = self.forward(
                    model, sample_lhand, sample_rhand,
                    sample_obj, obj_feat,
                    timesteps=t_hand,
                    obj_timesteps=(t_obj if L != 0 else None),
                    force_mask=True,
                    valid_mask_lhand=valid_mask_lhand,
                    valid_mask_rhand=valid_mask_rhand,
                    valid_mask_obj=valid_mask_obj
                )
                s = self.cfg_scale
                pred_X0_lhand = u_lhand + s * (pred_X0_lhand - u_lhand)
                pred_X0_rhand = u_rhand + s * (pred_X0_rhand - u_rhand)
                pred_X0_obj = u_obj + s * (pred_X0_obj - u_obj)

            if hand_on:
                sample_lhand = posterior(pred_X0_lhand, sample_lhand, t_hand, noise_lhand)
                sample_rhand = posterior(pred_X0_rhand, sample_rhand, t_hand, noise_rhand)
            if obj_on:
                sample_obj = posterior(pred_X0_obj, sample_obj, t_obj, noise_obj)

        return sample_lhand, sample_rhand, sample_obj

    def ddim_loop(
        self, sample_lhand, sample_rhand, sample_obj,
        model, obj_feat,
        valid_mask_lhand, valid_mask_rhand, valid_mask_obj,
        shared_noise_across_batch=False,
    ):
        """Deterministic DDIM (eta = 0) on the trained schedule, sample_steps timesteps evenly spaced
        from T-1 down to 0. The network predicts x0, so eps = (x_t - sqrt(ab_t) x0) / sqrt(1 - ab_t) and
        x_prev = sqrt(ab_prev) x0 + sqrt(1 - ab_prev) eps (ab_prev = 1 after the last step -> x0)."""
        assert self.obj_diffusion_lag == 0, "ddim sampler: obj_diffusion_lag must be 0"
        T = len(self.alpha_bars)
        S = self.sample_steps if self.sample_steps > 0 else T
        ts = sorted(set(torch.linspace(T - 1, 0, S).round().long().tolist()), reverse=True)
        for j, t in enumerate(tqdm.tqdm(ts, desc=f"ddim{len(ts)}", total=len(ts))):
            pl, pr, po = self.forward(
                model, sample_lhand, sample_rhand, sample_obj, obj_feat, timesteps=t,
                valid_mask_lhand=valid_mask_lhand, valid_mask_rhand=valid_mask_rhand, valid_mask_obj=valid_mask_obj)
            if self.cfg_scale != 1.0:
                ul, ur, uo = self.forward(
                    model, sample_lhand, sample_rhand, sample_obj, obj_feat, timesteps=t, force_mask=True,
                    valid_mask_lhand=valid_mask_lhand, valid_mask_rhand=valid_mask_rhand, valid_mask_obj=valid_mask_obj)
                s = self.cfg_scale
                pl, pr, po = ul + s * (pl - ul), ur + s * (pr - ur), uo + s * (po - uo)
            ab = self.alpha_bars[t]
            ab_prev = self.alpha_bars[ts[j + 1]] if j + 1 < len(ts) else torch.ones_like(ab)

            def step(x0, xt):
                eps = (xt - ab.sqrt() * x0) / (1 - ab).sqrt()
                return ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
            sample_lhand, sample_rhand, sample_obj = step(pl, sample_lhand), step(pr, sample_rhand), step(po, sample_obj)
        return sample_lhand, sample_rhand, sample_obj

    def ddpm_loop_replace(
        self, x0_lhand, x0_rhand, x0_obj,
        model, obj_feat,
        valid_mask_lhand, valid_mask_rhand, valid_mask_obj,
        fixed="hand_fixed", shared_noise_across_batch=False,
    ):
        """(Unused ablation) replacement sampling: clamp one branch to a clean value and
        denoise only the other branch from scratch.

        The clamped branch enters the model as x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps
        at every step, so the input stays in distribution.
        """
        T = len(self.alpha_bars)
        hand_fixed = fixed == "hand_fixed"

        def noised(x0, t):
            ab = self.alpha_bars[t]
            return ab.sqrt() * x0 + (1 - ab).sqrt() * torch.randn_like(x0)

        def posterior(x0, xt, t, noise):
            beta, alpha = self.betas[t], self.alphas[t]
            alpha_prev_bar, alpha_bar = self.alpha_prev_bars[t], self.alpha_bars[t]
            c_x0 = beta * torch.sqrt(alpha_prev_bar) / (1 - alpha_bar)
            c_noise = (1 - alpha_prev_bar) * torch.sqrt(alpha) / (1 - alpha_bar)
            return x0 * c_x0 + xt * c_noise + torch.exp(0.5 * self.posterior_log_variance_clipped[t]) * noise

        def draw(x, zero):
            if zero:
                return torch.zeros_like(x)
            return torch.randn_like(x[:1]).expand_as(x) if shared_noise_across_batch else torch.randn_like(x)

        sample_lhand = torch.randn_like(x0_lhand) if not hand_fixed else None
        sample_rhand = torch.randn_like(x0_rhand) if not hand_fixed else None
        sample_obj = torch.randn_like(x0_obj) if hand_fixed else None
        for t in tqdm.tqdm(range(T - 1, -1, -1), desc=f"replace[{fixed}]", total=T):
            if hand_fixed:
                xl, xr, xo = noised(x0_lhand, t), noised(x0_rhand, t), sample_obj
            else:
                xl, xr, xo = sample_lhand, sample_rhand, noised(x0_obj, t)
            pl, pr, po = self.forward(
                model, xl, xr, xo, obj_feat, timesteps=t,
                valid_mask_lhand=valid_mask_lhand, valid_mask_rhand=valid_mask_rhand,
                valid_mask_obj=valid_mask_obj)
            if hand_fixed:
                sample_obj = posterior(po, sample_obj, t, draw(sample_obj, t == 0))
            else:
                sample_lhand = posterior(pl, sample_lhand, t, draw(sample_lhand, t == 0))
                sample_rhand = posterior(pr, sample_rhand, t, draw(sample_rhand, t == 0))
        if hand_fixed:
            return x0_lhand, x0_rhand, sample_obj
        return sample_lhand, sample_rhand, x0_obj
