import math
import torch
import torch.nn.functional as F
from torch.nn import Module
from core.networks.diffusion_util import VarianceSchedule
import numpy as np


def face3d_raw_to_norm(face3d_raw, exp_min, exp_max):
    """

    Args:
        face3d_raw (_type_): (B, L, C_face3d)
        exp_min (_type_): (C_face3d)
        exp_max (_type_): (C_face3d)

    Returns:
        _type_: (B, L, C_face3d) in [-1, 1]
    """
    exp_min_expand = exp_min[None, None, :]
    exp_max_expand = exp_max[None, None, :]
    face3d_norm_01 = (face3d_raw - exp_min_expand) / (exp_max_expand - exp_min_expand)
    face3d_norm = face3d_norm_01 * 2 - 1
    return face3d_norm


def face3d_norm_to_raw(face3d_norm, exp_min, exp_max):
    """

    Args:
        face3d_norm (_type_): (B, L, C_face3d)
        exp_min (_type_): (C_face3d)
        exp_max (_type_): (C_face3d)

    Returns:
        _type_: (B, L, C_face3d)
    """
    exp_min_expand = exp_min[None, None, :]
    exp_max_expand = exp_max[None, None, :]
    face3d_norm_01 = (face3d_norm + 1) / 2
    face3d_raw = face3d_norm_01 * (exp_max_expand - exp_min_expand) + exp_min_expand
    return face3d_raw


class DiffusionNet(Module):
    def __init__(self, cfg, net, var_sched: VarianceSchedule):
        super().__init__()
        self.cfg = cfg
        self.net = net
        self.var_sched = var_sched
        self.face3d_latent_type = self.cfg.TRAIN.FACE3D_LATENT.TYPE
        self.predict_what = self.cfg.DIFFUSION.PREDICT_WHAT

        if self.cfg.CF_GUIDANCE.TRAINING:
            null_style_clip = torch.zeros(
                self.cfg.DATASET.STYLE_MAX_LEN, self.cfg.DATASET.FACE3D_DIM
            )
            self.register_buffer("null_style_clip", null_style_clip)

            null_pad_mask = torch.tensor([False] * self.cfg.DATASET.STYLE_MAX_LEN)
            self.register_buffer("null_pad_mask", null_pad_mask)

    def _face3d_to_latent(self, face3d):
        latent = None
        if self.face3d_latent_type == "face3d":
            latent = face3d
        elif self.face3d_latent_type == "normalized_face3d":
            latent = face3d_raw_to_norm(
                face3d, exp_min=self.exp_min, exp_max=self.exp_max
            )
        else:
            raise ValueError(f"Invalid face3d latent type: {self.face3d_latent_type}")
        return latent

    def _latent_to_face3d(self, latent):
        face3d = None
        if self.face3d_latent_type == "face3d":
            face3d = latent
        elif self.face3d_latent_type == "normalized_face3d":
            latent = torch.clamp(latent, min=-1, max=1)
            face3d = face3d_norm_to_raw(
                latent, exp_min=self.exp_min, exp_max=self.exp_max
            )
        else:
            raise ValueError(f"Invalid face3d latent type: {self.face3d_latent_type}")
        return face3d

    def ddim_sample(
        self,
        audio,
        style_clip,
        style_pad_mask,
        output_dim,
        flexibility=0.0,
        ret_traj=False,
        use_cf_guidance=False,
        cfg_scale=2.0,
        ddim_num_step=50,
        ready_style_code=None,
    ):
        """

        Args:
            audio (_type_): (B, L, W) or (B, L, W, C)
            style_clip (_type_): (B, L_clipmax, C_face3d)
            style_pad_mask : (B, L_clipmax)
            pose_dim (_type_): int
            flexibility (float, optional): _description_. Defaults to 0.0.
            ret_traj (bool, optional): _description_. Defaults to False.


        Returns:
            _type_: (B, L, C_face)
        """
        if self.predict_what != "x0":
            raise NotImplementedError(self.predict_what)

        if ready_style_code is not None and use_cf_guidance:
            raise NotImplementedError("not implement cfg for ready style code")

        c = self.var_sched.num_steps // ddim_num_step
        time_steps = torch.tensor(
            np.asarray(list(range(0, self.var_sched.num_steps, c))) + 1
        )
        assert len(time_steps) == ddim_num_step
        prev_time_steps = torch.cat((torch.tensor([0]), time_steps[:-1]))

        batch_size, output_len = audio.shape[:2]
        # batch_size = context.size(0)
        context = {
            "audio": audio,
            "style_clip": style_clip,
            "style_pad_mask": style_pad_mask,
            "ready_style_code": ready_style_code,
        }
        if use_cf_guidance:
            uncond_style_clip = self.null_style_clip.unsqueeze(0).repeat(
                batch_size, 1, 1
            )
            uncond_pad_mask = self.null_pad_mask.unsqueeze(0).repeat(batch_size, 1)

            context_double = {
                "audio": torch.cat([audio] * 2, dim=0),
                "style_clip": torch.cat([style_clip, uncond_style_clip], dim=0),
                "style_pad_mask": torch.cat([style_pad_mask, uncond_pad_mask], dim=0),
                "ready_style_code": None
                if ready_style_code is None
                else torch.cat(
                    [
                        ready_style_code,
                        self.net.style_encoder(uncond_style_clip, uncond_pad_mask),
                    ],
                    dim=0,
                ),
            }

        x_t = torch.randn([batch_size, output_len, output_dim]).to(audio.device)

        for idx in list(range(ddim_num_step))[::-1]:
            t = time_steps[idx]
            t_prev = prev_time_steps[idx]
            ddim_alpha = self.var_sched.alpha_bars[t]
            ddim_alpha_prev = self.var_sched.alpha_bars[t_prev]

            t_tensor = torch.tensor([t] * batch_size).to(audio.device).float()
            if use_cf_guidance:
                x_t_double = torch.cat([x_t] * 2, dim=0)
                t_tensor_double = torch.cat([t_tensor] * 2, dim=0)
                cond_output, uncond_output = self.net(
                    x_t_double, t=t_tensor_double, **context_double
                ).chunk(2)
                diff_output = uncond_output + cfg_scale * (cond_output - uncond_output)
            else:
                diff_output = self.net(x_t, t=t_tensor, **context)

            pred_x0 = diff_output
            eps = (x_t - torch.sqrt(ddim_alpha) * pred_x0) / torch.sqrt(1 - ddim_alpha)
            c1 = torch.sqrt(ddim_alpha_prev)
            c2 = torch.sqrt(1 - ddim_alpha_prev)

            x_t = c1 * pred_x0 + c2 * eps

        latent_output = x_t
        face3d_output = self._latent_to_face3d(latent_output)
        return face3d_output

    def sample(
        self,
        audio,
        style_clip,
        style_pad_mask,
        output_dim,
        flexibility=0.0,
        ret_traj=False,
        use_cf_guidance=False,
        cfg_scale=2.0,
        sample_method="ddpm",
        ddim_num_step=50,
        ready_style_code=None,
    ):
        # sample_method = kwargs["sample_method"]
        if sample_method == "ddpm":
            if ready_style_code is not None:
                raise NotImplementedError("ready style code in ddpm")
            return self.ddpm_sample(
                audio,
                style_clip,
                style_pad_mask,
                output_dim,
                flexibility=flexibility,
                ret_traj=ret_traj,
                use_cf_guidance=use_cf_guidance,
                cfg_scale=cfg_scale,
            )
        elif sample_method == "ddim":
            return self.ddim_sample(
                audio,
                style_clip,
                style_pad_mask,
                output_dim,
                flexibility=flexibility,
                ret_traj=ret_traj,
                use_cf_guidance=use_cf_guidance,
                cfg_scale=cfg_scale,
                ddim_num_step=ddim_num_step,
                ready_style_code=ready_style_code,
            )

    def ddpm_sample(
        self,
        audio,
        style_clip,
        style_pad_mask,
        output_dim,
        flexibility=0.0,
        ret_traj=False,
        use_cf_guidance=False,
        cfg_scale=2.0,
    ):
        """

        Args:
            audio (_type_): (B, L, W) or (B, L, W, C)
            style_clip (_type_): (B, L_clipmax, C_face3d)
            style_pad_mask : (B, L_clipmax)
            pose_dim (_type_): int
            flexibility (float, optional): _description_. Defaults to 0.0.
            ret_traj (bool, optional): _description_. Defaults to False.


        Returns:
            _type_: (B, L, C_face)
        """
        batch_size, output_len = audio.shape[:2]
        # batch_size = context.size(0)
        context = {
            "audio": audio,
            "style_clip": style_clip,
            "style_pad_mask": style_pad_mask,
        }
        if use_cf_guidance:
            uncond_style_clip = self.null_style_clip.unsqueeze(0).repeat(
                batch_size, 1, 1
            )
            uncond_pad_mask = self.null_pad_mask.unsqueeze(0).repeat(batch_size, 1)
            context_double = {
                "audio": torch.cat([audio] * 2, dim=0),
                "style_clip": torch.cat([style_clip, uncond_style_clip], dim=0),
                "style_pad_mask": torch.cat([style_pad_mask, uncond_pad_mask], dim=0),
            }

        x_T = torch.randn([batch_size, output_len, output_dim]).to(audio.device)
        traj = {self.var_sched.num_steps: x_T}
        for t in range(self.var_sched.num_steps, 0, -1):
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            alpha_bar_prev = self.var_sched.alpha_bars[t - 1]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            x_t = traj[t]
            t_tensor = torch.tensor([t] * batch_size).to(audio.device).float()
            if use_cf_guidance:
                x_t_double = torch.cat([x_t] * 2, dim=0)
                t_tensor_double = torch.cat([t_tensor] * 2, dim=0)
                cond_output, uncond_output = self.net(
                    x_t_double, t=t_tensor_double, **context_double
                ).chunk(2)
                diff_output = uncond_output + cfg_scale * (cond_output - uncond_output)
            else:
                diff_output = self.net(x_t, t=t_tensor, **context)

            if self.predict_what == "noise":
                c0 = 1.0 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                x_next = c0 * (x_t - c1 * diff_output) + sigma * z
            elif self.predict_what == "x0":
                d0 = torch.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar)
                d1 = torch.sqrt(alpha_bar_prev) * (1 - alpha) / (1 - alpha_bar)
                x_next = d0 * x_t + d1 * diff_output + sigma * z
            traj[t - 1] = x_next.detach()
            traj[t] = traj[t].cpu()
            if not ret_traj:
                del traj[t]

        if ret_traj:
            raise NotImplementedError
            return traj
        else:
            latent_output = traj[0]
            face3d_output = self._latent_to_face3d(latent_output)
            return face3d_output


if __name__ == "__main__":
    from core.networks.diffusion_util import NoisePredictor, VarianceSchedule

    diffnet = DiffusionNet(
        net=NoisePredictor(),
        var_sched=VarianceSchedule(
            num_steps=500, beta_1=1e-4, beta_T=0.02, mode="linear"
        ),
    )

    import torch

    gt_face3d = torch.randn(16, 64, 64)
    audio = torch.randn(16, 64, 11)
    style_clip = torch.randn(16, 256, 64)
    style_pad_mask = torch.ones(16, 256)

    context = {
        "audio": audio,
        "style_clip": style_clip,
        "style_pad_mask": style_pad_mask,
    }

    loss = diffnet.get_loss(gt_face3d, context)

    print("hello")
