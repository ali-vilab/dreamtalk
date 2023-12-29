import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
from core.networks import get_network
from core.utils import sinusoidal_embedding


class VarianceSchedule(Module):
    def __init__(self, num_steps, beta_1, beta_T, mode="linear"):
        super().__init__()
        assert mode in ("linear",)
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)  # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[
                i
            ]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sigmas_flex", sigmas_flex)
        self.register_buffer("sigmas_inflex", sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps + 1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (
            1 - flexibility
        )
        return sigmas


class NoisePredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        content_encoder_class = get_network(cfg.CONTENT_ENCODER_TYPE)
        self.content_encoder = content_encoder_class(**cfg.CONTENT_ENCODER)

        style_encoder_class = get_network(cfg.STYLE_ENCODER_TYPE)
        cfg.defrost()
        cfg.STYLE_ENCODER.input_dim = cfg.DATASET.FACE3D_DIM
        cfg.freeze()
        self.style_encoder = style_encoder_class(**cfg.STYLE_ENCODER)

        decoder_class = get_network(cfg.DECODER_TYPE)
        cfg.defrost()
        cfg.DECODER.output_dim = cfg.DATASET.FACE3D_DIM
        cfg.freeze()
        self.decoder = decoder_class(**cfg.DECODER)

        self.content_xt_to_decoder_input_wo_time = nn.Sequential(
            nn.Linear(cfg.D_MODEL + cfg.DATASET.FACE3D_DIM, cfg.D_MODEL),
            nn.ReLU(),
            nn.Linear(cfg.D_MODEL, cfg.D_MODEL),
            nn.ReLU(),
            nn.Linear(cfg.D_MODEL, cfg.D_MODEL),
        )

        self.time_sinusoidal_dim = cfg.D_MODEL
        self.time_embed_net = nn.Sequential(
            nn.Linear(cfg.D_MODEL, cfg.D_MODEL),
            nn.SiLU(),
            nn.Linear(cfg.D_MODEL, cfg.D_MODEL),
        )

    def forward(self, x_t, t, audio, style_clip, style_pad_mask, ready_style_code=None):
        """_summary_

        Args:
            x_t (_type_): (B, L, C_face)
            t (_type_): (B,) dtype:float32
            audio (_type_): (B, L, W)
            style_clip (_type_): (B, L_clipmax, C_face3d)
            style_pad_mask : (B, L_clipmax)
            ready_style_code: (B, C_model)
        Returns:
            e_theta : (B, L, C_face)
        """
        W = audio.shape[2]
        content = self.content_encoder(audio)
        # (B, L, W, C_model)
        x_t_expand = x_t.unsqueeze(2).repeat(1, 1, W, 1)
        # (B, L, C_face) -> (B, L, W, C_face)
        content_xt_concat = torch.cat((content, x_t_expand), dim=3)
        # (B, L, W, C_model+C_face)
        decoder_input_without_time = self.content_xt_to_decoder_input_wo_time(
            content_xt_concat
        )
        # (B, L, W, C_model)

        time_sinusoidal = sinusoidal_embedding(t, self.time_sinusoidal_dim)
        # (B, C_embed)
        time_embedding = self.time_embed_net(time_sinusoidal)
        # (B, C_model)
        B, C = time_embedding.shape
        time_embed_expand = time_embedding.view(B, 1, 1, C)
        decoder_input = decoder_input_without_time + time_embed_expand
        # (B, L, W, C_model)

        if ready_style_code is not None:
            style_code = ready_style_code
        else:
            style_code = self.style_encoder(style_clip, style_pad_mask)
        # (B, C_model)

        e_theta = self.decoder(decoder_input, style_code)
        # (B, L, C_face)
        return e_theta
