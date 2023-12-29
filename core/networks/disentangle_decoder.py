import torch
from torch import nn

from .transformer import (
    PositionalEncoding,
    TransformerDecoderLayer,
    TransformerDecoder,
)
from core.networks.dynamic_fc_decoder import DynamicFCDecoderLayer, DynamicFCDecoder
from core.utils import _reset_parameters


def get_decoder_network(
    network_type,
    d_model,
    nhead,
    dim_feedforward,
    dropout,
    activation,
    normalize_before,
    num_decoder_layers,
    return_intermediate_dec,
    dynamic_K,
    dynamic_ratio,
):
    decoder = None
    if network_type == "TransformerDecoder":
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        norm = nn.LayerNorm(d_model)
        decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            norm,
            return_intermediate_dec,
        )
    elif network_type == "DynamicFCDecoder":
        d_style = d_model
        decoder_layer = DynamicFCDecoderLayer(
            d_model,
            nhead,
            d_style,
            dynamic_K,
            dynamic_ratio,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )
        norm = nn.LayerNorm(d_model)
        decoder = DynamicFCDecoder(
            decoder_layer, num_decoder_layers, norm, return_intermediate_dec
        )
    elif network_type == "DynamicFCEncoder":
        d_style = d_model
        decoder_layer = DynamicFCEncoderLayer(
            d_model,
            nhead,
            d_style,
            dynamic_K,
            dynamic_ratio,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )
        norm = nn.LayerNorm(d_model)
        decoder = DynamicFCEncoder(decoder_layer, num_decoder_layers, norm)

    else:
        raise ValueError(f"Invalid network_type {network_type}")

    return decoder


class DisentangleDecoder(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_decoder_layers=3,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        pos_embed_len=80,
        upper_face3d_indices=tuple(list(range(19)) + list(range(46, 51))),
        lower_face3d_indices=tuple(range(19, 46)),
        network_type="None",
        dynamic_K=None,
        dynamic_ratio=None,
        **_,
    ) -> None:
        super().__init__()

        self.upper_face3d_indices = upper_face3d_indices
        self.lower_face3d_indices = lower_face3d_indices

        # upper_decoder_layer = TransformerDecoderLayer(
        #     d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        # )
        # upper_decoder_norm = nn.LayerNorm(d_model)
        # self.upper_decoder = TransformerDecoder(
        #     upper_decoder_layer,
        #     num_decoder_layers,
        #     upper_decoder_norm,
        #     return_intermediate=return_intermediate_dec,
        # )
        self.upper_decoder = get_decoder_network(
            network_type,
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            num_decoder_layers,
            return_intermediate_dec,
            dynamic_K,
            dynamic_ratio,
        )
        _reset_parameters(self.upper_decoder)

        # lower_decoder_layer = TransformerDecoderLayer(
        #     d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        # )
        # lower_decoder_norm = nn.LayerNorm(d_model)
        # self.lower_decoder = TransformerDecoder(
        #     lower_decoder_layer,
        #     num_decoder_layers,
        #     lower_decoder_norm,
        #     return_intermediate=return_intermediate_dec,
        # )
        self.lower_decoder = get_decoder_network(
            network_type,
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            num_decoder_layers,
            return_intermediate_dec,
            dynamic_K,
            dynamic_ratio,
        )
        _reset_parameters(self.lower_decoder)

        self.pos_embed = PositionalEncoding(d_model, pos_embed_len)

        tail_hidden_dim = d_model // 2
        self.upper_tail_fc = nn.Sequential(
            nn.Linear(d_model, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, len(upper_face3d_indices)),
        )
        self.lower_tail_fc = nn.Sequential(
            nn.Linear(d_model, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, len(lower_face3d_indices)),
        )

    def forward(self, content, style_code):
        """

        Args:
            content (_type_): (B, num_frames, window, C_dmodel)
            style_code (_type_): (B, C_dmodel)

        Returns:
            face3d: (B, L_clip, C_3dmm)
        """
        B, N, W, C = content.shape
        style = style_code.reshape(B, 1, 1, C).expand(B, N, W, C)
        style = style.permute(2, 0, 1, 3).reshape(W, B * N, C)
        # (W, B*N, C)

        content = content.permute(2, 0, 1, 3).reshape(W, B * N, C)
        # (W, B*N, C)
        tgt = torch.zeros_like(style)
        pos_embed = self.pos_embed(W)
        pos_embed = pos_embed.permute(1, 0, 2)

        upper_face3d_feat = self.upper_decoder(
            tgt, content, pos=pos_embed, query_pos=style
        )[0]
        # (W, B*N, C)
        upper_face3d_feat = upper_face3d_feat.permute(1, 0, 2).reshape(B, N, W, C)[
            :, :, W // 2, :
        ]
        # (B, N, C)
        upper_face3d = self.upper_tail_fc(upper_face3d_feat)
        # (B, N, C_exp)

        lower_face3d_feat = self.lower_decoder(
            tgt, content, pos=pos_embed, query_pos=style
        )[0]
        lower_face3d_feat = lower_face3d_feat.permute(1, 0, 2).reshape(B, N, W, C)[
            :, :, W // 2, :
        ]
        lower_face3d = self.lower_tail_fc(lower_face3d_feat)
        C_exp = len(self.upper_face3d_indices) + len(self.lower_face3d_indices)
        face3d = torch.zeros(B, N, C_exp).to(upper_face3d)
        face3d[:, :, self.upper_face3d_indices] = upper_face3d
        face3d[:, :, self.lower_face3d_indices] = lower_face3d
        return face3d


if __name__ == "__main__":
    import sys

    sys.path.append("/home/mayifeng/Research/styleTH")

    from configs.default import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.merge_from_file("configs/styleTH_unpair_lsfm_emotion.yaml")
    cfg.freeze()

    # content_encoder = ContentEncoder(**cfg.CONTENT_ENCODER)

    # dummy_audio = torch.randint(0, 41, (5, 64, 11))
    # dummy_content = content_encoder(dummy_audio)

    # style_encoder = StyleEncoder(**cfg.STYLE_ENCODER)
    # dummy_face3d_seq = torch.randn(5, 64, 64)
    # dummy_style_code = style_encoder(dummy_face3d_seq)

    decoder = DisentangleDecoder(**cfg.DECODER)
    dummy_content = torch.randn(5, 64, 11, 256)
    dummy_style = torch.randn(5, 256)
    dummy_output = decoder(dummy_content, dummy_style)

    print("hello")
