import torch
from torch import nn

from .transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
    PositionalEncoding,
    TransformerDecoderLayer,
    TransformerDecoder,
)
from core.utils import _reset_parameters
from core.networks.self_attention_pooling import SelfAttentionPooling


# class ContentEncoder(nn.Module):
#     def __init__(
#         self,
#         d_model=512,
#         nhead=8,
#         num_encoder_layers=6,
#         dim_feedforward=2048,
#         dropout=0.1,
#         activation="relu",
#         normalize_before=False,
#         pos_embed_len=80,
#         ph_embed_dim=128,
#     ):
#         super().__init__()

#         encoder_layer = TransformerEncoderLayer(
#             d_model, nhead, dim_feedforward, dropout, activation, normalize_before
#         )
#         encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
#         self.encoder = TransformerEncoder(
#             encoder_layer, num_encoder_layers, encoder_norm
#         )

#         _reset_parameters(self.encoder)

#         self.pos_embed = PositionalEncoding(d_model, pos_embed_len)

#         self.ph_embedding = nn.Embedding(41, ph_embed_dim)
#         self.increase_embed_dim = nn.Linear(ph_embed_dim, d_model)

#     def forward(self, x):
#         """

#         Args:
#             x (_type_): (B, num_frames, window)

#         Returns:
#             content: (B, num_frames, window, C_dmodel)
#         """
#         x_embedding = self.ph_embedding(x)
#         x_embedding = self.increase_embed_dim(x_embedding)
#         # (B, N, W, C)
#         B, N, W, C = x_embedding.shape
#         x_embedding = x_embedding.reshape(B * N, W, C)
#         x_embedding = x_embedding.permute(1, 0, 2)
#         # (W, B*N, C)

#         pos = self.pos_embed(W)
#         pos = pos.permute(1, 0, 2)
#         # (W, 1, C)

#         content = self.encoder(x_embedding, pos=pos)
#         # (W, B*N, C)
#         content = content.permute(1, 0, 2).reshape(B, N, W, C)
#         # (B, N, W, C)

#         return content


class ContentW2VEncoder(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        pos_embed_len=80,
        ph_embed_dim=128,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        _reset_parameters(self.encoder)

        self.pos_embed = PositionalEncoding(d_model, pos_embed_len)

        self.increase_embed_dim = nn.Linear(1024, d_model)

    def forward(self, x):
        """
        Args:
            x (_type_): (B, num_frames, window, C_wav2vec)

        Returns:
            content: (B, num_frames, window, C_dmodel)
        """
        x_embedding = self.increase_embed_dim(
            x
        )  # [16, 64, 11, 1024] -> [16, 64, 11, 256]
        # (B, N, W, C)
        B, N, W, C = x_embedding.shape
        x_embedding = x_embedding.reshape(B * N, W, C)
        x_embedding = x_embedding.permute(1, 0, 2)  # [11, 1024, 256]
        # (W, B*N, C)

        pos = self.pos_embed(W)
        pos = pos.permute(1, 0, 2)  # [11, 1, 256]
        # (W, 1, C)

        content = self.encoder(x_embedding, pos=pos)  # [11, 1024, 256]
        # (W, B*N, C)
        content = content.permute(1, 0, 2).reshape(B, N, W, C)
        # (B, N, W, C)

        return content


class StyleEncoder(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        pos_embed_len=80,
        input_dim=128,
        aggregate_method="average",
    ):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )
        _reset_parameters(self.encoder)

        self.pos_embed = PositionalEncoding(d_model, pos_embed_len)

        self.increase_embed_dim = nn.Linear(input_dim, d_model)

        self.aggregate_method = None
        if aggregate_method == "self_attention_pooling":
            self.aggregate_method = SelfAttentionPooling(d_model)
        elif aggregate_method == "average":
            pass
        else:
            raise ValueError(f"Invalid aggregate method {aggregate_method}")

    def forward(self, x, pad_mask=None):
        """

        Args:
            x (_type_): (B, num_frames(L), C_exp)
            pad_mask: (B, num_frames)

        Returns:
            style_code: (B, C_model)
        """
        x = self.increase_embed_dim(x)
        # (B, L, C)
        x = x.permute(1, 0, 2)
        # (L, B, C)

        pos = self.pos_embed(x.shape[0])
        pos = pos.permute(1, 0, 2)
        # (L, 1, C)

        style = self.encoder(x, pos=pos, src_key_padding_mask=pad_mask)
        # (L, B, C)

        if self.aggregate_method is not None:
            permute_style = style.permute(1, 0, 2)
            # (B, L, C)
            style_code = self.aggregate_method(permute_style, pad_mask)
            return style_code

        if pad_mask is None:
            style = style.permute(1, 2, 0)
            # (B, C, L)
            style_code = style.mean(2)
            # (B, C)
        else:
            permute_style = style.permute(1, 0, 2)
            # (B, L, C)
            permute_style[pad_mask] = 0
            sum_style_code = permute_style.sum(dim=1)
            # (B, C)
            valid_token_num = (~pad_mask).sum(dim=1).unsqueeze(-1)
            # (B, 1)
            style_code = sum_style_code / valid_token_num
            # (B, C)

        return style_code


class Decoder(nn.Module):
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
        output_dim=64,
        **_,
    ) -> None:
        super().__init__()

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        _reset_parameters(self.decoder)

        self.pos_embed = PositionalEncoding(d_model, pos_embed_len)

        tail_hidden_dim = d_model // 2
        self.tail_fc = nn.Sequential(
            nn.Linear(d_model, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, tail_hidden_dim),
            nn.ReLU(),
            nn.Linear(tail_hidden_dim, output_dim),
        )

    def forward(self, content, style_code):
        """

        Args:
            content (_type_): (B, num_frames, window, C_dmodel)
            style_code (_type_): (B, C_dmodel)

        Returns:
            face3d: (B, num_frames, C_3dmm)
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
        face3d_feat = self.decoder(tgt, content, pos=pos_embed, query_pos=style)[0]
        # (W, B*N, C)
        face3d_feat = face3d_feat.permute(1, 0, 2).reshape(B, N, W, C)[:, :, W // 2, :]
        # (B, N, C)
        face3d = self.tail_fc(face3d_feat)
        # (B, N, C_exp)
        return face3d


if __name__ == "__main__":
    import sys

    sys.path.append("/home/mayifeng/Research/styleTH")

    from configs.default import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.merge_from_file("configs/styleTH_bp.yaml")
    cfg.freeze()

    # content_encoder = ContentEncoder(**cfg.CONTENT_ENCODER)

    # dummy_audio = torch.randint(0, 41, (5, 64, 11))
    # dummy_content = content_encoder(dummy_audio)

    # style_encoder = StyleEncoder(**cfg.STYLE_ENCODER)
    # dummy_face3d_seq = torch.randn(5, 64, 64)
    # dummy_style_code = style_encoder(dummy_face3d_seq)

    decoder = Decoder(**cfg.DECODER)
    dummy_content = torch.randn(5, 64, 11, 512)
    dummy_style = torch.randn(5, 512)
    dummy_output = decoder(dummy_content, dummy_style)

    print("hello")
