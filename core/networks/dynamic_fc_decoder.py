import torch.nn as nn
import torch

from core.networks.transformer import _get_activation_fn, _get_clones
from core.networks.dynamic_linear import DynamicLinear


class DynamicFCDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        d_style,
        dynamic_K,
        dynamic_ratio,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear1 = DynamicLinear(d_model, dim_feedforward, d_style, K=dynamic_K, ratio=dynamic_ratio)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # self.linear2 = DynamicLinear(dim_feedforward, d_model, d_style, K=dynamic_K, ratio=dynamic_ratio)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        style,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        # q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(tgt, tgt, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=tgt, key=memory, value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt, style))), style)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt, style))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    # def forward_pre(
    #     self,
    #     tgt,
    #     memory,
    #     tgt_mask=None,
    #     memory_mask=None,
    #     tgt_key_padding_mask=None,
    #     memory_key_padding_mask=None,
    #     pos=None,
    #     query_pos=None,
    # ):
    #     tgt2 = self.norm1(tgt)
    #     # q = k = self.with_pos_embed(tgt2, query_pos)
    #     tgt2 = self.self_attn(tgt2, tgt2, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
    #     tgt = tgt + self.dropout1(tgt2)
    #     tgt2 = self.norm2(tgt)
    #     tgt2 = self.multihead_attn(
    #         query=tgt2, key=memory, value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask
    #     )[0]
    #     tgt = tgt + self.dropout2(tgt2)
    #     tgt2 = self.norm3(tgt)
    #     tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
    #     tgt = tgt + self.dropout3(tgt2)
    #     return tgt

    def forward(
        self,
        tgt,
        memory,
        style,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        if self.normalize_before:
            raise NotImplementedError
            # return self.forward_pre(
            #     tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos
            # )
        return self.forward_post(
            tgt, memory, style, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos
        )


class DynamicFCDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        style = query_pos[0]
        # (B*N, C)
        output = tgt + pos + query_pos

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                style,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


if __name__ == "__main__":
    query = torch.randn(11, 1024, 256)
    content = torch.randn(11, 1024, 256)
    style = torch.randn(1024, 256)
    pos = torch.randn(11, 1, 256)
    m = DynamicFCDecoderLayer(256, 4, 256, 4, 4, 1024)

    out = m(query, content, style, pos=pos)
    print(out.shape)
