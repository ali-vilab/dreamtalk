import torch
import torch.nn as nn
from core.networks.mish import Mish


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Sequential(nn.Linear(input_dim, input_dim), Mish(), nn.Linear(input_dim, 1))
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask=None):
        """
        N: batch size, T: sequence length, H: Hidden dimension
        input:
            batch_rep : size (N, T, H)
        attention_weight:
            att_w : size (N, T, 1)
        att_mask:
            att_mask: size (N, T): if True, mask this item.
        return:
            utter_rep: size (N, H)
        """

        att_logits = self.W(batch_rep).squeeze(-1)
        # (N, T)
        if att_mask is not None:
            att_mask_logits = att_mask.to(dtype=batch_rep.dtype) * -100000.0
            # (N, T)
            att_logits = att_mask_logits + att_logits

        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep


if __name__ == "__main__":
    batch = torch.randn(8, 64, 256)
    self_attn_pool = SelfAttentionPooling(256)
    att_mask = torch.zeros(8, 64)
    att_mask[:, 60:] = 1
    att_mask = att_mask.to(torch.bool)
    output = self_attn_pool(batch, att_mask)
    # (8, 256)

    print("hello")
