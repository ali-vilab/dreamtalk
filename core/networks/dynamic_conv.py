import math

import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, cond_planes, ratio, K, temperature=30, init_weight=True):
        super().__init__()
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.temprature = temperature
        assert cond_planes > ratio
        hidden_planes = cond_planes // ratio
        self.net = nn.Sequential(
            nn.Conv2d(cond_planes, hidden_planes, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes, K, kernel_size=1, bias=False),
        )

        if init_weight:
            self._initialize_weights()

    def update_temprature(self):
        if self.temprature > 1:
            self.temprature -= 1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, cond):
        """

        Args:
            cond (_type_): (B, C_style)

        Returns:
            _type_: (B, K)
        """

        # att = self.avgpool(cond)  # bs,dim,1,1
        att = cond.view(cond.shape[0], cond.shape[1], 1, 1)
        att = self.net(att).view(cond.shape[0], -1)  # bs,K
        return F.softmax(att / self.temprature, -1)


class DynamicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        cond_planes,
        kernel_size,
        stride,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        K=4,
        temperature=30,
        ratio=4,
        init_weight=True,
    ):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.cond_planes = cond_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.init_weight = init_weight
        self.attention = Attention(
            cond_planes=cond_planes, ratio=ratio, K=K, temperature=temperature, init_weight=init_weight
        )

        self.weight = nn.Parameter(
            torch.randn(K, out_planes, in_planes // groups, kernel_size, kernel_size), requires_grad=True
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(K, out_planes), requires_grad=True)
        else:
            self.bias = None

        if self.init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                if fan_in != 0:
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, cond):
        """

        Args:
            x (_type_): (B, C_in, L, 1)
            cond (_type_): (B, C_style)

        Returns:
            _type_: (B, C_out, L, 1)
        """
        bs, in_planels, h, w = x.shape
        softmax_att = self.attention(cond)  # bs,K
        x = x.view(1, -1, h, w)
        weight = self.weight.view(self.K, -1)  # K,-1
        aggregate_weight = torch.mm(softmax_att, weight).view(
            bs * self.out_planes, self.in_planes // self.groups, self.kernel_size, self.kernel_size
        )  # bs*out_p,in_p,k,k

        if self.bias is not None:
            bias = self.bias.view(self.K, -1)  # K,out_p
            aggregate_bias = torch.mm(softmax_att, bias).view(-1)  # bs*out_p
            output = F.conv2d(
                x, # 1, bs*in_p, L, 1
                weight=aggregate_weight,
                bias=aggregate_bias,
                stride=self.stride,
                padding=self.padding,
                groups=self.groups * bs,
                dilation=self.dilation,
            )
        else:
            output = F.conv2d(
                x,
                weight=aggregate_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                groups=self.groups * bs,
                dilation=self.dilation,
            )

        output = output.view(bs, self.out_planes, h, w)
        return output


if __name__ == "__main__":
    input = torch.randn(3, 32, 64, 64)
    m = DynamicConv(in_planes=32, out_planes=64, kernel_size=3, stride=1, padding=1, bias=True)
    out = m(input)
    print(out.shape)
