# coding=utf-8
# @FileName:FSIM.py
# @Time:2024/3/1 
# @Author: CZH
# coding=utf-8
# @FileName:FSIM.py
# @Time:2023/12/2
# @Author: gyy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FreBlock(nn.Module):
    def __init__(self):
        super(FreBlock, self).__init__()

    def forward(self, x):
        x = x + 1e-8
        mag = torch.abs(x)
        pha = torch.angle(x)

        return mag, pha


class FSIM(nn.Module):
    def __init__(self, n_feats=64):
        super().__init__()
        self.FF = FreBlock()

    def forward(self, x):
        b, c, H, W = x.shape
        # fft
        mix = x
        mix_mag, mix_pha = self.FF(mix)
        constant = mix_mag.mean()  # 振幅设成常量
        pha_only = constant * np.e ** (1j * mix_pha)
        # irfft
        x_out_main = torch.abs(torch.fft.irfft2(pha_only, s=(H, W), norm='backward')) + 1e-8
        out1 = x_out_main + mix
        return out1


if __name__ == '__main__':
    input = torch.randn(56, 1, 288, 144)
    model = FSIM()
    output = model(input)
    print(output.shape)
