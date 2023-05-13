''' PCEVA '''
import torch
from torch import nn
from torch.nn import functional as F


class PConv():
    ''' PConv (no clone) '''
    def __init__(self, *args, act=None, **kwargs):
        super().__init__(*args, **kwargs)
        if act is not None:
            self.register_module('act', act)
        else:
            self.act = None

    def forward(self, x: torch.Tensor):
        ''' forward '''
        # pylint: disable=not-callable
        if self.training:
            out = super().forward(x[:, :self.in_channels])
            if self.act is not None:
                out = self.act(out)
            return torch.cat([out, x[:, self.in_channels:]], dim=1)
        x[:, :self.in_channels] = super().forward(x[:, :self.in_channels])
        if self.act is not None:
            x[:, :self.in_channels] = self.act(x[:, :self.in_channels])
        return x


class ApproximateVariancePooling(nn.Module):
    ''''''
    def __init__(self, kernel_size) -> None:
        super().__init__()
        self.ap = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size//2, count_include_pad=False)
        self.mp = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        ''' forward '''
        return self.mp((self.ap(x) - x) ** 2)


class VarAttention(nn.Module):
    ''' attention based on local variance'''
    def __init__(self, channels, zip_channels):
        super().__init__()
        f = zip_channels
        self.body = nn.Sequential(
            # zip channel and hw
            nn.Conv2d(channels, f, 3, padding=1),
            nn.MaxPool2d(7, stride=3),
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            # approximate var pooling
            ApproximateVariancePooling(7),
            # fc
            nn.Conv2d(f, f, 3, padding=1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(f, channels, kernel_size=1),
            # to heatmap 
            nn.Sigmoid()
        )

    def forward(self, x):
        ''' forward '''
        # interpolate the heat map
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return x * w


class PBlock(nn.Module):
    ''' PBlock '''
    def __init__(self, channels, sub_channels, zip_channels, pre_conv=True):
        super().__init__()
        self.body = nn.Sequential()
        if pre_conv:
            self.body.append(nn.Conv2d(channels, channels, 1))
            self.body.append(nn.PReLU(channels))
        # PConvs
        for sub_channel in sub_channels:
            if sub_channel < channels:
                self.body.append(PConv(sub_channel, sub_channel, 3, padding=1, act=nn.PReLU(sub_channel)))      
            else:
                self.body.append(nn.Conv2d(sub_channel, sub_channel, 3, padding=1))
                self.body.append(nn.PReLU(sub_channel))
        # 基于局部var信息的attention
        self.esa = VarAttention(channels, zip_channels)

    def forward(self, input):
        ''' forward '''
        out = self.body(input)
        return self.esa(out)


class PCEVA(nn.Module):
    ''' PCEVA '''
    def __init__(self, channels, sub_channels, zip_channels, blocks, scale, pre_conv=True):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.PReLU(channels),
            *[PBlock(channels, sub_channels, zip_channels, pre_conv=pre_conv) for _ in range(blocks)],
            nn.Conv2d(channels, 3 * scale ** 2, 3, padding=1),
        )
        self.upsampler = nn.PixelShuffle(scale)

    def forward(self, x):
        ''' forward '''
        shortcut = torch.repeat_interleave(x, self.upsampler.upscale_factor ** 2, dim=1)
        x = self.body(x) + shortcut
        output = self.upsampler(x)
        return output



class Unshuffle_PCEVA(nn.Module):
    ''' PCEVA '''
    def __init__(self, channels, sub_channels, zip_channels, blocks, scale, pre_conv=True):
        super().__init__()

        self.scale = scale
        self.body = nn.Sequential(
            nn.Conv2d(3 * 4, channels, 2),
            nn.PReLU(channels),
            *[PBlock(channels, sub_channels, zip_channels, pre_conv=pre_conv) for _ in range(blocks)],
            nn.Conv2d(channels, 3 * (scale * 2) ** 2, 2),
        )

        self.upsampler = nn.PixelShuffle(scale * 2)
        self.repeat = nn.Conv2d(3 * 4, 3 * self.upsampler.upscale_factor ** 2, kernel_size=1, bias=False, groups=3)

        self.init_weight()

    def init_weight(self):
        self.repeat.weight.data[:] = 1
        self.repeat.weight.requires_grad_(False)

    def forward(self, x):
        ''' forward '''
        h, w = x.shape[-2:]
        if h % 2 == 1:
            x = F.pad(x, [0, 0, 0, 1])
        if w % 2 == 1:
            x = F.pad(x, [0, 1, 0, 0])
        x = F.pixel_unshuffle(x, 2)
        shortcut = self.repeat(x)
        x = self.body(x) + shortcut
        output = self.upsampler(x)
        if h % 2 == 1:
            output = output[:, :, :-self.scale, :]
        if w % 2 == 1:
            output = output[:, :, :, :-self.scale]
        return output


def pceva(model_name, scale):
    if model_name == "PCEVA-S":
        return Unshuffle_PCEVA(channels=64, sub_channels=[64, 48, 48, 32, 32], zip_channels=16, blocks=1, scale=scale, pre_conv=False)
    elif model_name == "PCEVA-M":
        return PCEVA(channels=32, sub_channels=[32, 16], zip_channels=16, blocks=3, scale=scale, pre_conv=False)
    elif model_name == "PCEVA-L":
        return PCEVA(channels=64, sub_channels=[64, 48, 32], zip_channels=16, blocks=4, scale=scale, pre_conv=False)