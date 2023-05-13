''' PCEVA '''
import torch
from torch import nn
from torch.nn import functional as F


class SeqConv3x3(nn.Module):
    ''' SeqConv3x3 '''
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier):
        super().__init__()

        self.type = seq_type
        self.out_planes = out_planes

        if self.type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = torch.nn.Conv2d(inp_planes,
                                    self.mid_planes,
                                    kernel_size=1,
                                    padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            conv1 = torch.nn.Conv2d(self.mid_planes,
                                    self.out_planes,
                                    kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias
        elif self.type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(inp_planes,
                                    self.out_planes,
                                    kernel_size=1,
                                    padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes, ))
            self.bias = nn.Parameter(bias)
            self.mask = torch.zeros((self.out_planes, 1, 3, 3),
                                    dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        elif self.type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(inp_planes,
                                    self.out_planes,
                                    kernel_size=1,
                                    padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes, ))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            self.mask = torch.zeros((self.out_planes, 1, 3, 3),
                                    dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        elif self.type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(inp_planes,
                                    self.out_planes,
                                    kernel_size=1,
                                    padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes, ))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            self.mask = torch.zeros((self.out_planes, 1, 3, 3),
                                    dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        ''' forward '''
        if self.type == 'conv1x1-conv3x3':
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        else:
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            y1 = F.conv2d(input=y0,
                          weight=self.scale * self.mask,
                          bias=self.bias,
                          stride=1,
                          groups=self.out_planes)
        return y1

    def rep_params(self):
        ''' rep_params '''
        device = self.k0.get_device()
        if device < 0:
            device = None
        if self.type == 'conv1x1-conv3x3':
            RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
            RB = torch.ones(1, self.mid_planes, 3, 3,
                            device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=self.k1).view(-1, ) + self.b1
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3),
                             device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias
            RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            RB = torch.ones(1, self.out_planes, 3, 3,
                            device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=k1).view(-1, ) + b1
        return RK, RB


class ECB(nn.Module):
    ''' ECB block '''
    def __init__(self,
                 inp_planes,
                 out_planes,
                 depth_multiplier):
        super().__init__()
        self.in_channels = inp_planes

        self.conv3x3 = torch.nn.Conv2d(inp_planes,
                                       out_planes,
                                       kernel_size=3,
                                       padding=1)
        self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', inp_planes,
                                      out_planes, depth_multiplier)
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', inp_planes, out_planes,
                                      -1)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', inp_planes, out_planes,
                                      -1)
        self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', inp_planes,
                                      out_planes, -1)

    def forward(self, x):
        ''' forward '''
        if self.training:
            y = self.conv3x3(x) + self.conv1x1_3x3(x) + self.conv1x1_sbx(
                x) + self.conv1x1_sby(x) + self.conv1x1_lpl(x)
        else:
            RK, RB = self.rep_params()
            y = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
        return y

    def rep_params(self):
        ''' rep params '''
        K0, B0 = self.conv3x3.weight, self.conv3x3.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K2, B2 = self.conv1x1_sbx.rep_params()
        K3, B3 = self.conv1x1_sby.rep_params()
        K4, B4 = self.conv1x1_lpl.rep_params()
        RK, RB = (K0 + K1 + K2 + K3 + K4), (B0 + B1 + B2 + B3 + B4)
        return RK, RB


class PConv(ECB):
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
            ECB(channels, f, 2),
            # nn.Conv2d(channels, f, 3, padding=1),       # NO ECB
            nn.MaxPool2d(7, stride=3),
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            # approximate var pooling
            ApproximateVariancePooling(7),
            # fc
            ECB(f, f, 2),
            # nn.Conv2d(f, f, 3, padding=1),       # NO ECB
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
                self.body.append(PConv(sub_channel, sub_channel, 2, act=nn.PReLU(sub_channel)))
                # self.body.append(PConv(sub_channel, sub_channel, 3, padding=1, act=nn.PReLU(sub_channel)))       # NO ECB
            else:
                self.body.append(ECB(sub_channel, sub_channel, 2))
                # self.body.append(nn.Conv2d(sub_channel, sub_channel, 3, padding=1))       # NO ECB
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
            ECB(3, channels, 2),       # NO ECB
            # nn.Conv2d(3, channels, 3, padding=1),
            nn.PReLU(channels),
            *[PBlock(channels, sub_channels, zip_channels, pre_conv=pre_conv) for _ in range(blocks)],
            ECB(channels, 3 * scale ** 2, 2),       # NO ECB
            # nn.Conv2d(channels, 3 * scale ** 2, 3, padding=1),
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
            ECB(3 * 4, channels, 2),
            nn.PReLU(channels),
            *[PBlock(channels, sub_channels, zip_channels, pre_conv=pre_conv) for _ in range(blocks)],
            ECB(channels, 3 * (scale * 2) ** 2, 2),
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
    