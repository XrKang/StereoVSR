#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# here put the import lib

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from skimage import morphology
import numpy as np
from torch.autograd import Variable


class VTAM(nn.Module):  # stereo attention block
    def __init__(self, embed_ch):
        super(VTAM, self).__init__()
        self.conv_l = nn.Conv2d(embed_ch, embed_ch, 1, 1, 0, bias=True)
        self.conv_r = nn.Conv2d(embed_ch, embed_ch, 1, 1, 0, bias=True)
        self.rb = ResBlock(embed_ch)
        self.softmax = nn.Softmax(-1)
        self.conv_out = nn.Conv2d(embed_ch * 2 + 1, embed_ch, 1, 1, 0, bias=True)

    def forward(self, x_left, x_right_nbr):  # B * C * H * W
        b, c, h, w = x_left.shape
        buffer_left = self.rb(x_left)
        buffer_right = self.rb(x_right_nbr)
        # print("buffer_right", buffer_right.shape)
        # M_{right_to_left}

        F0 = self.conv_l(buffer_left).contiguous().view(b, -1, h * w).permute(0, 2, 1)  # B, H * W, C
        F1 = self.conv_r(buffer_right).contiguous().view(b, -1, h * w)  # B, C, H * W
        # print(F0.shape)
        # print(F1.shape)

        attention = torch.bmm(F0, F1)  # B, H * W, H * W
        # view相当于reshape：若之前进行了permute，必须先进行contiguous
        # bmm相当于矩阵相乘，仅对后两维进行运算，并仅对三维torch进行运算。

        M_right_to_left = self.softmax(attention)  # B, H * W, H * W

        out_f = x_right_nbr.contiguous().view(b, -1, h * w).permute(0, 2, 1)  # B, H * W, C
        out_f = torch.bmm(M_right_to_left, out_f).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)
        # B * C * H * W

        return out_f,  M_right_to_left

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


def conv_extractor(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )


def upconv_extractor(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


def morphologic_process(mask):
    device = mask.device
    b, _, _, _ = mask.shape
    mask = ~mask  # ~：逐位取反
    mask_np = mask.cpu().numpy().astype(bool)
    mask_np = morphology.remove_small_objects(mask_np, 20, 2)  # 形态学处理，去除小于20的区域，2表示8邻接
    mask_np = morphology.remove_small_holes(mask_np, 10, 2)  # 形态学处理，去除小于10的孔，2表示8邻接
    for idx in range(b):
        buffer = np.pad(mask_np[idx, 0, :, :], ((3, 3), (3, 3)), 'constant')
        buffer = morphology.binary_closing(buffer, morphology.disk(3))  # closing 先膨胀再腐蚀
        mask_np[idx, 0, :, :] = buffer[3:-3, 3:-3]
    mask_np = 1 - mask_np
    mask_np = mask_np.astype(float)
    return torch.from_numpy(mask_np).float().to(device)


class ResBlock(nn.Module):
    def __init__(self, embed_ch):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(embed_ch, embed_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(embed_ch, embed_ch, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def __call__(self, x):
        res = self.body(x)
        return res + x


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class Upsampler(nn.Sequential):  # 上采样/放大器
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):
        # scale:放大倍数
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):  # 2^n,循环n次
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                # PixelShuffle,(1,c*4,h,w)----->(1,c,h*2,w*2) 把通道的像素迁移到size上
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)




class PAM(nn.Module):  # stereo attention block
    def __init__(self, embed_ch):
        super(PAM, self).__init__()
        self.conv_l = nn.Conv2d(embed_ch, embed_ch, 1, 1, 0, bias=True)
        self.conv_r = nn.Conv2d(embed_ch, embed_ch, 1, 1, 0, bias=True)
        self.rb = ResBlock(embed_ch)
        self.softmax = nn.Softmax(-1)
        self.conv_out = nn.Conv2d(embed_ch * 2 + 1, embed_ch, 1, 1, 0, bias=True)

    def forward(self, x_left, x_right):  # B * C * H * W
        b, c, h, w = x_left.shape
        buffer_left = self.rb(x_left)
        buffer_right = self.rb(x_right)
        # M_{right_to_left}
        F0 = self.conv_l(buffer_left).permute(0, 2, 3, 1)  # B * H * W * C
        F1 = self.conv_r(buffer_right).permute(0, 2, 1, 3)  # B * H * C * W
        S = torch.bmm(F0.contiguous().view(-1, w, c),
                      F1.contiguous().view(-1, c, w))  # (B*H) * W * W
        # view相当于reshape：若之前进行了permute，必须先进行contiguous
        # bmm相当于矩阵相乘，仅对后两维进行运算，并仅对三维torch进行运算。

        M_right_to_left = self.softmax(S)  # (B*H) * W * W
        # right map transfer to left
        S_T = S.permute(0, 2, 1)  # (B*H) * W * W
        M_left_to_right = self.softmax(S_T)

        # valid mask for transfer information from Feature(right->left)
        V_left_to_right = torch.sum(M_left_to_right.detach(), 1) > 0.1  # (B*H) * 1 * W
        V_left_to_right = V_left_to_right.view(b, 1, h, w)  # B * 1 * H * W
        V_left_to_right = morphologic_process(V_left_to_right)  # 形态学处理


        # V_right_to_left = torch.sum(M_right_to_left.detach(), 1) > 0.1
        # V_right_to_left = V_right_to_left.view(b, 1, h, w)  # B * 1 * H * W
        # V_right_to_left = morphologic_process(V_right_to_left)

        out_f = x_right.permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # (B*H) * W * C
        out_f = torch.bmm(M_right_to_left, out_f).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)
        # B * C * H * W

        return out_f, V_left_to_right, (M_left_to_right, M_right_to_left)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'
    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output

def flow_warping(x, flo):
    """warp an image/tensor (im2) back to im1, according to the optical flow
     x: [B, C, H, W] (im2)
     flo: [B, 2, H, W] flow   """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    grid.requires_grad = False
    grid = grid.type_as(x)

    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid)
        # mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        # mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

        # mask[mask < 0.9999] = 0
        # mask[mask > 0] = 1

    return output







class ResidualUpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualUpSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels,   1, stride=1, padding=0, bias=bias),
                                nn.PReLU(),
                                nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1,bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
                                nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top+bot
        return out


class UpSampleFusion(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2, kernel_size=3):
        super(UpSampleFusion, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(ResidualUpSample(in_channels))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x