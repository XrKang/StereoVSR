#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# here put the import lib

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import arch_util as arch_util
try:
    from dcn.deform_conv import ModulatedDeformConvPack as DCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')



class ST_Fusion(nn.Module):

    def __init__(self, nf, nframes, center, groups):
        super(ST_Fusion, self).__init__()
        self.center = center
        self.nframes = nframes

        # temporal attention & Deform_Conv
        self.tAtt_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)


        self.offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                           extra_offset_mask=True)
        self.center_Resblock = arch_util.ResBlock(nf)
        self.fea_fusion_tam = nn.Conv2d(nframes * nf + nf, nf, 1, 1, bias=True)

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(nframes * nf + nf, nf, 1, 1, bias=True)

        # spatial attention (after fusion conv)
        self.sAtt_1 = nn.Conv2d(nframes * nf + nf, nf, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)
        self.sAtt_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_4 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_L1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_L2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.sAtt_L3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_add_1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_add_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, fea):
        B, N, C, H, W = fea.size()  # N video frames
        # temporal attention

        ref = fea[:, self.center, :, :, :].clone()
        cor_l = []
        aligned_fea = []
        for i in range(N):
            nbr = fea[:, i, :, :, :].clone()

            # deformable_conv
            offset = torch.cat([nbr, ref], dim=1)
            offset = self.lrelu(self.offset_conv1(offset))
            offset = self.lrelu(self.offset_conv2(offset))
            aligned_nbr = self.lrelu(self.dcnpack([nbr, offset]))
            aligned_fea.append(aligned_nbr)

            emb_nbr = self.tAtt_1(aligned_nbr)  # [B, N, C(nf), H, W]
            emb_ref = self.tAtt_2(ref)
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        aligned_fea = torch.cat(aligned_fea, dim=1)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)  # B, N*nf, H, W

        fea_tam = fea.view(B, -1, H, W) * cor_prob   # B, N*nf, H, W
        ref_resblock = self.center_Resblock(ref)     # B, nf, H, W

        fea_tam = torch.cat([fea_tam, ref_resblock], dim=1)     # B, N*nf + nf, H, W
        fea_tam = self.lrelu(self.fea_fusion_tam(fea_tam))      # B, nf, H, W

        fea_tam = torch.cat([fea_tam, aligned_fea], dim=1)       # B, N*nf + nf, H, W


        # fusion
        fea = self.lrelu(self.fea_fusion(fea_tam))


        # spatial attention
        att = self.lrelu(self.sAtt_1(fea_tam))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        # pyramid levels
        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L, scale_factor=2, mode='bilinear', align_corners=False)

        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        fea = fea * att * 2 + att_add
        return fea


class VTAM_singleOut(nn.Module):

    def __init__(self, nf, nframes, center):
        super(VTAM_singleOut, self).__init__()
        self.center = center
        self.nframes = nframes
        self.n_view_nbr = self.nframes-1

        self.pam = arch_util.PAM(nf)
        self.pam_conv = nn.Conv2d(2 * nf + 1, nf, 1, 1, bias=True)

        vtam_layers = []
        vtam_conv = []
        for _ in range(self.n_view_nbr):
            vtam_layers.append(arch_util.VTAM(nf))
            vtam_conv.append(nn.Conv2d(2 * nf, nf, 1, 1, bias=True))
        self.vtam_layers = nn.Sequential(*vtam_layers)
        self.vtam_conv = nn.Sequential(*vtam_conv)

        self.vtam_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        self.out_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
        self.out_conv_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.out_conv_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, x_left_center, x_right):
        x_right_center = x_right[:, self.center, :, :, :].contiguous()

        pam_fea, V_t_R2L, M_t_R2L = self.pam(x_left_center, x_right_center)

        vtam_fea = []   # Concat with feature of PAM
        vtam_fea_out = []   # Concat at final point
        M_att = []
        index_frame = 0
        for i in range(self.nframes):
            if i != self.center:
                vtam_fea_single, M_single = self.vtam_layers[index_frame](x_left_center,
                                                                          x_right[:, index_frame, :, :, :])
                vtam_fea.append(vtam_fea_single)
                M_att.append(M_single)

                vtam_fea_out_single = torch.cat([vtam_fea_single, x_left_center], dim=1)
                vtam_fea_out_single = self.lrelu(self.vtam_conv[index_frame](vtam_fea_out_single))
                vtam_fea_out.append(vtam_fea_out_single)
                index_frame = index_frame + 1

        vtam_fea = torch.cat(vtam_fea, dim=1)
        vtam_fea = self.lrelu(self.vtam_fusion(torch.cat([vtam_fea, x_left_center], dim=1)))
        pam_fea_out = self.lrelu(self.pam_conv(torch.cat([pam_fea, vtam_fea, V_t_R2L], dim=1)))

        vtam_fea_out = torch.cat(vtam_fea_out, dim=1)
        out_feat = self.lrelu(self.vtam_fusion(torch.cat([vtam_fea_out, pam_fea_out], dim=1)))
        out_feat = self.lrelu(self.out_conv_2(self.out_conv_1(out_feat)))
        return out_feat


class SVSRNet(nn.Module):
    def __init__(self, arg):
        #  nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
        #          predeblur=False, HR_in=False, w_TSA=True
        super(SVSRNet, self).__init__()
        self.nf = arg.embed_ch
        self.nframes = arg.nframes
        self.center = int(self.nframes/2)//2
        self.groups = arg.groups
        self.front_RBs = arg.front_RBs
        self.back_RBs = arg.back_RBs

        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=self.nf)

        #### extract features (for each frame)
        self.conv_first = nn.Conv2d(3, self.nf, 3, 1, 1, bias=True)
        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, self.front_RBs)

        self.st_fusion_left = ST_Fusion(nf=self.nf, nframes=int(self.nframes/2), center=self.center, groups=self.groups)
        self.st_fusion_right = ST_Fusion(nf=self.nf, nframes=int(self.nframes/2), center=self.center, groups=self.groups)

        self.vtam_left = VTAM_singleOut(nf=self.nf, nframes=int(self.nframes/2), center=self.center)
        self.vtam_right = VTAM_singleOut(nf=self.nf, nframes=int(self.nframes/2), center=self.center)

        #### reconstruction
        self.fusion_right = nn.Conv2d(2 * self.nf, self.nf, 1, 1, bias=True)
        self.recon_trunk_right = arch_util.make_layer(ResidualBlock_noBN_f, self.back_RBs)

        #### upsampling
        self.upconv1_right = nn.Conv2d(self.nf, self.nf * 4, 3, 1, 1, bias=True)
        self.upconv2_right = nn.Conv2d(self.nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle_right = nn.PixelShuffle(2)
        self.HRconv_right = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last_right = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        #### reconstruction
        self.fusion = nn.Conv2d(2 * self.nf, self.nf, 1, 1, bias=True)
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, self.back_RBs)

        #### upsampling
        self.upconv1 = nn.Conv2d(self.nf, self.nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(self.nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)



        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames
        # print(self.center)
        # print(self.center + int(self.nframes/2))
        x_left_center = x[:, self.center, :, :, :].contiguous()
        x_right_center = x[:, self.center + int(self.nframes/2), :, :, :].contiguous()

        #### extract LR features
        fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
        fea = self.feature_extraction(fea)
        fea = fea.view(B, N, -1, H, W)
        # print("fea_left_center", fea_left_center.shape)

        fea_left = fea[:, :(self.nframes//2), :, :, :]
        fea_left_center = fea_left[:, self.center, :, :, :].contiguous()

        # print("fea_left", fea_left.shape)

        fea_right = fea[:, (self.nframes//2):, :, :, :]
        fea_right_center = fea_right[:, self.center, :, :, :].contiguous()

        # print("fea_right", fea_right.shape)

        fea_left_f = self.st_fusion_left(fea_left)
        fea_right_f = self.st_fusion_left(fea_right)

        fea_left_v = self.vtam_left(fea_left_center, fea_right)
        fea_right_v = self.vtam_right(fea_right_center, fea_left)

        fea_left_out = self.lrelu(self.fusion(torch.cat([fea_left_f, fea_left_v], dim=1)))
        fea_right_out = self.lrelu(self.fusion_right(torch.cat([fea_right_f, fea_right_v], dim=1)))

        out_left = self.recon_trunk(fea_left_out)
        out_left = self.lrelu(self.pixel_shuffle(self.upconv1(out_left)))
        out_left = self.lrelu(self.pixel_shuffle(self.upconv2(out_left)))
        out_left = self.lrelu(self.HRconv(out_left))
        out_left = self.conv_last(out_left)
        base_left = F.interpolate(x_left_center, scale_factor=4, mode='bilinear', align_corners=False)
        out_left += base_left

        out_right = self.recon_trunk_right(fea_right_out)
        out_right = self.lrelu(self.pixel_shuffle_right(self.upconv1(out_right)))
        out_right = self.lrelu(self.pixel_shuffle_right(self.upconv2(out_right)))
        out_right = self.lrelu(self.HRconv_right(out_right))
        out_right = self.conv_last_right(out_right)
        base_right = F.interpolate(x_right_center, scale_factor=4, mode='bilinear', align_corners=False)
        out_right += base_right

        return out_left, out_right

if __name__ == '__main__':
    import argparse

    # x1 = torch.rand(1, 6, 3, 310//4*4, 93//4*4).cuda()
    # x1 = torch.rand(1, 6, 3, 310//4*4, 93//4*4).cuda()
    x1 = torch.rand(1, 6, 3, 240//4*4, 135//4*4).cuda()

    print(x1.shape)
    parser = argparse.ArgumentParser(description="StereoVSR")
    arg = parser.parse_args()
    arg.in_ch = 3
    arg.scale = 4
    arg.embed_ch = 64

    arg.nframes = 3*2
    arg.groups = 8
    arg.front_RBs = 5
    arg.back_RBs = 10
    with torch.no_grad():
        model = SVSRNet(arg).cuda()
        a, b = model(x1)
        print(a.shape)
        print(b.shape)