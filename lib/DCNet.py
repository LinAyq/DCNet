import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
# from .Res2Net_v1b import res2net50_v1b_26w_4s
from .Swin_Transformer import *
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from math import log
affine_par = True

class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class Conv3x3(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 3,padding=1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MSCA(nn.Module):
    def __init__(self, channels, r=16):
        super(MSCA, self).__init__()
        out_channels = int(channels // r)
        # local att
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # global att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):

        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sig(xlg)
        x = x * wei
        return x

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class aggregation(nn.Module):
    def __init__(self, channel=32):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.ra1_1_up = TransBasicConv2d(32, 32, 2, 2)
        self.ra1_2_up = TransBasicConv2d(32, 32, 2, 2)
        self.ra1_3_up = TransBasicConv2d(32, 32, 2, 2)
        self.ra1_4_up = TransBasicConv2d(32, 32, 2, 2)
        self.ra2_up = TransBasicConv2d(32, 32, 2, 2)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_concat12 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat123 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.msca = MSCA(3*channel)
        self.rcab = RCAB(3*channel)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        if x3.size() !=x2.size():
            x3 = F.interpolate(x3, x2.size()[2:], mode='bilinear', align_corners=False)
        if x1.size() !=x2.size():
            x1 = F.interpolate(x1, x2.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv_concat3(x)
        x = self.conv4(x)
        x = self.rcab(x)
        x = self.conv5(x)
        return x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class CFF(nn.Module):
    def __init__(self, in_channel_h, in_channel_l, out_channel):
        self.init__ = super(CFF, self).__init__()

        act_fn = nn.ReLU(inplace=True)

        self.layer0 = BasicConv2d(in_channel_h, out_channel // 2, 1)
        self.layer1 = BasicConv2d(in_channel_l, out_channel // 2, 1)
        self.layer2 = BasicConv2d(in_channel_h, out_channel, 1)
        self.layer3 = BasicConv2d(in_channel_l, out_channel, 1)

        self.layer3_1 = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(out_channel // 2), act_fn)
        self.layer3_2 = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(out_channel // 2), act_fn)

        self.layer5_1 = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),
                                      nn.BatchNorm2d(out_channel // 2), act_fn)
        self.layer5_2 = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),
                                      nn.BatchNorm2d(out_channel // 2), act_fn)

        self.layer_out = nn.Sequential(nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(out_channel), act_fn)
        self.layer_out1 = nn.Sequential(nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(out_channel), act_fn)

        self.msca = MSCA(out_channel)
        self.conv_x0 = BasicConv2d(out_channel, out_channel, 3, stride=1, padding=1)
        self.conv_x1 = BasicConv2d(out_channel, out_channel, 3, stride=1, padding=1)

    def forward(self, xh, xl):
        
        xh = F.interpolate(xh, xl.size()[2:], mode='bilinear', align_corners=False)
        x0_1 = self.layer0(xh)
        x1_1 = self.layer1(xl)
        x2_1 = self.layer2(xh)
        x3_1 = self.layer3(xl)

        x_3_1 = self.layer3_1(torch.cat((x0_1, x1_1), dim=1))
        x_5_1 = self.layer5_1(torch.cat((x1_1, x0_1), dim=1))

        x_3_2 = self.layer3_2(torch.cat((x_3_1, x_5_1), dim=1))
        x_5_2 = self.layer5_2(torch.cat((x_5_1, x_3_1), dim=1))

        out = self.msca(self.layer_out(x0_1 + x1_1 + torch.mul(x_3_2, x_5_2)))
        x2_1 = x2_1*out + x2_1
        x3_1 = x3_1*out + x3_1
        out = self.layer_out1(self.conv_x1(x2_1) + self.conv_x0(x3_1))

        return out

class MFAM1(nn.Module):
    def __init__(self, in_channel_h, in_channel_l, out_channels):
        self.init__ = super(MFAM1, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv_1_0 = BasicConv2d(in_channel_h, out_channels, 1)
        self.conv_1_1 = BasicConv2d(in_channel_l, out_channels, 1)

        self.conv_1_2 = BasicConv2d(in_channel_h, out_channels, 1)
        self.conv_1_3 = BasicConv2d(in_channel_l, out_channels, 1)

        self.conv_1_5 = BasicConv2d(out_channels, out_channels, 3, stride=1, padding=1)

        self.conv_3_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.conv_5_1 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.conv_5_2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.conv_5_3 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)

        self.conv_x0 = BasicConv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.conv_x1 = BasicConv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.conv_out = BasicConv2d(out_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, xh, xl):
        
        xh = F.interpolate(xh, xl.size()[2:], mode='bilinear', align_corners=False)
        x0 = self.conv_1_0(xh)
        x1 = self.conv_1_1(xl)

        x2 = self.conv_1_2(xh)
        x3 = self.conv_1_3(xl)

        x_3_1 = self.relu(self.conv_3_1(x2))  ## (BS, 32, , )
        x_5_1 = self.relu(self.conv_5_1(x3))  ## (BS, 32, , )

        x_3_2 = self.relu(self.conv_3_2(x_3_1 + x_5_1))  ## (BS, 64, , )
        x_5_2 = self.relu(self.conv_5_2(x_5_1 + x_3_1))  ## (BS, 64, , )

        x_3_3 = self.relu(self.conv_3_3(x_3_2 + x_5_2))
        x_5_3 = self.relu(self.conv_5_3(x_5_2 + x_3_2))  ## (BS, 64, , )

        x_mul = torch.mul(x_3_3, x_5_3)

        x_sig = torch.sigmoid(self.conv_1_5(x_mul + x_3_1 + x_5_1))
        x0 = x0 * x_sig + x0
        x1 = x1 * x_sig + x1
        out = self.conv_out(self.conv_x1(x1) + self.conv_x0(x0))

        return out

class CLF(nn.Module):
    def __init__(self, in_channel_h, in_channel_l, out_channel):
        self.init__ = super(CLF, self).__init__()
        self.high_fusion1 = CFF(in_channel_h // 2, in_channel_l // 2, out_channel // 2)
        self.mfam1 = MFAM1(in_channel_h // 2, in_channel_l // 2, out_channel // 2)
        self.conv = BasicConv2d(out_channel, out_channel, kernel_size=3, padding=1)


    def forward(self, x0, x1):

        x0 = torch.chunk(x0, 2, dim=1)
        x1 = torch.chunk(x1, 2, dim=1)

        high_fusion1 = self.high_fusion1(x0[0], x1[0])
        mfam1 = self.mfam1(x0[1], x1[1])
        out = self.conv(torch.cat((high_fusion1, mfam1), dim=1))
        
        return out


class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel, RFB = True):
        super(Reduction, self).__init__()
        if(RFB):
            self.reduce = nn.Sequential(
                RFB_modified(in_channel,out_channel),
            )
        else:
            self.reduce = nn.Sequential(
                BasicConv2d(in_channel, out_channel, 1),
            )
    def forward(self, x):
        return self.reduce(x)

class EFM0(nn.Module):
    def __init__(self, channel):
        super(EFM0, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.cat = ConvBNR(2*channel, channel, 3)
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, att):
        if c.size() != att.size():
            c = F.interpolate(c, att.size()[2:], mode='bilinear', align_corners=False)
        x = c * att + c
        x = self.conv2d(x)
        wei = self.avg_pool(x)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = x * wei
        return x

class ETR(nn.Module):
    def __init__(self, in_channels, out_channels, cat=True):
        super(ETR, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv_up = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1)
                                     )
        self.cat = nn.Conv2d(out_channels, out_channels, 1)
        self.efm_e = EFM0(out_channels)
        self.efm_t = EFM0(out_channels)


    def forward(self, x, edge, texture):
        x = self.conv_up(x)

        edge = torch.sigmoid(edge)
        texture = torch.sigmoid(texture)

        x_edge = self.efm_e(x, edge)
        x_texture = self.efm_t(x, texture)

        return x_edge, x_texture

class ETA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ETA, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.cat = ConvBNR(2*in_channels, out_channels, 3)
        self.rcab_e = RCAB(in_channels)
        self.rcab_t = RCAB(in_channels)

    def forward(self, x, edge, texture):

        edge = torch.sigmoid(edge)
        texture = torch.sigmoid(texture)
        edge = F.interpolate(edge, x.size()[2:], mode='bilinear', align_corners=False)
        texture = F.interpolate(texture, x.size()[2:], mode='bilinear', align_corners=False)

        x_edge = x*edge + x
        x_edge = self.rcab_e(x_edge)
        x_texture = x*texture + x
        x_texture = self.rcab_t(x_texture)
        fuse = torch.cat((x_texture, x_edge), dim=1)
        fuse = self.cat(fuse)

        return fuse



class ETA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ETA4, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.cat = ConvBNR(2*in_channels, out_channels, 3)
        self.rcab_e = CBAM(in_channels)
        self.rcab_t = CBAM(in_channels)

    def forward(self, x, edge, texture):

        edge = torch.sigmoid(edge)
        texture = torch.sigmoid(texture)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        edge = F.interpolate(edge, scale_factor=2, mode='bilinear', align_corners=False)
        texture = F.interpolate(texture, scale_factor=2, mode='bilinear', align_corners=False)


        x_edge = x*edge + x
        x_edge = self.rcab_e(x_edge)
        x_texture = x*texture + x
        x_texture = self.rcab_t(x_texture)
        fuse = torch.cat((x_texture, x_edge), dim=1)
        fuse = self.cat(fuse)

        return fuse


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        # self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out =self.shared_MLP(self.avg_pool(x))# self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out =self.shared_MLP(self.max_pool(x))# self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class EC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EC, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.cat1 = ConvBNR(2*in_channels, out_channels, 3)
        self.cat2 = ConvBNR(out_channels, out_channels, 3)
        self.rcab_e = RCAB(in_channels)
        self.rcab_t = RCAB(in_channels)
        self.avg_pool = nn.AdaptiveMaxPool2d(96)

    def forward(self, edge, edge23):

        edge23 = F.interpolate(edge23, edge.size()[2:], mode='bilinear', align_corners=False)

        e = self.cat1(torch.cat((edge, edge23), dim=1))
        mul = self.avg_pool(edge23 * edge)
        edge = e + mul

        edge = self.cat2(edge)

        return edge


class TC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TC, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.cat1 = ConvBNR(2 * in_channels, out_channels, 3)
        self.cat2 = ConvBNR(out_channels, out_channels, 3)
        self.rcab_e = RCAB(in_channels)
        self.rcab_t = RCAB(in_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(96)

    def forward(self, texture, texture23):
        texture = F.interpolate(texture, size=(96,96), mode='bilinear', align_corners=False)
        texture23 = F.interpolate(texture23, size=(96,96) , mode='bilinear', align_corners=False)

        t = self.cat1(torch.cat((texture, texture23), dim=1))
        mul = self.avg_pool(texture23 * texture)
        texture = t + mul

        texture = self.cat2(texture)

        return texture

class FirstOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = torch.nn.Conv2d(in_channels, int(alpha * in_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels, in_channels - int(alpha * in_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        if self.stride ==2:
            x = self.h2g_pool(x)

        X_h2l = self.h2g_pool(x)
        X_h = x
        X_h = self.h2h(X_h)
        X_l = self.h2l(X_h2l)

        return X_h, X_l

class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride
        self.l2l = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2l = torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2l = self.h2g_pool(X_h)

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)

        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)

        # X_l2h = self.upsample(X_l2h)
        X_l2h = F.interpolate(X_l2h, (int(X_h2h.size()[2]),int(X_h2h.size()[3])), mode='bilinear')
        # print('X_l2h:{}'.format(X_l2h.shape))
        # print('X_h2h:{}'.format(X_h2h.shape))
        X_h = X_l2h + X_h2h
        X_l = X_h2l + X_l2l

        return X_h, X_l

class LastOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(LastOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.l2h = torch.nn.Conv2d(int(alpha * out_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(out_channels - int(alpha * out_channels),
                                   out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2h = self.h2h(X_h) # 高频组对齐通道
        X_l2h = self.l2h(X_l) # 低频组对齐通道
        # 低频组对齐长宽尺寸
        X_l2h = F.interpolate(X_l2h, (int(X_h2h.size()[2]), int(X_h2h.size()[3])), mode='bilinear')

        X_h = X_h2h + X_l2h  # 本来的设置：高频低频融合输出
        # return X_h       #都输出

        # return X_h2h  #只输出高频组
        # return X_l2h    #只输出低频组

        return X_h, X_h2h, X_l2h

class Octave(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super(Octave, self).__init__()
        # 第一层，将特征分为高频和低频
        self.fir = FirstOctaveConv(in_channels, out_channels, kernel_size)
        # 第二层，低高频输入，低高频输出
        self.mid1 = OctaveConv(in_channels, in_channels, kernel_size)
        self.mid2 = OctaveConv(in_channels, out_channels, kernel_size)
        # 第三层，将低高频汇合后输出
        self.lst = LastOctaveConv(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x0 = x
        x_h, x_l = self.fir(x)                   # (1,64,64,64) ,(1,64,32,32)
        x_hh, x_ll = x_h, x_l,
        # x_1 = x_hh +x_ll
        x_h_1, x_l_1 = self.mid1((x_h, x_l))     # (1,64,64,64) ,(1,64,32,32)
        x_h_2, x_l_2 = self.mid1((x_h_1, x_l_1)) # (1,64,64,64) ,(1,64,32,32)
        x_h_5, x_l_5 = self.mid2((x_h_2, x_l_2)) # (1,32,64,64) ,(1,32,32,32)
        X_h, X_h2h, X_l2h = self.lst((x_h_5, x_l_5)) # (1,64,64,64)
        return X_h, X_h2h, X_l2h

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    # paper: Image Super-Resolution Using Very DeepResidual Channel Attention Networks
    # input: B*C*H*W
    # output: B*C*H*W
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class E(nn.Module):
    def __init__(self, in_channels):
        super(E, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv_up = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1))
        self.conv_edge = RFB_E(in_channels, in_channels)

    def forward(self, x):
        x = self.conv_up(x)

        edge = self.relu(self.conv_edge(x))

        return edge

class T(nn.Module):
    def __init__(self, in_channels):
        super(T, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv_up = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1))
        self.conv_texture = RFB_E(in_channels, in_channels)

    def forward(self, x):
        x = self.conv_up(x)

        texture = self.relu(self.conv_texture(x))

        return texture

class RFB_T(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_T, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = x_cat + self.conv_res(x)
        return x


class RFB_E(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_E, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = x_cat + self.conv_res(x)
        return x

class Net(nn.Module):
    def __init__(self, channel=32):
        super(Net, self).__init__()
        self.swin = SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])


        self.high_fusion43 = CLF(1024, 512, channel)
        self.high_fusion32 = CLF(512, 256, channel)
        self.high_fusion21 = CLF(256, 128, channel)


        self.predictorf4_e = nn.Conv2d(channel, 1, 1)
        self.predictorf3_e = nn.Conv2d(channel, 1, 1)
        self.predictorf2_e = nn.Conv2d(channel, 1, 1)
        self.predictorf1_e = nn.Conv2d(channel, 1, 1)

        self.predictorf4_t = nn.Conv2d(channel, 1, 1)
        self.predictorf3_t = nn.Conv2d(channel, 1, 1)
        self.predictorf2_t = nn.Conv2d(channel, 1, 1)
        self.predictorf1_t = nn.Conv2d(channel, 1, 1)

        self.predictorfuse4 = nn.Conv2d(channel, 1, 1)
        self.predictorfuse3 = nn.Conv2d(channel, 1, 1)
        self.predictorfuse2 = nn.Conv2d(channel, 1, 1)
        self.predictorfuse1 = nn.Conv2d(channel, 1, 1)


        self.Octave4 = Octave(1024, 1024)
        self.Octave3 = Octave(512, 512)
        self.Octave2 = Octave(256, 256)
        self.Octave1 = Octave(128, 128)


        self.e = E(channel)
        self.e23 = E(channel)
        self.t = T(channel)
        self.t23 = T(channel)
        self.agg = aggregation(channel)
        self.ec = EC(channel,channel)
        self.tc = TC(channel,channel)

        self.etr3 = ETR(512, channel)
        self.etr2 = ETR(256, channel)
        self.etr1 = ETR(128, channel)
        self.etr4 = ETR(1, channel)

        self.eta4 = ETA(channel,channel)
        self.eta3 = ETA(channel,channel)
        self.eta2 = ETA(channel,channel)
        self.eta1 = ETA(channel,channel)



    def forward(self, x):
        batch_size, channels, height, width = x.size()
        rgb_list = self.swin(x)
        #
        x1 = rgb_list[0]  # 128  96*96
        x2 = rgb_list[1]  # 256  48*48
        x3 = rgb_list[2]  # 512  24*24
        x4 = rgb_list[3]  # 1024  12*12

        _, _, x4 = self.Octave4(x4)
        x3, _, _ = self.Octave3(x3)
        x2, _, _ = self.Octave2(x2)
        _, x1, _ = self.Octave1(x1)
        high_fusion43 = self.high_fusion43(x4, x3) #24
        high_fusion23 = self.high_fusion32(x3, x2) #48
        high_fusion21 = self.high_fusion21(x2, x1) #96
        edge = self.e(high_fusion21) #96
        texture = self.t(high_fusion43) #24

        edge23 = self.e23(high_fusion23) #48
        texture23 = self.t23(high_fusion23) #48
        edge = self.ec(edge, edge23) #96
        texture = self.tc(texture, texture23) #96
        feat = self.agg(high_fusion43, edge, texture) #96

        f4_e,f4_t = self.etr4(feat, edge, texture) #96,96,96  f4_e,f4_t 96,96
        f3_e,f3_t = self.etr3(x3, f4_e, f4_t) #24,96,96       f3_e,f3_t 96,96
        f2_e,f2_t = self.etr2(x2, f3_e, f3_t) #48,96,96       f2_e,f2_t 96,96
        f1_e,f1_t = self.etr1(x1, f2_e, f2_t) #96,96          f1_e,f1_t 96,96

        fuse4 = self.eta4(high_fusion21, f4_e, f4_t) #192
        fuse3 = self.eta3(fuse4, f3_e, f3_t) #192
        fuse2 = self.eta2(fuse3, f2_e, f2_t) #192
        fuse1 = self.eta1(fuse2, f1_e, f1_t) #192



        f4_e = self.predictorf4_e(f4_e)
        f3_e = self.predictorf3_e(f3_e)
        f2_e = self.predictorf2_e(f2_e)
        f1_e = self.predictorf1_e(f1_e)

        f4_t = self.predictorf4_t(f4_t)
        f3_t = self.predictorf3_t(f3_t)
        f2_t = self.predictorf2_t(f2_t)
        f1_t = self.predictorf1_t(f1_t)

        fuse4 = self.predictorfuse4(fuse4)
        fuse3 = self.predictorfuse3(fuse3)
        fuse2 = self.predictorfuse2(fuse2)
        fuse1 = self.predictorfuse1(fuse1)


        f4_e = F.interpolate(f4_e, size=(height, width), mode='bilinear', align_corners=False)
        f3_e = F.interpolate(f3_e, size=(height, width), mode='bilinear', align_corners=False)
        f2_e = F.interpolate(f2_e, size=(height, width), mode='bilinear', align_corners=False)
        f1_e = F.interpolate(f1_e, size=(height, width), mode='bilinear', align_corners=False)

        f4_t = F.interpolate(f4_t, size=(height, width), mode='bilinear', align_corners=False)
        f3_t = F.interpolate(f3_t, size=(height, width), mode='bilinear', align_corners=False)
        f2_t = F.interpolate(f2_t, size=(height, width), mode='bilinear', align_corners=False)
        f1_t = F.interpolate(f1_t, size=(height, width), mode='bilinear', align_corners=False)

        fuse4 = F.interpolate(fuse4, size=(height, width), mode='bilinear', align_corners=False)
        fuse3 = F.interpolate(fuse3, size=(height, width), mode='bilinear', align_corners=False)
        fuse2 = F.interpolate(fuse2, size=(height, width), mode='bilinear', align_corners=False)
        fuse1 = F.interpolate(fuse1, size=(height, width), mode='bilinear', align_corners=False)


        return fuse4, fuse3, fuse2, fuse1, f4_e, f4_t, f3_e, f3_t, f2_e, f2_t, f1_e, f1_t

    def load_pre(self, pre_model):
        self.swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"RGB Swin loading pre_model ${pre_model}")

