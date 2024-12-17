"""Data helper."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020-2022(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 11月 02日 星期一 17:52:14 CST
# ***
# ************************************************************************************/
#

import pdb

import torch


def rgb2xyz(rgb):  # rgb from [0,1]
    # [0.412453, 0.357580, 0.180423],
    # [0.212671, 0.715160, 0.072169],
    # [0.019334, 0.119193, 0.950227]

    mask = (rgb > 0.04045).float().to(rgb.device)
    rgb = (((rgb + 0.055) / 1.055) ** 2.4) * mask + rgb / 12.92 * (1.0 - mask)

    x = 0.412453 * rgb[:, 0, :, :] + 0.357580 * rgb[:, 1, :, :] + 0.180423 * rgb[:, 2, :, :]
    y = 0.212671 * rgb[:, 0, :, :] + 0.715160 * rgb[:, 1, :, :] + 0.072169 * rgb[:, 2, :, :]
    z = 0.019334 * rgb[:, 0, :, :] + 0.119193 * rgb[:, 1, :, :] + 0.950227 * rgb[:, 2, :, :]

    return torch.cat((x[:, None, :, :], y[:, None, :, :], z[:, None, :, :]), dim=1)


def xyz2rgb(xyz):
    # [ 3.24048134, -1.53715152, -0.49853633],
    # [-0.96925495,  1.87599   ,  0.04155593],
    # [ 0.05564664, -0.20404134,  1.05731107]

    r = 3.24048134 * xyz[:, 0, :, :] - 1.53715152 * xyz[:, 1, :, :] - 0.49853633 * xyz[:, 2, :, :]
    g = -0.96925495 * xyz[:, 0, :, :] + 1.87599 * xyz[:, 1, :, :] + 0.04155593 * xyz[:, 2, :, :]
    b = 0.05564664 * xyz[:, 0, :, :] - 0.20404134 * xyz[:, 1, :, :] + 1.05731107 * xyz[:, 2, :, :]

    rgb = torch.cat((r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]), dim=1)
    # Some times reaches a small negative number, which causes NaNs
    rgb = torch.max(rgb, torch.zeros_like(rgb))

    mask = (rgb > 0.0031308).float().to(rgb.device)

    return (1.055 * (rgb ** (1.0 / 2.4)) - 0.055) * mask + 12.92 * rgb * (1 - mask)


def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.tensor((0.95047, 1.0, 1.08883))[None, :, None, None]
    # sc.size() torch.Size([1, 3, 1, 1])

    if xyz.is_cuda:
        sc = sc.cuda()

    xyz_scale = xyz / sc

    mask = (xyz_scale > 0.008856).float().to(xyz.device)

    xyz_int = xyz_scale ** (1 / 3.0) * mask + (7.787 * xyz_scale + 16.0 / 116.0) * (1 - mask)

    L = 116.0 * xyz_int[:, 1, :, :] - 16.0
    a = 500.0 * (xyz_int[:, 0, :, :] - xyz_int[:, 1, :, :])
    b = 200.0 * (xyz_int[:, 1, :, :] - xyz_int[:, 2, :, :])

    return torch.cat((L[:, None, :, :], a[:, None, :, :], b[:, None, :, :]), dim=1)


def lab2xyz(lab):
    y_int = (lab[:, 0, :, :] + 16.0) / 116.0
    x_int = (lab[:, 1, :, :] / 500.0) + y_int
    z_int = y_int - (lab[:, 2, :, :] / 200.0)
    z_int = torch.max(torch.tensor((0,)).to(lab.device), z_int)

    out = torch.cat((x_int[:, None, :, :], y_int[:, None, :, :], z_int[:, None, :, :]), dim=1)
    mask = (out > 0.2068966).float().to(lab.device)

    out = (out ** 3.0) * mask + (out - 16.0 / 116.0) / 7.787 * (1 - mask)

    sc = torch.tensor((0.95047, 1.0, 1.08883))[None, :, None, None]
    sc = sc.to(out.device)

    return out * sc


def rgb2lab(rgb):
    lab = xyz2lab(rgb2xyz(rgb))
    # xyz2lab(rgb2xyz(rgb)) parameters:
    # input: rgb in [0, 1.0]
    # output: l in [0, 100], ab in [-110, 110]

    l_rs = lab[:, [0], :, :] - 50.0
    ab_rs = lab[:, 1:, :, :]
    out = torch.cat((l_rs, ab_rs), dim=1)

    # out.size() -- [1, 3, 1024, 1024], l_rs in [-50.0, 50.0]
    return out


def lab2rgb(lab_rs):
    l = lab_rs[:, [0], :, :] + 50.0
    ab = lab_rs[:, 1:, :, :]
    lab = torch.cat((l, ab), dim=1)

    # lab range: l->[0, 100], ab in [-110, 110] ==> rgb: [0, 1.0]
    out = xyz2rgb(lab2xyz(lab))
    # out in [0.0, 1.0]
    return out.clamp(0.0, 1.0)
