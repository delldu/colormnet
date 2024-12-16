import torch
import torch.nn as nn
import torch.nn.functional as F
from spatial_correlation_sampler import SpatialCorrelationSampler
import todos

import pdb

class DWConv2d(nn.Module):
    def __init__(self, indim, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv2d(indim,
                              indim,
                              5,
                              dilation=1,
                              padding=2,
                              groups=indim,
                              bias=False)
        # self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, x, size_2d):
        h, w = size_2d
        _, bs, c = x.size()
        x = x.view(h, w, bs, c).permute(2, 3, 0, 1)
        x = self.conv(x)
        # x = self.dropout(x)
        x = x.view(bs, c, h * w).permute(2, 0, 1)
        return x

# LocalAttention -- LA
class LocalGatedPropagation(nn.Module):
    def __init__(self, d_qk = 64, d_vu = 1024, max_dis=7, d_att=64):
        super().__init__()
        self.window_size = 2 * max_dis + 1
        assert self.window_size == 15
        self.max_dis = max_dis
        assert self.max_dis == 7

        self.hidden_dim = d_vu
        self.d_att = d_att
        assert self.d_att == 64

        self.relative_emb_k = nn.Conv2d(self.d_att, self.window_size * self.window_size, kernel_size=1)
        # self.relative_emb_k -- Conv2d(64, 225, kernel_size=(1, 1), stride=(1, 1))

        # xxxx_debug
        self.correlation_sampler = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=self.window_size, # 15
            stride=1, padding=0, dilation=1, dilation_patch=1)

        self.dw_conv = DWConv2d(d_vu)
        self.projection = nn.Linear(d_vu, d_vu)
        # self.dropout = nn.Dropout(dropout)

        self.local_mask = None

    def forward(self, q, k, v):
        # tensor [q] size: [1, 64, 35, 56], min: -2.75, max: 3.162109, mean: -0.143807
        # tensor [k] size: [1, 64, 35, 56], min: -2.755859, max: 3.140625, mean: -0.143588
        # tensor [v] size: [1, 1024, 35, 56], min: -9.921875, max: 5.101562, mean: -0.014141
        # --------------------------------------------------------------------------------
        assert q.size()[2:] == k.size()[2:]
        assert q.size()[2:] == v.size()[2:]
        B, C, H, W = v.size()

        relative_emb = self.relative_emb_k(q)
        relative_emb = relative_emb.view(B, self.window_size * self.window_size, H * W)

        # Scale
        q = q / (self.d_att**0.5) # 8
        assert q.size() == q.view(-1, self.d_att, H, W).size()
        q = q.view(-1, self.d_att, H, W)

        assert k.size() == k.view(-1, self.d_att, H, W).size()
        k = k.view(-1, self.d_att, H, W).contiguous()
        v = v.view(-1, 1, self.hidden_dim, H * W)
        
        # tensor [q] size: [1, 64, 35, 56], min: -0.342285, max: 0.398682, mean: -0.017892
        # tensor [k] size: [1, 64, 35, 56], min: -2.755859, max: 3.140625, mean: -0.143588
        # tensor [qk] size: [15, 15, 35, 56], min: 0.0, max: 5.730469, mean: 3.024395
        # --------------------------------------------------------------------------------
        qk = self.correlation_sampler(q, k).view(B, self.window_size * self.window_size, H * W)
        # qk.size() -- [1, 225, 1960], 1960 == 35 * 56

        qk = qk + relative_emb
        # local_attn = torch.softmax(qk, dim=2)
        local_attn = torch.softmax(qk, dim=1)

        # tensor [local_attn] size: [1, 1, 225, 1960], min: 0.0, max: 0.519583, mean: 0.004444
        global_attn = self.local2global(local_attn, H, W)
        # tensor [global_attn] size: [1, 1, 1960, 1960], min: 0.0, max: 0.519583, mean: 0.00051

        agg_value = (global_attn @ v.transpose(-2, -1)).permute(2, 0, 1, 3).reshape(H * W, B, -1)
        output = self.dw_conv(agg_value, (H, W))
        output = self.projection(output)

        # tensor [output] size: [1960, 1, 1024], min: -2.011719, max: 1.611328, mean: 0.004996
        return output

    def local2global(self, local_attn, height, width):
        # tensor [local_attn] size: [1, 1, 225, 1960], min: 0.0, max: 0.519583, mean: 0.004444
        B = local_attn.size()[0]

        pad_height = height + 2 * self.max_dis
        pad_width = width + 2 * self.max_dis

        if self.local_mask is not None:
            local_mask = self.local_mask
        else:
            ky, kx = torch.meshgrid([
                torch.arange(0, pad_height, device=local_attn.device),
                torch.arange(0, pad_width, device=local_attn.device)
            ])
            # tensor [ky] size: [49, 70], min: 0.0, max: 48.0, mean: 23.999998
            # tensor [kx] size: [49, 70], min: 0.0, max: 69.0, mean: 34.5
            # ky --
            # tensor([[ 0,  0,  0,  ...,  0,  0,  0],
            #         [ 1,  1,  1,  ...,  1,  1,  1],
            #         [ 2,  2,  2,  ...,  2,  2,  2],
            #         ...,
            #         [46, 46, 46,  ..., 46, 46, 46],
            #         [47, 47, 47,  ..., 47, 47, 47],
            #         [48, 48, 48,  ..., 48, 48, 48]], device='cuda:0')
            qy, qx = torch.meshgrid([
                torch.arange(0, height, device=local_attn.device),
                torch.arange(0, width, device=local_attn.device)
            ])
            # tensor [qy] size: [35, 56], min: 0.0, max: 34.0, mean: 17.0
            # tensor [qx] size: [35, 56], min: 0.0, max: 55.0, mean: 27.499998

            offset_y = qy.reshape(-1, 1) - ky.reshape(1, -1) + self.max_dis
            offset_x = qx.reshape(-1, 1) - kx.reshape(1, -1) + self.max_dis
            # tensor [offset_y] size: [1960, 3430], min: -41.0, max: 41.0, mean: 0.0
            # tensor [offset_x] size: [1960, 3430], min: -62.0, max: 62.0, mean: 0.0
            local_mask = (offset_y.abs() <= self.max_dis) & (offset_x.abs() <= self.max_dis)
            local_mask = local_mask.view(1, height * width, pad_height, pad_width)
            self.local_mask = local_mask

        # tensor [local_mask] size: [1, 1960, 49, 70], min: 0.0, max: 1.0, mean: 0.065598
        global_attn = torch.zeros((B, height * width, pad_height, pad_width), device=local_attn.device)
        # global_attn.size() -- [1, 1960, 49, 70]

        # local_mask.size() -- torch.Size([1, 1960, 49, 70])
        # (Pdb) local_attn.size() -- [1, 1, 225, 1960]
        # (Pdb) local_attn.transpose(-1, -2).size() -- [1, 1, 1960, 225]
        # local_attn.transpose(-1, -2).reshape(-1).size() -- [441000]
        # local_mask.expand(B, -1, -1, -1).size() -- [1, 1960, 49, 70]
        # 1960*49*70 -- 6722800
        global_attn[local_mask.expand(B, -1, -1, -1)] = local_attn.transpose(-1, -2).reshape(-1)
        global_attn = global_attn[:, :, self.max_dis:-self.max_dis, self.max_dis:-self.max_dis].reshape(B, height * width, height * width)
        # tensor [global_attn] size: [1, 1, 1960, 1960], min: 0.0, max: 0.519583, mean: 0.00051
        # ==>  # tensor [global_attn] size: [1, 1960, 1960], min: 0.0, max: 0.519583, mean: 0.000508
        return global_attn #[1, 1960, 1960]

    # def pad_and_unfold(self, x):
    #     pad_pixel = self.max_dis
    #     x = F.pad(x, (pad_pixel, pad_pixel, pad_pixel, pad_pixel), mode='constant', value=0)
    #     x = F.unfold(x,
    #                  kernel_size=(self.window_size, self.window_size),
    #                  stride=(1, 1),
    #                  dilation=1)
    #     return x
