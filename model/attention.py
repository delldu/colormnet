import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basic import DWConv2d
from spatial_correlation_sampler import SpatialCorrelationSampler

import pdb

def linear_gate(x, dim=-1):
    # return F.relu_(x).pow(2.) / x.size()[dim]
    return torch.softmax(x, dim=dim)

# LocalAttention -- LA
class LocalGatedPropagation(nn.Module):
    def __init__(self,
                 d_qk,
                 d_vu,
                 num_head,
                 dropout=0.,
                 max_dis=7,
                 d_att=64):
        super().__init__()
        # d_qk=64, # 256
        # d_vu=512 * 2,
        # num_head=1,
        # dropout=0,
        # d_att=64, # 128
        # max_dis=7,

        # self.d_qk = d_qk
        # self.d_vu = d_vu
        self.window_size = 2 * max_dis + 1
        assert self.window_size == 15
        self.max_dis = max_dis
        assert self.max_dis == 7

        self.num_head = num_head
        self.hidden_dim = d_vu // num_head

        self.d_att = d_att
        assert self.d_att == 64

        self.T = self.d_att**0.5
        assert self.T == 8

        self.relative_emb_k = nn.Conv2d(self.d_att * self.num_head,
                                        num_head * self.window_size * self.window_size,
                                        kernel_size=1,
                                        groups=num_head)

        # xxxx_debug
        self.correlation_sampler = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=self.window_size,
            stride=1,
            padding=0,
            dilation=1,
        dilation_patch=1)

        self.dw_conv = DWConv2d(d_vu)
        self.projection = nn.Linear(d_vu, d_vu)
        # self.dropout = nn.Dropout(dropout)

        self.local_mask = None
        self.last_size_2d = None
        self.qk_mask = None

    def forward(self, q, k, v, size_2d):
        n, c, h, w = v.size()

        if self.qk_mask is not None and (h, w) == self.last_size_2d:
            qk_mask = self.qk_mask
        else:
            memory_mask = torch.ones((1, 1, h, w), device=v.device).float()
            unfolded_k_mask = self.pad_and_unfold(memory_mask).view(
                1, 1, self.window_size * self.window_size, h * w)
            qk_mask = 1 - unfolded_k_mask
            self.qk_mask = qk_mask

        relative_emb = self.relative_emb_k(q)

        # Scale
        q = q / self.T
        q = q.view(-1, self.d_att, h, w)
        k = k.view(-1, self.d_att, h, w).contiguous()
        v = v.view(-1, self.num_head, self.hidden_dim, h * w)
        
        relative_emb = relative_emb.view(n, self.num_head, self.window_size * self.window_size, h * w)
        qk = self.correlation_sampler(q, k).view(n, self.num_head, self.window_size * self.window_size, h * w)
        qk = qk + relative_emb

        # assert qk.dtype == torch.float32
        qk -= qk_mask * 1e+8 if qk.dtype == torch.float32 else qk_mask * 1e+4

        local_attn = linear_gate(qk, dim=2)
        # local_attn = self.dropout(local_attn)

        global_attn = self.local2global(local_attn, h, w)
        agg_value = (global_attn @ v.transpose(-2, -1)).permute(2, 0, 1, 3).reshape(h * w, n, -1)

        output = self.dw_conv(agg_value, size_2d)
        output = self.projection(output)

        self.last_size_2d = (h, w)
        return output

    def local2global(self, local_attn, height, width):
        batch_size = local_attn.size()[0]

        pad_height = height + 2 * self.max_dis
        pad_width = width + 2 * self.max_dis

        # assert self.local_mask == None
        if self.local_mask is not None and (height, width) == self.last_size_2d:
            local_mask = self.local_mask
        else:
            ky, kx = torch.meshgrid([
                torch.arange(0, pad_height, device=local_attn.device),
                torch.arange(0, pad_width, device=local_attn.device)
            ])
            qy, qx = torch.meshgrid([
                torch.arange(0, height, device=local_attn.device),
                torch.arange(0, width, device=local_attn.device)
            ])

            offset_y = qy.reshape(-1, 1) - ky.reshape(1, -1) + self.max_dis
            offset_x = qx.reshape(-1, 1) - kx.reshape(1, -1) + self.max_dis

            local_mask = (offset_y.abs() <= self.max_dis) & (offset_x.abs() <= self.max_dis)
            local_mask = local_mask.view(1, 1, height * width, pad_height, pad_width)
            self.local_mask = local_mask

        global_attn = torch.zeros((batch_size, self.num_head, height * width, pad_height, pad_width), 
            device=local_attn.device)
        global_attn[local_mask.expand(batch_size, self.num_head, -1, -1, -1)] = \
                        local_attn.transpose(-1, -2).reshape(-1)
        global_attn = global_attn[:, :, :, self.max_dis:-self.max_dis, self.max_dis:-self.max_dis].reshape(
                        batch_size, self.num_head, height * width, height * width)

        return global_attn

    def pad_and_unfold(self, x):
        pad_pixel = self.max_dis
        x = F.pad(x, (pad_pixel, pad_pixel, pad_pixel, pad_pixel), mode='constant', value=0)
        x = F.unfold(x,
                     kernel_size=(self.window_size, self.window_size),
                     stride=(1, 1),
                     dilation=1)
        return x
