# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020-2024(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 11月 02日 星期一 17:52:14 CST
# ***
# ************************************************************************************/
#

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov2 import DinoVisionTransformer
from spatial_correlation_sampler import SpatialCorrelationSampler

from . import resnet
from . import data
import todos

import pdb


class ColorMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.MAX_H = 1024
        self.MAX_W = 1024
        self.MAX_TIMES = 112
        # ------------------------------------------------------------------------------
        self.key_dim = 64
        self.value_dim = 512
        self.hidden_dim = 64

        self.key_encoder = DINOv2_v6()
        self.key_proj = KeyProjection()

        # self.value_encoder = ValueEncoder(self.value_dim, self.hidden_dim)
        self.value_encoder = ValueEncoder()
        self.short_term_attn = LocalAttention(d_vu=512 * 2, d_att=64, max_dis=7)
        # self.decoder = Decoder(self.value_dim, self.hidden_dim)
        self.decoder = Decoder()

        self.load_weights()
        # from ggml_engine import create_network
        # create_network(self)


    def forward(self, image_tensor, reference_tensor):
        B2, C2, H2, W2 = image_tensor.size()

        image_tensor = self.resize_pad(image_tensor)
        B, C, H, W = image_tensor.size()
        reference_tensor = F.interpolate(reference_tensor, size=(H, W), mode="bilinear", align_corners=False)

        return image_tensor


    def resize_pad(self, x):
        # Need Resize ?
        B, C, H, W = x.size()
        if H > self.MAX_H or W > self.MAX_W:
            s = min(self.MAX_H / H, self.MAX_W / W)
            SH, SW = int(s * H), int(s * W)
            x = F.interpolate(x, size=(SH, SW), mode="bilinear", align_corners=False)

        # Need Pad ?
        B, C, H, W = x.size()
        pad_h = (self.MAX_TIMES - (H % self.MAX_TIMES)) % self.MAX_TIMES
        pad_w = (self.MAX_TIMES - (W % self.MAX_TIMES)) % self.MAX_TIMES
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')

        return x

    def load_weights(self, model_path="models/video_color.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        self.load_state_dict(torch.load(checkpoint))


    def encode_key(self, frame): 
        # Determine input shape
        # (Pdb) frame.shape -- [1, 3, 560, 896]
        assert len(frame.shape) == 4
        f16, f8, f4 = self.key_encoder(frame)
        key, shrinkage, selection = self.key_proj(f16)
        return key, shrinkage, selection, f16, f8, f4


    def encode_value(self, frame, f16, h16, ref_ab): 
        # tensor [ref_ab] size: [1, 2, 560, 896], min: -0.476372, max: 0.657849, mean: 0.02309
        g16, h16 = self.value_encoder(frame, f16, h16, ref_ab)
        return g16, h16

    
    def decode_color(self, multi_scale_features, color_feature, hidden_state):
        # multi_scale_features is tuple: len = 3
        #     tensor [item] size: [1, 1024, 35, 56], min: 0.0, max: 2.601784, mean: 0.063031
        #     tensor [item] size: [1, 512, 70, 112], min: 0.0, max: 1.79675, mean: 0.090695
        #     tensor [item] size: [1, 256, 140, 224], min: 0.0, max: 6.709424, mean: 0.200673
        # tensor [color_feature] size: [2, 512, 35, 56], min: -9.328125, max: 4.738281, mean: -0.007783
        # tensor [hidden_state] size: [2, 64, 35, 56], min: -1.0, max: 0.999023, mean: -0.009137
        return  self.decoder(*multi_scale_features, hidden_state, color_feature)


class DINOv2_v6(nn.Module):
    def __init__(self):
        super().__init__()
        # network = resnet.resnet50()
        # self.conv1 = network.conv1
        # self.bn1 = network.bn1
        # self.maxpool = network.maxpool
        # self.res2 = network.layer1
        # self.layer2 = network.layer2
        # self.layer3 = network.layer3

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res2 = resnet.make_bottleneck_layer(64, 64, 3, stride=1)
        self.layer2 = resnet.make_bottleneck_layer(256, 128, 4, stride=2)
        self.layer3 = resnet.make_bottleneck_layer(512, 256, 6, stride=2)

        self.network2 = Segmentor()

        self.fuse1 = resnet.Fuse(384 * 4, 1024) # n = [8, 9, 10, 11]
        self.fuse2 = resnet.Fuse(384 * 4, 512)
        self.fuse3 = resnet.Fuse(384 * 4, 256)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')


    def forward(self, f):
        # tensor [f] size: [1, 3, 560, 896], min: -0.994517, max: 1.0, mean: -0.189531

        x = self.conv1(f) 
        x = self.bn1(x)
        x = F.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)   # 1/4, 256
        f8 = self.layer2(f4) # 1/8, 512
        f16 = self.layer3(f8) # 1/16, 1024
        # tensor [f16] size: [1, 1024, 35, 56], min: 0.0, max: 2.109375, mean: 0.067145

        dino_f16 = self.network2(f) # 1/14, 384  ->   interp to 1/16
        # tensor [dino_f16] size: [1, 1536, 35, 56], min: 0.0, max: 10.015625, mean: 0.865097

        g16 = self.fuse1(dino_f16, f16)
        g8 = self.fuse2(self.upsample2(dino_f16), f8)
        g4 = self.fuse3(self.upsample4(dino_f16), f4)
        # tensor [g16] size: [1, 1024, 35, 56], min: 0.0, max: 2.594945, mean: 0.063114
        # tensor [g8] size: [1, 512, 70, 112], min: 0.0, max: 1.842727, mean: 0.090533
        # tensor [g4] size: [1, 256, 140, 224], min: 0.0, max: 6.625021, mean: 0.200046

        return g16, g8, g4

class Segmentor(nn.Module):
    '''一个线性设置的增强版本。将最后4层的patch token concat起来预测类logits'''
    def __init__(self):
        super().__init__()

        self.backbone = DinoVisionTransformer()
        self.conv3 = nn.Conv2d(1536, 1536, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(1536)
        # self.backbone -- DinoVisionTransformer()
        # self.backbone.forward.__code__
        #   -- <file "/home/dell/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/models/vision_transformer.py", line 324>

    def forward(self, x):
        # tensor [x] size: [1, 3, 560, 896], min: -0.994517, max: 1.0, mean: -0.189531
        tokens = self.backbone(x) #.get_intermediate_layers(x, n=[8, 9, 10, 11], reshape=True) # last n=4 [8, 9, 10, 11]
        # tokens is tuple: len = 4
        #     tensor [item] size: [1, 384, 40, 64], min: -64.093712, max: 65.633827, mean: 0.04372
        #     tensor [item] size: [1, 384, 40, 64], min: -60.8563, max: 49.656631, mean: 0.003902
        #     tensor [item] size: [1, 384, 40, 64], min: -46.128963, max: 40.135544, mean: 0.009809
        #     tensor [item] size: [1, 384, 40, 64], min: -21.549391, max: 19.685974, mean: 0.007802
        f16 = torch.cat(tokens, dim=1)
        # tensor [f16] size: [1, 1536, 40, 64], min: -64.093712, max: 65.633827, mean: 0.016308
        f16 = self.conv3(f16)
        f16 = self.bn3(f16)
        f16 = F.relu(f16)
        old_size = (f16.shape[2], f16.shape[3])
        new_size = (int(old_size[0]*14/16), int(old_size[1]*14/16))
        f16 = F.interpolate(f16, size=new_size, mode='bilinear', align_corners=False) # scale_factor=3.5
        # tensor [f16] size: [1, 1536, 35, 56], min: 0.0, max: 10.015625, mean: 0.865097
        return f16


class ValueEncoder(nn.Module):
    def __init__(self):
    # def __init__(self, value_dim, hidden_dim):
        super().__init__()
        # assert value_dim == 512

        # network = resnet.resnet18()
        # self.conv1 = network.conv1
        # self.bn1 = network.bn1
        # self.maxpool = network.maxpool
        # self.layer1 = network.layer1 # 1/4, 64
        # self.layer2 = network.layer2 # 1/8, 128
        # self.layer3 = network.layer3 # 1/16, 256

        value_dim = 512
        hidden_dim = 64
        self.conv1 = nn.Conv2d(3 + 2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = resnet.make_basicblock_layer(64, 64, 2, stride=1)
        self.layer2 = resnet.make_basicblock_layer(64, 128, 2, stride=2)
        self.layer3 = resnet.make_basicblock_layer(128, 256, 2, stride=2)

        self.fuser = FeatureFusionBlock(256)
        self.hidden_reinforce = HiddenReinforcer()

    def forward(self, image, f16, h16, ref_ab):
        # tensor [image] size: [1, 3, 560, 896], min: -5.011174, max: 5.030801, mean: 0.000676
        # tensor [f16] size: [1, 1024, 35, 56], min: 0.0, max: 2.263417, mean: 0.06662
        # tensor [h16] size: [2, 64, 35, 56], min: -4.67093, max: 4.75749, mean: 0.000638
        # tensor [ref_ab] size: [1, 2, 560, 896], min: -4.593725, max: 4.704186, mean: -0.002239
        s0 = ref_ab[:, 0:1, :, :]
        s1 = ref_ab[:, 1:2, :, :]
        ref_ba = torch.cat([s1, s0], dim = 1)
        # tensor [ref_ba] size: [1, 2, 560, 896], min: -4.593725, max: 4.704186, mean: -0.002239

        g = torch.cat([ref_ab, ref_ba], dim=0)
        B, C, H, W = g.size()
        g = torch.cat([image.repeat(B, 1, 1, 1), g], dim=1)

        g = self.conv1(g)
        g = self.bn1(g)     # 1/2, 64
        g = self.maxpool(g) # 1/4, 64
        g = F.relu(g) 

        g = self.layer1(g) # 1/4
        g = self.layer2(g) # 1/8
        g = self.layer3(g) # 1/16

        # handle dim problem raised by vit
        g = F.interpolate(g, f16.shape[2:], mode='bilinear', align_corners=False)

        g = self.fuser(f16, g)

        # tensor [g] size: [2, 512, 35, 56], min: -28.850132, max: 14.702946, mean: -0.759376
        # tensor [h16] size: [2, 64, 35, 56], min: -4.463562, max: 4.321184, mean: 0.000787
        h = self.hidden_reinforce(g, h16)
        # tensor [h] size: [2, 64, 35, 56], min: -4.807474, max: 4.858111, mean: 0.004066

        return g, h

class KeyProjection(nn.Module):
    # def __init__(self, in_dim, keydim):
    def __init__(self):
        super().__init__()
        # assert in_dim == 1024
        # assert keydim == 64
        in_dim = 1024
        keydim = 64
        self.key_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1) # shrinkage
        self.d_proj = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1) # selection
        self.e_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)

    def forward(self, x):
        key = self.key_proj(x)
        shrinkage = self.d_proj(x)**2 + 1
        selection = torch.sigmoid(self.e_proj(x))
        return key, shrinkage, selection

# ----------------------------------------
class FeatureFusionBlock(nn.Module):
    def __init__(self, g_in_dim):
    # def __init__(self):
        super().__init__()
        x_in_dim = 1024
        g_mid_dim = 512
        g_out_dim = 512
        self.block1 = GroupResBlock(x_in_dim+g_in_dim, g_mid_dim)
        self.attention = CBAM(g_mid_dim)
        self.block2 = GroupResBlock(g_mid_dim, g_out_dim)

    def forward(self, x, g):
        B, C, H, W = g.size()
        g = torch.cat([x.repeat(B, 1, 1, 1), g], dim = 1)
        g = self.block1(g)
        r = self.attention(g)
        g = self.block2(g + r)
        return g

# ----------------------------------------
class HiddenReinforcer(nn.Module):
    # Used in the value encoder, a single GRU
    def __init__(self):
        super().__init__()
        g_dim = 512
        hidden_dim = 64

        self.hidden_dim = hidden_dim
        self.transform = nn.Conv2d(g_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)

    def forward(self, g, h):
        # tensor [g] size: [2, 512, 35, 56], min: -28.850132, max: 14.702946, mean: -0.759376
        # tensor [h] size: [2, 64, 35, 56], min: -4.463562, max: 4.321184, mean: 0.000787

        g = torch.cat([g, h], dim=1)
        # tensor [g] size: [2, 576, 35, 56], min: -28.850132, max: 14.702946, mean: -0.674914
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:, :self.hidden_dim])
        update_gate = torch.sigmoid(values[:, self.hidden_dim:self.hidden_dim*2])
        new_value = torch.tanh(values[:, self.hidden_dim*2:])

        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value
        # tensor [new_h] size: [2, 64, 35, 56], min: -3.942453, max: 3.99185, mean: 0.111976
        return new_h


class DWConv2d(nn.Module):
    def __init__(self, indim):
        super().__init__()
        self.conv = nn.Conv2d(indim, indim, 5, dilation=1, padding=2, groups=indim, bias=False)
        # self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, x, size_2d):
        # tensor [x] size: [1960, 1, 1024], min: -9.280723, max: 4.028006, mean: -0.008225
        # size_2d -- (35, 36)
        h, w = size_2d
        n, bs, c = x.size() # 1, 1024
        x = x.view(h, w, bs, c).permute(2, 3, 0, 1) # [35, 36, 1, 1024] ==> [1, 1024, 35, 36]
        x = self.conv(x)
        # x = self.dropout(x)
        x = x.view(bs, c, h * w).permute(2, 0, 1) # [1, 1024, 1960] ==> [1960, 1, 1024]
        # tensor [x] size: [1960, 1, 1024], min: -6.485817, max: 5.726138, mean: -0.003478
        return x

class LocalAttention(nn.Module):
    '''LocalGatedPropagation'''
    def __init__(self, d_vu = 1024, max_dis=7, d_att=64):
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
        B, C, H, W = v.size()
        v = v.view(1, 1024, H, W)

        assert q.size()[2:] == k.size()[2:]
        # assert q.size()[2:] == v.size()[2:]

        relative_emb = self.relative_emb_k(q)
        relative_emb = relative_emb.view(1, self.window_size * self.window_size, H * W)

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
        qk = self.correlation_sampler(q, k).view(1, self.window_size * self.window_size, H * W)
        # qk.size() -- [1, 225, 1960], 1960 == 35 * 56

        qk = qk + relative_emb
        # local_attn = torch.softmax(qk, dim=2)
        local_attn = torch.softmax(qk, dim=1)

        # tensor [local_attn] size: [1, 1, 225, 1960], min: 0.0, max: 0.519583, mean: 0.004444
        global_attn = self.local2global(local_attn, H, W)
        # tensor [global_attn] size: [1, 1, 1960, 1960], min: 0.0, max: 0.519583, mean: 0.00051

        agg_value = (global_attn @ v.transpose(-2, -1)).permute(2, 0, 1, 3).reshape(H * W, 1, -1)
        output = self.dw_conv(agg_value, (H, W))
        output = self.projection(output)

        # tensor [output] size: [1960, 1, 1024], min: -2.011719, max: 1.611328, mean: 0.004996
        # ==> [2, 512, 35, 56]

        # batch, num_objects, value_dim, h, w = self.last_ti_value.shape
        # last_ti_value = self.last_ti_value.flatten(start_dim=1, end_dim=2)
        # # tensor [key] size: [1, 64, 35, 56], min: -2.75, max: 3.166016, mean: -0.143513
        # # tensor [self.last_ti_key] size: [1, 64, 35, 56], min: -2.753906, max: 3.142578, mean: -0.143365
        # # tensor [last_ti_value] size: [1, 1024, 35, 56], min: -9.9375, max: 5.101562, mean: -0.014165
        # memory_value_short = self.network.short_term_attn(key, self.last_ti_key, last_ti_value)
        # # tensor [memory_value_short] size: [1960, 1, 1024], min: -2.007812, max: 1.608398, mean: 0.005006

        # memory_value_short = memory_value_short.permute(1, 2, 0).view(batch, num_objects, value_dim, h, w)
        output = output.permute(1, 2, 0).view(2, 512, H, W)
        return output # ==> [2, 512, 35, 56]

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
            ], indexing="ij")
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
            ], indexing="ij")
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

# ----------------------------------------
class GroupResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        if in_dim == out_dim:
            self.downsample = None
        else:
            # xxxx_debug
            self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
 
    def forward(self, g):
        # tensor [g] size: [2, 1280, 35, 56], min: 0.0, max: 27.27334, mean: 0.346497
        out_g = self.conv1(F.relu(g))
        out_g = self.conv2(F.relu(out_g))
        
        if self.downsample is not None:
            g = self.downsample(g)

        return out_g + g

class BasicConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, dilation=1, groups=1, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, pool_types=['avg', 'max']):
        super().__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // 16),
            nn.ReLU(),
            nn.Linear(gate_channels // 16, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class ChannelPool(nn.Module):
    def forward(self, x):
        # tensor [x] size: [2, 512, 35, 56], min: -5.116851, max: 2.511841, mean: -0.022028
        # (Pdb) torch.max(x,1)[0].size() -- torch.Size([2, 35, 56]) ==> [2, 1, 35, 56]
        # torch.mean(x,1).size() -- [2, 35, 56] ==> [2, 1, 35, 35]
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1) #[2, 2, 35, 56]

class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv()
        
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

# CBAM(g_mid_dim)
class CBAM(nn.Module):
    def __init__(self, gate_channels, pool_types=['avg', 'max']):
        super().__init__()
        self.ChannelGate = ChannelGate(gate_channels, pool_types)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out

class Decoder(nn.Module):
    # def __init__(self, value_dim, hidden_dim):
    def __init__(self):
        super().__init__()
        value_dim = 512
        hidden_dim = 64
        self.fuser = FeatureFusionBlock(value_dim+hidden_dim) # 576
        self.hidden_update = HiddenUpdater() 
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4
        self.pred = nn.Conv2d(256, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, f16, f8, f4, hidden_state, color_feature):
        g16 = self.fuser(f16, torch.cat([color_feature, hidden_state], dim=1))
        # tensor [g16] size: [2, 512, 35, 56], min: -89.383621, max: 14.023798, mean: -1.546733

        g8 = self.up_16_8(f8, g16)
        g4 = self.up_8_4(f4, g8)
        # tensor [g4] size: [2, 256, 140, 224], min: -34.172653, max: 25.263411, mean: -7.309633

        logits = self.pred(F.relu(g4))
        g4 = torch.cat([g4, logits], 1)
        hidden_state = self.hidden_update([g16, g8, g4], hidden_state)
        # tensor [hidden_state] size: [2, 64, 35, 56], min: -0.999481, max: 0.999002, mean: -0.085589
        
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        logits = logits.permute(1, 0, 2, 3).contiguous() # (C, B, H, W) --> (B, C, H, W)
        # tensor [logits] size: [1, 2, 560, 896], min: -0.472656, max: 0.702148, mean: 0.024722
        predict_color_ab = torch.tanh(logits)

        return hidden_state, predict_color_ab

class HiddenUpdater(nn.Module):
    # Used in the decoder, multi-scale feature + GRU
    def __init__(self):
        super().__init__()
        self.hidden_dim = 64
        self.g16_conv = nn.Conv2d(512, 256, kernel_size=1)
        self.g8_conv = nn.Conv2d(256, 256, kernel_size=1)
        self.g4_conv = nn.Conv2d(257, 256, kernel_size=1)
        self.transform = nn.Conv2d(320, self.hidden_dim*3, kernel_size=3, padding=1)


    def forward(self, g, h):
        # g is list: len = 3
        #     tensor [item] size: [2, 512, 35, 56], min: -82.737953, max: 16.419943, mean: -1.540743
        #     tensor [item] size: [2, 256, 70, 112], min: -27.510799, max: 26.222929, mean: -1.975606
        #     tensor [item] size: [2, 257, 140, 224], min: -33.898613, max: 25.28302, mean: -7.281233
        # tensor [h] size: [2, 64, 35, 56], min: -3.735293, max: 4.35715, mean: 0.105984

        g = self.g16_conv(g[0]) + \
            self.g8_conv(F.interpolate(g[1], scale_factor=0.5, mode='area', align_corners=None)) + \
            self.g4_conv(F.interpolate(g[2], scale_factor=0.25, mode='area', align_corners=None))

        g = torch.cat([g, h], dim=1)

        # defined slightly differently than standard GRU, 
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an 
        # implementation error that I never bothered fixing
        values = self.transform(g)
        # tensor [values] size: [2, 192, 35, 56], min: -46.875889, max: 65.977158, mean: 4.075093

        forget_gate = torch.sigmoid(values[:,:self.hidden_dim])
        update_gate = torch.sigmoid(values[:,self.hidden_dim:self.hidden_dim*2])
        new_value = torch.tanh(values[:,self.hidden_dim*2:])

        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value
        # tensor [new_h] size: [2, 64, 35, 56], min: -3.057797, max: 3.071628, mean: 0.050769

        return new_h


class UpsampleBlock(nn.Module):
    def __init__(self, skip_dim, g_up_dim, g_out_dim):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_dim, g_up_dim, kernel_size=3, padding=1)
        self.out_conv = GroupResBlock(g_up_dim, g_out_dim)

    def forward(self, skip_f, up_g):
        # tensor [skip_f] size: [1, 512, 70, 112], min: 0.0, max: 1.505345, mean: 0.09193
        # tensor [up_g] size: [2, 512, 35, 56], min: -89.383621, max: 14.023798, mean: -1.546733

        skip_f = self.skip_conv(skip_f)
        g = F.interpolate(up_g, scale_factor=2.0, mode='bilinear', align_corners=False)
        # tensor [skip_f] size: [1, 512, 70, 112], min: -3.976562, max: 2.03125, mean: 0.014867
        # tensor [g] size: [2, 512, 70, 112], min: -84.804108, max: 13.753417, mean: -1.546733

        B, C, H, W = g.size()
        g = skip_f.repeat(B, 1, 1, 1) + g
        # tensor [g] size: [2, 512, 70, 112], min: -16.578125, max: 9.945312, mean: -0.034318

        g = self.out_conv(g)
        # tensor [g] size: [2, 256, 70, 112], min: -7.59375, max: 18.921875, mean: -0.155025

        return g

