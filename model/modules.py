"""
modules.py - This file stores the rather boring network blocks.

x - usually means features that only depends on the image
g - usually means features that also depends on the mask. 
    They might have an extra "group" or "num_objects" dimension, hence
    batch_size * num_objects * num_channels * H * W

The trailing number of a variable usually denote the stride

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.group_modules import *
from model import resnet
from model.cbam import CBAM
import todos
import pdb

class FeatureFusionBlock(nn.Module):
    def __init__(self, x_in_dim, g_in_dim, g_mid_dim, g_out_dim):
        super().__init__()
        assert x_in_dim == 1024

        self.distributor = MainToGroupDistributor()
        self.block1 = GroupResBlock(x_in_dim+g_in_dim, g_mid_dim)
        self.attention = CBAM(g_mid_dim)
        self.block2 = GroupResBlock(g_mid_dim, g_out_dim)

    def forward(self, x, g):
        batch_size, num_objects = g.shape[:2]

        g = self.distributor(x, g)
        g = self.block1(g) # xxxx_3333
        r = self.attention(g.flatten(start_dim=0, end_dim=1))
        r = r.view(batch_size, num_objects, *r.shape[1:])

        g = self.block2(g+r)

        return g


class HiddenUpdater(nn.Module):
    # Used in the decoder, multi-scale feature + GRU
    # # [512, 256, 256+1], 256, hidden_dim, 1
    def __init__(self, g_dims, mid_dim, hidden_dim):
        super().__init__()
        assert hidden_dim == 64
        self.hidden_dim = hidden_dim
        self.g16_conv = GConv2D(g_dims[0], mid_dim, kernel_size=1)
        self.g8_conv = GConv2D(g_dims[1], mid_dim, kernel_size=1)
        self.g4_conv = GConv2D(g_dims[2], mid_dim, kernel_size=1)

        self.transform = GConv2D(mid_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)


    def forward(self, g, h):
        # h -- xxxx_gggg
        # g is list: len = 3
        #     tensor [item] size: [1, 2, 512, 35, 56], min: -17.046875, max: 10.992188, mean: -0.049185
        #     tensor [item] size: [1, 2, 256, 70, 112], min: -7.59375, max: 18.921875, mean: -0.155025
        #     tensor [item] size: [1, 2, 257, 140, 224], min: -11.390625, max: 17.90625, mean: -1.676876
        # tensor [h] size: [1, 2, 64, 35, 56], min: -1.0, max: 0.999023, mean: -0.009327
        g = self.g16_conv(g[0]) + \
            self.g8_conv(downsample_groups(g[1], ratio=1/2)) + \
            self.g4_conv(downsample_groups(g[2], ratio=1/4))

        g = torch.cat([g, h], 2)

        # defined slightly differently than standard GRU, 
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an 
        # implementation error that I never bothered fixing
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:,:,:self.hidden_dim])
        update_gate = torch.sigmoid(values[:,:,self.hidden_dim:self.hidden_dim*2])
        new_value = torch.tanh(values[:,:,self.hidden_dim*2:])
        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value

        return new_h


class HiddenReinforcer(nn.Module):
    # Used in the value encoder, a single GRU
    def __init__(self, g_dim, hidden_dim):
        super().__init__()
        assert g_dim == 512
        assert hidden_dim == 64

        self.hidden_dim = hidden_dim
        self.transform = GConv2D(g_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h): # xxxx_gggg
        # tensor [h] size: [1, 2, 64, 35, 56], min: 0.0, max: 0.0, mean: 0.0
        g = torch.cat([g, h], 2)
        # defined slightly differently than standard GRU, 
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an 
        # implementation error that I never bothered fixing
        values = self.transform(g) # xxxx_gggg
        forget_gate = torch.sigmoid(values[:, :, :self.hidden_dim])
        update_gate = torch.sigmoid(values[:, :, self.hidden_dim:self.hidden_dim*2])
        new_value = torch.tanh(values[:, :, self.hidden_dim*2:])
        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value

        # tensor [new_h] size: [1, 2, 64, 35, 56], min: -1.0, max: 0.999023, mean: -0.009327
        return new_h # xxxx_gggg


class ValueEncoder(nn.Module):
    def __init__(self, value_dim, hidden_dim):
        super().__init__()
        assert value_dim == 512

        network = resnet.resnet18(pretrained=True, extra_dim=2)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.layer1 = network.layer1 # 1/4, 64
        self.layer2 = network.layer2 # 1/8, 128
        self.layer3 = network.layer3 # 1/16, 256

        self.distributor = MainToGroupDistributor()
        self.fuser = FeatureFusionBlock(1024, 256, value_dim, value_dim) # (1024 256) -> (384 256) -> (384 256)
        self.hidden_reinforce = HiddenReinforcer(value_dim, hidden_dim)

    def forward(self, image, image_feat_f16, h, masks, others): # xxxx_gggg
        # image_feat_f16 is the feature from the key encoder
        g = torch.stack([masks, others], 2)
        g = self.distributor(image, g)

        batch_size, num_objects = g.shape[:2]
        g = g.flatten(start_dim=0, end_dim=1) # xxxx_3333

        g = self.conv1(g)
        g = self.bn1(g) # 1/2, 64
        g = self.maxpool(g)  # 1/4, 64
        g = self.relu(g) 

        g = self.layer1(g) # 1/4
        g = self.layer2(g) # 1/8
        g = self.layer3(g) # 1/16

        # handle dim problem raised by vit
        g = F.interpolate(g, image_feat_f16.shape[2:], mode='bilinear', align_corners=False)

        g = g.view(batch_size, num_objects, *g.shape[1:]) # xxxx_3333

        # tensor [image_feat_f16] size: [1, 1024, 35, 56], min: 0.0, max: 2.594402, mean: 0.063101
        # tensor [g] size: [1, 2, 256, 35, 56], min: 0.0, max: 13.382812, mean: 0.155741
        g = self.fuser(image_feat_f16, g)
        # tensor [g] size: [1, 2, 512, 35, 56], min: -9.921875, max: 5.101562, mean: -0.014141

        h = self.hidden_reinforce(g, h) # xxxx_gggg
        # tensor [h] size: [1, 2, 64, 35, 56], min: -1.0, max: 0.999023, mean: -0.009327

        return g, h # xxxx_3333

class KeyEncoder_DINOv2_v6(nn.Module):
    def __init__(self):
        super().__init__()
        network = resnet.resnet50(pretrained=True)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.res2 = network.layer1 # 1/4, 256
        self.layer2 = network.layer2 # 1/8, 512
        self.layer3 = network.layer3 # 1/16, 1024

        self.network2 = resnet.Segmentor()

        self.fuse1 = resnet.Fuse(384 * 4, 1024) # n = [8, 9, 10, 11]
        self.fuse2 = resnet.Fuse(384 * 4, 512)
        self.fuse3 = resnet.Fuse(384 * 4, 256)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, f):
        # tensor [f] size: [1, 3, 560, 896], min: -0.994517, max: 1.0, mean: -0.189531

        x = self.conv1(f) 
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)   # 1/4, 256
        f8 = self.layer2(f4) # 1/8, 512
        f16 = self.layer3(f8) # 1/16, 1024
        # tensor [f16] size: [1, 1024, 35, 56], min: 0.0, max: 2.109375, mean: 0.067145

        f16_dino = self.network2(f) # 1/14, 384  ->   interp to 1/16
        # tensor [f16_dino] size: [1, 1536, 35, 56], min: 0.0, max: 10.015625, mean: 0.865097

        g16 = self.fuse1(f16_dino, f16)
        g8 = self.fuse2(self.upsample2(f16_dino), f8)
        g4 = self.fuse3(self.upsample4(f16_dino), f4)
        # tensor [g16] size: [1, 1024, 35, 56], min: 0.0, max: 2.594945, mean: 0.063114
        # tensor [g8] size: [1, 512, 70, 112], min: 0.0, max: 1.842727, mean: 0.090533
        # tensor [g4] size: [1, 256, 140, 224], min: 0.0, max: 6.625021, mean: 0.200046

        return g16, g8, g4

class UpsampleBlock(nn.Module):
    def __init__(self, skip_dim, g_up_dim, g_out_dim):
        super().__init__()

        self.skip_conv = nn.Conv2d(skip_dim, g_up_dim, kernel_size=3, padding=1)
        self.distributor = MainToGroupDistributor(method='add')
        self.out_conv = GroupResBlock(g_up_dim, g_out_dim)

    def forward(self, skip_f, up_g):
        # tensor [skip_f] size: [1, 512, 70, 112], min: 0.0, max: 1.79675, mean: 0.090695
        # tensor [up_g] size: [1, 2, 512, 35, 56], min: -17.046875, max: 10.992188, mean: -0.049185

        skip_f = self.skip_conv(skip_f)
        g = upsample_groups(up_g, ratio=2)
        # tensor [skip_f] size: [1, 512, 70, 112], min: -3.976562, max: 2.03125, mean: 0.014867
        # tensor [g] size: [1, 2, 512, 70, 112], min: -16.765625, max: 9.71875, mean: -0.049185

        g = self.distributor(skip_f, g) # xxxx_3333
        # tensor [g] size: [1, 2, 512, 70, 112], min: -16.578125, max: 9.945312, mean: -0.034318

        g = self.out_conv(g) # xxxx_3333
        # tensor [g] size: [1, 2, 256, 70, 112], min: -7.59375, max: 18.921875, mean: -0.155025

        return g
    
    
class KeyProjection(nn.Module):
    def __init__(self, in_dim, keydim):
        super().__init__()
        assert in_dim == 1024
        assert keydim == 64

        self.key_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)
        # shrinkage
        self.d_proj = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        # selection
        self.e_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)

        # nn.init.orthogonal_(self.key_proj.weight.data)
        # nn.init.zeros_(self.key_proj.bias.data)
    def forward(self, x):
        shrinkage = self.d_proj(x)**2 + 1
        selection = torch.sigmoid(self.e_proj(x))

        return self.key_proj(x), shrinkage, selection


class Decoder(nn.Module):
    def __init__(self, val_dim, hidden_dim):
        super().__init__()
        assert val_dim == 512
        assert hidden_dim == 64

        self.fuser = FeatureFusionBlock(1024, val_dim+hidden_dim, 512, 512) 
        self.hidden_update = HiddenUpdater([512, 256, 256+1], 256, hidden_dim) 
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4
        self.pred = nn.Conv2d(256, 1, kernel_size=3, padding=1, stride=1)

    # xxxx_gggg
    def forward(self, f16, f8, f4, hidden_state, memory_readout, h_out=True):
        assert h_out == True
        batch_size, num_objects = memory_readout.shape[:2]

        # todos.debug.output_var("add-f16", torch.cat([memory_readout, hidden_state], 2))
        # tensor [f16] size: [1, 1024, 35, 56], min: 0.0, max: 2.601784, mean: 0.063031
        # tensor [add-f16] size: [1, 2, 576, 35, 56], min: -9.328125, max: 4.703125, mean: -0.008664
        # xxxx_gggg
        g16 = self.fuser(f16, torch.cat([memory_readout, hidden_state], 2))
        # tensor [g16] size: [1, 2, 512, 35, 56], min: -17.046875, max: 10.992188, mean: -0.049185

        g8 = self.up_16_8(f8, g16)
        g4 = self.up_8_4(f4, g8)

        logits = self.pred(F.relu(g4.flatten(start_dim=0, end_dim=1)))
        if h_out:
            g4 = torch.cat([g4, logits.view(batch_size, num_objects, 1, *logits.shape[-2:])], 2)
            hidden_state = self.hidden_update([g16, g8, g4], hidden_state) # xxxx_gggg
        
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        logits = logits.view(batch_size, num_objects, *logits.shape[-2:])

        # tensor [hidden_state] size: [1, 2, 64, 35, 56], min: -0.999481, max: 0.999002, mean: -0.085589
        # tensor [logits] size: [1, 2, 560, 896], min: -0.472656, max: 0.702148, mean: 0.024722
        return hidden_state, logits
