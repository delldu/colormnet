"""
Group-specific modules
They handle features that also depends on the mask. 
Features are typically of shape
    batch_size * num_objects * num_channels * H * W

All of them are permutation equivariant w.r.t. to the num_objects dimension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import todos
import pdb

def interpolate_groups(g, ratio, mode, align_corners):
    assert len(g.shape) == 5
    if len(g.shape) == 4:
        g = F.interpolate(g, scale_factor=ratio, mode=mode, align_corners=align_corners)
    elif len(g.shape) == 5:
        batch_size, num_objects = g.shape[:2]
        g = F.interpolate(g.flatten(start_dim=0, end_dim=1), 
                    scale_factor=ratio, mode=mode, align_corners=align_corners)
        g = g.view(batch_size, num_objects, *g.shape[1:])
    return g

def upsample_groups(g, ratio=2, mode='bilinear', align_corners=False):
    return interpolate_groups(g, ratio, mode, align_corners)

def downsample_groups(g, ratio=1/2, mode='area', align_corners=None):
    return interpolate_groups(g, ratio, mode, align_corners)


class GConv2D(nn.Conv2d): # xxxx_gggg
    def forward(self, g):
        # tensor [g] size: [1, 2, 1280, 35, 56], min: 0.0, max: 13.382812, mean: 0.081629
        batch_size, num_objects = g.shape[:2]
        # g.flatten(start_dim=0, end_dim=1).size() -- [2, 1280, 35, 56]

        g = super().forward(g.flatten(start_dim=0, end_dim=1))

        return g.view(batch_size, num_objects, *g.shape[1:])


class GroupResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        if in_dim == out_dim:
            self.downsample = None
        else:
            # xxxx_debug
            self.downsample = GConv2D(in_dim, out_dim, kernel_size=3, padding=1)

        self.conv1 = GConv2D(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = GConv2D(out_dim, out_dim, kernel_size=3, padding=1)
 
    def forward(self, g):
        out_g = self.conv1(F.relu(g))
        out_g = self.conv2(F.relu(out_g))
        
        if self.downsample is not None:
            g = self.downsample(g)

        return out_g + g


class MainToGroupDistributor(nn.Module):
    def __init__(self, method='cat'):
        super().__init__()
        self.method = method

    def forward(self, x, g):
        num_objects = g.shape[1]

        # xxxx_debug
        if self.method == 'cat':
            g = torch.cat([x.unsqueeze(1).expand(-1, num_objects, -1, -1, -1), g], 2)
        elif self.method == 'add':
            g = x.unsqueeze(1).expand(-1, num_objects, -1, -1, -1) + g
        else:
            raise NotImplementedError

        return g
