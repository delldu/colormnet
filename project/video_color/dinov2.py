# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import todos
import pdb

class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)
    """

    def __init__(self,
        img_size  = 518,
        patch_size  = 14,
        in_chans  = 3,
        embed_dim  = 384,
        norm_layer  = None,
    ):
        super().__init__()
        # img_size = 518
        # patch_size = 14
        # in_chans = 3
        # embed_dim = 384
        # norm_layer = None

        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size) 
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        assert norm_layer == None
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        x = x.flatten(2).transpose(1, 2)  # B HW C
        # x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads = 6):
        super().__init__()
        # dim = 384
        # num_heads = 6

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        B, N, C = x.shape
        # tensor [x] size: [1, 2561, 384], min: -7.700042, max: 5.273503, mean: 0.004173

        ## tensor [qkv(x)] size: [1, 2561, 1152], min: -11.863246, max: 11.457247, mean: 0.032135
        # qkv2 = self.qkv(x).reshape(B, N, 3*self.num_heads, C//self.num_heads)
        # q2 = qkv2[:, :, 0:self.num_heads, :] * self.scale
        # q2 = q2.permute(0, 2, 1, 3)  # [1, 6, 2561, 64] --> [1, 2561, 6, 64]
        # k2 = qkv2[:, :, self.num_heads:2*self.num_heads, :]
        # k2 = k2.permute(0, 2, 1, 3)  # [1, 6, 2561, 64] --> [1, 2561, 6, 64]
        # v2 = qkv2[:, :, 2*self.num_heads:3*self.num_heads, :]
        # v2 = v2.permute(0, 2, 1, 3) # [1, 6, 2561, 64] --> [1, 2561, 6, 64]

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # tensor [qkv] size: [3, 1, 6, 2561, 64], min: -11.863246, max: 11.457247, mean: 0.03211
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        # tensor [k] size: [1, 6, 2561, 64], min: -11.863246, max: 11.457247, mean: 0.10765
        attn = q @ k.transpose(-2, -1)
        # tensor [attn] size: [1, 6, 2561, 2561], min: 0.342996, max: 53.578362, mean: 20.544596
        attn = attn.softmax(dim=-1)

        # tensor [attn@v] size: [1, 6, 2561, 64], min: -1.033432, max: 1.115259, mean: -0.009298
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x # [1, 2561, 384] ???

class LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # dim = 384
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        # in_features = 384
        # hidden_features = 1536
        # out_features = 384
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class NestedTensorBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        # dim = 384
        # num_heads = 6

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.ls1 = LayerScale(dim)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=dim * 4,
            out_features=dim
        )
        self.ls2 = LayerScale(dim)


    def forward(self, x):
        def attn_residual_func(x):
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x):
            return self.ls2(self.mlp(self.norm2(x)))

        x = x + attn_residual_func(x)
        x = x + ffn_residual_func(x)
        return x

class DinoVisionTransformer(nn.Module):
    '''Small dino v2'''
    def __init__(self,
        img_size=518,
        patch_size=14,
        in_chans=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
    ):
        super().__init__()
        self.num_tokens = 1
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches # 1369

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        # self.pos_embed.size() -- [1, 1370, 384]

        blocks_list = [ NestedTensorBlock(dim=embed_dim, num_heads=num_heads) for i in range(depth) ]
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim)) # !!! useless !!!

    def interpolate_pos_encoding(self, x, H, W):
        # x.size() -- [1, 2561, 384], H -- 560, W -- 896
        B, NP, D = x.size()
        assert D == self.embed_dim

        N = self.pos_embed.shape[1] - 1
        if N == NP - 1 and W == H:
            return self.pos_embed

        pos_embed = self.pos_embed.float() # [1, 1370, 384]
        class_pos_embed = pos_embed[:, 0:1]
        patch_pos_embed = pos_embed[:, 1:]

        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        NH = (H + self.patch_size - 1) // self.patch_size
        NW = (W + self.patch_size - 1) // self.patch_size

        # tensor [patch_pos_embed] size: [1, 1369, 384], min: -0.1611, max: 0.126807, mean: 8.3e-05
        patch_pos_embed = patch_pos_embed.reshape(1, M, M, D)  # (1, M, M, D) -- (1, 37, 37, 384)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2) # [1. 37, 37, 384] --> [1, 384, 37, 37]

        patch_pos_embed = F.interpolate(patch_pos_embed, size=(NH, NW), mode="bicubic", antialias=False)
        # tensor [patch_pos_embed] size: [1, 384, 40, 64], min: -0.162149, max: 0.127178, mean: 8.2e-05
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, D)
        # tensor [patch_pos_embed] size: [1, 2560, 384], min: -0.162149, max: 0.127178, mean: 8.2e-05

        # class_pos_embed.size() -- [1, 384]
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1).to(x.dtype)

    def forward(self, x):
        # tensor [x] size: [1, 3, 560, 896], min: -0.495599, max: 0.496816, mean: -0.109927
        B, C, H, W = x.shape

        x = self.patch_embed(x)
        # tensor [x] size: [1, 2560, 384], min: -0.818547, max: 0.587891, mean: -0.002679

        x = torch.cat((self.cls_token.expand(B, 1, -1), x), dim=1)
        # tensor [x] size: [1, 2561, 384], min: -0.818547, max: 0.587891, mean: -0.002678

        x = x + self.interpolate_pos_encoding(x, H, W)

        NH = (H + self.patch_size - 1) // self.patch_size
        NW = (W + self.patch_size - 1) // self.patch_size

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in [8, 9, 10, 11]:
                out = self.norm(x)  # [1, 2561, 384]
                out = out[:, 1:, :] # [1, 2560, 384]
                # w // self.patch_size, h // self.patch_size === 40, 64
                out = out.reshape(B, NH, NW, -1).permute(0, 3, 1, 2).contiguous()
                # [1, 40, 60, 384] --> [1, 384, 40, 64]
                outputs.append(out)

        # outputs is list: len = 4
        #     tensor [item] size: [1, 384, 40, 64], min: -64.29377, max: 62.932507, mean: 0.046734
        #     tensor [item] size: [1, 384, 40, 64], min: -58.107525, max: 53.356197, mean: 0.016807
        #     tensor [item] size: [1, 384, 40, 64], min: -48.493, max: 43.823879, mean: 0.01582
        #     tensor [item] size: [1, 384, 40, 64], min: -22.330799, max: 15.610704, mean: 0.011709
        return outputs
