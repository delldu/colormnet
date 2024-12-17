# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)

        return x


class Attention(nn.Module):
    def __init__(self,
        dim: int,
        num_heads: int = 8,
    ):
        super().__init__()
        # dim = 384
        # num_heads = 6

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        # pdb.set_trace()
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

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
        # ==> pdb.set_trace()
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class NestedTensorBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
    ):
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
    def __init__(
        self,
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

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches # 1369

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        # self.pos_embed.size() -- [1, 1370, 384]

        blocks_list = [ NestedTensorBlock(dim=embed_dim, num_heads=num_heads) for i in range(depth) ]
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))


    def interpolate_pos_encoding(self, x, w, h):
        # x.size() -- [1, 2561, 384]
        # w = 560
        # h = 896
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if npatch == N and w == h: # False
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
        # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
        sx = float(w0 + 0.1) / M
        sy = float(h0 + 0.1) / M

        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=False,
            scale_factor=(sx, sy)
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        return x

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output


    def get_intermediate_layers(self, x, n = 1):
        # n = [8, 9, 10, 11]
        # reshape = True
        # norm = True
        # assert self.chunked_blocks == False
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1:] for out in outputs]

        B, _, w, h = x.shape
        outputs = [
            out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
            for out in outputs
        ]
        return tuple(outputs)


    def forward(self, x):
        return self.get_intermediate_layers(x, n=[8, 9, 10, 11])
