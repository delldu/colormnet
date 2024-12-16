# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py
import os
import warnings

from functools import partial
import math
from typing import Sequence, Tuple, Union, Callable, Optional, List

import torch
import torch.nn as nn
import pdb


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ):
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    # def flops(self) -> float:
    #     Ho, Wo = self.patches_resolution
    #     flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
    #     if self.norm is not None:
    #         flops += Ho * Wo * self.embed_dim
    #     return flops

# class SwiGLUFFN(nn.Module):
#     def __init__(
#         self,
#         in_features: int,
#         hidden_features: Optional[int] = None,
#         out_features: Optional[int] = None,
#         act_layer: Callable[..., nn.Module] = None,
#         bias: bool = True,
#     ) -> None:
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
#         self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x12 = self.w12(x)
#         x1, x2 = x12.chunk(2, dim=-1)
#         hidden = F.silu(x1) * x2
#         return self.w3(hidden)
        

# class SwiGLUFFNFused(SwiGLUFFN):
#     def __init__(
#         self,
#         in_features: int,
#         hidden_features: Optional[int] = None,
#         out_features: Optional[int] = None,
#         act_layer: Callable[..., nn.Module] = None,
#         bias: bool = True,
#     ) -> None:
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
#         super().__init__(
#             in_features=in_features,
#             hidden_features=hidden_features,
#             out_features=out_features,
#             bias=bias,
#         )


# def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
#     if not depth_first and include_root:
#         fn(module=module, name=name)
#     for child_name, child_module in module.named_children():
#         child_name = ".".join((name, child_name)) if name else child_name
#         named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
#     if depth_first and include_root:
#         fn(module=module, name=name)
#     return module


# class BlockChunk(nn.ModuleList):
#     def forward(self, x):
#         for b in self:
#             x = b(x)
#         return x

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
        pdb.set_trace()

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x, attn_bias=None):
        assert attn_bias == None

        if not XFORMERS_AVAILABLE: # False
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        return x


class LayerScale(nn.Module):
    def __init__(self,
        dim: int,
        init_values = 1.0,
        inplace: bool = False,
    ):
        super().__init__()
        # dim = 384
        # init_values = 1.0
        # inplace = False        
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        # ==> pdb.set_trace()
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Mlp(nn.Module):
    def __init__(self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        bias: bool = True,
    ):
        super().__init__()
        # in_features = 384
        # hidden_features = 1536
        # out_features = 384
        # act_layer = <class 'torch.nn.modules.activation.GELU'>
        # bias = True
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        # ==> pdb.set_trace()
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = MemEffAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        # dim = 384
        # num_heads = 6
        # mlp_ratio = 4
        # drop_path = 0.0
        # act_layer = <class 'torch.nn.modules.activation.GELU'>
        # norm_layer = functools.partial(<class 'torch.nn.modules.normalization.LayerNorm'>, eps=1e-06)
        # attn_class = <class 'model.dinov2.MemEffAttention'>
        # ffn_layer = <class 'model.dinov2.Mlp'>


        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
        )
        self.ls1 = LayerScale(dim, init_values=1.0)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            bias=True,
        )
        self.ls2 = LayerScale(dim, init_values=1.0)

        self.sample_drop_ratio = drop_path

    def forward(self, x):
        def attn_residual_func(x):
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x):
            return self.ls2(self.mlp(self.norm2(x)))

        x = x + attn_residual_func(x)
        x = x + ffn_residual_func(x)
        return x


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)
        pdb.set_trace()

        # if self.training and self.sample_drop_ratio > 0.0:
        #     pdb.set_trace()

        #     def attn_residual_func(x: torch.Tensor, attn_bias=None) -> torch.Tensor:
        #         return self.attn(self.norm1(x), attn_bias=attn_bias)

        #     def ffn_residual_func(x: torch.Tensor, attn_bias=None) -> torch.Tensor:
        #         return self.mlp(self.norm2(x))

        #     x_list = drop_add_residual_stochastic_depth_list(
        #         x_list,
        #         residual_func=attn_residual_func,
        #         sample_drop_ratio=self.sample_drop_ratio,
        #         scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
        #     )
        #     x_list = drop_add_residual_stochastic_depth_list(
        #         x_list,
        #         residual_func=ffn_residual_func,
        #         sample_drop_ratio=self.sample_drop_ratio,
        #         scaling_vector=self.ls2.gamma if isinstance(self.ls1, LayerScale) else None,
        #     )
        #     return x_list
        # else:

        #     def attn_residual_func(x: torch.Tensor, attn_bias=None) -> torch.Tensor:
        #         return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

        #     def ffn_residual_func(x: torch.Tensor, attn_bias=None) -> torch.Tensor:
        #         return self.ls2(self.mlp(self.norm2(x)))

        #     attn_bias, x = get_attn_bias_and_cat(x_list)
        #     x = x + attn_residual_func(x, attn_bias=attn_bias)
        #     x = x + ffn_residual_func(x)
        #     return attn_bias.split(x)

    def forward(self, x_or_x_list):
        pdb.set_trace()

        # if isinstance(x_or_x_list, torch.Tensor):
        #     return super().forward(x_or_x_list)
        # elif isinstance(x_or_x_list, list):
        #     if not XFORMERS_AVAILABLE:
        #         raise AssertionError("xFormers is required for using nested tensors")
        #     return self.forward_nested(x_or_x_list)
        # else:
        #     raise AssertionError


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        in_chans=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        init_values=1.0,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=NestedTensorBlock, # Block,
        ffn_layer="mlp",
        interpolate_offset=0.1,
    ):
        super().__init__()
        # self = DinoVisionTransformer(
        #   (patch_embed): PatchEmbed(
        #     (proj): Conv2d(3, 384, kernel_size=(14, 14), stride=(14, 14))
        #     (norm): Identity()
        #   )
        #   (blocks): ModuleList(
        #     (0-11): 12 x Block(
        #       (norm1): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        #       (attn): MemEffAttention(
        #         (qkv): Linear(in_features=384, out_features=1152, bias=True)
        #         (proj): Linear(in_features=384, out_features=384, bias=True)
        #       )
        #       (ls1): LayerScale()
        #       (drop_path1): Identity()
        #       (norm2): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        #       (mlp): Mlp(
        #         (fc1): Linear(in_features=384, out_features=1536, bias=True)
        #         (act): GELU(approximate='none')
        #         (fc2): Linear(in_features=1536, out_features=384, bias=True)
        #       )
        #       (ls2): LayerScale()
        #     )
        #   )
        #   (norm): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        #   (head): Identity()
        # )


        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches # 1369

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        # self.pos_embed.size() -- [1, 1370, 384]

        assert ffn_layer == "mlp"
        ffn_layer = Mlp

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                # init_values=1.0,
            )
            for i in range(depth)
        ]
        # xxxx_1111
        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

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
        kwargs = {}
        if self.interpolate_offset: # self.interpolate_offset === 0.1
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            pdb.set_trace()
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=False,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        assert masks == None

        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        return x

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        # ==> ???
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

    def _get_intermediate_layers_chunked(self, x, n=1):
        pdb.set_trace()
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x,
        n = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        # n = [8, 9, 10, 11]
        # reshape = True
        # return_class_token = False
        # norm = True

        assert self.chunked_blocks == False
        if self.chunked_blocks: # False
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1:] for out in outputs]
        if reshape: # True, xxxx_1111
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    # def forward(self, *args, is_training=False, **kwargs):
    #     ret = self.forward_features(*args, **kwargs)
    #     if is_training:
    #         return ret
    #     else:
    #         return self.head(ret["x_norm_clstoken"])

    def forward(self, x):
        return x



def vit_small(patch_size=16, **kwargs):
    # patch_size = 14
    # kwargs = {'img_size': 518, interpolate_offset': 0.1}

    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        **kwargs,
    )
    return model
