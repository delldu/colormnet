"""
resnet.py - A modified ResNet structure
We append extra channels to the first conv by some network surgery
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from einops import rearrange
import todos
import pdb

# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
# }

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, dilation=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    #block(self.inplanes, planes, stride, downsample)
    def __init__(self, inplanes, planes, stride=1, downsample=False):
        super().__init__()
        # inplanes -- 64, 128, 256, 512 ?
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        # if stride != 1 or self.inplanes != planes * block.expansion:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(planes * block.expansion),
        #     )
        self.downsample = None
        if downsample and (stride != 1 or inplanes != planes * self.expansion):
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def make_basicblock_layer(inplanes, planes, blocks, stride=1):
    layers = [BasicBlock(inplanes, planes, stride, downsample=True)]
    for i in range(1, blocks):
        layers.append(BasicBlock(planes, planes, stride=1, downsample=False))

    return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    expansion = 4
    # block(self.inplanes, planes, stride, downsample)
    def __init__(self, inplanes, planes, stride=1, downsample=False):
        super().__init__()
        # inplanes -- 64, 128, 256, 512 ?
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        # if stride != 1 or self.inplanes != planes * block.expansion:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(planes * block.expansion),
        #     )
        self.downsample = None
        if downsample and (stride != 1 or inplanes != planes * self.expansion):
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def make_bottleneck_layer(inplanes, planes, blocks, stride=1):
    # planess -- [64, 128, 256, 512]
    # blocks -- [3, 4, 6, 3]
    layers = [Bottleneck(inplanes, planes, stride, downsample=True)]
    inplanes = planes * 4
    for i in range(1, blocks):
        layers.append(Bottleneck(inplanes, planes, stride=1, downsample=False))

    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3), extra_dim=0):
        self.inplanes = 64
        super().__init__()
        assert self.inplanes == 64

        self.conv1 = nn.Conv2d(3+extra_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # for resnet18: layers == [2, 2, 2, 2]
        # for resnet50: layers == [3, 4, 6, 3]
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        # assert block.expansion == 1
        # assert block == BasicBlock

        # downsample = None

        # if stride != 1 or self.inplanes != planes * block.expansion:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(planes * block.expansion),
        #     )
        layers = [block(self.inplanes, planes, stride=stride, downsample=True)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, downsample=False))

        return nn.Sequential(*layers)

def resnet18():
    model = ResNet(BasicBlock, [2, 2, 2, 2], extra_dim=2)
    return model

def resnet50():
    model = ResNet(Bottleneck, [3, 4, 6, 3], extra_dim=0)
    return model


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
    
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    
class CrossChannelAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True)
        self.to_q_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=True)

        self.to_k = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True)
        self.to_k_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=True)

        self.to_v = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True)
        self.to_v_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=True)

        self.to_out = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1, 1, 0),
        )

    def forward(self, encoder, decoder):
        b, c, h, w = encoder.shape

        q = self.to_q_dw(self.to_q(encoder))

        k = self.to_k_dw(self.to_k(decoder))
        v = self.to_v_dw(self.to_v(decoder))
        # xxxx_debug
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.heads, h=h, w=w)

        return self.to_out(out)

class Fuse(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()

        self.encode_enc = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1)
        self.norm1 = LayerNorm2d(out_feat)
        self.norm2 = LayerNorm2d(out_feat)
        self.crossattn = CrossChannelAttention(dim=out_feat)
        self.norm3 = LayerNorm2d(out_feat)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, enc, dec):
        enc = self.encode_enc(enc)
        res = enc
        enc = self.norm1(enc)
        dec = self.norm2(dec)
        output = self.crossattn(enc, dec) + res

        output = self.norm3(output)
        output = self.relu3(output)

        return output
