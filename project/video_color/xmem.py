import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import todos
import pdb


def get_similarity(mem_key, mem_shrinkage, query_key, query_selection):
    # tensor [mem_key] size: [1, 64, 1960], min: -2.803887, max: 3.253764, mean: -0.159051
    # tensor [mem_shrinkage] size: [1, 1960, 1], min: 23.295803, max: 45.933445, mean: 32.660492
    # tensor [query_key] size: [1, 64, 1960], min: -2.80991, max: 3.298451, mean: -0.158671
    # tensor [query_selection] size: [1, 64, 1960], min: 0.0, max: 0.905321, mean: 0.5004

    CK = mem_key.shape[1] # 64

    mem_key = mem_key.transpose(1, 2)
    a_sq = (mem_key.pow(2) @ query_selection)
    # tensor [a_sq] size: [1, 1960, 1960], min: 5.804688, max: 25.171875, mean: 13.989838

    two_ab = 2 * (mem_key @ (query_key * query_selection))
    # tensor [two_ab] size: [1, 1960, 1960], min: 6.933594, max: 44.0, mean: 22.547285

    b_sq = (query_selection * query_key.pow(2)).sum(1, keepdim=True)
    # tensor [b_sq] size: [1, 1, 1960], min: 7.427909, max: 22.625141, mean: 14.256322

    similarity = (-a_sq + two_ab - b_sq)
    similarity = similarity * mem_shrinkage / math.sqrt(CK)   # B*N*HW
    # tensor [similarity] size: [1, 1960, 1960], min: -87.556335, max: -0.000922, mean: -19.917494

    return similarity

def do_softmax(similarity, top_k):
    # xxxx_debug

    # tensor [similarity] size: [1, 1960, 1960], min: -87.945297, max: 0.012002, mean: -21.991934
    values, indices = torch.topk(similarity, k=top_k, dim=1)
    # tensor [values] size: [1, 30, 1960], min: -9.706834, max: 0.012002, mean: -3.975312
    x_exp = values.exp_()
    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    #  x_exp.size() -- [1, 30, 1960]

    affinity = torch.zeros_like(similarity).scatter_(1, indices, x_exp) # B*N*HW
    # tensor [affinity] size: [1, 1960, 1960], min: 0.0, max: 0.442188, mean: 0.000255
    return affinity

class XMemCache():
    def __init__(self, device, H, W, S):
        super().__init__()
        self.S = S # size -- buffer size
        self.H = H
        self.W = W

        self.k = torch.zeros(1, 64, self.H*self.W, S).to(device)      # key -- hidden_dim == 64
        self.s = torch.zeros(1, self.H*self.W, 1, S).to(device)       # shrinkage
        self.v = torch.zeros(2, 512, self.H*self.W, S).to(device)     # value -- color_ab == 2, value_dim == 512
        self.index = 0
        self.count = 0

    @property
    def key(self):
        return self.k[:, :, :, 0:self.count]

    @property
    def shrinkage(self):
        return self.s[:, :, :, 0:self.count]

    @property
    def value(self):
        return self.v[:, :, :, 0:self.count]

    # def check(self, k, s, v):
    #     assert k.size() == (1, 64, self.H, self.W)
    #     assert s.size() == (1, 1, self.H, self.W)
    #     assert v.size() == (2, 512, self.H, self.W)
    #     return True

    def set(self, k, s, v):
        # self.check(k, s, v)
        i = self.index % self.S
        self.k[:, :, :, i:i+1] = k.view(1, 64, self.H*self.W, 1)
        self.s[:, :, :, i:i+1] = s.view(1, self.H*self.W, 1, 1)
        self.v[:, :, :, i:i+1] = v.view(2, 512, self.H*self.W, 1)
        self.index = self.index + 1
        if (self.count < self.S):
            self.count = self.count + 1

class XMem(nn.Module):
    def __init__(self, device, H, W, WORK_SIZE=8, LONG_SIZE=8, TOP_K=30):
        super().__init__()
        self.H = H
        self.W = W
        self.TOP_K = TOP_K
        self.sensory = torch.zeros(2, 64, H, W).to(device)   # Sensory Memory
        self.lastkey = torch.zeros(1, 64, H, W).to(device)   # Last Key
        self.lastval = torch.zeros(2, 512, H, W).to(device)  # Last Value
        self.workmem = XMemCache(device, H, W, WORK_SIZE)    # Short-Term, working Memory
        self.longmem = XMemCache(device, H, W, LONG_SIZE)    # Long-Term Memory

        # from ggml_engine import create_network
        # create_network(self)

    def set_hidden(self, h):
        assert h.size() == self.sensory.size()
        self.sensory = h

    def set_last_key(self, k):
        assert k.size() == self.lastkey.size()
        self.lastkey = k

    def set_last_value(self, v):
        assert v.size() == self.lastval.size()
        self.lastval = v

    def set_work_memory(self, k, s, v):
        self.workmem.set(k, s, v)

    def set_long_memory(self, k, s, v):
        self.longmem.set(k, s, v)

    def get_hidden(self):
        return self.sensory

    def get_last_key(self):
        return self.lastkey

    def get_last_value(self):
        return self.lastval

    def _get_key(self):
        c = self.longmem.count + self.workmem.count
        return torch.cat([self.longmem.key, self.workmem.key], dim=3).view(1, 64, c*self.H*self.W)

    def _get_shringage(self):
        c = self.longmem.count + self.workmem.count
        return torch.cat([self.longmem.shrinkage, self.workmem.shrinkage], dim=3).view(1, c * self.H*self.W, 1)

    def _get_value(self):
        c = self.longmem.count + self.workmem.count
        return torch.cat([self.longmem.value, self.workmem.value], dim=3).view(2, 512, c * self.H*self.W)

    def get_value(self, q_key, q_selection):
        # assert q_key.size() == (1, 64, self.H, self.W)
        # assert q_selection.size() == (1, 64, self.H, self.W)
        q_key = q_key.view(1, 64, self.H*self.W)
        q_selection = q_selection.view(1, 64, self.H*self.W)

        mem_key = self._get_key()
        mem_shrinkage = self._get_shringage()
        mem_value = self._get_value()

        # 9800 === 5*HW
        similarity = get_similarity(mem_key, mem_shrinkage, q_key, q_selection)

        affinity = do_softmax(similarity, self.TOP_K)

        final_value = mem_value @ affinity
        final_value = final_value.view(2, 512, self.H, self.W)

        return final_value

    def forward(self, q_key, q_selection):
        return self.get_value(q_key, q_selection)
