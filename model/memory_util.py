import math
import numpy as np
import torch
from typing import Optional
import todos
import pdb

# get_similarity(self.work_mem.key, self.work_mem.shrinkage, query_key, selection)
def get_similarity(mk, ms, qk, qe):
    # used for training/inference and memory reading/memory potentiation
    # mk: B x CK x [N]    - Memory keys
    # ms: B x  1 x [N]    - Memory shrinkage
    # qk: B x CK x [HW/P] - Query keys
    # qe: B x CK x [HW/P] - Query selection
    # Dimensions in [] are flattened
    CK = mk.shape[1]
    mk = mk.flatten(start_dim=2)
    ms = ms.flatten(start_dim=1).unsqueeze(2) if ms is not None else None
    qk = qk.flatten(start_dim=2)
    qe = qe.flatten(start_dim=2) if qe is not None else None

    # See appendix for derivation
    # or you can just trust me ヽ(ー_ー )ノ
    # tensor [mk] size: [1, 64, 3920], min: -2.755859, max: 3.140625, mean: -0.143588
    # tensor [qk] size: [1, 64, 1960], min: -2.75, max: 3.162109, mean: -0.143807
    # tensor [qe] size: [1, 64, 1960], min: 0.0, max: 0.9375, mean: 0.4714

    mk = mk.transpose(1, 2)
    # tensor [mk] size: [1, 3920, 64], min: -2.755859, max: 3.140625, mean: -0.143588

    a_sq = (mk.pow(2) @ qe)
    # tensor [a_sq] size: [1, 3920, 1960], min: 5.804688, max: 25.171875, mean: 13.989838

    two_ab = 2 * (mk @ (qk * qe))
    # tensor [two_ab] size: [1, 3920, 1960], min: 6.933594, max: 44.0, mean: 22.547285

    b_sq = (qe * qk.pow(2)).sum(1, keepdim=True)
    # tensor [b_sq] size: [1, 1, 1960], min: 7.427909, max: 22.625141, mean: 14.256322

    similarity = (-a_sq+two_ab-b_sq)
    # tensor [similarity] size: [1, 3920, 1960], min: -22.395203, max: 0.004377, mean: -5.698874

    # tensor [ms] size: [1, 3920, 1], min: 15.327408, max: 42.592422, mean: 31.011293
    similarity = similarity * ms / math.sqrt(CK)   # B*N*HW

    return similarity

def do_softmax(similarity, top_k: Optional[int]=None, return_usage=False):
    # normalize similarity with top-k softmax
    # similarity: B x N x [HW/P]
    # assert top_k is not None

    # tensor [similarity] size: [1, 1960, 1960], min: -87.945297, max: 0.012002, mean: -21.991936

    # s2 = similarity.clone()

    # tensor [similarity] size: [1, 3920, 1960], min: -87.945297, max: 0.012002, mean: -21.991934
    values, indices = torch.topk(similarity, k=top_k, dim=1)
    # values.size() -- [1, 30, 1960]
    # tensor [values] size: [1, 30, 1960], min: -9.706834, max: 0.012002, mean: -3.975312
    x_exp = values.exp_()
    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    #  x_exp.size() -- [1, 30, 1960]

    affinity = torch.zeros_like(similarity).scatter_(1, indices, x_exp) # B*N*HW
    # tensor [affinity] size: [1, 3920, 1960], min: 0.0, max: 0.442188, mean: 0.000255

    # todos.debug.output_var("do_softmax", (affinity - values.softmax(dim=1)).abs())
    # tensor [do_softmax] size: [1, 1960, 1960], min: 0.0, max: 0.112351, mean: 7.3e-05
    if return_usage:
        return affinity, affinity.sum(dim=2)

    return affinity

def get_affinity(mk, ms, qk, qe):
    # shorthand used in training with no top-k
    similarity = get_similarity(mk, ms, qk, qe)
    affinity = do_softmax(similarity)
    return affinity

def readout(affinity, mv):
    B, CV, T, H, W = mv.shape

    mo = mv.view(B, CV, T*H*W) 
    mem = torch.bmm(mo, affinity)
    mem = mem.view(B, CV, H, W)

    return mem
