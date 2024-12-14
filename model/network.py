"""
This file defines XMem, the highest level nn.Module interface
During training, it is used by trainer.py
During evaluation, it is used by inference_core.py

It further depends on modules.py which gives more detailed implementations of sub-modules
"""

import torch
import torch.nn as nn

# from model.aggregate import aggregate
from model.modules import *
from model.memory_util import *

from model.attention import LocalGatedPropagation
import todos
import pdb

class ColorMNet(nn.Module):
    def __init__(self, config, model_path=None, map_location=None):
        super().__init__()
        self.key_dim = 64
        self.value_dim = 512
        self.hidden_dim = 64
        config['key_dim'] = self.key_dim
        config['value_dim'] = self.value_dim
        config['hidden_dim'] = self.hidden_dim

        self.key_encoder = KeyEncoder_DINOv2_v6()
        # Projection from f16 feature space to key/value space
        self.key_proj = KeyProjection(1024, self.key_dim) # 1024 -> 384 -> 3072

        self.value_encoder = ValueEncoder(self.value_dim, self.hidden_dim)
        self.short_term_attn = LocalGatedPropagation(d_qk=64, d_vu=512 * 2, num_head=1, dropout=0, d_att=64, max_dis=7)
        self.decoder = Decoder(self.value_dim, self.hidden_dim)

    def encode_key(self, frame, need_sk=True): 
        # Determine input shape
        # (Pdb) frame.shape -- [1, 3, 560, 896]
        assert len(frame.shape) == 4
        f16, f8, f4 = self.key_encoder(frame)
        key, shrinkage, selection = self.key_proj(f16, need_sk)
        return key, shrinkage, selection, f16, f8, f4

    def encode_value(self, frame, image_feat_f16, h16, masks): 
        # tensor [masks] size: [1, 2, 560, 896], min: -0.476372, max: 0.657849, mean: 0.02309
        num_objects = masks.shape[1]
        assert num_objects == 2

        others = torch.cat([torch.sum(masks[:, [j for j in range(num_objects) if i!=j]] , dim=1, keepdim=True)
        for i in range(num_objects)], 1)
        # tensor [others] size: [1, 2, 560, 896], min: -0.476372, max: 0.657849, mean: 0.02309

        # s1 = torch.sum(masks[:, [1]], dim=1, keepdim=True)
        # s0 = torch.sum(masks[:, [0]], dim=1, keepdim=True)
        # others2 = torch.cat([s1, s0], 1)
        # todos.debug.output_var("|others - others2|", (others - others2).abs())
        g16, h16 = self.value_encoder(frame, image_feat_f16, h16, masks, others)

        return g16, h16

    # # Used in training only. 
    # # This step is replaced by MemoryManager in test time
    # def read_memory(self, query_key, query_selection, memory_key, 
    #                 memory_shrinkage, memory_value):
    #     """
    #     query_key       : B * CK * H * W
    #     query_selection : B * CK * H * W
    #     memory_key      : B * CK * T * H * W
    #     memory_shrinkage: B * 1  * T * H * W
    #     memory_value    : B * num_objects * CV * T * H * W
    #     """
    #     pdb.set_trace()
    #     batch_size, num_objects = memory_value.shape[:2]
    #     memory_value = memory_value.flatten(start_dim=1, end_dim=2)

    #     affinity = get_affinity(memory_key, memory_shrinkage, query_key, query_selection)
    #     memory = readout(affinity, memory_value)
    #     memory = memory.view(batch_size, num_objects, self.value_dim, *memory.shape[-2:])

    #     return memory

    # def read_memory_short(self, query_key, memory_key, memory_value):
    #     """
    #     query_key       : B * CK * H * W
    #     query_selection : B * CK * H * W
    #     memory_key      : B * CK * T * H * W
    #     memory_shrinkage: B * 1  * T * H * W
    #     memory_value    : B * num_objects * CV * T * H * W
    #     """
    #     pdb.set_trace()
    #     batch_size, num_objects = memory_value.shape[:2]
    #     memory_value = memory_value.flatten(start_dim=1, end_dim=2)

    #     size_2d = query_key.shape[-2:]
    #     memory_value_short = self.short_term_attn(query_key, memory_key, memory_value, size_2d)

    #     memory_value_short = memory_value_short.permute(1, 2, 0).view(batch_size, num_objects, self.value_dim, *memory_value.shape[-2:])

    #     return memory_value_short
    
    def segment(self, multi_scale_features, memory_readout, hidden_state, h_out=True):
        # multi_scale_features is tuple: len = 3
        #     tensor [item] size: [1, 1024, 35, 56], min: 0.0, max: 2.601784, mean: 0.063031
        #     tensor [item] size: [1, 512, 70, 112], min: 0.0, max: 1.79675, mean: 0.090695
        #     tensor [item] size: [1, 256, 140, 224], min: 0.0, max: 6.709424, mean: 0.200673
        # tensor [memory_readout] size: [1, 2, 512, 35, 56], min: -9.328125, max: 4.738281, mean: -0.007783
        # tensor [hidden_state] size: [1, 2, 64, 35, 56], min: -1.0, max: 0.999023, mean: -0.009137
        # assert h_out == True
        hidden_state, logits = self.decoder(*multi_scale_features, hidden_state, memory_readout, h_out=h_out)
        logits = torch.tanh(logits)

        return hidden_state, logits

    def forward(self, mode, *args, **kwargs):
        pdb.set_trace()
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'read_memory':
            return self.read_memory(*args, **kwargs)
        elif mode == 'read_memory_short':
            return self.read_memory_short(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError


    def load_weights(self, src_dict, init_as_zero_if_needed=False):
        # Maps SO weight (without other_mask) to MO weight (with other_mask)
        for k in list(src_dict.keys()):
            if k == 'value_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    print('Converting weights from single object to multiple objects.')
                    pads = torch.zeros((64,1,7,7), device=src_dict[k].device)
                    if not init_as_zero_if_needed:
                        print('Randomly initialized padding.')
                        nn.init.orthogonal_(pads)
                    else:
                        print('Zero-initialized padding.')
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)

        self.load_state_dict(src_dict)
