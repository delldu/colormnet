import torch
import warnings

from inference.kv_memory_store import KeyValueMemoryStore
from model.memory_util import *

import todos
import pdb

class MemoryManager:
    """
    Manages all three memory stores and the transition between working/long-term memory
    """
    def __init__(self, config):
        self.hidden_dim = config['hidden_dim'] # 64
        self.top_k = config['top_k'] # 30
        self.max_mt_frames = config['max_mid_term_frames'] # 10
        self.min_mt_frames = config['min_mid_term_frames'] # 5
        self.num_prototypes = config['num_prototypes'] # 128
        self.max_long_elements = config['max_long_term_elements'] # 10000

        # dimensions will be inferred from input later
        self.CK = self.CV = None
        self.H = self.W = None

        # The hidden state will be stored in a single tensor for all objects
        # B x num_objects x CH x H x W
        self.hidden = None

        self.work_mem = KeyValueMemoryStore()
        self.reset_config = True

        # pdb.set_trace()

    def _readout(self, affinity, v):
        # this function is for a single object group
        return v @ affinity

    def match_memory(self, query_key, selection):
        # query_key: B x C^k x H x W
        # selection:  B x C^k x H x W
        num_groups = self.work_mem.num_groups # === 0
        h, w = query_key.shape[-2:]

        query_key = query_key.flatten(start_dim=2)
        selection = selection.flatten(start_dim=2)

        """
        Memory readout using keys
        """
        # tensor [self.work_mem.key] size: [1, 64, 1960], min: -2.755859, max: 3.140625, mean: -0.143588
        # tensor [self.work_mem.shrinkage] size: [1, 1, 1960], min: 15.327408, max: 42.592422, mean: 31.011292
        # tensor [query_key] size: [1, 64, 1960], min: -2.75, max: 3.162109, mean: -0.143807
        # tensor [selection] size: [1, 64, 1960], min: 0.0, max: 0.9375, mean: 0.4714

        # No long-term memory
        similarity = get_similarity(self.work_mem.key, self.work_mem.shrinkage, query_key, selection)
        # tensor [similarity] size: [1, 1960, 1960], min: -87.945297, max: 0.012002, mean: -21.991936

        affinity, usage = do_softmax(similarity, top_k=self.top_k, return_usage=True)
        # tensor [affinity] size: [1, 1960, 1960], min: 0.0, max: 0.876976, mean: 0.00051
        # usage.size() -- [1, 1960]

        # Record memory usage for working memory
        self.work_mem.update_usage(usage.flatten())

        affinity = [affinity]
        # compute affinity group by group as later groups only have a subset of keys
        for gi in range(1, num_groups): # num_groups == 1
            pdb.set_trace()
            affinity_one_group = do_softmax(similarity[:, -self.work_mem.get_v_size(gi):], top_k=self.top_k)
            affinity.append(affinity_one_group)
            
        all_memory_value = self.work_mem.value
        # len(self.work_mem.value) -- 1, self.work_mem.value[0].size() -- [2, 512, 1960]

        # Shared affinity within each group
        # affinity[0].size() -- [1, 1960, 1960]
        all_readout_mem = torch.cat([self._readout(affinity[gi], gv) for gi, gv in enumerate(all_memory_value)], 0)
        # tensor [all_readout_mem] size: [2, 512, 1960], min: -9.492188, max: 4.671875, mean: -0.013572
        return all_readout_mem.view(all_readout_mem.shape[0], self.CV, h, w) # self.CV --- 512

    def add_memory(self, key, shrinkage, value, objects, selection=None):
        # key: 1*C*H*W
        # value: 1*num_objects*C*H*W
        # objects contain a list of object indices
        # assert self.reset_config == True
        if self.H is None or self.reset_config:
            self.reset_config = False
            self.H, self.W = key.shape[-2:]
            self.HW = self.H*self.W
            # convert from num. frames to num. nodes
            self.min_work_elements = self.min_mt_frames*self.HW
            self.max_work_elements = self.max_mt_frames*self.HW
        
        # key:   1*C*N
        # value: num_objects*C*N
        key = key.flatten(start_dim=2)
        shrinkage = shrinkage.flatten(start_dim=2) 
        value = value[0].flatten(start_dim=2)

        self.CK = key.shape[1]
        self.CV = value.shape[1]

        assert selection is not None
        selection = selection.flatten(start_dim=2)

        self.work_mem.add(key, value, shrinkage, selection, objects)


    def create_hidden_state(self, n, sample_key):
        h, w = sample_key.shape[-2:]
        assert self.hidden is None
        self.hidden = torch.zeros((1, n, self.hidden_dim, h, w), device=sample_key.device)

    def set_hidden(self, hidden):
        self.hidden = hidden

    def get_hidden(self):
        return self.hidden

