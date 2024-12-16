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

        self.work_mem = KeyValueMemoryStore(count_usage=True)
        self.long_mem = KeyValueMemoryStore(count_usage=False)
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
        selection = selection.flatten(start_dim=2) if selection is not None else None

        """
        Memory readout using keys
        """
        if self.long_mem.engaged():
            pdb.set_trace()
            
            # Use long-term memory
            long_mem_size = self.long_mem.size
            memory_key = torch.cat([self.long_mem.key, self.work_mem.key], -1)
            shrinkage = torch.cat([self.long_mem.shrinkage, self.work_mem.shrinkage], -1) 

            todos.debug.output_var("memory_key", memory_key)
            todos.debug.output_var("shrinkage", shrinkage)
            todos.debug.output_var("query_key", query_key)
            todos.debug.output_var("selection", selection)

            similarity = get_similarity(memory_key, shrinkage, query_key, selection)
            todos.debug.output_var("similarity", similarity)

            work_mem_similarity = similarity[:, long_mem_size:]
            long_mem_similarity = similarity[:, :long_mem_size]

            # get the usage with the first group
            # the first group always have all the keys valid
            affinity, usage = do_softmax(
                    torch.cat([long_mem_similarity[:, -self.long_mem.get_v_size(0):], work_mem_similarity], 1), 
                    top_k=self.top_k, return_usage=True)
            todos.debug.output_var("affinity", affinity)


            affinity = [affinity]

            # compute affinity group by group as later groups only have a subset of keys
            pdb.set_trace()
            for gi in range(1, num_groups):
                if gi < self.long_mem.num_groups:
                    # merge working and lt similarities before softmax
                    affinity_one_group = do_softmax(
                        torch.cat([long_mem_similarity[:, -self.long_mem.get_v_size(gi):], 
                                    work_mem_similarity[:, -self.work_mem.get_v_size(gi):]], 1), top_k=self.top_k)
                else:
                    # no long-term memory for this group
                    affinity_one_group = do_softmax(work_mem_similarity[:, -self.work_mem.get_v_size(gi):], top_k=self.top_k)
                affinity.append(affinity_one_group)

            all_memory_value = []
            for gi, gv in enumerate(self.work_mem.value):
                # merge the working and lt values before readout
                if gi < self.long_mem.num_groups:
                    all_memory_value.append(torch.cat([self.long_mem.value[gi], self.work_mem.value[gi]], -1))
                else:
                    all_memory_value.append(gv)

            """
            Record memory usage for working and long-term memory
            """
            # ignore the index return for long-term memory
            work_usage = usage[:, long_mem_size:]
            self.work_mem.update_usage(work_usage.flatten())

            # ignore the index return for working memory
            long_usage = usage[:, :long_mem_size]
            self.long_mem.update_usage(long_usage.flatten())
        else:
            # ==> pdb.set_trace()
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
        if selection is not None:
            selection = selection.flatten(start_dim=2)

        self.work_mem.add(key, value, shrinkage, selection, objects)

        # long-term memory cleanup
        # Do memory compressed if needed
        if self.work_mem.size >= self.max_work_elements:
            pdb.set_trace()
            # Remove obsolete features if needed
            if self.long_mem.size >= (self.max_long_elements-self.num_prototypes):
                self.long_mem.remove_obsolete_features(self.max_long_elements-self.num_prototypes)
                
            self.compress_features()


    def create_hidden_state(self, n, sample_key):
        # n is the TOTAL number of objects
        h, w = sample_key.shape[-2:]
        assert self.hidden is None

        if self.hidden is None:
            self.hidden = torch.zeros((1, n, self.hidden_dim, h, w), device=sample_key.device)
        elif self.hidden.shape[1] != n:
            pdb.set_trace()
            self.hidden = torch.cat([
                self.hidden, 
                torch.zeros((1, n-self.hidden.shape[1], self.hidden_dim, h, w), device=sample_key.device)
            ], 1)

        assert(self.hidden.shape[1] == n)

    def set_hidden(self, hidden):
        self.hidden = hidden

    def get_hidden(self):
        return self.hidden

    def compress_features(self):
        pdb.set_trace()
        HW = self.HW
        candidate_value = []
        total_work_mem_size = self.work_mem.size
        for gv in self.work_mem.value:
            # Some object groups might be added later in the video
            # So not all keys have values associated with all objects
            # We need to keep track of the key->value validity
            mem_size_in_this_group = gv.shape[-1]
            if mem_size_in_this_group == total_work_mem_size:
                # full LT
                candidate_value.append(gv[:,:,HW:-self.min_work_elements+HW])
            else:
                # mem_size is smaller than total_work_mem_size, but at least HW
                assert HW <= mem_size_in_this_group < total_work_mem_size
                if mem_size_in_this_group > self.min_work_elements+HW:
                    # part of this object group still goes into LT
                    candidate_value.append(gv[:,:,HW:-self.min_work_elements+HW])
                else:
                    # this object group cannot go to the LT at all
                    candidate_value.append(None)

        # perform memory consolidation
        prototype_key, prototype_value, prototype_shrinkage = self.consolidation(
            *self.work_mem.get_all_sliced(HW, -self.min_work_elements+HW), candidate_value)

        # remove consolidated working memory
        self.work_mem.sieve_by_range(HW, -self.min_work_elements+HW, min_size=self.min_work_elements+HW)

        # add to long-term memory
        self.long_mem.add(prototype_key, prototype_value, prototype_shrinkage, selection=None, objects=None)

    def consolidation(self, candidate_key, candidate_shrinkage, candidate_selection, usage, candidate_value):
        pdb.set_trace()
        # keys: 1*C*N
        # values: num_objects*C*N
        N = candidate_key.shape[-1]

        # find the indices with max usage
        _, max_usage_indices = torch.topk(usage, k=self.num_prototypes, dim=-1, sorted=True)
        prototype_indices = max_usage_indices.flatten()

        # Prototypes are invalid for out-of-bound groups
        validity = [prototype_indices >= (N-gv.shape[2]) if gv is not None else None for gv in candidate_value]

        prototype_key = candidate_key[:, :, prototype_indices]
        prototype_selection = candidate_selection[:, :, prototype_indices] if candidate_selection is not None else None

        """
        Potentiation step
        """
        similarity = get_similarity(candidate_key, candidate_shrinkage, prototype_key, prototype_selection)

        # convert similarity to affinity
        # need to do it group by group since the softmax normalization would be different
        affinity = [
            do_softmax(similarity[:, -gv.shape[2]:, validity[gi]]) if gv is not None else None
            for gi, gv in enumerate(candidate_value)
        ]

        # some values can be have all False validity. Weed them out.
        affinity = [
            aff if aff is None or aff.shape[-1] > 0 else None for aff in affinity
        ]

        # readout the values
        prototype_value = [
            self._readout(affinity[gi], gv) if affinity[gi] is not None else None
            for gi, gv in enumerate(candidate_value)
        ]

        # readout the shrinkage term
        prototype_shrinkage = self._readout(affinity[0], candidate_shrinkage) if candidate_shrinkage is not None else None

        return prototype_key, prototype_value, prototype_shrinkage
