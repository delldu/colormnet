from inference.memory_manager import MemoryManager
from model.network import ColorMNet

from util.tensor_util import pad_divide_by, unpad
import torch

import todos
import pdb

class InferenceCore:
    def __init__(self, network:ColorMNet, config):
        self.config = config
        self.network = network
        self.mem_every = config['mem_every'] # 5
        self.clear_memory()
        self.all_labels = None

        self.last_ti_key = None
        self.last_ti_value = None

    def clear_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        self.memory = MemoryManager(config=self.config)

    def set_all_labels(self, all_labels):
        self.all_labels = all_labels

    def step(self, image, mask=None):
        # image: 3*H*W
        # mask: num_objects*H*W or None
        self.curr_ti += 1
        divide_by = 112 # 16
        image, self.pad = pad_divide_by(image, divide_by)
        image = image.unsqueeze(0) # add the batch dimension
        # tensor [image] size: [3, 480, 832], min: -1.0, max: 1.0, mean: -0.183782
        # [mask] type: <class 'NoneType'>
        # tensor [image] size: [3, 560, 896], min: -1.0, max: 1.0, mean: -0.146276
        # tensor [image] size: [1, 3, 560, 896], min: -1.0, max: 1.0, mean: -0.146276

        is_mem_frame = ((self.curr_ti-self.last_mem_ti >= self.mem_every) or (mask is not None))
        # need_segment = (self.curr_ti > 0) and ((valid_labels is None) or (len(self.all_labels) != len(valid_labels)))
        # need_segment = (self.curr_ti > 0) # and (mask is None)
        need_segment = (mask is None)

        # xxxx_debug
        key, shrinkage, selection, f16, f8, f4 = self.network.encode_key(image, need_sk=is_mem_frame)
        multi_scale_features = (f16, f8, f4)

        # segment the current frame is needed
        if need_segment: # True || False
            memory_readout = self.memory.match_memory(key, selection).unsqueeze(0)

            # short term memory 
            batch, num_objects, value_dim, h, w = self.last_ti_value.shape
            last_ti_value = self.last_ti_value.flatten(start_dim=1, end_dim=2)
            memory_value_short = self.network.short_term_attn(key, self.last_ti_key, last_ti_value, key.shape[-2:])
            memory_value_short = memory_value_short.permute(1, 2, 0).view(batch, num_objects, value_dim, h, w)
            memory_readout += memory_value_short

            hidden, pred_prob_with_bg = self.network.segment(multi_scale_features, memory_readout, 
                                    self.memory.get_hidden(), h_out=not is_mem_frame)
            # remove batch dim
            pred_prob_with_bg = pred_prob_with_bg[0]

            if not is_mem_frame:
                self.memory.set_hidden(hidden)
        else:
            mask, _ = pad_divide_by(mask, divide_by)
            pred_prob_with_bg = mask

            self.memory.create_hidden_state(2, key) # xxxx_debug


        # save as memory if needed
        if is_mem_frame: # True
            # self.memory.get_hidden().size() -- [1, 2, 64, 35, 56]
            # self.network -- ColorMNet
            value, hidden = self.network.encode_value(image, f16, self.memory.get_hidden(), pred_prob_with_bg.unsqueeze(0))
            self.memory.add_memory(key, shrinkage, value, self.all_labels, selection=selection)
            self.last_mem_ti = self.curr_ti

            self.last_ti_key = key
            self.last_ti_value = value

            self.memory.set_hidden(hidden)
                
        return unpad(pred_prob_with_bg, self.pad)

