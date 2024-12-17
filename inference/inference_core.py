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
        self.clear_memory()
        
        self.all_labels = None

        self.last_ti_key = None
        self.last_ti_value = None

    def clear_memory(self):
        self.memory = MemoryManager(config=self.config)

    def set_all_labels(self, all_labels):
        self.all_labels = all_labels

    def step(self, image, mask=None):
        # image: 3*H*W
        # mask: num_objects*H*W or None
        divide_by = 112 # 16
        image, self.pad = pad_divide_by(image, divide_by)
        image = image.unsqueeze(0) # add the batch dimension
        # tensor [image] size: [3, 480, 832], min: -1.0, max: 1.0, mean: -0.183782
        # [mask] type: <class 'NoneType'>
        # tensor [image] size: [3, 560, 896], min: -1.0, max: 1.0, mean: -0.146276
        # tensor [image] size: [1, 3, 560, 896], min: -1.0, max: 1.0, mean: -0.146276

        has_mask = (mask is not None)

        key, shrinkage, selection, f16, f8, f4 = self.network.encode_key(image)
        # tensor [key] size: [1, 64, 35, 56], min: -2.763672, max: 3.185547, mean: -0.142968
        # tensor [shrinkage] size: [1, 1, 35, 56], min: 15.297852, max: 43.097794, mean: 31.045326
        # tensor [selection] size: [1, 64, 35, 56], min: 0.0, max: 0.937012, mean: 0.470785

        # multi_scale_features ---
        #   tensor [f16] size: [1, 1024, 35, 56], min: 0.0, max: 2.594945, mean: 0.063114
        #   tensor [f8] size: [1, 512, 70, 112], min: 0.0, max: 1.842727, mean: 0.090533
        #   tensor [f4] size: [1, 256, 140, 224], min: 0.0, max: 6.625021, mean: 0.200046

        multi_scale_features = (f16, f8, f4)

        if has_mask:
            mask, _ = pad_divide_by(mask, divide_by)
            predict_ab = mask

            self.memory.create_hidden_state(2, key) # xxxx_debug
            # tensor [key] size: [1, 64, 35, 56], min: -2.763672, max: 3.185547, mean: -0.142968
            # tensor [self.memory.get_hidden()] size: [1, 2, 64, 35, 56], min: 0.0, max: 0.0, mean: 0.0

            # xxxx_gggg
            value, hidden = self.network.encode_value(image, f16, self.memory.get_hidden(), predict_ab.unsqueeze(0))
            # tensor [value] size: [1, 2, 512, 35, 56], min: -9.9375, max: 5.132812, mean: -0.01333
            # tensor [hidden] size: [1, 2, 64, 35, 56], min: -1.0, max: 0.999023, mean: -0.009137
            # --------------------------------------------------------------------------------

            # Save (key, shrinkage, selection), (value, hidden)

            self.memory.add_memory(key, shrinkage, value, self.all_labels, selection=selection)

            self.last_ti_key = key
            self.last_ti_value = value

            self.memory.set_hidden(hidden)

        else:
            memory_readout = self.memory.match_memory(key, selection).unsqueeze(0)
            # tensor [memory_readout] size: [1, 2, 512, 35, 56], min: -9.5, max: 4.726562, mean: -0.012821

            # short term memory 
            batch, num_objects, value_dim, h, w = self.last_ti_value.shape
            last_ti_value = self.last_ti_value.flatten(start_dim=1, end_dim=2)
            # tensor [key] size: [1, 64, 35, 56], min: -2.75, max: 3.166016, mean: -0.143513
            # tensor [self.last_ti_key] size: [1, 64, 35, 56], min: -2.753906, max: 3.142578, mean: -0.143365
            # tensor [last_ti_value] size: [1, 1024, 35, 56], min: -9.9375, max: 5.101562, mean: -0.014165
            memory_value_short = self.network.short_term_attn(key, self.last_ti_key, last_ti_value)
            # tensor [memory_value_short] size: [1960, 1, 1024], min: -2.007812, max: 1.608398, mean: 0.005006

            memory_value_short = memory_value_short.permute(1, 2, 0).view(batch, num_objects, value_dim, h, w)
            memory_readout += memory_value_short

            # todos.debug.output_var("memory_value_short", memory_value_short)
            # todos.debug.output_var("memory_readout", memory_readout)

            # xxxx_gggg
            hidden, predict_ab = self.network.decode_color(multi_scale_features, memory_readout, 
                                    self.memory.get_hidden(), h_out=not has_mask)

            # todos.debug.output_var("hidden", hidden)
            # todos.debug.output_var("predict_ab", predict_ab)
            # print("-" * 80)
            # tensor [memory_value_short] size: [1, 2, 512, 35, 56], min: -2.029297, max: 1.620117, mean: 0.005038
            # tensor [memory_readout] size: [1, 2, 512, 35, 56], min: -9.328125, max: 4.738281, mean: -0.007783
            # tensor [hidden] size: [1, 2, 64, 35, 56], min: -0.999528, max: 0.999268, mean: -0.085303
            # tensor [predict_ab] size: [1, 2, 560, 896], min: -0.441895, max: 0.613281, mean: 0.02429
            # --------------------------------------------------------------------------------


            # remove batch dim
            predict_ab = predict_ab[0]


            self.memory.set_hidden(hidden)
                
        return unpad(predict_ab, self.pad)

