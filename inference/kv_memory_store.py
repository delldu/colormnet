import torch
from typing import List
import todos
import pdb

class KeyValueMemoryStore:
    """
    Works for key/value pairs type storage
    e.g., working and long-term memory
    """

    """
    An object group is created when new objects enter the video
    Objects in the same group share the same temporal extent
    i.e., objects initialized in the same frame are in the same group
    For DAVIS/interactive, there is only one object group
    For YouTubeVOS, there can be multiple object groups
    """

    def __init__(self):
        # keys are stored in a single tensor and are shared between groups/objects
        # values are stored as a list indexed by object groups
        self.k = None
        self.v = []
        self.obj_groups = []

        # shrinkage and selection are also single tensors
        self.s = self.e = None

        # usage
        self.use_count = self.life_count = None

    def add(self, key, value, shrinkage, selection, objects: List[int]):
        new_count = torch.zeros((key.shape[0], 1, key.shape[2]), device=key.device, dtype=torch.float32)
        new_life = torch.zeros((key.shape[0], 1, key.shape[2]), device=key.device, dtype=torch.float32) + 1e-7

        # add the key
        assert self.k is None
        self.k = key
        self.s = shrinkage
        self.e = selection
        self.use_count = new_count
        self.life_count = new_life

        # add the value
        # When objects is given, v is a tensor; used in working memory
        assert isinstance(value, torch.Tensor)
        # First consume objects that are already in the memory bank
        # cannot use set here because we need to preserve order
        # shift by one as background is not part of value
        # objects === [1, 2], come from labels ...

        remaining_objects = [obj-1 for obj in objects] # ==> [0, 1]
        for gi, group in enumerate(self.obj_groups): # self.obj_groups === []
            pdb.set_trace()
            for obj in group:
                # should properly raise an error if there are overlaps in obj_groups
                remaining_objects.remove(obj)
            self.v[gi] = torch.cat([self.v[gi], value[group]], -1)

        # If there are remaining objects, add them as a new group
        if len(remaining_objects) > 0:
            new_group = list(remaining_objects) # === [0, 1]
            # tensor [value] size: [2, 512, 1960], min: -9.921875, max: 5.101562, mean: -0.014141
            self.v.append(value[new_group])
            self.obj_groups.append(new_group)
            
        # ------------------------------------------------------------------------------------------------------
        # (Pdb) len(self.v) -- 1
        # self.v[0].size() -- [2, 512, 1960]
        # self.obj_groups -- [[0, 1]]

    def update_usage(self, usage):
        # increase all life count by 1
        # increase use of indexed elements
        self.use_count += usage.view_as(self.use_count)
        self.life_count += 1


    def get_usage(self):
        usage = self.use_count / self.life_count
        return usage

    def get_v_size(self, ni: int):
        return self.v[ni].shape[2]

    def engaged(self):
        return self.k is not None

    @property
    def size(self):
        if self.k is None:
            return 0
        else:
            return self.k.shape[-1]

    @property
    def num_groups(self):
        return len(self.v)

    @property
    def key(self):
        return self.k

    @property
    def value(self):
        return self.v

    @property
    def shrinkage(self):
        return self.s

    @property
    def selection(self):
        return self.e

