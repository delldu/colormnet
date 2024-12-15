import os
from os import path

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as Ff
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization, im_rgb2lab_normalization, ToTensor, RGB2Lab
import pdb

class VideoReader_221128_TransColorization(Dataset):
    """
    This class is used to read a video, one frame at a time
    """
    def __init__(self, vid_name, image_dir, mask_dir, size=-1, to_save=None, size_dir=None):
        """
        image_dir - points to a directory of jpg images
        mask_dir - points to a directory of png masks
        size - resize min. side to size. Does nothing if <0.
        to_save - optionally contains a list of file names without extensions 
            where the segmentation mask is required
        use_all_mask - when true, read all available mask in mask_dir.
            Default false. Set to true for YouTubeVOS validation.
        """
        self.vid_name = vid_name
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.to_save = to_save

        self.frames = [img for img in sorted(os.listdir(self.image_dir)) if (img.endswith('.jpg') or img.endswith('.png')) and not img.startswith('.')]
        # self.frames = self.frames[:16] # xxxx_debug

        # self.palette = Image.open(path.join(mask_dir, sorted([msk for msk in os.listdir(mask_dir) if not msk.startswith('.')])[0])).getpalette()
        # self.suffix = self.first_gt_path.split('.')[-1]

        self.im_transform = transforms.Compose([
            RGB2Lab(),
            ToTensor(),
            im_rgb2lab_normalization,
        ])
        self.size = size

        self.first_gt_path = path.join(self.mask_dir, sorted([msk for msk in os.listdir(self.mask_dir) if not msk.startswith('.')])[0])


    def __getitem__(self, idx):
        frame = self.frames[idx]
        info = {}
        data = {}
        info['frame'] = frame
        info['vid_name'] = self.vid_name
        info['save'] = (self.to_save is None) or (frame[:-4] in self.to_save)

        im_path = path.join(self.image_dir, frame)
        img = Image.open(im_path).convert('RGB')
        shape = np.array(img).shape[:2]

        img = self.im_transform(img)
        img_l = img[:1,:,:]
        img_lll = img_l.repeat(3,1,1)

        # self.first_gt_path -- 'ref/blackswan/00000.png'
        gt_path = path.join(self.mask_dir, sorted(os.listdir(self.mask_dir))[idx]) if idx < len(os.listdir(self.mask_dir)) else None 
        load_mask = (gt_path == self.first_gt_path)
        if load_mask and path.exists(gt_path):
            mask = Image.open(gt_path).convert('RGB')
            mask = self.im_transform(mask)
            data['mask'] = mask

        info['shape'] = shape
        info['need_resize'] = not (self.size < 0) # False
        data['rgb'] = img_lll
        data['info'] = info

        return data


    def __len__(self):
        return len(self.frames)
