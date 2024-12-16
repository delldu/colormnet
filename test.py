import os
from os import path
from argparse import ArgumentParser
import shutil

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from torch.utils.data import DataLoader
from inference.data.video_reader import VideoReader_221128_TransColorization

from model.network import ColorMNet
from inference.inference_core import InferenceCore

from tqdm import tqdm
from dataset.range_transform import inv_lll2rgb_trans
from skimage import color


import todos
import pdb

def detach_to_cpu(x):
    return x.detach().cpu()

def tensor_to_np_float(image):
    image_np = image.numpy().astype('float32')
    return image_np

def lab2rgb_transform_PIL(mask):
    mask_d = detach_to_cpu(mask)
    mask_d = inv_lll2rgb_trans(mask_d)
    # tensor [mask] size: [3, 480, 832], min: -0.994517, max: 1.0, mean: -0.060036
    # tensor [mask_d] size: [3, 480, 832], min: -0.994517, max: 1.0, mean: -0.060036
    # tensor [mask_d] size: [3, 480, 832], min: -52.40094, max: 100.0, mean: 14.825279

    im = tensor_to_np_float(mask_d)
    if len(im.shape) == 3:
        im = im.transpose((1, 2, 0))
    else:
        im = im[:, :, None]
    im = color.lab2rgb(im)
    # array [im] shape: (3, 480, 832), min: -52.40093994140625, max: 100.0, mean: 14.825279235839844
    # array [im] shape: (480, 832, 3), min: 0.0, max: 1.0, mean: 0.331046998500824
    return im.clip(0, 1)

def main():
    """
    Arguments loading
    """
    parser = ArgumentParser()
    parser.add_argument('--model', default='saves/DINOv2FeatureV6_LocalAtten_s2_154000.pth')
    parser.add_argument('--FirstFrameIsNotExemplar', help='Whether the provided reference frame is exactly the first input frame', action='store_true')

    # dataset setting
    parser.add_argument('--d16_batch_path', default='input/blackswan')
    parser.add_argument('--ref_path', default='ref/blackswan')
    # parser.add_argument('--d16_batch_path', default='input/v32')
    # parser.add_argument('--ref_path', default='ref/v32')

    parser.add_argument('--output', default='result')

    # For generic (G) evaluation, point to a folder that contains "JPEGImages" and "Annotations"
    parser.add_argument('--dataset', help='D16/D17/Y18/Y19/LV1/LV3/G', default='D16_batch')
    parser.add_argument('--split', help='val/test', default='val')
    parser.add_argument('--save_all', action='store_true', 
                help='Save all frames. Useful only in YouTubeVOS/long-time video', )
    parser.add_argument('--benchmark', action='store_true', help='enable to disable amp for FPS benchmarking')
            
    # Long-term memory options
    parser.add_argument('--disable_long_term', action='store_true')
    parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=10)
    parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=5)
    parser.add_argument('--max_long_term_elements', help='LT_max in paper, increase if objects disappear for a long time', 
                                                    type=int, default=10000)
    parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128)

    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--mem_every', help='r in paper. Increase to improve running speed.', type=int, default=5)
    parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int, default=-1)

    # Multi-scale options
    parser.add_argument('--save_scores', action='store_true')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--size', default=-1, type=int, 
                help='Resize the shorter side to this size. -1 to use original resolution. ')

    args = parser.parse_args()
    config = vars(args)
    config['enable_long_term'] = not config['disable_long_term'] # True

    if args.output is None:
        pdb.set_trace()
        args.output = f'.output/{args.dataset}_{args.split}'
        print(f'Output path not provided. Defaulting to {args.output}')

    # meta_dataset = VideoReader_221128_TransColorization("blackswan", args.d16_batch_path, args.ref_path)
    meta_dataset = VideoReader_221128_TransColorization("v32", args.d16_batch_path, args.ref_path)
    torch.autograd.set_grad_enabled(False)

    # Load our checkpoint
    network = ColorMNet(config, args.model).cuda().eval()
    # args.model -- 'saves/DINOv2FeatureV6_LocalAtten_s2_154000.pth'
    model_weights = torch.load(args.model)
    network.load_weights(model_weights, init_as_zero_if_needed=True)

    total_process_time = 0
    total_frames = 0

    loader = DataLoader(meta_dataset, batch_size=1, shuffle=False, num_workers=0)
    vid_name = "blackswan"
    vid_length = len(meta_dataset) #len(loader)

    processor = InferenceCore(network, config=config)
    progress_bar = tqdm(total=vid_length)

    for ti, data in enumerate(loader):
        progress_bar.update(1)

        with torch.cuda.amp.autocast(enabled=not args.benchmark):
            rgb = data['rgb'].cuda()[0]
            # tensor [rgb] size: [3, 480, 832], min: -0.994517, max: 1.0, mean: -0.238128
            
            msk = data.get('mask')
            # tensor [msk] size: [1, 3, 480, 832], min: -0.996459, max: 0.998028, mean: -0.054033

            # data.get('mask').size() -- [1, 3, 480, 832]
            msk = msk[:,1:3,:,:] if msk is not None else None
            # tensor [msk] size: [1, 2, 480, 832], min: -0.476372, max: 0.657849, mean: 0.02901
                
            info = data['info']
            # {'frame': ['00000.png'], 'vid_name': ['blackswan'], 'save': tensor([True]), 
            # 'shape': [tensor([480]), tensor([832])], 'need_resize': tensor([False])}

            frame = info['frame'][0]

            """
            For timing see https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
            Seems to be very similar in testing as my previous timing method 
            with two cuda sync + time.time() in STCN though 
            """
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            # Map possibly non-continuous labels to continuous ones
            if msk is not None: # True
                # msk.size() -- [1, 2, 480, 832]
                msk = torch.Tensor(msk[0]).cuda()
                processor.set_all_labels(list(range(1,3)))

    
            # Run the model on this frame
            # tensor [rgb] size: [3, 480, 832], min: -0.994517, max: 1.0, mean: -0.238128
            # tensor [msk] size: [2, 480, 832], min: -0.476372, max: 0.657849, mean: 0.02901
            # [labels] type: <class 'range'>
            prob = processor.step(rgb, msk)
            # tensor [prob] size: [2, 480, 832], min: -0.476372, max: 0.657849, mean: 0.02901
            end.record()
            torch.cuda.synchronize()
            total_process_time += (start.elapsed_time(end)/1000)
            total_frames += 1

            # Save the mask
            if args.save_all or info['save'][0]: # True
                this_out_path = path.join(args.output, vid_name)
                os.makedirs(this_out_path, exist_ok=True)

                out_mask_final = lab2rgb_transform_PIL(torch.cat([rgb[:1,:,:], prob], dim=0))
                out_mask_final = out_mask_final * 255
                out_mask_final = out_mask_final.astype(np.uint8)

                out_img = Image.fromarray(out_mask_final)
                out_img.save(os.path.join(this_out_path, frame[:-4]+'.png'))

    print(f'Total processing time: {total_process_time}')
    print(f'Total processed frames: {total_frames}')
    print(f'FPS: {total_frames / total_process_time}')
    print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}')


if __name__ == '__main__':  
    main()