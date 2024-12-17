"""Video Color Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch

import redos
import todos

from . import data, color

import pdb


def get_color_model():
    """Create model.
    pre-trained model video_color.pth comes from
    https://github.com/delldu/TorchScript.git/video_color
    """

    model = color.ColorMNet()
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    # model = torch.jit.script(model)
    # todos.data.mkdir("output")
    # if not os.path.exists("output/video_color.torch"):
    #     model.save("output/video_color.torch")

    return model, device

def test_color_model(model, device):
    # tensor [image] size: [1, 3, 560, 896], min: -1.0, max: 1.0, mean: -0.183782
    image = torch.randn(1, 3, 560, 896).to(device)
    B, C, H, W = image.size()

    color_ab = torch.randn(B, 2, H, W).to(device)
    hidden_state = torch.randn(2, 64, H//16, W//16).to(device)

    key, shrinkage, selection, f16, f8, f4 = model.encode_key(image)

    todos.debug.output_var("key", key)
    todos.debug.output_var("shrinkage", shrinkage)
    todos.debug.output_var("selection", selection)
    todos.debug.output_var("f16", f16)
    todos.debug.output_var("f8", f8)
    todos.debug.output_var("f4", f4)
    print("-" * 80)
    # tensor [key] size: [1, 64, 35, 56], min: -2.006974, max: 2.265148, mean: -0.119895
    # tensor [shrinkage] size: [1, 1, 35, 56], min: 19.956394, max: 43.019291, mean: 37.951973
    # tensor [selection] size: [1, 64, 35, 56], min: 0.0, max: 0.93948, mean: 0.570642
    # tensor [f16] size: [1, 1024, 35, 56], min: 0.0, max: 2.395351, mean: 0.066758
    # tensor [f8] size: [1, 512, 70, 112], min: 0.0, max: 1.421202, mean: 0.09145
    # tensor [f4] size: [1, 256, 140, 224], min: 0.0, max: 5.96357, mean: 0.215469

    value, hidden = model.encode_value(image, f16, hidden_state, color_ab)
    todos.debug.output_var("value", value)
    todos.debug.output_var("hidden", hidden)
    print("-" * 80)

    # tensor [value] size: [2, 512, 35, 56], min: -31.887749, max: 15.17862, mean: -0.761654
    # tensor [hidden] size: [2, 64, 35, 56], min: -3.855072, max: 4.748185, mean: 0.110133

    multi_scale_features = (f16, f8, f4)
    hidden, predict = model.decode_color(multi_scale_features, value, hidden)
    todos.debug.output_var("hidden", hidden)
    todos.debug.output_var("predict", predict)
    print("-" * 80)

    # tensor [key] size: [1, 64, 35, 56], min: -2.75, max: 3.166016, mean: -0.143513
    # tensor [self.last_ti_key] size: [1, 64, 35, 56], min: -2.753906, max: 3.142578, mean: -0.143365
    # tensor [last_ti_value] size: [1, 1024, 35, 56], min: -9.9375, max: 5.101562, mean: -0.014165
    key = torch.randn(1, 64, 35, 56).to(device)
    key2 = torch.randn(1, 64, 35, 56).to(device)
    val2 = torch.randn(1, 1024, 35, 56).to(device)
    local_value = model.short_term_attn(key, key2, val2)
    todos.debug.output_var("local_value", local_value)
    # tensor [local_value] size: [1960, 1, 1024], min: -0.21165, max: 0.222777, mean: 0.001241


def video_predict(input_file, color_file, output_file):
    # load video
    video = redos.video.Reader(input_file)
    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_color_model()

    # return test_color_model(model, device)

    # Reference
    reference_tensor = todos.data.load_tensor(color_file)

    print(f"  Color {input_file} with {color_file}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)

    def color_video_frame(no, datax):
        # print(f"frame: {no} -- {data.shape}")
        progress_bar.update(1)

        input_tensor = todos.data.frame_totensor(datax)

        # convert tensor from 1x4xHxW to 1x3xHxW
        input_tensor = input_tensor[:, 0:3, :, :]
        output_rgb = todos.model.two_forward(model, device, input_tensor, reference_tensor)

        # save the frames
        temp_output_file = "{}/{:06d}.png".format(output_dir, no + 1)
        todos.data.save_tensor(output_rgb, temp_output_file)

    video.forward(callback=color_video_frame)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i + 1)
        os.remove(temp_output_file)

    os.removedirs(output_dir)
    todos.model.reset_device()

    return True
