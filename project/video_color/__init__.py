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
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import redos
import todos

from . import data, color
from .xmem import XMem

import pdb


class ResizePad(nn.Module):
    def __init__(self):
        super().__init__()
        self.MAX_H = 1024
        self.MAX_W = 1024
        self.MAX_TIMES = 112

    def forward(self, x):
        # Need Resize ?
        B, C, H, W = x.size()
        if H > self.MAX_H or W > self.MAX_W:
            s = min(self.MAX_H / H, self.MAX_W / W)
            SH, SW = int(s * H), int(s * W)
            x = F.interpolate(x, size=(SH, SW), mode="bilinear", align_corners=False)
        # Need Pad ?
        B, C, H, W = x.size()
        pad_h = (self.MAX_TIMES - (H % self.MAX_TIMES)) % self.MAX_TIMES
        pad_w = (self.MAX_TIMES - (W % self.MAX_TIMES)) % self.MAX_TIMES
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')
            # x = F.pad(x, (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2))
        return x


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
    # image = torch.randn(1, 3, 560, 896).to(device)
    # B, C, H, W = image.size()

    # color_ab = torch.randn(B, 2, H, W).to(device)
    # hidden_state = torch.randn(2, 64, H//16, W//16).to(device)

    # key, shrinkage, selection, f16, f8, f4 = model.encode_key(image)

    # todos.debug.output_var("key", key)
    # todos.debug.output_var("shrinkage", shrinkage)
    # todos.debug.output_var("selection", selection)
    # todos.debug.output_var("f16", f16)
    # todos.debug.output_var("f8", f8)
    # todos.debug.output_var("f4", f4)
    # print("-" * 80)
    # tensor [key] size: [1, 64, 35, 56], min: -2.006974, max: 2.265148, mean: -0.119895
    # tensor [shrinkage] size: [1, 1, 35, 56], min: 19.956394, max: 43.019291, mean: 37.951973
    # tensor [selection] size: [1, 64, 35, 56], min: 0.0, max: 0.93948, mean: 0.570642
    # tensor [f16] size: [1, 1024, 35, 56], min: 0.0, max: 2.395351, mean: 0.066758
    # tensor [f8] size: [1, 512, 70, 112], min: 0.0, max: 1.421202, mean: 0.09145
    # tensor [f4] size: [1, 256, 140, 224], min: 0.0, max: 5.96357, mean: 0.215469

    # value, hidden = model.encode_value(image, f16, hidden_state, color_ab)
    # todos.debug.output_var("value", value)
    # todos.debug.output_var("hidden", hidden)
    # print("-" * 80)

    # # tensor [value] size: [2, 512, 35, 56], min: -31.887749, max: 15.17862, mean: -0.761654
    # # tensor [hidden] size: [2, 64, 35, 56], min: -3.855072, max: 4.748185, mean: 0.110133

    # multi_scale_features = (f16, f8, f4)
    # hidden, predict_ab = model.decode_color(multi_scale_features, value, hidden)
    # todos.debug.output_var("hidden", hidden)
    # todos.debug.output_var("predict_ab", predict_ab)
    # print("-" * 80)

    # tensor [key] size: [1, 64, 35, 56], min: -2.75, max: 3.166016, mean: -0.143513
    # tensor [self.last_ti_key] size: [1, 64, 35, 56], min: -2.753906, max: 3.142578, mean: -0.143365
    # tensor [last_ti_value] size: [1, 1024, 35, 56], min: -9.9375, max: 5.101562, mean: -0.014165
    key = torch.randn(1, 64, 35, 56).to(device)
    key2 = torch.randn(1, 64, 35, 56).to(device)
    val2 = torch.randn(1, 1024, 35, 56).to(device)
    local_value = model.short_term_attn(key, key2, val2)
    todos.debug.output_var("local_value", local_value)
    # tensor [local_value] size: [1960, 1, 1024], min: -0.21165, max: 0.222777, mean: 0.001241

# def test_self_color(model, device, input_rgb):
#     B, C, H, W = input_rgb.size()

#     MAX_TIMES = 112
#     pad_h = (MAX_TIMES - (H % MAX_TIMES)) % MAX_TIMES
#     pad_w = (MAX_TIMES - (W % MAX_TIMES)) % MAX_TIMES
#     if pad_h > 0 or pad_w > 0:
#         input_rgb = F.pad(input_rgb, (0, pad_w, 0, pad_h), 'reflect')

#     # new size ?
#     B2, C2, H2, W2 = input_rgb.size()

#     input_lab = data.rgb2lab(input_rgb).to(device)
#     output_l = input_lab[:, 0:1, :, :] 
#     input_l = input_lab[:, 0:1, :, :]/100.0
#     input_l = (input_l - 0.5) * 2.0 # in [-1.0, 1.0]

#     input_ab = input_lab[:, 1:3, :, :]/110.0 # in [-1.0, 1.0]
#     input_lll = input_l.repeat(1, 3, 1, 1)

#     input_hidden = torch.zeros(2, 64, H2//16, W2//16).to(device)
#     key, shrinkage, selection, f16, f8, f4 = model.encode_key(input_lll)
#     value, hidden = model.encode_value(input_lll, f16, input_hidden, input_ab)

#     multi_scale_features = (f16, f8, f4)
#     hidden, predict_ab = model.decode_color(multi_scale_features, value, hidden)
#     output_ab = predict_ab * 110.0
#     output_lab = torch.cat([output_l, output_ab], dim=1)

#     output_rgb = data.lab2rgb(output_lab)
#     output_rgb = output_rgb[:, :, 0:H, 0:W].cpu()
#     del input_lab, input_ab, input_lll, input_hidden, value, hidden, predict_ab, output_l, output_ab, output_lab

#     return output_rgb

def rgb_lab(input_rgb):
    # tensor [input_rgb] size: [1, 3, 560, 896], min: 0.0, max: 1.0, mean: 0.339481
    input_lab = data.rgb2lab(input_rgb)
    # tensor [input_l] size: [1, 1, 560, 896], min: -49.55986, max: 49.681641, mean: -10.992733
    # tensor [input_ab] size: [1, 2, 560, 896], min: -51.83556, max: 72.052292, mean: 3.19213
    # input_lab[:, 0:1, :, :] in [-50.0, 50]
    input_lll = input_lab[:, 0:1, :, :]/100.0
    input_lll = input_lll.repeat(1, 3, 1, 1)
    input_ab = input_lab[:, 1:3, :, :]/110
    return input_lll, input_ab

def video_predict(input_file, color_files, output_file):
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

    resize_model = ResizePad()
    resize_model = resize_model
    resize_model.eval()

    # return test_color_model(model, device)
    fake_frame = torch.zeros(1, 3, video.height, video.width)
    with torch.no_grad():
        fake_frame = resize_model(fake_frame)
    B, C, H, W = fake_frame.size()
    xmem = XMem(device, H//16, W//16, 4, 4)

    # examples image must be reverse sorted !!!
    example_images = sorted(todos.data.load_files(color_files), reverse=True)

    hidden_state = torch.zeros(2, 64, H//16, W//16).to(device)
    for f in example_images:
        print(f"encode examples {f} ...")
        image = todos.data.load_tensor(f)
        B2, C2, H2, W2 = image.size()
        if H2 != H or W2 != W:        
            image = F.interpolate(image, size=(H, W), mode="bilinear")
        image = image.to(device)
        image_lll, image_ab = rgb_lab(image)
        with torch.no_grad():
            key, shrinkage, selection, f16, f8, f4 = model.encode_key(image_lll)
            value, hidden = model.encode_value(image_lll, f16, hidden_state, image_ab)

        xmem.set_long_memory(key, shrinkage, value)
        xmem.set_hidden(hidden)
        xmem.set_last_key(key)
        xmem.set_last_value(value)


    print(f"  Color {input_file} with {color_files}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)

    def color_video_frame(no, datax):
        # print(f"frame: {no} -- {data.shape}")
        progress_bar.update(1)

        input_tensor = todos.data.frame_totensor(datax)
        B2, C2, H2, W2 = input_tensor.size()
        if H2 != H or W2 != W:        
            input_tensor = F.interpolate(input_tensor, size=(H, W), mode="bilinear")

        # convert tensor from 1x4xHxW to 1x3xHxW
        input_rgb = input_tensor[:, 0:3, :, :].to(device)
        # input_rgb = color_to_gray(input_rgb)
        image_lll, image_ab = rgb_lab(input_rgb)

        with torch.no_grad():
            key, shrinkage, selection, f16, f8, f4 = model.encode_key(image_lll)
            multi_scale_features = (f16, f8, f4)

            value = xmem.forward(key, selection)
            hidden = xmem.get_hidden()

            # reference local frame ...
            last_key = xmem.get_last_key()
            last_value = xmem.get_last_value()
            short_value = model.short_term_attn(key, last_key, last_value)
            value = value + short_value

            hidden, predict_ab = model.decode_color(multi_scale_features, value, hidden)
            output_l = image_lll[:, 0:1, :, :] * 100.0
            output_ab = predict_ab * 110.0
            output_lab = torch.cat([output_l, output_ab], dim=1)
            output_rgb = data.lab2rgb(output_lab)

        # save the frames
        temp_output_file = "{}/{:06d}.png".format(output_dir, no + 1)
        todos.data.save_tensor(output_rgb, temp_output_file)
        if no % 5 == 0: # key frames ...
            # update work memory
            with torch.no_grad():
                value, hidden = model.encode_value(image_lll, f16, hidden, predict_ab[:, 0:2, :, :])
            xmem.set_work_memory(key, shrinkage, value)
            xmem.set_hidden(hidden)
            xmem.set_last_key(key)
            xmem.set_last_value(value)
        del image_ab, selection, f8, f4, output_l, output_ab, output_lab

    video.forward(callback=color_video_frame)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i + 1)
        os.remove(temp_output_file)

    os.removedirs(output_dir)
    todos.model.reset_device()

    return True


def image_predict(input_files, color_files, output_file):
    # load input files
    gray_images = todos.data.load_files(input_files)
    assert len(gray_images) > 0

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_color_model()

    resize_model = ResizePad()
    resize_model = resize_model
    resize_model.eval()

    # return test_color_model(model, device)
    fake_frame = todos.data.load_tensor(gray_images[0]) # [1, 3, 480, 832]
    with torch.no_grad():
        fake_frame = resize_model(fake_frame)
    B, C, H, W = fake_frame.size() # [1, 3, 560, 896]
    xmem = XMem(device, H//16, W//16, 4, 4)

    # examples image must be reverse sorted !!!
    example_images = sorted(todos.data.load_files(color_files), reverse=True)

    hidden_state = torch.zeros(2, 64, H//16, W//16).to(device)
    for f in example_images:
        print(f"encode examples {f} ...")

        image = todos.data.load_tensor(f)
        B2, C2, H2, W2 = image.size()
        if H2 != H or W2 != W:        
            image = F.interpolate(image, size=(H, W), mode="bilinear")
        image = image.to(device)
        image_lll, image_ab = rgb_lab(image)
        with torch.no_grad():
            # tensor [image_lll] size: [1, 3, 560, 896], min: -0.495599, max: 0.496816, mean: -0.109927
            key, shrinkage, selection, f16, f8, f4 = model.encode_key(image_lll)
            # tensor [key] size: [1, 64, 35, 56], min: -2.803887, max: 3.253764, mean: -0.159051
            # tensor [shrinkage] size: [1, 1, 35, 56], min: 23.295803, max: 45.933445, mean: 32.660492
            # tensor [selection] size: [1, 64, 35, 56], min: 0.0, max: 0.902618, mean: 0.500047
            # tensor [f16] size: [1, 1024, 35, 56], min: 0.0, max: 2.623592, mean: 0.064568
            # tensor [f8] size: [1, 512, 70, 112], min: 0.0, max: 1.733069, mean: 0.093956
            # tensor [f4] size: [1, 256, 140, 224], min: 0.0, max: 6.521586, mean: 0.203084

            todos.debug.output_var("image_lll", image_lll)
            todos.debug.output_var("f16", f16)
            todos.debug.output_var("hidden_state", hidden_state)
            todos.debug.output_var("image_ab", image_ab)
            value, hidden = model.encode_value(image_lll, f16, hidden_state, image_ab)
            todos.debug.output_var("value", value)
            todos.debug.output_var("hidden", hidden)
            print("-" * 80)

        xmem.set_long_memory(key, shrinkage, value)
        xmem.set_hidden(hidden)
        xmem.set_last_key(key)
        xmem.set_last_value(value)

    print(f"  Color {input_files} with {color_files}, save to {output_file} ...")
    progress_bar = tqdm(total=len(gray_images))
    no = 0
    for f in gray_images:
        progress_bar.update(1)

        input_tensor = todos.data.load_tensor(f)

        # convert tensor from 1x4xHxW to 1x3xHxW
        input_rgb = input_tensor[:, 0:3, :, :]
        B2, C2, H2, W2 = input_rgb.size()
        if H2 != H or W2 != W:        
            input_rgb = F.interpolate(input_rgb, size=(H, W), mode="bilinear")

        # input_rgb = color_to_gray(input_rgb)
        input_rgb = input_rgb.to(device)
        image_lll, image_ab = rgb_lab(input_rgb)
        with torch.no_grad():
            key, shrinkage, selection, f16, f8, f4 = model.encode_key(image_lll)

            multi_scale_features = (f16, f8, f4)
            value = xmem.forward(key, selection)
            # pdb.set_trace()

            hidden = xmem.get_hidden()

            # reference local frame ...
            last_key = xmem.get_last_key()
            last_value = xmem.get_last_value()
            short_value = model.short_term_attn(key, last_key, last_value)
            value = value + short_value

            # hidden, predict_ab = model.decode_color(multi_scale_features, value, hidden)
            predict_ab = model.decode_color(multi_scale_features, value, hidden)

            output_l = image_lll[:, 0:1, :, :] * 100.0
            output_ab = predict_ab * 110.0
            output_lab = torch.cat([output_l, output_ab], dim=1)
            output_rgb = data.lab2rgb(output_lab)
            del image_ab, selection, f8, f4, output_l, output_ab, output_lab

        # save the frames
        temp_output_file = "{}/{:06d}.png".format(output_dir, no + 1)
        todos.data.save_tensor(output_rgb, temp_output_file)

        if no % 5 == 0:
            # update work memory
            value, hidden = model.encode_value(image_lll, f16, hidden, predict_ab[:, 0:2, :, :])
            xmem.set_work_memory(key, shrinkage, value)
            xmem.set_hidden(hidden)            
            xmem.set_last_key(key)
            xmem.set_last_value(value)

        no = no + 1
    todos.model.reset_device()

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(len(gray_images)):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i + 1)
        os.remove(temp_output_file)

    os.removedirs(output_dir)

    return True
