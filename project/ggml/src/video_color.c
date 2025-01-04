/************************************************************************************
***
***	Copyright 2024 Dell Du(18588220928@163.com), All Rights Reserved.
***
***	File Author: Dell, Tue 02 Apr 2024 03:49:53 PM CST
***
************************************************************************************/

#include "video_xmem.h"
#include "video_color.h"

#define GGML_ENGINE_IMPLEMENTATION
#include <ggml_engine.h>
#define GGML_NN_IMPLEMENTATION
#include <ggml_nn.h>

#include <sys/stat.h> // for chmod()
#include <glob.h>
#include <nimage/tensor.h>

int tensor_are_same_shape(TENSOR *t1, TENSOR *t2)
{
    return t1->batch == t2->batch && t1->chan == t2->chan && t1->height == t2->height && t1->width == t2->width;
}

void tensor_debug_show(char *prompt, TENSOR* tensor)
{
    char buffer[512];

    // min, max, mean
    {
        double d;
        float min, max, mean;
        int n = tensor->batch * tensor->chan * tensor->height * tensor->width;

        d = 0.0;
        min = tensor->data[0];
        max = tensor->data[0];
        for (int i = 0; i < n; i++) {
            d += tensor->data[i];
            min = MIN(min, tensor->data[i]);
            max = MAX(max, tensor->data[i]);
        }
        d /= n;
        mean = (float)d;
        snprintf(buffer, sizeof(buffer), "min: %.4f, max: %.4f, mean: %.4f", min, max, mean);
    }

    syslog_info("%s Tensor: %dx%dx%dx%d, %s", prompt, tensor->batch, tensor->chan,
        tensor->height, tensor->width, buffer);
}


static TENSOR *lab_lll(TENSOR *lab)
{
    CHECK_TENSOR(lab);

    TENSOR *lll = tensor_create(1, 3, lab->height, lab->width);
    CHECK_TENSOR(lll);

    int n = lab->height * lab->width;
    float *d = lll->data;
    for (int i = 0; i < 3; i++) {
        memcpy(d, lab->data, n * sizeof(float));
        d += n;
    }
    return lll;
}

static TENSOR *lab_ab(TENSOR *lab)
{
    CHECK_TENSOR(lab);

    TENSOR *ab = tensor_create(1, 2, lab->height, lab->width);
    CHECK_TENSOR(ab);

    int n = lab->height * lab->width;
    memcpy(ab->data, lab->data + n, 2 * n * sizeof(float));

    return ab;
}

static int lab_update(TENSOR *lab, TENSOR *new_ab)
{
    check_tensor(lab);
    check_tensor(new_ab);

    int n = lab->height * lab->width;
    memcpy(lab->data + n, new_ab->data, 2 * n * sizeof(float));

    return RET_OK;
}

int video_color_predict(VideoColorNetwork *color_net, VideoXMemNetwork *xmem_net, char *example_files, char *gray_files, char *output_dir)
{
    int B, C, H, W;
    TENSOR *rgb_tensor, *lab_tensor, *image_lll, *image_ab, *predict_ab, *key, *value; 
    glob_t gray_glob, example_glob;
    char output_filename[512];

    glob(gray_files, GLOB_TILDE, NULL, &gray_glob);
    if (gray_glob.gl_pathc < 1) {
        syslog_error("There are any gray files (%s) to be colored.", gray_files);
        return RET_ERROR;        
    }

    glob(example_files, GLOB_TILDE, NULL, &example_glob);
    if (example_glob.gl_pathc < 1) {
        syslog_error("There are any color files (%s) for reference.", example_files);
        return RET_ERROR;        
    }

    make_dir(output_dir);

    // Create xmem cache
    rgb_tensor = tensor_load_image(gray_glob.gl_pathv[0], 0 /*with_alpha */ );
    tensor_resizepad_(rgb_tensor, color_net->MAX_H, color_net->MAX_W, color_net->MAX_TIMES);
    B = rgb_tensor->batch;
    C = rgb_tensor->chan;
    H = rgb_tensor->height;
    W = rgb_tensor->width;
    xmem_net->alloc_cache(H/16, W/16, 8 /*WORKMEM_SIZE*/, example_glob.gl_pathc /*LONGMEM_SIZE*/);
    tensor_destroy(rgb_tensor);


    // Decode examples
    // hidden_state = torch.zeros(2, 64, H//16, W//16).to(device)
    // for f in example_images:
    //     print(f"encode examples {f} ...")
    //     image = todos.data.load_tensor(f)
    //     B2, C2, H2, W2 = image.size()
    //     if H2 != H or W2 != W:        
    //         image = F.interpolate(image, size=(H, W), mode="bilinear")
    //     image = image.to(device)
    //     image_lll, image_ab = rgb_lab(image)
    //     with torch.no_grad():
    //         key, shrinkage, selection, f16, f8, f4 = model.encode_key(image_lll)
    //         value, hidden = model.encode_value(image_lll, f16, hidden_state, image_ab)
    //     xmem.set_long_memory(key, shrinkage, value)
    //     xmem.set_hidden(hidden)
    //     xmem.set_last_key(key)
    //     xmem.set_last_value(value)

    TENSOR *zero_hidden = tensor_create(2, 64, H/16, W/16);
    check_tensor(zero_hidden);
    for (int i = example_glob.gl_pathc - 1; i >= 0; i--) {
        CheckPoint("%s ...", example_glob.gl_pathv[i]);

        rgb_tensor = tensor_load_image(example_glob.gl_pathv[i], 0 /*with_alpha */ );
        tensor_zoom_(rgb_tensor, H, W);
        lab_tensor = tensor_rgb2lab(rgb_tensor);
        tensor_destroy(rgb_tensor);

        // tensor_debug_show("--lab_tensor", lab_tensor);

        image_lll = lab_lll(lab_tensor);
        image_ab  = lab_ab(lab_tensor);
        // tensor_debug_show("--image_lll", image_lll);
        // tensor_debug_show("--image_ab", image_ab);
        tensor_destroy(lab_tensor);

        key = color_net->encode_key(image_lll);
        // tensor_debug_show("--key", key);
        // tensor_debug_show("--KEY", color_net->KEY);
        // tensor_debug_show("--SHRINKAGE", color_net->SHRINKAGE);
        // tensor_debug_show("--SELECTION", color_net->SELECTION);
        // tensor_debug_show("--F16", color_net->F16);
        // tensor_debug_show("--F8", color_net->F8);
        // tensor_debug_show("--F4", color_net->F4);

        value = color_net->encode_value(image_lll, color_net->F16, zero_hidden, image_ab);
        // tensor_debug_show("--value", value);
        // tensor_debug_show("--VALUE", color_net->VALUE);
        // tensor_debug_show("--HIDDEN", color_net->HIDDEN);

        // tensor [image_lll] size: [1, 3, 560, 896], min: -0.495599, max: 0.496816, mean: -0.109927
        // tensor [f16] size: [1, 1024, 35, 56], min: 0.0, max: 2.623592, mean: 0.064568
        // tensor [hidden_state] size: [2, 64, 35, 56], min: 0.0, max: 0.0, mean: 0.0
        // tensor [image_ab] size: [1, 2, 560, 896], min: -0.471232, max: 0.655021, mean: 0.029019

        // tensor [value] size: [2, 512, 35, 56], min: -11.473879, max: 5.339447, mean: -0.008071
        // tensor [hidden] size: [2, 64, 35, 56], min: -0.999958, max: 0.999501, mean: -0.020198
        // --------------------------------------------------------------------------------
        xmem_net->set_long_memory(color_net->KEY, color_net->SHRINKAGE, color_net->VALUE, color_net->HIDDEN);
        tensor_destroy(image_ab);
        tensor_destroy(image_lll);
        tensor_destroy(value);
        tensor_destroy(key);
    }
    tensor_destroy(zero_hidden);
    xmem_net->dump();

    // Decode gray images ...
    // input_rgb = input_rgb.to(device)
    // image_lll, image_ab = rgb_lab(input_rgb)
    // with torch.no_grad():
    //     key, shrinkage, selection, f16, f8, f4 = model.encode_key(image_lll)

    //     multi_scale_features = (f16, f8, f4)
    //     value = xmem.forward(key, selection)
    //     hidden = xmem.get_hidden()

    //     # reference local frame ...
    //     last_key = xmem.get_last_key()
    //     last_value = xmem.get_last_value()
    //     short_value = model.short_term_attn(key, last_key, last_value)
    //     value = value + short_value

    //     # hidden, predict_ab = model.decode_color(multi_scale_features, value, hidden)
    //     predict_ab = model.decode_color(multi_scale_features, value, hidden)

    //     output_l = image_lll[:, 0:1, :, :] * 100.0
    //     output_ab = predict_ab * 110.0
    //     output_lab = torch.cat([output_l, output_ab], dim=1)
    //     output_rgb = data.lab2rgb(output_lab)
    //     del image_ab, selection, f8, f4, output_l, output_ab, output_lab

    // # save the frames
    // temp_output_file = "{}/{:06d}.png".format(output_dir, no + 1)
    // todos.data.save_tensor(output_rgb, temp_output_file)

    // if no % 5 == 0:
    //     # update work memory
    //     value, hidden = model.encode_value(image_lll, f16, hidden, predict_ab[:, 0:2, :, :])
    //     xmem.set_work_memory(key, shrinkage, value)
    //     xmem.set_hidden(hidden)            
    //     xmem.set_last_key(key)
    //     xmem.set_last_value(value)

    for (int i = 0; i < gray_glob.gl_pathc; i++) {
        rgb_tensor = tensor_load_image(gray_glob.gl_pathv[i], 0 /*with_alpha */ );
        tensor_zoom_(rgb_tensor, H, W);
        lab_tensor = tensor_rgb2lab(rgb_tensor);
        tensor_destroy(rgb_tensor);
        // CheckPoint();

        image_lll = lab_lll(lab_tensor);
        key = color_net->encode_key(image_lll);

        // CheckPoint();
        value = xmem_net->query_value(color_net->KEY, color_net->SELECTION);

        // CheckPoint();
        // # multi_scale_features(f16, f8, f4) is tuple: len = 3
        // #     tensor [item] size: [1, 1024, 35, 56], min: 0.0, max: 2.601784, mean: 0.063031
        // #     tensor [item] size: [1, 512, 70, 112], min: 0.0, max: 1.79675, mean: 0.090695
        // #     tensor [item] size: [1, 256, 140, 224], min: 0.0, max: 6.709424, mean: 0.200673
        // # tensor [value] size: [2, 512, 35, 56], min: -9.328125, max: 4.738281, mean: -0.007783
        // # tensor [hidden] size: [2, 64, 35, 56], min: -1.0, max: 0.999023, mean: -0.009137      
        tensor_debug_show("---- F16", color_net->F16);
        tensor_debug_show("---- F8", color_net->F8);
        tensor_debug_show("---- F4", color_net->F4);
        tensor_debug_show("---- value", value);
        tensor_debug_show("---- HIDDEN", color_net->HIDDEN);

        predict_ab = color_net->decode_color(color_net->F16, color_net->F8, color_net->F4, value, color_net->HIDDEN);
        tensor_debug_show("---- predict_ab", predict_ab);
        tensor_destroy(value);

        // CheckPoint();

        // output ...
        lab_update(lab_tensor, predict_ab);
        rgb_tensor = tensor_lab2rgb(lab_tensor);
        snprintf(output_filename, sizeof(output_filename), "%s/%06d.png", output_dir, i + 1);
        tensor_saveas_image(rgb_tensor, 0 /*batch*/, output_filename);
        tensor_destroy(rgb_tensor);
        tensor_destroy(lab_tensor);

        // CheckPoint();

        // if (i % 5 == 0) {
        //     value = color_net->encode_value(image_lll, color_net->F16, color_net->HIDDEN, predict_ab);
        //     xmem_net->set_work_memory(color_net->KEY, color_net->SHRINKAGE, color_net->VALUE, color_net->HIDDEN);
        //     tensor_destroy(value);
        // }

        // CheckPoint();

        tensor_destroy(image_lll);
        tensor_destroy(lab_tensor);
        tensor_destroy(key);
        tensor_destroy(predict_ab);
    }

    // snprintf(output_filename, sizeof(output_filename), "%s.mp4", output_dir);
    // video_encode(output_dir, output_filename);

    globfree(&gray_glob);
    globfree(&example_glob);

    return RET_OK;
}
