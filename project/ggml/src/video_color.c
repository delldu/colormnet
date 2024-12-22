/************************************************************************************
***
***	Copyright 2024 Dell Du(18588220928@163.com), All Rights Reserved.
***
***	File Author: Dell, Tue 02 Apr 2024 03:49:53 PM CST
***
************************************************************************************/

#include "video_matte.h"

#define GGML_ENGINE_IMPLEMENTATION
#include <ggml_engine.h>
#define GGML_NN_IMPLEMENTATION
#include <ggml_nn.h>

#include <sys/stat.h> // for chmod()

int make_divisible(int v, int divisor) {
    return ((v + divisor - 1)/divisor) * divisor;
}

int video_matte_predict(VideoMatteNetwork *matte_net, char *input_file, char *output_file)
{
    TENSOR *input_tensor, *output_tensor;
    {
        input_tensor = tensor_load_image(input_file, 0 /*input_with_alpha*/);
        check_tensor(input_tensor);
    }

    {
        output_tensor = matte_net->forward(input_tensor);
        check_tensor(output_tensor);
        tensor_destroy(input_tensor);

        // TENSOR *xxxx_test;
        // xxxx_test = matte_net->net.get_output_tensor("R1");
        // if (tensor_valid(xxxx_test)) {
        //     tensor_show("********************** R1", xxxx_test);
        //     tensor_destroy(xxxx_test);
        // }

        tensor_saveas_image(output_tensor, 0 /*batch 0*/, output_file);
        chmod(output_file, 0644);

        tensor_destroy(output_tensor);
    }

    return RET_OK;
}
