/************************************************************************************
***
*** Copyright 2021-2024 Dell Du(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, 2021年 11月 22日 星期一 14:33:18 CST
***
************************************************************************************/

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <glob.h>

#include <ggml_engine.h>
#include <nimage/tensor.h>

// #include "video_color.h"
#include "video_xmem.h"
#include "video_color.h"

#define DEFAULT_DEVICE 1
#define DEFAULT_OUTPUT "output"

// int video_color_predict(VideoColorNetwork *color_net, char *input_file, char *output_file);

static void video_color_help(char* cmd)
{
    printf("Usage: %s [option] image_files\n", cmd);
    printf("    -h, --help                   Display this help, version %s.\n", ENGINE_VERSION);
    printf("    -d, --device <no>            Set device (0 -- cpu, 1 -- cuda0, 2 -- cuda1, ..., default: %d)\n", DEFAULT_DEVICE);
    printf("    -e, --examples <file names>  Set color example files\n");

    printf("    -o, --output                 output dir, default: %s.\n", DEFAULT_OUTPUT);

    exit(1);
}

int main(int argc, char** argv)
{
    int optc;
    int option_index = 0;
    int device_no = DEFAULT_DEVICE;
    char* output_dir = (char*)DEFAULT_OUTPUT;

    glob_t examples;
    int examples_set = 0;

    char *p, output_filename[1024];

    struct option long_opts[] = {
        { "help", 0, 0, 'h' },
        { "device", 1, 0, 'd' },
        { "examples", 1, 0, 'e' },
        { "output", 1, 0, 'o' },
        { 0, 0, 0, 0 }

    };

    if (argc <= 1)
        video_color_help(argv[0]);


    while ((optc = getopt_long(argc, argv, "h d: e: o:", long_opts, &option_index)) != EOF) {
        switch (optc) {
        case 'd':
            device_no = atoi(optarg);
            break;
        case 'e':
            glob(optarg, GLOB_TILDE, NULL, &examples);
            examples_set = 1;
            break;
        case 'o':
            output_dir = optarg;
            break;
        case 'h': // help
        default:
            video_color_help(argv[0]);
            break;
        }
    }

    // client
    if (optind == argc) // no input image, nothing to do ...
        return 0;

    VideoXMemNetwork xmem_net;
    VideoColorNetwork color_net;

    // int network
    {
        xmem_net.init(device_no);
        color_net.init(device_no);
    }

    if (examples_set) {
        for (int i = 0; i < examples.gl_pathc; i++) {
            printf("examples: %s\n", examples.gl_pathv[i]);
        }
        printf("--------------\n");
    } else {
        printf("--examples (-e) are required.\n");
        video_color_help(argv[0]);
    }


    for (int i = optind; i < argc; i++) {
        p = strrchr(argv[i], '/');
        if (p != NULL) {
            snprintf(output_filename, sizeof(output_filename), "%s/%s", output_dir, p + 1);
        } else {
            snprintf(output_filename, sizeof(output_filename), "%s/%s", output_dir, argv[i]);
        }
        printf("images: %s\n", argv[i]);

        // video_color_predict(&color_net, argv[i], output_filename);
    }

    // free network ...
    {
        color_net.exit();
        xmem_net.exit();
    }

    if (examples_set) {
        globfree(&examples);
    }
    return 0;
}
