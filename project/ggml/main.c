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
// #include <glob.h>

#include <ggml_engine.h>
#include <nimage/tensor.h>

// #include "video_color.h"
#include "video_xmem.h"
#include "video_color.h"

#define DEFAULT_DEVICE 1
#define DEFAULT_OUTPUT "output"

int video_color_predict(VideoColorNetwork *color_net, VideoXMemNetwork *xmem_net, char *example_files, char *gray_files, char *output_dir);

static void video_color_help(char* cmd)
{
    printf("Usage: %s [option] gray_files\n", cmd);
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
    char *example_files = NULL;

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
            example_files = optarg;
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

    if (! example_files) {
        printf("--examples (-e) are required.\n");
        video_color_help(argv[0]);
    }

    if (optind < argc - 1) {
        printf("gray_files are required.\n");
        video_color_help(argv[0]);
    }


    for (int i = optind; i < argc; i++) {
        video_color_predict(&color_net, &xmem_net, example_files, argv[i], output_dir);
    }

    // free network ...
    {
        color_net.exit();
        xmem_net.exit();
    }

    return 0;
}
