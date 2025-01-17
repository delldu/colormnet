#ifndef __COLORMNET__H__
#define __COLORMNET__H__
#include "ggml_engine.h"
#include "ggml_nn.h"

#pragma GCC diagnostic ignored "-Wformat-truncation"
// x = ggml_cont(ctx, x);
// ggml_set_name(x, "x");
// ggml_set_output(x);

struct GroupResBlock {
    int in_dim = 256;
    int out_dim = 256;

    struct Conv2d conv1;
    struct Conv2d conv2;
    struct Conv2d downsample;

    void create_weight_tensors(struct ggml_context* ctx) {
        if (in_dim != out_dim) {
            downsample.in_channels = in_dim;
            downsample.out_channels = out_dim;
            downsample.kernel_size = {3, 3};
            downsample.stride = { 1, 1 };
            downsample.padding = { 1, 1 };
            downsample.create_weight_tensors(ctx);
        }

        conv1.in_channels = in_dim;
        conv1.out_channels = out_dim;
        conv1.kernel_size = {3, 3};
        conv1.stride = { 1, 1 };
        conv1.padding = { 1, 1 };
        conv1.create_weight_tensors(ctx);

        conv2.in_channels = out_dim;
        conv2.out_channels = out_dim;
        conv2.kernel_size = {3, 3};
        conv2.stride = { 1, 1 };
        conv2.padding = { 1, 1 };
        conv2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        if (in_dim != out_dim) {
            snprintf(s, sizeof(s), "%s%s", prefix, "downsample.");
            downsample.setup_weight_names(s);
        }

        snprintf(s, sizeof(s), "%s%s", prefix, "conv1.");
        conv1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv2.");
        conv2.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* g) {
        ggml_tensor_t *out_g;
        out_g = ggml_relu(ctx, g);
        out_g = conv1.forward(ctx, out_g);
        out_g = ggml_relu(ctx, out_g);
        out_g = conv2.forward(ctx, out_g);

        if (in_dim != out_dim) {
            // g    f32 [56, 35, 1600, 2], 
            g = downsample.forward(ctx, g);
            // g    f32 [56, 35, 512, 2], 
        }
        out_g = ggml_add(ctx, out_g, g);

        return out_g;
    }
};

struct UpsampleBlock {
    int skip_dim = 256;
    int g_up_dim = 256;
    const int g_out_dim = 256;
    
    // network params
    struct Conv2d skip_conv;
    struct GroupResBlock out_conv;

    void create_weight_tensors(struct ggml_context* ctx) {
        skip_conv.in_channels = skip_dim;
        skip_conv.out_channels = g_up_dim;
        skip_conv.kernel_size = {3, 3};
        skip_conv.stride = { 1, 1 };
        skip_conv.padding = { 1, 1 };
        skip_conv.create_weight_tensors(ctx);

        out_conv.in_dim = g_up_dim;
        out_conv.out_dim = g_out_dim;
        out_conv.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "skip_conv.");
        skip_conv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "out_conv.");
        out_conv.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* skip_f, ggml_tensor_t* up_g) {
        int W = (int)up_g->ne[0];
        int H = (int)up_g->ne[1];
        int B = (int)up_g->ne[3];

        skip_f = skip_conv.forward(ctx, skip_f);

        ggml_tensor_t *g;
        g = ggml_interpolate(ctx, up_g, 0, 2*W);
        g = ggml_interpolate(ctx, g, 1, 2*H);

        skip_f = ggml_repeat_ext(ctx, skip_f, 1, 1, 1, B);
        g = ggml_add(ctx, skip_f, g);
        g = out_conv.forward(ctx, g);

    	return g;
    }
};

// // useless
// struct HiddenUpdater {
//     const int hidden_dim = 64;

//     struct Conv2d g16_conv;
//     struct Conv2d g8_conv;
//     struct Conv2d g4_conv;
//     struct Conv2d transform;

//     void create_weight_tensors(struct ggml_context* ctx) {
//         g16_conv.in_channels = 512;
//         g16_conv.out_channels = 256;
//         g16_conv.kernel_size = {1, 1};
//         g16_conv.stride = { 1, 1 };
//         g16_conv.padding = { 0, 0 };
//         g16_conv.create_weight_tensors(ctx);

//         g8_conv.in_channels = 256;
//         g8_conv.out_channels = 256;
//         g8_conv.kernel_size = {1, 1};
//         g8_conv.stride = { 1, 1 };
//         g8_conv.padding = { 0, 0 };
//         g8_conv.create_weight_tensors(ctx);

//         g4_conv.in_channels = 257;
//         g4_conv.out_channels = 256;
//         g4_conv.kernel_size = {1, 1};
//         g4_conv.stride = { 1, 1 };
//         g4_conv.padding = { 0, 0 };
//         g4_conv.create_weight_tensors(ctx);

//         transform.in_channels = 320; // mid_dim + hidden_dim;
//         transform.out_channels =  192; // hidden_dim*3;
//         transform.kernel_size = {3, 3};
//         transform.stride = { 1, 1 };
//         transform.padding = { 1, 1 };
//         transform.create_weight_tensors(ctx);
//     }

//     void setup_weight_names(const char *prefix) {
//         char s[GGML_MAX_NAME];

//         snprintf(s, sizeof(s), "%s%s", prefix, "g16_conv.");
//         g16_conv.setup_weight_names(s);

//         snprintf(s, sizeof(s), "%s%s", prefix, "g8_conv.");
//         g8_conv.setup_weight_names(s);

//         snprintf(s, sizeof(s), "%s%s", prefix, "g4_conv.");
//         g4_conv.setup_weight_names(s);

//         snprintf(s, sizeof(s), "%s%s", prefix, "transform.");
//         transform.setup_weight_names(s);
//     }

//     ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* g0, ggml_tensor_t* g1, ggml_tensor_t* g2, 
//         ggml_tensor_t* h) {
//         // # g is list: len = 3
//         // #     tensor [item] size: [2, 512, 35, 56], min: -82.737953, max: 16.419943, mean: -1.540743
//         // #     tensor [item] size: [2, 256, 70, 112], min: -27.510799, max: 26.222929, mean: -1.975606
//         // #     tensor [item] size: [2, 257, 140, 224], min: -33.898613, max: 25.28302, mean: -7.281233
//         // # tensor [h] size: [2, 64, 35, 56], min: -3.735293, max: 4.35715, mean: 0.105984

//         int H, W;
//         ggml_tensor_t *g, *forget_gate, *update_gate, *new_value, *new_h;

//         // g = self.g16_conv(g[0]) + \
//         //     self.g8_conv(F.interpolate(g[1], scale_factor=0.5, mode='area', align_corners=None)) + \
//         //     self.g4_conv(F.interpolate(g[2], scale_factor=0.25, mode='area', align_corners=None))

//         g0 = g16_conv.forward(ctx, g0);

//         W = (int)g1->ne[0];
//         H = (int)g1->ne[1];
//         g1 = ggml_interpolate(ctx, g1, 0/*on W*/, W/2);
//         g1 = ggml_interpolate(ctx, g1, 1/*on H*/, H/2);
//         g1 = g8_conv.forward(ctx, g1);

//         W = (int)g2->ne[0];
//         H = (int)g2->ne[1];
//         g2 = ggml_interpolate(ctx, g2, 0/*on W*/, W/4);
//         g2 = ggml_interpolate(ctx, g2, 1/*on H*/, H/4);
//         g2 = g4_conv.forward(ctx, g2);

//         // ---- g0    f32 [56, 35, 256, 2], 
//         // ---- g1    f32 [56, 35, 256, 2], 
//         // ---- g2    f32 [56, 35, 256, 2], 
//         g = ggml_add(ctx, g0, g1);
//         g = ggml_add(ctx, g, g2);

//         // # tensor [h] size: [2, 64, 35, 56], min: -3.735293, max: 4.35715, mean: 0.105984
//         g = ggml_concat(ctx, g, h, 2/*dim on channel*/);

//         // g = torch.cat([g, h], dim=1)
//         g = transform.forward(ctx, g);
//         // # tensor [g] size: [2, 192, 35, 56], min: -46.875889, max: 65.977158, mean: 4.075093

//         forget_gate = ggml_sigmoid(ctx, ggml_nn_slice(ctx, g, 2/*dim*/, 0, hidden_dim, 1/*step*/));
//         update_gate = ggml_sigmoid(ctx, ggml_nn_slice(ctx, g, 2/*dim*/, hidden_dim, 2*hidden_dim, 1/*step*/));
//         new_value = ggml_tanh(ctx, ggml_nn_slice(ctx, g, 2/*dim*/, 2*hidden_dim, 3*hidden_dim, 1/*step*/));

//         g = ggml_dup(ctx, update_gate);
//         g = ggml_constant(ctx, g, 1.0f);
//         g = ggml_sub(ctx, g, update_gate);
//         g = ggml_mul(ctx, h, g);

//         forget_gate = ggml_mul(ctx, forget_gate, g);
//         update_gate = ggml_mul(ctx, update_gate, new_value);

//         new_h = ggml_add(ctx, forget_gate, update_gate);
//     	return new_h;
//     }
// };

struct Flatten {
    void create_weight_tensors(struct ggml_context* ctx) {
        GGML_UNUSED(ctx);
    }

    void setup_weight_names(const char *prefix) {
        GGML_UNUSED(prefix);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        int WHC = (int)x->ne[0] * x->ne[1] * x->ne[2];
        int B = (int)x->ne[3];
        x = ggml_reshape_2d(ctx, x, WHC, B);
        return x;
    }
};

struct ChannelGate {
    int gate_channels = 512;

    // network params
    struct Flatten mlp_0;
    struct Linear mlp_1;
    struct Linear mlp_3;

    void create_weight_tensors(struct ggml_context* ctx) {
        mlp_0.create_weight_tensors(ctx);

        mlp_1.in_features = gate_channels;
        mlp_1.out_features = gate_channels/16;
        mlp_1.has_bias = true; // Fixed default
        mlp_1.create_weight_tensors(ctx);

        mlp_3.in_features = gate_channels/16;
        mlp_3.out_features = gate_channels;
        mlp_3.has_bias = true; // Fixed default
        mlp_3.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "mlp.1.");
        mlp_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "mlp.3.");
        mlp_3.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        ggml_tensor_t *avg, *max, *scale;
        // tensor [x] size: [2, 512, 35, 56], min: -8.184954, max: 4.269275, mean: -0.079296
        int W = (int)x->ne[0];
        int H = (int)x->ne[1];
        int C = (int)x->ne[2];
        int B = (int)x->ne[3];

        avg = ggml_pool_2d(ctx, x, GGML_OP_POOL_AVG, W, H, W, H, (float)0.0, (float)0.0);
        // tensor [avg] size: [2, 512, 1, 1], min: -1.965386, max: 1.042446, mean: -0.079296
        avg = mlp_0.forward(ctx, avg);
        avg = mlp_1.forward(ctx, avg);
        avg = ggml_relu(ctx, avg);
        avg = mlp_3.forward(ctx, avg);

        max = ggml_pool_2d(ctx, x, GGML_OP_POOL_MAX, W, H, W, H, (float)0.0, (float)0.0);
        // tensor [max_pool] size: [2, 512, 1, 1], min: 0.13838, max: 4.269275, mean: 1.23257
        max = mlp_0.forward(ctx, max);
        max = mlp_1.forward(ctx, max);
        max = ggml_relu(ctx, max);
        max = mlp_3.forward(ctx, max);

        scale = ggml_add(ctx, avg, max);
        // tensor [scale] size: [2, 512], min: -4.608468, max: 2.956863, mean: -0.669225

        // scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        scale = ggml_sigmoid(ctx, scale);
        scale = ggml_reshape_4d(ctx, scale, 1, 1, C, B);
        scale = ggml_repeat_ext(ctx, scale, W, H, 1, 1);

        x = ggml_mul(ctx, x, scale);
        // tensor [x] size: [2, 512, 35, 56], min: -5.115581, max: 2.511071, mean: -0.022023
        return x;
    }
};

struct BasicConv {
    struct Conv2d conv;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv.in_channels = 2;
        conv.out_channels = 1;
        conv.kernel_size = {7, 7};
        conv.stride = { 1, 1 };
        conv.padding = { 3, 3 };
        conv.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.");
        conv.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        return conv.forward(ctx, x);
    }
};

struct ChannelPool {
    void create_weight_tensors(struct ggml_context* ctx) {
        GGML_UNUSED(ctx);        
    }

    void setup_weight_names(const char *prefix) {
        GGML_UNUSED(prefix);        
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        ggml_tensor_t *m1 = ggml_max(ctx, x, 2/*dim on channel*/); 
        ggml_tensor_t *m2 = ggml_mean_ext(ctx, x, 2/*dim on channel*/);

        return ggml_concat(ctx, m1, m2, 2/*dim on channel*/);
    }
};

struct SpatialGate {
    struct ChannelPool compress;
    struct BasicConv spatial;

    void create_weight_tensors(struct ggml_context* ctx) {
        compress.create_weight_tensors(ctx);
        spatial.create_weight_tensors(ctx);
    }


    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "compress.");
        compress.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "spatial.");
        spatial.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        ggml_tensor_t *x_out = compress.forward(ctx, x);
        x_out = spatial.forward(ctx, x_out);
        x_out = ggml_sigmoid(ctx, x_out);

        return ggml_mul(ctx, x, x_out);
    }
};

struct CBAM {
    int gate_channels = 512;

    struct ChannelGate channel_gate;
    struct SpatialGate spatial_gate;

    void create_weight_tensors(struct ggml_context* ctx) {
        channel_gate.gate_channels = gate_channels;
        channel_gate.create_weight_tensors(ctx);

        spatial_gate.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "ChannelGate.");
        channel_gate.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "SpatialGate.");
        spatial_gate.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // x_out = self.ChannelGate(x)
        // x_out = self.SpatialGate(x_out)
        // return x_out
        x = channel_gate.forward(ctx, x);
        x = spatial_gate.forward(ctx, x);
        return x;
    }
};

struct FeatureFusionBlock {
    int g_in_dim = 512;

    const int x_in_dim = 1024;
    const int g_mid_dim = 512;
    const int g_out_dim = 512;

    // network params
    struct GroupResBlock block1;
    struct CBAM attention;
    struct GroupResBlock block2;

    void create_weight_tensors(struct ggml_context* ctx) {
        block1.in_dim = x_in_dim + g_in_dim;
        block1.out_dim = g_mid_dim;
        block1.create_weight_tensors(ctx);

        attention.gate_channels = g_mid_dim;
        attention.create_weight_tensors(ctx);

        block2.in_dim = g_mid_dim;
        block2.out_dim = g_out_dim;
        block2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "block1.");
        block1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "attention.");
        attention.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "block2.");
        block2.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x, ggml_tensor_t* g) {
        ggml_tensor_t *r;
        int B = g->ne[3];
        x = ggml_repeat_ext(ctx, x, 1, 1, 1, B);
        g = ggml_concat(ctx, x, g, 2/*dim on Channel*/);
        g = block1.forward(ctx, g);
        r = attention.forward(ctx, g);
        g = ggml_add(ctx, g, r);
        g = block2.forward(ctx, g);
        return g;
    }
};

struct ColorDecoder : GGMLNetwork {
    int val_dim = 512;
    int hidden_dim = 64;

    // network params
    struct FeatureFusionBlock fuser;
    // struct HiddenUpdater hidden_update; // useless ...
    struct UpsampleBlock up_16_8;
    struct UpsampleBlock up_8_4;
    struct Conv2d pred;

    void create_weight_tensors(struct ggml_context* ctx) {
        // self.fuser = FeatureFusionBlock(val_dim+hidden_dim) 
        fuser.g_in_dim = val_dim + hidden_dim;
        fuser.create_weight_tensors(ctx);

        // self.hidden_update = HiddenUpdater([512, 256, 256+1], 256, hidden_dim) 
        // hidden_update.create_weight_tensors(ctx);

        // self.up_16_8 = UpsampleBlock(512, 512)
        up_16_8.skip_dim = 512;
        up_16_8.g_up_dim = 512;
        up_16_8.create_weight_tensors(ctx);

        // self.up_8_4 = UpsampleBlock(256, 256)
        up_8_4.skip_dim = 256;
        up_8_4.g_up_dim = 256;
        up_8_4.create_weight_tensors(ctx);

        // self.pred = nn.Conv2d(256, 1, kernel_size=3, padding=1, stride=1)
        pred.in_channels = 256;
        pred.out_channels = 1;
        pred.kernel_size = {3, 3};
        pred.stride = { 1, 1 };
        pred.padding = { 1, 1 };
        pred.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "fuser.");
        fuser.setup_weight_names(s);

        // snprintf(s, sizeof(s), "%s%s", prefix, "hidden_update.");
        // hidden_update.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "up_16_8.");
        up_16_8.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "up_8_4.");
        up_8_4.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "pred.");
        pred.setup_weight_names(s);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, int argc, ggml_tensor_t* argv[]) {
        GGML_ASSERT(argc == 5);
        ggml_tensor_t* f16 = argv[0];
        ggml_tensor_t* f8 = argv[1];
        ggml_tensor_t* f4 = argv[2];
        ggml_tensor_t* color_feature = argv[3];
        ggml_tensor_t* hidden_state = argv[4];

        ggml_tensor_t *g16, *g8, *g4, *logits, *color_ab;

        g16 = ggml_concat(ctx, color_feature, hidden_state, 2/*dim on channel*/);
        g16 = fuser.forward(ctx, f16, g16);
        g8 = up_16_8.forward(ctx, f8, g16);

        g4 = up_8_4.forward(ctx, f4, g8);

        logits = pred.forward(ctx, ggml_relu(ctx, g4));

        int W = (int)logits->ne[0];
        int H = (int)logits->ne[1];
        logits = ggml_interpolate(ctx, logits, 0/*on W*/, 4*W);
        logits = ggml_interpolate(ctx, logits, 1/*on H*/, 4*H);

        logits = ggml_cont(ctx, ggml_permute(ctx, logits, 0, 1, 3, 2)); // [W, H, B, C] -> [W, H, C, B]
        color_ab = ggml_tanh(ctx, logits);

        return color_ab;
    }
};

struct DWConv2d {
    int indim = 1024;

    struct Conv2d conv;  // torch.float32, [1024, 1, 5, 5]

    void create_weight_tensors(struct ggml_context* ctx) {
        conv.in_channels = indim;
        conv.out_channels = indim;
        conv.kernel_size = {5, 5};
        conv.stride = { 1, 1 };
        conv.padding = { 2, 2 };
        conv.is_depthwise = true;
        conv.has_bias = false;

        conv.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.");
        conv.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x, int H, int W) {
        //# tensor [x] size: [1960, 1, 1024], min: -9.280723, max: 4.028006, mean: -0.008225

        int B = (int)x->ne[1];
        int C = (int)x->ne[0]; // 1024
        // int HW = (int)x->ne[2];
        x = ggml_reshape_4d(ctx, x, B, C, H, W);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 3, 2, 1, 0)); // [B, C, H, W] --> [W, H, C, B]
        x = conv.forward(ctx, x); // Depthwise Conv2d
        x = ggml_cont(ctx, ggml_permute(ctx, x, 3, 2, 0, 1)); // [W, H, C, B] -> [C, B, H, W]
        x = ggml_reshape_3d(ctx, x, C, B, H*W);
    	return x;
    }
};


struct LocalAttention : GGMLNetwork {
    const int window_size = 15;
    const int hidden_dim = 512;
    const int d_att = 64;
    int max_dis = window_size/2; // 7

    // network params
    struct Conv2d relative_emb_k;
    struct DWConv2d dw_conv;

    struct Linear projection;  // torch.float32, [1024, 1024] 

    void create_weight_tensors(struct ggml_context* ctx) {
        relative_emb_k.in_channels = d_att;
        relative_emb_k.out_channels = window_size * window_size;
        relative_emb_k.kernel_size = {1, 1};
        relative_emb_k.stride = { 1, 1 };
        relative_emb_k.padding = { 0, 0 };
        relative_emb_k.create_weight_tensors(ctx);


        dw_conv.indim = 2*hidden_dim; // d_uv, 1024
        dw_conv.create_weight_tensors(ctx);

        projection.in_features = 2*hidden_dim;
        projection.out_features = 2*hidden_dim;
        projection.has_bias = true; // Fixed default
        projection.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "relative_emb_k.");
        relative_emb_k.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "dw_conv.");
        dw_conv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "projection.");
        projection.setup_weight_names(s);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, int argc, ggml_tensor_t* argv[]) {
        GGML_ASSERT(argc == 3);
        ggml_tensor_t* q = argv[0];
        ggml_tensor_t* k = argv[1];
        ggml_tensor_t* v = argv[2];
        // # tensor [q] size: [1, 64, 35, 56], min: -2.75, max: 3.162109, mean: -0.143807
        // # tensor [k] size: [1, 64, 35, 56], min: -2.755859, max: 3.140625, mean: -0.143588
        // # tensor [v] size: [2, 512, 35, 56], min: -9.921875, max: 5.101562, mean: -0.014141

        int W = (int)v->ne[0];
        int H = (int)v->ne[1];
        int C = (int)v->ne[2];
        int B = (int)v->ne[3];

        // # Scale
        q = ggml_scale(ctx, q, 0.125f); // sqrtf(d_att);

        ggml_tensor_t *qk = ggml_corr(ctx, q, k, window_size); // [225, 35, 56]
        // # tensor [qk] size: [15, 15, 35, 56], min: 0.0, max: 5.730469, mean: 3.024395

        qk = ggml_reshape_3d(ctx, qk, H*W, window_size * window_size, 1); // [1960, 225, 1]
        ggml_tensor_t *rel_emb = relative_emb_k.forward(ctx, q);
        rel_emb = ggml_reshape_3d(ctx, rel_emb, H * W, window_size * window_size, 1);
        qk = ggml_add(ctx, qk, rel_emb);
        // # tensor [qk] size: [1, 225, 1960], min: -0.646284, max: 6.405645, mean: 3.097403

        ggml_tensor_t *local_attn = ggml_softmax(ctx, qk, 1/*dim*/); // [1960, 225, 1]
        local_attn = ggml_cont(ctx, ggml_transpose(ctx, local_attn)); // [225, 1960, 1]
        local_attn = ggml_reshape_1d(ctx, local_attn, -1);

        ggml_tensor_t *global_attn = ggml_global_attn(ctx, local_attn, H, W, max_dis); // [70x49, 35x56]

        // global_attn = global_attn.view(1, H * W, H + 2*MAX_DISP, W + 2*MAX_DISP)
        global_attn = ggml_reshape_3d(ctx, global_attn, W+2*max_dis, H+2*max_dis, H*W);
        global_attn = ggml_nn_slice(ctx, global_attn, 0/*on W*/, max_dis, W + max_dis, 1/*step*/);
        global_attn = ggml_nn_slice(ctx, global_attn, 1/*on H*/, max_dis, H + max_dis, 1/*step*/);
        global_attn = ggml_reshape_3d(ctx, global_attn, H*W, H*W, 1);
        // # tensor [global_attn] size: [1, 1960, 1960], min: 0.0, max: 0.511266, mean: 0.00051

        v = ggml_reshape_4d(ctx, v, H*W, 2*hidden_dim, 1, -1);
        // # tensor [v] size: [1, 1, 1024, 1960], min: -11.473879, max: 5.339447, mean: -0.008071

        v = ggml_cont(ctx, ggml_transpose(ctx, v));
        ggml_tensor_t *agg_value = ggml_nn_mul_mat(ctx, global_attn, v); // [512, HW, 1]

        // tensor [agg_value] size: [1, 1, 1960, 1024], min: -5.614281, max: 2.84555, mean: -0.008132
        agg_value = ggml_cont(ctx, ggml_permute(ctx, agg_value, 0, 3, 1, 2)); // [1024, 1960, 1, 1] -> [1024, 1, 1, 1960]
        // tensor [agg_value] size: [1960, 1, 1, 1024], min: -5.614281, max: 2.84555, mean: -0.008132
        agg_value = ggml_reshape_3d(ctx, agg_value, 2*hidden_dim, 1, H*W);
        // tensor [agg_value] size: [1960, 1, 1024], min: -5.614281, max: 2.84555, mean: -0.008132

        ggml_tensor_t *output = dw_conv.forward(ctx, agg_value, H, W);

        output = projection.forward(ctx, output);
        output = ggml_cont(ctx, ggml_permute(ctx, output, 1, 2, 0, 3)); // [1024, 1, 1960] -->[1960, 1024, 1]
        output = ggml_reshape_4d(ctx, output, W, H, hidden_dim, 2);

        return output; // [56, 35, 512, 2]
    }
};

struct HiddenReinforcer {
    int g_dim = 512;
    int hidden_dim = 64;

    struct Conv2d transform;

    void create_weight_tensors(struct ggml_context* ctx) {
        transform.in_channels = g_dim + hidden_dim;
        transform.out_channels = hidden_dim * 3;
        transform.kernel_size = {3, 3};
        transform.stride = { 1, 1 };
        transform.padding = { 1, 1 };
        transform.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "transform.");
        transform.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* g, ggml_tensor_t* h) {
        ggml_tensor_t *forget_gate, *update_gate, *new_h, *t;
        g = ggml_concat(ctx, g, h, 2/*dim on C*/);
        g = transform.forward(ctx, g);
        t = ggml_nn_slice(ctx, g, 2/*dim on C*/, 0, hidden_dim, 1/*step*/);
        forget_gate = ggml_sigmoid(ctx, t);
        t = ggml_nn_slice(ctx, g, 2/*dim on C*/, hidden_dim, 2*hidden_dim, 1/*step*/);
        update_gate = ggml_sigmoid(ctx, t);
        t = ggml_nn_slice(ctx, g, 2/*dim on C*/, 2*hidden_dim, 3*hidden_dim, 1/*step*/);
        g = ggml_tanh(ctx, t);

        t = ggml_dup(ctx, update_gate);
        t = ggml_constant(ctx, t, 1.0f);
        t = ggml_sub(ctx, t, update_gate);
        h = ggml_mul(ctx, h, t);
        forget_gate = ggml_mul(ctx, forget_gate, h);
        update_gate = ggml_mul(ctx, update_gate, g);

        new_h = ggml_add(ctx, forget_gate, update_gate);
        return new_h;
    }
};

struct BasicBlock {
    int inplanes;
    int planes;
    int stride = 1;
    bool downsample = false;
    const int expansion = 1; // Fixed

    // network params
    struct Conv2d conv1;
    struct BatchNorm2d bn1;

    struct Conv2d conv2;
    struct BatchNorm2d bn2;

    struct Conv2d downsample_conv;
    struct BatchNorm2d downsample_bn;

    void create_weight_tensors(struct ggml_context* ctx) {
        // self.conv1 = conv3x3(inplanes, planes, stride=stride)
        // self.bn1 = nn.BatchNorm2d(planes)

        conv1.in_channels = inplanes;
        conv1.out_channels = planes;
        conv1.kernel_size = {3, 3};
        conv1.stride = { stride, stride };
        conv1.padding = { 1, 1 };
        conv1.has_bias = false;
        conv1.create_weight_tensors(ctx);

        bn1.num_features = planes;
        bn1.create_weight_tensors(ctx);

        // self.conv2 = conv3x3(planes, planes, stride=1)
        // self.bn2 = nn.BatchNorm2d(planes)
        conv2.in_channels = planes;
        conv2.out_channels = planes;
        conv2.kernel_size = {3, 3};
        conv2.stride = { 1, 1 };
        conv2.padding = { 1, 1 };
        conv2.has_bias = false;
        conv2.create_weight_tensors(ctx);

        bn2.num_features = planes;
        bn2.create_weight_tensors(ctx);

        if (downsample) {
            downsample = (stride != 1 || inplanes != planes * expansion);
        }
        if (downsample) {
            // self.downsample = nn.Sequential(
            //     nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
            //     nn.BatchNorm2d(planes * self.expansion))
            downsample_conv.in_channels = inplanes;
            downsample_conv.out_channels = planes * expansion;
            downsample_conv.kernel_size = {1, 1};
            downsample_conv.stride = { stride, stride };
            downsample_conv.padding = { 0, 0 };
            downsample_conv.has_bias = false;
            downsample_conv.create_weight_tensors(ctx);

            downsample_bn.num_features = planes * expansion;
            downsample_bn.create_weight_tensors(ctx);
        }
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv1.");
        conv1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "bn1.");
        bn1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv2.");
        conv2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "bn2.");
        bn2.setup_weight_names(s);

        if (downsample) {
            snprintf(s, sizeof(s), "%s%s", prefix, "downsample.0.");
            downsample_conv.setup_weight_names(s);

            snprintf(s, sizeof(s), "%s%s", prefix, "downsample.1.");
            downsample_bn.setup_weight_names(s);
        }
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        ggml_tensor_t *residual = x;
        if (downsample) {
            residual = downsample_conv.forward(ctx, x);
            residual = downsample_bn.forward(ctx, residual);
        }
        x = conv1.forward(ctx, x);
        x = bn1.forward(ctx, x);
        x = ggml_relu(ctx, x);
        x = conv2.forward(ctx, x);
        x = bn2.forward(ctx, x);

        x = ggml_add(ctx, x, residual);
        x = ggml_relu(ctx, x);

    	return x;
    }
};

// make_basicblock_layer
struct BasicBlockLayer {
    int inplanes;
    int planes;
    int stride = 1;
    const int blocks = 2;

    struct BasicBlock layers[2];

    void create_weight_tensors(struct ggml_context* ctx) {
        GGML_ASSERT(ARRAY_SIZE(layers) >= blocks);

        layers[0].inplanes = inplanes;
        layers[0].planes = planes;
        layers[0].stride = stride;
        layers[0].downsample = true;
        layers[0].create_weight_tensors(ctx);

        for (int i = 1; i < blocks; i++) {
            layers[i].inplanes = planes;
            layers[i].planes = planes;
            layers[i].stride = 1;
            layers[i].downsample = false;
            layers[i].create_weight_tensors(ctx);
        }
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "0.");
        layers[0].setup_weight_names(s);

        for (int i = 1; i < blocks; i++) {
            snprintf(s, sizeof(s), "%s%d.", prefix, i);
            layers[i].setup_weight_names(s);
        }
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        for (int i = 0; i < blocks; i++) {
            x = layers[i].forward(ctx, x);
        }

        return x;
    }
};


struct ValueEncoder : GGMLNetwork {
    int value_dim = 512;
    int hidden_dim = 64;
    
    // network params
    struct Conv2d conv1;
    struct BatchNorm2d bn1;
    struct MaxPool2d maxpool;

    struct BasicBlockLayer layer1;
    struct BasicBlockLayer layer2;
    struct BasicBlockLayer layer3;

    struct FeatureFusionBlock fuser;
    struct HiddenReinforcer hidden_reinforce;

    void create_weight_tensors(struct ggml_context* ctx) {
        // self.conv1 = nn.Conv2d(3 + 2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        conv1.in_channels = 5;
        conv1.out_channels = 64;
        conv1.kernel_size = {7, 7};
        conv1.stride = { 2, 2 };
        conv1.padding = { 3, 3 };
        conv1.has_bias = false;
        conv1.create_weight_tensors(ctx);

        bn1.num_features = 64;
        bn1.create_weight_tensors(ctx);

        // self.layer1 = resnet.make_basicblock_layer(64, 64, 2, stride=1)
        layer1.inplanes = 64;
        layer1.planes = 64;
        // layer1.blocks = 2;
        layer1.stride = 1;
        layer1.create_weight_tensors(ctx);

        // self.layer2 = resnet.make_basicblock_layer(64, 128, 2, stride=2)
        layer2.inplanes = 64;
        layer2.planes = 128;
        // layer2.blocks = 2;
        layer2.stride = 2;
        layer2.create_weight_tensors(ctx);

        // self.layer3 = resnet.make_basicblock_layer(128, 256, 2, stride=2)
        layer3.inplanes = 128;
        layer3.planes = 256;
        // layer1.blocks = 2;
        layer3.stride = 2;
        layer3.create_weight_tensors(ctx);

        // self.fuser = FeatureFusionBlock(value_dim)
        fuser.g_in_dim = 256;
        fuser.create_weight_tensors(ctx);

        // self.hidden_reinforce = HiddenReinforcer()
        hidden_reinforce.create_weight_tensors(ctx);

        maxpool.kernel_size = 3;
        maxpool.stride = 2;
        maxpool.padding = 1;
        maxpool.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv1.");
        conv1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "bn1.");
        bn1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "layer1.");
        layer1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer2.");
        layer2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer3.");
        layer3.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "fuser.");
        fuser.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "hidden_reinforce.");
        hidden_reinforce.setup_weight_names(s);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, int argc, ggml_tensor_t* argv[]) {
        GGML_ASSERT(argc == 4);
        ggml_tensor_t* image = argv[0]; // image_lll
        ggml_tensor_t* f16 = argv[1];
        ggml_tensor_t* h16 = argv[2];
        ggml_tensor_t* ref_ab = argv[3];        

        ggml_tensor_t* s0, *s1, *ref_ba, *g, *h;
        s0 = ggml_nn_slice(ctx, ref_ab, 2/*dim on C*/, 0, 1, 1/*step*/);
        s1 = ggml_nn_slice(ctx, ref_ab, 2/*dim on C*/, 1, 2, 1/*step*/);
        ref_ba = ggml_concat(ctx, s1, s0, 2/*dim on C*/);
        g = ggml_concat(ctx, ref_ab, ref_ba, 3/*dim on B*/);
        // int W = (int)g->ne[0];
        // int H = (int)g->ne[1];
        // int C = (int)g->ne[2];
        int B = (int)g->ne[3];

        image = ggml_repeat_ext(ctx, image, 1, 1, 1, B);
        g = ggml_concat(ctx, image, g, 2/*dim on C*/);

        g = conv1.forward(ctx, g);
        g = bn1.forward(ctx, g);
        g = maxpool.forward(ctx, g);
        g = ggml_relu(ctx, g);

        g = layer1.forward(ctx, g);
        g = layer2.forward(ctx, g);
        g = layer3.forward(ctx, g);

        int W = (int)f16->ne[0];
        int H = (int)f16->ne[1];
        g = ggml_interpolate(ctx, g, 0/*on W*/, W);
        g = ggml_interpolate(ctx, g, 1/*on H*/, H);
        g = fuser.forward(ctx, f16, g);
        h = hidden_reinforce.forward(ctx, g, h16);
        // Save g, h
        {
            g = ggml_cont(ctx, g);
            ggml_set_name(g, "VALUE");
            ggml_set_output(g);

            h = ggml_cont(ctx, h);
            ggml_set_name(h, "HIDDEN");
            ggml_set_output(h);
        }

        return ggml_package(ctx, 2, g, h);
    }
};

struct KeyProjection {
    // network params
    struct Conv2d key_proj;
    struct Conv2d d_proj;
    struct Conv2d e_proj;

    void create_weight_tensors(struct ggml_context* ctx) {
        key_proj.in_channels = 1024;
        key_proj.out_channels = 64;
        key_proj.kernel_size = {3, 3};
        key_proj.stride = { 1, 1 };
        key_proj.padding = { 1, 1 };
        key_proj.create_weight_tensors(ctx);

        d_proj.in_channels = 1024;
        d_proj.out_channels = 1;
        d_proj.kernel_size = {3, 3};
        d_proj.stride = { 1, 1 };
        d_proj.padding = { 1, 1 };
        d_proj.create_weight_tensors(ctx);

        e_proj.in_channels = 1024;
        e_proj.out_channels = 64;
        e_proj.kernel_size = {3, 3};
        e_proj.stride = { 1, 1 };
        e_proj.padding = { 1, 1 };
        e_proj.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "key_proj.");
        key_proj.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "d_proj.");
        d_proj.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "e_proj.");
        e_proj.setup_weight_names(s);                
    }

    ggml_tensor_t * forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // key = self.key_proj(x)
        // shrinkage = self.d_proj(x)**2 + 1
        // selection = torch.sigmoid(self.e_proj(x))
        // return key, shrinkage, selection

        ggml_tensor_t *key = key_proj.forward(ctx, x);
        ggml_tensor_t *shrinkage = d_proj.forward(ctx, x);
        shrinkage = ggml_mul(ctx, shrinkage, shrinkage);
        shrinkage = ggml_add_constant(ctx, shrinkage, 1.0);
        ggml_tensor_t *selection = e_proj.forward(ctx, x);
        selection = ggml_sigmoid(ctx, selection);
        // Save shrinkage, selection
        {
            key = ggml_cont(ctx, key);
            ggml_set_name(key, "KEY");
            ggml_set_output(key);

            shrinkage = ggml_cont(ctx, shrinkage);
            ggml_set_name(shrinkage, "SHRINKAGE");
            ggml_set_output(shrinkage);

            selection = ggml_cont(ctx, selection);
            ggml_set_name(selection, "SELECTION");
            ggml_set_output(selection);
        }

        return ggml_package(ctx, 3, key, shrinkage, selection);
    }
};

// --------------------------------------------------------------------------------------------------
struct CrossChannelAttention {
    int dim = 256;
    int heads = 8;

    // network params
    ggml_tensor_t *temperature;

    struct Conv2d to_q;
    struct Conv2d to_q_dw;
    struct Conv2d to_k;
    struct Conv2d to_k_dw;
    struct Conv2d to_v;
    struct Conv2d to_v_dw;
    struct Conv2d to_out;

    void create_weight_tensors(struct ggml_context* ctx) {
        temperature = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, heads);

        // self.to_q = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True)
        to_q.in_channels = dim;
        to_q.out_channels = dim * 2;
        to_q.kernel_size = {1, 1};
        to_q.stride = { 1, 1 };
        to_q.padding = { 0, 0 };
        to_q.create_weight_tensors(ctx);

        // self.to_q_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=True)
        to_q_dw.in_channels = dim * 2;
        to_q_dw.out_channels = dim * 2;
        to_q_dw.kernel_size = {3, 3};
        to_q_dw.stride = { 1, 1 };
        to_q_dw.padding = { 1, 1 };
        to_q_dw.is_depthwise = true;
        to_q_dw.create_weight_tensors(ctx);

        // self.to_k = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True)
        to_k.in_channels = dim;
        to_k.out_channels = dim * 2;
        to_k.kernel_size = {1, 1};
        to_k.stride = { 1, 1 };
        to_k.padding = { 0, 0 };
        to_k.create_weight_tensors(ctx);

        // self.to_k_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=True)
        to_k_dw.in_channels = dim * 2;
        to_k_dw.out_channels = dim * 2;
        to_k_dw.kernel_size = {3, 3};
        to_k_dw.stride = { 1, 1 };
        to_k_dw.padding = { 1, 1 };
        to_k_dw.is_depthwise = true;
        to_k_dw.create_weight_tensors(ctx);

        // self.to_v = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True)
        to_v.in_channels = dim;
        to_v.out_channels = dim * 2;
        to_v.kernel_size = {1, 1};
        to_v.stride = { 1, 1 };
        to_v.padding = { 0, 0 };
        to_v.create_weight_tensors(ctx);

        // self.to_v_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=True)
        to_v_dw.in_channels = dim * 2;
        to_v_dw.out_channels = dim * 2;
        to_v_dw.kernel_size = {3, 3};
        to_v_dw.stride = { 1, 1 };
        to_v_dw.padding = { 1, 1 };
        to_v_dw.is_depthwise = true;
        to_v_dw.create_weight_tensors(ctx);

        // self.to_out = nn.Sequential(nn.Conv2d(dim*2, dim, 1, 1, 0), )
        to_out.in_channels = dim * 2;
        to_out.out_channels = dim;
        to_out.kernel_size = {1, 1};
        to_out.stride = { 1, 1 };
        to_out.padding = { 0, 0 };
        to_out.create_weight_tensors(ctx);                                        
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        ggml_format_name(temperature, "%s%s", prefix, "temperature");

        snprintf(s, sizeof(s), "%s%s", prefix, "to_q.");
        to_q.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "to_q_dw.");
        to_q_dw.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "to_k.");
        to_k.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "to_k_dw.");
        to_k_dw.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "to_v.");
        to_v.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "to_v_dw.");
        to_v_dw.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "to_out.0.");
        to_out.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* encoder, ggml_tensor_t* decoder) {
        ggml_tensor_t *q, *k, *v, *attn, *out;
        int W = (int)encoder->ne[0];
        int H = (int)encoder->ne[1];
        int C = (int)encoder->ne[2];
        int B = (int)encoder->ne[3];
        q = to_q.forward(ctx, encoder);
        q = to_q_dw.forward(ctx, q);
        k = to_k.forward(ctx, decoder);
        k = to_k_dw.forward(ctx, k);
        v = to_v.forward(ctx, decoder);
        v = to_v_dw.forward(ctx, v);

        // # tensor [q] size: [1, 2048, 35, 56], min: -0.950127, max: 1.107969, mean: 0.006086
        // # tensor [k] size: [1, 2048, 35, 56], min: -3.406554, max: 3.649786, mean: -0.084804
        // # tensor [v] size: [1, 2048, 35, 56], min: -8.682275, max: 10.331948, mean: 0.076861
        // q = ggml_reshape_3d(ctx, q, H*W, -1, B);
        q = ggml_reshape_4d(ctx, q, H*W, -1, heads, B);
        // k = ggml_reshape_3d(ctx, k, H*W, -1, B);
        k = ggml_reshape_4d(ctx, k, H*W, -1, heads, B);
        // v = ggml_reshape_3d(ctx, v, H*W, -1, B);
        v = ggml_reshape_4d(ctx, v, H*W, -1, heads, B);
        // # tensor [q] size: [1, 8, 256, 1960], min: -0.950127, max: 1.107969, mean: 0.006086
        // # tensor [k] size: [1, 8, 256, 1960], min: -3.406554, max: 3.649786, mean: -0.084804
        // # tensor [v] size: [1, 8, 256, 1960], min: -8.682275, max: 10.331948, mean: 0.076861

        // q = F.normalize(q, dim=-1)
        // k = F.normalize(k, dim=-1)
        q = ggml_normalize(ctx, q, 0/*dim*/, 1e-12);
        k = ggml_normalize(ctx, k, 0/*dim*/, 1e-12);

        // attn = (q @ k.transpose(-2, -1)) * self.temperature
        // attn = attn.softmax(dim=-1)
        // out = (attn @ v)
        k = ggml_transpose(ctx, k);
        attn = ggml_nn_mul_mat(ctx, q, k);
        attn = ggml_mul(ctx, attn, temperature);
        attn = ggml_softmax(ctx, attn, 0/*dim*/);
        out = ggml_nn_mul_mat(ctx, attn, v);

        // out = out.view(b, -1, h*w).view(b, -1, h, w)
        // out = ggml_reshape_3d(ctx, out, H * W, -1, B);
        out = ggml_reshape_4d(ctx, out, W, H, -1, B);

        out = to_out.forward(ctx, out);
        // # tensor [out1] size: [1, 8, 256, 1960], min: -0.91977, max: 1.421521, mean: 0.080392

        return out;
    }
};

// ------------------------------------------------------------------------------------------------------
struct LayerNorm2d {
    int64_t normalized_shape;
    const int dim = 2;  // dim on C
    const float eps = 1e-6;

    ggml_tensor_t* w;
    ggml_tensor_t* b;

    void create_weight_tensors(ggml_context_t* ctx)
    {
        GGML_ASSERT(normalized_shape > 0);

        w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, normalized_shape);
        b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, normalized_shape);
    }

    void setup_weight_names(const char* prefix)
    {
        ggml_format_name(w, "%s%s", prefix, "weight");
        ggml_format_name(b, "%s%s", prefix, "bias");
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* x)
    {
        x = ggml_norm_ext(ctx, x, dim, eps);

        // int C = (int)w->ne[0];
        ggml_tensor_t *w0 = ggml_reshape_4d(ctx, w, 1, 1, normalized_shape, 1);
        ggml_tensor_t *b0 = ggml_reshape_4d(ctx, b, 1, 1, normalized_shape, 1);
        x = ggml_mul(ctx, x, w0);
        x = ggml_add(ctx, x, b0);

        return x;
    }
};


struct Fuse {
    int in_feat;
    int out_feat;

    // network params
    struct Conv2d encode_enc;

    struct LayerNorm2d norm1;
    struct LayerNorm2d norm2;
    struct CrossChannelAttention crossattn;
    struct LayerNorm2d norm3;

    void create_weight_tensors(struct ggml_context* ctx) {
        encode_enc.in_channels = in_feat;
        encode_enc.out_channels = out_feat;
        encode_enc.kernel_size = {3, 3};
        encode_enc.stride = { 1, 1 };
        encode_enc.padding = { 1, 1 };
        encode_enc.create_weight_tensors(ctx);

        norm1.normalized_shape = out_feat;
        norm1.create_weight_tensors(ctx);

        norm2.normalized_shape = out_feat;
        norm2.create_weight_tensors(ctx);

        crossattn.dim = out_feat;
        crossattn.create_weight_tensors(ctx);

        norm3.normalized_shape = out_feat;
        norm3.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "encode_enc.");
        encode_enc.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "norm1.");
        norm1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "norm2.");
        norm2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "crossattn.");
        crossattn.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "norm3.");
        norm3.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* enc, ggml_tensor_t* dec) {
        ggml_tensor_t *res, *output;

        enc = encode_enc.forward(ctx, enc);
        res = enc;
        enc = norm1.forward(ctx, enc);
        dec = norm2.forward(ctx, dec);
        // # tensor [enc] size: [1, 1024, 35, 56], min: -11.505666, max: 2.44047, mean: 0.000167
        // # tensor [dec] size: [1, 1024, 35, 56], min: -0.646519, max: 10.31074, mean: 0.000101

        output = crossattn.forward(ctx, enc, dec);
        output = ggml_add(ctx, output, res);

        output = norm3.forward(ctx, output);
        output = ggml_relu(ctx, output);

        // tensor [enc] size: [1, 1024, 35, 56], min: -11.505666, max: 2.44047, mean: 0.000167
        // tensor [dec] size: [1, 1024, 35, 56], min: -0.646519, max: 10.31074, mean: 0.000101
        // tensor [output1] size: [1, 1024, 35, 56], min: -318.123413, max: 70.843498, mean: 9.195792
        // tensor [output2] size: [1, 1024, 35, 56], min: 0.0, max: 2.623592, mean: 0.064568

    	return output;
    }
};

struct Mlp {
    int in_features = 384;
    int hidden_features = 1536;
    int out_features = 384;

    // network params
    struct Linear fc1;
    struct Linear fc2;

    void create_weight_tensors(struct ggml_context* ctx) {
        fc1.in_features = in_features;
        fc1.out_features = hidden_features;
        fc1.has_bias = true; // Fixed default
        fc1.create_weight_tensors(ctx);

        fc2.in_features = hidden_features;
        fc2.out_features = out_features;
        fc2.has_bias = true; // Fixed default
        fc2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "fc1.");
        fc1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "fc2.");
        fc2.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        x = fc1.forward(ctx, x);
        x = ggml_gelu(ctx, x);
        x = fc2.forward(ctx, x);

    	return x;
    }
};

struct LayerScale {
    int dim;

    ggml_tensor_t *gamma;    

    void create_weight_tensors(struct ggml_context* ctx) {
        gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);        
    }

    void setup_weight_names(const char *prefix) {
        // char s[GGML_MAX_NAME];
        ggml_format_name(gamma, "%s%s", prefix, "gamma");
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        return ggml_mul(ctx, x, gamma);
    }
};

struct Attention {
    int dim = 384;
    int num_heads = 6;
    float scale = 0.125;

    // network params
    struct Linear qkv;
    struct Linear proj;

    void create_weight_tensors(struct ggml_context* ctx) {
        qkv.in_features = dim;
        qkv.out_features = dim * 3;
        qkv.has_bias = true; // Fixed default
        qkv.create_weight_tensors(ctx);

        proj.in_features = dim;
        proj.out_features = dim;
        proj.has_bias = true; // Fixed default
        proj.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "qkv.");
        qkv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "proj.");
        proj.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        ggml_tensor_t *q, *k, *v, *attn;

        // # tensor [x] size: [1, 2561, 384], min: -7.700042, max: 5.273503, mean: 0.004173
        int C = (int)x->ne[0]; // 384
        int N = (int)x->ne[1]; // 2561
        int B = (int)x->ne[2];

        x = qkv.forward(ctx, x);
        // # tensor [x] size: [1, 2561, 1152], min: -11.863246, max: 11.457247, mean: 0.032135
        x = ggml_reshape_4d(ctx, x, C/num_heads, 3*num_heads, N, B);
        q = ggml_nn_slice(ctx, x, 1, 0*num_heads, 1*num_heads, 1/*step*/);
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)); // [64, 2561, 6, 1] --> [64, 6, 2561, 1]
        q = ggml_scale(ctx, q, scale);
        k = ggml_nn_slice(ctx, x, 1, 1*num_heads, 2*num_heads, 1/*step*/);
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)); // [64, 2561, 6, 1] --> [64, 6, 2561, 1]
        v = ggml_nn_slice(ctx, x, 1, 2*num_heads, 3*num_heads, 1/*step*/);
        v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3)); // [64, 2561, 6, 1] --> [64, 6, 2561, 1]

        k = ggml_transpose(ctx, k);
        attn = ggml_nn_mul_mat(ctx, q, k);

        attn = ggml_softmax(ctx, attn, 0/*dim*/);
        attn = ggml_nn_mul_mat(ctx, attn, v);
        // # tensor [attn@v] size: [1, 6, 2561, 64], min: -1.033432, max: 1.115259, mean: -0.009298

        x = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3)); // [64, 2561, 6, 1] -> [64, 6, 2561, 1] -> [384, 2561, 1]
        x = ggml_reshape_3d(ctx, x, C, N, B); // [64, 6, 2561, 1] -> [384, 2561, 1]
        x = proj.forward(ctx, x);

        return x;
    }
};

struct NestedTensorBlock {
    int dim = 384;
    int num_heads = 6;

    // network params
    struct LayerNorm norm1;
    struct Attention attn;
    struct LayerScale ls1;
    struct LayerNorm norm2;
    struct Mlp mlp;
    struct LayerScale ls2;

    void create_weight_tensors(struct ggml_context* ctx) {
        norm1.normalized_shape = dim;
        norm1.create_weight_tensors(ctx);

        attn.dim = dim;
        attn.num_heads = num_heads;
        attn.create_weight_tensors(ctx);

        ls1.dim = dim;
        ls1.create_weight_tensors(ctx);

        norm2.normalized_shape = dim;
        norm2.create_weight_tensors(ctx);

        mlp.in_features = dim;
        mlp.hidden_features = dim * 4;
        mlp.out_features = dim;
        mlp.create_weight_tensors(ctx);

        ls2.dim = dim;
        ls2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "norm1.");
        norm1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "attn.");
        attn.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "ls1.");
        ls1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "norm2.");
        norm2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "mlp.");
        mlp.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "ls2.");
        ls2.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        ggml_tensor_t *y = norm1.forward(ctx, x);
        y = attn.forward(ctx, y);
        y = ls1.forward(ctx, y);
        x = ggml_add(ctx, x, y);
        // ---------------------------------
        y = norm2.forward(ctx, x);
        y = mlp.forward(ctx, y);
        y = ls2.forward(ctx, y);
        x = ggml_add(ctx, x, y);

        return x;
    }
};

struct PatchEmbed {
    struct Conv2d proj;

    void create_weight_tensors(struct ggml_context* ctx) {
        proj.in_channels = 3;
        proj.out_channels = 384; // embed_dim
        proj.kernel_size = {14, 14}; // 14 -- patch_size
        proj.stride = { 14, 14 };
        proj.padding = { 0, 0 };
        proj.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "proj.");
        proj.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // B, C, H, W = x.shape
        // x = self.proj(x)  # B C H W
        // x = x.flatten(2).transpose(1, 2)  # B HW C
        x = proj.forward(ctx, x);
        int W = (int)x->ne[0];
        int H = (int)x->ne[1];
        int C = (int)x->ne[2];
        int B = (int)x->ne[3];
        x = ggml_reshape_3d(ctx, x, W*H, C, B);

        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)); // [HW, C, B] -> [C, HW, B]
        // tensor [x] size: [1, 2560, 384], min: -0.818547, max: 0.587891, mean: -0.002679

    	return x;
    }
};

// -----------------------------------------------------------------
struct DinoVisionTransformer {
    const int patch_size = 14;
    const int num_patches = 1369;
    const int num_tokens = 1;
    const int num_heads = 6;
    const int embed_dim = 384;

    // network params
    struct PatchEmbed patch_embed;
    ggml_tensor_t *cls_token;
    ggml_tensor_t *pos_embed;
    struct NestedTensorBlock blocks[12];
    struct LayerNorm norm;
    // ggml_tensor_t *mask_token;

    void create_weight_tensors(struct ggml_context* ctx) {
        patch_embed.create_weight_tensors(ctx);

        // self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        // self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        cls_token = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, embed_dim, 1, 1);
        pos_embed = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, embed_dim, num_tokens + num_patches, 1);

        // blocks_0.dim = 384; // embed_dim
        // blocks_0.num_heads = 6;
        // blocks_0.create_weight_tensors(ctx);
        for (int i = 0; i < 12; i++) {
            blocks[i].dim = embed_dim;
            blocks[i].num_heads = num_heads;
            blocks[i].create_weight_tensors(ctx);
        }

        norm.normalized_shape = embed_dim;
        norm.eps = 1e-6;
        norm.create_weight_tensors(ctx);
        // mask_token = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embed_dim, 1);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "patch_embed.");
        patch_embed.setup_weight_names(s);

        ggml_format_name(cls_token, "%s%s", prefix, "cls_token");
        ggml_format_name(pos_embed, "%s%s", prefix, "pos_embed");

        for (int i = 0; i < 12; i++) {
            snprintf(s, sizeof(s), "%sblocks.%d.", prefix, i);
            blocks[i].setup_weight_names(s);
        }

        snprintf(s, sizeof(s), "%s%s", prefix, "norm.");
        norm.setup_weight_names(s);
        // ggml_format_name(mask_token, "%s%s", prefix, "mask_token");
    }

    ggml_tensor_t* interpolate_pos_encoding(struct ggml_context* ctx, ggml_tensor_t* x, int H, int W) {
        // # x.size() -- [1, 2561, 384], H -- 560, W -- 896
        if (H == W && x->ne[1] == pos_embed->ne[1]) {
            return pos_embed;
        }
        int M = (int)sqrtf(num_patches);
        int NH = (H + patch_size - 1)/patch_size;
        int NW = (W + patch_size - 1)/patch_size;
        ggml_tensor_t *class_pos_embed, *patch_pos_embed;
        class_pos_embed = ggml_nn_slice(ctx, pos_embed, 1 /*dim*/, 0, num_tokens, 1/*step*/);
        patch_pos_embed = ggml_nn_slice(ctx, pos_embed, 1 /*dim*/, num_tokens, num_tokens + num_patches, 1/*step*/);

        patch_pos_embed = ggml_reshape_4d(ctx, patch_pos_embed, embed_dim, M, M, 1);
        patch_pos_embed = ggml_interpolate(ctx, patch_pos_embed, 1/*on W*/, NW);
        patch_pos_embed = ggml_interpolate(ctx, patch_pos_embed, 2/*on H*/, NH);

        patch_pos_embed = ggml_reshape_3d(ctx, patch_pos_embed, embed_dim, -1 /*HW*/, 1);
        // # tensor [patch_pos_embed] size: [1, 2560, 384], min: -0.162149, max: 0.127178, mean: 8.2e-05

        x = ggml_concat(ctx, class_pos_embed, patch_pos_embed, 1/*middle dim*/);

        return x;
    }

    std::vector<ggml_tensor_t *> forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // # tensor [x] size: [1, 3, 560, 896], min: -0.495599, max: 0.496816, mean: -0.109927
        std::vector<ggml_tensor_t *> xlist;
        ggml_tensor_t *out;

        int W = (int) x->ne[0];
        int H = (int) x->ne[1];
        int C = (int) x->ne[2];
        int B = (int) x->ne[3];

        x = patch_embed.forward(ctx, x);
        // # tensor [x] size: [1, 2560, 384], min: -0.818547, max: 0.587891, mean: -0.002679

        x = ggml_concat(ctx, ggml_repeat_ext(ctx, cls_token, 1, 1, B, 1), x, 1/*dim*/);
        // # tensor [x] size: [1, 2561, 384], min: -0.818547, max: 0.587891, mean: -0.002678

        // x = x + self.interpolate_pos_encoding(x, H, W)
        ggml_tensor_t *pos = interpolate_pos_encoding(ctx, x, H, W);
        x = ggml_add(ctx, x, pos);

        int NH = (H + patch_size - 1)/patch_size;
        int NW = (W + patch_size - 1)/patch_size;

        for (int i = 0; i < 12; i++) {
            x = blocks[i].forward(ctx, x);
            if (i < 8)
                continue;

            { // i == 8 || i == 9 || i == 10 || i == 11
                out = norm.forward(ctx, x);
                C = (int)out->ne[1];
                out = ggml_nn_slice(ctx, out, 1/*dim*/, 1, C, 1/*step*/);
                out = ggml_reshape_4d(ctx, out, embed_dim, NW, NH, B);
                out = ggml_cont(ctx, ggml_permute(ctx, out, 2, 0, 1, 3)); // [384, 64, 40, 1] -> [64, 40, 384, 1]
                xlist.push_back(out);
            }
        }

    	return xlist;
    }
};

// --------------------------------------------------------------------------------------------------------
struct Segmentor {
    // network params
    struct DinoVisionTransformer backbone;
    struct Conv2d conv3;  // torch.float32, [1536, 1536, 1, 1] 
    struct BatchNorm2d bn3;

    void create_weight_tensors(struct ggml_context* ctx) {
        backbone.create_weight_tensors(ctx);

        conv3.in_channels = 1536;
        conv3.out_channels = 1536;
        conv3.kernel_size = {1, 1};
        conv3.stride = { 1, 1 };
        conv3.padding = { 0, 0 };
        conv3.has_bias = false;
        conv3.create_weight_tensors(ctx);

        bn3.num_features = 1536;
        bn3.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "backbone.");
        backbone.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv3.");
        conv3.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "bn3.");
        bn3.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // # tensor [x] size: [1, 3, 560, 896], min: -0.994517, max: 1.0, mean: -0.189531

        std::vector<ggml_tensor_t *> tokens = backbone.forward(ctx, x);
        // tokens = self.backbone(x) #.get_intermediate_layers(x, n=[8, 9, 10, 11], reshape=True) # last n=4 [8, 9, 10, 11]
        // # tokens is tuple: len = 4
        // #     tensor [item] size: [1, 384, 40, 64], min: -64.093712, max: 65.633827, mean: 0.04372
        // #     tensor [item] size: [1, 384, 40, 64], min: -60.8563, max: 49.656631, mean: 0.003902
        // #     tensor [item] size: [1, 384, 40, 64], min: -46.128963, max: 40.135544, mean: 0.009809
        // #     tensor [item] size: [1, 384, 40, 64], min: -21.549391, max: 19.685974, mean: 0.007802
        ggml_tensor_t *f16 = ggml_cat(ctx, 4, tokens[0], tokens[1], tokens[2], tokens[3], 2/*dim on C*/);
        // # tensor [f16] size: [1, 1536, 40, 64], min: -64.088036, max: 62.895596, mean: 0.022864
        f16 = conv3.forward(ctx, f16);
        // f16    f32 [64, 40, 1536, 1], 
        f16 = bn3.forward(ctx, f16);
        f16 = ggml_relu(ctx, f16);
        // # tensor [f16] size: [1, 1536, 40, 64], min: 0.0, max: 11.556356, mean: 0.868597

        int W = (int)f16->ne[0];
        int H = (int)f16->ne[1];
        W = (W * 14)/16;
        H = (H * 14)/16;
        f16 = ggml_interpolate(ctx, f16, 0 /*dim on W*/, W);
        f16 = ggml_interpolate(ctx, f16, 1 /*dim on H*/, H);
        // # tensor [f16] size: [1, 1536, 35, 56], min: 0.0, max: 11.285922, mean: 0.868653

        return f16;
    }
};

// ----------------------------------------------------------------------------------------------------
struct Bottleneck {
    int inplanes = 1024;
    int planes = 256;
    int stride = 1;
    bool downsample = false;
    const int expansion = 4; // Fixed

    // network params
    struct Conv2d conv1;
    struct BatchNorm2d bn1;
    struct Conv2d conv2;
    struct BatchNorm2d bn2;
    struct Conv2d conv3;
    struct BatchNorm2d bn3;

    struct Conv2d downsample_conv;
    struct BatchNorm2d downsample_bn;

    void create_weight_tensors(struct ggml_context* ctx) {
        // self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        // self.bn1 = nn.BatchNorm2d(planes)
        conv1.in_channels = inplanes;
        conv1.out_channels = planes;
        conv1.kernel_size = {1, 1};
        conv1.stride = { 1, 1 };
        conv1.padding = { 0, 0 };
        conv1.has_bias = false;
        conv1.create_weight_tensors(ctx);

        bn1.num_features = planes;
        bn1.create_weight_tensors(ctx);

        // self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=1, padding=1, bias=False)
        // self.bn2 = nn.BatchNorm2d(planes)
        conv2.in_channels = planes;
        conv2.out_channels = planes;
        conv2.kernel_size = {3, 3};
        conv2.stride = { stride, stride };
        conv2.padding = { 1, 1 };
        conv2.has_bias = false;
        conv2.create_weight_tensors(ctx);

        bn2.num_features = planes;
        bn2.create_weight_tensors(ctx);

        // self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        // self.bn3 = nn.BatchNorm2d(planes * 4)
        conv3.in_channels = planes;
        conv3.out_channels = planes * 4;
        conv3.kernel_size = {1, 1};
        conv3.stride = { 1, 1 };
        conv3.padding = { 0, 0 };
        conv3.has_bias = false;
        conv3.create_weight_tensors(ctx);

        bn3.num_features = planes * 4;
        bn3.create_weight_tensors(ctx);


        if (downsample) {
            downsample = (stride != 1 || inplanes != planes * expansion);
        }
        if (downsample) {
            //     if downsample and (stride != 1 or inplanes != planes * self.expansion):
            //         self.downsample = nn.Sequential(
            //             nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
            //             nn.BatchNorm2d(planes * self.expansion))
            downsample_conv.in_channels = inplanes;
            downsample_conv.out_channels = planes * expansion;
            downsample_conv.kernel_size = {1, 1};
            downsample_conv.stride = { stride, stride };
            downsample_conv.padding = { 0, 0 };
            downsample_conv.has_bias = false;
            downsample_conv.create_weight_tensors(ctx);

            downsample_bn.num_features = planes * expansion;
            downsample_bn.create_weight_tensors(ctx);
        }
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv1.");
        conv1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "bn1.");
        bn1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv2.");
        conv2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "bn2.");
        bn2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv3.");
        conv3.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "bn3.");
        bn3.setup_weight_names(s);

        if (downsample) {
            snprintf(s, sizeof(s), "%s%s", prefix, "downsample.0.");
            downsample_conv.setup_weight_names(s);
            snprintf(s, sizeof(s), "%s%s", prefix, "downsample.1.");
            downsample_bn.setup_weight_names(s);
        }        
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        ggml_tensor_t *residual = x;
        if (downsample) {
            residual = downsample_conv.forward(ctx, x);
            residual = downsample_bn.forward(ctx, residual);
        }
        x = conv1.forward(ctx, x);
        x = bn1.forward(ctx, x);
        x = ggml_relu(ctx, x);

        x = conv2.forward(ctx, x);
        x = bn2.forward(ctx, x);
        x = ggml_relu(ctx, x);

        x = conv3.forward(ctx, x);
        x = bn3.forward(ctx, x);
        // # tensor [out] size: [1, 256, 140, 224], min: -0.594325, max: 0.780308, mean: 0.012659
        // # tensor [residual] size: [1, 256, 140, 224], min: -0.548443, max: 0.556349, mean: 0.061899
        x = ggml_add(ctx, x, residual);
        x = ggml_relu(ctx, x);

        return x;
    }
};

// make_bottleneck_layer
struct BottleneckLayer {
    int inplanes;
    int planes;
    int stride = 1;
    int blocks = 3; // 3, 4, 6 ...

    struct Bottleneck layers[6];

    void create_weight_tensors(struct ggml_context* ctx) {
        GGML_ASSERT(ARRAY_SIZE(layers) >= blocks);

        layers[0].inplanes = inplanes;
        layers[0].planes = planes;
        layers[0].stride = stride;
        layers[0].downsample = true;
        layers[0].create_weight_tensors(ctx);

        for (int i = 1; i < blocks; i++) {
            layers[i].inplanes = planes * 4; // expansion -- 4
            layers[i].planes = planes;
            layers[i].stride = 1;
            layers[i].downsample = false;
            layers[i].create_weight_tensors(ctx);
        }
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "0.");
        layers[0].setup_weight_names(s);

        for (int i = 1; i < blocks; i++) {
            snprintf(s, sizeof(s), "%s%d.", prefix, i);
            layers[i].setup_weight_names(s);
        }
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        for (int i = 0; i < blocks; i++) {
            x = layers[i].forward(ctx, x);
        }

        return x;
    }
};

struct DINOv2_v6 {
    struct Conv2d conv1;
    struct BatchNorm2d bn1;

    struct MaxPool2d maxpool;

    struct BottleneckLayer res2;
    struct BottleneckLayer layer2;
    struct BottleneckLayer layer3;

    struct Segmentor network2;

    struct Fuse fuse1;
    struct Fuse fuse2;
    struct Fuse fuse3;

    void create_weight_tensors(struct ggml_context* ctx) {
        // self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        // self.bn1 = nn.BatchNorm2d(64)
        conv1.in_channels = 3;
        conv1.out_channels = 64;
        conv1.kernel_size = {7, 7};
        conv1.stride = { 2, 2 };
        conv1.padding = { 3, 3 };
        conv1.has_bias = false;
        conv1.create_weight_tensors(ctx);

        bn1.num_features = 64;
        bn1.create_weight_tensors(ctx);

        maxpool.kernel_size = 3;
        maxpool.stride = 2;
        maxpool.padding = 1;
        maxpool.create_weight_tensors(ctx);

        // self.res2 = resnet.make_bottleneck_layer(64, 64, 3, stride=1)
        res2.inplanes = 64;
        res2.planes = 64;
        res2.blocks = 3; // 3, 4, 6 ...
        res2.stride = 1;
        res2.create_weight_tensors(ctx);

        // self.layer2 = resnet.make_bottleneck_layer(256, 128, 4, stride=2)
        layer2.inplanes = 256;
        layer2.planes = 128;
        layer2.blocks = 4; // 3, 4, 6 ...
        layer2.stride = 2;
        layer2.create_weight_tensors(ctx);

        // self.layer3 = resnet.make_bottleneck_layer(512, 256, 6, stride=2)
        layer3.inplanes = 512;
        layer3.planes = 256;
        layer3.blocks = 6; // 3, 4, 6 ...
        layer3.stride = 2;
        layer3.create_weight_tensors(ctx);

        network2.create_weight_tensors(ctx);

        // self.fuse1 = resnet.Fuse(384 * 4, 1024) # n = [8, 9, 10, 11]
        fuse1.in_feat = 384 * 4;
        fuse1.out_feat = 1024;
        fuse1.create_weight_tensors(ctx);

        // self.fuse2 = resnet.Fuse(384 * 4, 512)
        fuse2.in_feat = 384 * 4;
        fuse2.out_feat = 512;
        fuse2.create_weight_tensors(ctx);

        // self.fuse3 = resnet.Fuse(384 * 4, 256)
        fuse3.in_feat = 384 * 4;
        fuse3.out_feat = 256;
        fuse3.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv1.");
        conv1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "bn1.");
        bn1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "res2.");
        res2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer2.");
        layer2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer3.");
        layer3.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "network2.");
        network2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "fuse1.");
        fuse1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "fuse2.");
        fuse2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "fuse3.");
        fuse3.setup_weight_names(s);
    }


   ggml_tensor_t * forward(struct ggml_context* ctx, ggml_tensor_t* f) {
        ggml_tensor_t *x, *f16, *f8, *f4, *dino_f16, *g16, *g8, *g4;

        x = conv1.forward(ctx, f);
        x = bn1.forward(ctx, x);
        x = ggml_relu(ctx, x);
        x = maxpool.forward(ctx, x);

        f4 = res2.forward(ctx, x);
        f8 = layer2.forward(ctx, f4);
        f16 = layer3.forward(ctx, f8);

        dino_f16 = network2.forward(ctx, f);

        // # tensor [dino_f16] size: [1, 1536, 35, 56], min: 0.0, max: 10.015625, mean: 0.865097
        int W = (int)dino_f16->ne[0];
        int H = (int)dino_f16->ne[1];

        // g16 = self.fuse1(dino_f16, f16)
        // tensor [dino_f16] size: [1, 1536, 35, 56], min: 0.0, max: 11.285922, mean: 0.868653
        // tensor [f16] size: [1, 1024, 35, 56], min: 0.0, max: 1.415327, mean: 0.062972
        g16 = fuse1.forward(ctx, dino_f16, f16);
        // tensor [g16] size: [1, 1024, 35, 56], min: 0.0, max: 2.623592, mean: 0.064568

        // g8 = self.fuse2(self.upsample2(dino_f16), f8)
        g8 = ggml_interpolate(ctx, dino_f16, 0/*on W*/, 2*W);
        g8 = ggml_interpolate(ctx, g8, 1/*on H*/, 2*H);
        // tensor [g8] size: [1, 1536, 70, 112], min: 0.0, max: 11.116426, mean: 0.868653
        // tensor [f8] size: [1, 512, 70, 112], min: 0.0, max: 1.790387, mean: 0.066411
        g8 = fuse2.forward(ctx, g8, f8);
        // tensor [g8] size: [1, 512, 70, 112], min: 0.0, max: 1.733069, mean: 0.093956

        // g4 = self.fuse3(self.upsample4(dino_f16), f4)
        g4 = ggml_interpolate(ctx, dino_f16, 0/*on W*/, 4*W);
        g4 = ggml_interpolate(ctx, g4, 1/*on H*/, 4*H);
        // tensor [g4] size: [1, 1536, 140, 224], min: 0.0, max: 11.212543, mean: 0.868653
        // tensor [f4] size: [1, 256, 140, 224], min: 0.0, max: 0.849516, mean: 0.11216
        g4 = fuse3.forward(ctx, g4, f4);
        // tensor [g4] size: [1, 256, 140, 224], min: 0.0, max: 6.521586, mean: 0.203084

        // Save f16, f8, f4
        {
            g16 = ggml_cont(ctx, g16);
            ggml_set_name(g16, "F16");
            ggml_set_output(g16);

            g8 = ggml_cont(ctx, g8);
            ggml_set_name(g8, "F8");
            ggml_set_output(g8);

            g4 = ggml_cont(ctx, g4);
            ggml_set_name(g4, "F4");
            ggml_set_output(g4);
        }

        return ggml_package(ctx, 3, g16, g8, g4);
    }
};

struct KeyEncoder : GGMLNetwork {
    struct DINOv2_v6 key_encoder;
    struct KeyProjection key_proj;

    void create_weight_tensors(struct ggml_context* ctx) {
        key_encoder.create_weight_tensors(ctx);
        key_proj.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "key_encoder.");
        key_encoder.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "key_proj.");
        key_proj.setup_weight_names(s);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, int argc, ggml_tensor_t* argv[]) {
        GGML_ASSERT(argc == 1);
        ggml_tensor_t *image_lll = argv[0];

        ggml_tensor_t *f16 = key_encoder.forward(ctx, image_lll);
        // F16, F8, F4

        ggml_tensor_t *key = key_proj.forward(ctx, f16);
        // KEY, SHRINKAGE, SELECTION

        return key;
   }
};

struct VideoColorNetwork {
    const int MAX_H = 1024;
    const int MAX_W = 1024;
    const int MAX_TIMES = 112;

    KeyEncoder key_net;
    ValueEncoder value_net;
    ColorDecoder color_net;

#ifdef ENABLE_LOCAL_ATTENTION
    LocalAttention local_net;
#endif    

    GGMLModel model;

    // encode_key output
    TENSOR *KEY = NULL; // key;
    TENSOR *SHRINKAGE = NULL; // shrinkage;
    TENSOR *SELECTION = NULL; // selection;
    TENSOR *F16 = NULL; // f16;
    TENSOR *F8 = NULL; //f8;
    TENSOR *F4 = NULL; // f4;

    // encode_value output
    TENSOR *VALUE = NULL;
    TENSOR *HIDDEN = NULL;

    int init(int device) {
        // -----------------------------------------------------------------------------------------
        key_net.set_device(device);
        key_net.start_engine();
        // key_net.dump();

        value_net.set_device(device);
        value_net.start_engine();
        // value_net.dump();

        color_net.set_device(device);
        color_net.start_engine();
        // color_net.dump();

#ifdef ENABLE_LOCAL_ATTENTION
        local_net.set_device(device); // xxxx_debug
        local_net.start_engine();
        // local_net.dump();
#endif

        check_point(model.preload("models/video_color_f16.gguf") == RET_OK);
        load();

        return RET_OK;
    }

    int load() {
        key_net.load_weight(&model, "");
        value_net.load_weight(&model, "value_encoder.");
        color_net.load_weight(&model, "decoder.");

#ifdef ENABLE_LOCAL_ATTENTION
        return local_net.load_weight(&model, "short_term_attn.");
#endif
        return RET_OK;
    }

    // KeyEncoder
    TENSOR *encode_key(TENSOR *image_lll) {
        TENSOR *argv[1];
        argv[0] = image_lll;

        TENSOR *key = key_net.engine_forward(ARRAY_SIZE(argv), argv);

        // Update shrinkage, selection, f16, f8, f4
        {
            // TENSOR *x;
            // x = key_net.get_output_tensor((char *)"DO_SOFTMAX");
            // if (tensor_valid(x)) {
            //     tensor_debug_show("-------- DO_SOFTMAX", x);
            //     tensor_destroy(x);
            // }
            tensor_destroy(KEY);
            KEY = key_net.get_output_tensor((char *)"KEY");

            tensor_destroy(SHRINKAGE);
            SHRINKAGE = key_net.get_output_tensor((char *)"SHRINKAGE");

            tensor_destroy(SELECTION);
            SELECTION = key_net.get_output_tensor((char *)"SELECTION");

            tensor_destroy(F16);
            F16 = key_net.get_output_tensor((char *)"F16");

            tensor_destroy(F8);
            F8 = key_net.get_output_tensor((char *)"F8");

            tensor_destroy(F4);
            F4 = key_net.get_output_tensor((char *)"F4");
        }

        return key;
    }

    // ValueEncoder
    TENSOR *encode_value(TENSOR *image_lll, TENSOR *f16, TENSOR *hidden, TENSOR *image_ab) {
        TENSOR *argv[4];
        argv[0] = image_lll;
        argv[1] = f16;
        argv[2] = hidden;
        argv[3] = image_ab;

        TENSOR *value = value_net.engine_forward(ARRAY_SIZE(argv), argv);
        // Update hidden
        {
            tensor_destroy(VALUE);
            VALUE = value_net.get_output_tensor((char *)"VALUE");

            tensor_destroy(HIDDEN);
            HIDDEN = value_net.get_output_tensor((char *)"HIDDEN");
        }

        return value;
    }

    // ColorDecoder
    TENSOR *decode_color(TENSOR *f16, TENSOR *f8, TENSOR *f4, TENSOR *value, TENSOR *hidden) {
        TENSOR *argv[5];
        argv[0] = f16;
        argv[1] = f8;
        argv[2] = f4;
        argv[3] = value;
        argv[4] = hidden;

        // # multi_scale_features(f16, f8, f4) is tuple: len = 3
        // #     tensor [item] size: [1, 1024, 35, 56], min: 0.0, max: 2.601784, mean: 0.063031
        // #     tensor [item] size: [1, 512, 70, 112], min: 0.0, max: 1.79675, mean: 0.090695
        // #     tensor [item] size: [1, 256, 140, 224], min: 0.0, max: 6.709424, mean: 0.200673
        // # tensor [value] size: [2, 512, 35, 56], min: -9.328125, max: 4.738281, mean: -0.007783
        // # tensor [hidden] size: [2, 64, 35, 56], min: -1.0, max: 0.999023, mean: -0.009137        

        TENSOR *color_ab = color_net.engine_forward(ARRAY_SIZE(argv), argv);

        return color_ab;
    }

#ifdef ENABLE_LOCAL_ATTENTION
    // LocalAttention
    TENSOR *encode_local(TENSOR *key, TENSOR *last_key, TENSOR *last_value) {
        TENSOR *argv[3];
        argv[0] = key;
        argv[1] = last_key;
        argv[2] = last_value;

        TENSOR *value = local_net.engine_forward(ARRAY_SIZE(argv), argv);

        return value;
    }
#endif

    void exit() {
        // encode_value
        tensor_destroy(VALUE);
        tensor_destroy(HIDDEN);

        // encode_key
        tensor_destroy(KEY);
        tensor_destroy(SHRINKAGE);
        tensor_destroy(SELECTION);
        tensor_destroy(F16);
        tensor_destroy(F8);
        tensor_destroy(F4);

        model.clear();

#ifdef ENABLE_LOCAL_ATTENTION
        local_net.stop_engine();
#endif        
        color_net.stop_engine();
        value_net.stop_engine();
        key_net.stop_engine();
    }
};
#endif // __COLORMNET__H__
