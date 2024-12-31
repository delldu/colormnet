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
            g = downsample.forward(ctx, g);
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
        skip_f = ggml_repeat_ext(ctx, skip_f, B, 1, 1, 1);

        ggml_tensor_t *g;
        g = ggml_interpolate(ctx, up_g, 0, 2*W);
        g = ggml_interpolate(ctx, g, 1, 2*H);
        g = ggml_add(ctx, skip_f, g);
        g = out_conv.forward(ctx, g);

    	return g;
    }
};

struct HiddenUpdater {
    const int hidden_dim = 64;

    struct Conv2d g16_conv;
    struct Conv2d g8_conv;
    struct Conv2d g4_conv;
    struct Conv2d transform;

    void create_weight_tensors(struct ggml_context* ctx) {
        g16_conv.in_channels = 512;
        g16_conv.out_channels = 256;
        g16_conv.kernel_size = {1, 1};
        g16_conv.stride = { 1, 1 };
        g16_conv.padding = { 0, 0 };
        g16_conv.create_weight_tensors(ctx);

        g8_conv.in_channels = 256;
        g8_conv.out_channels = 256;
        g8_conv.kernel_size = {1, 1};
        g8_conv.stride = { 1, 1 };
        g8_conv.padding = { 0, 0 };
        g8_conv.create_weight_tensors(ctx);

        g4_conv.in_channels = 257;
        g4_conv.out_channels = 256;
        g4_conv.kernel_size = {1, 1};
        g4_conv.stride = { 1, 1 };
        g4_conv.padding = { 0, 0 };
        g4_conv.create_weight_tensors(ctx);

        transform.in_channels = 320; // mid_dim + hidden_dim;
        transform.out_channels =  192; // hidden_dim*3;
        transform.kernel_size = {3, 3};
        transform.stride = { 1, 1 };
        transform.padding = { 1, 1 };
        transform.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "g16_conv.");
        g16_conv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "g8_conv.");
        g8_conv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "g4_conv.");
        g4_conv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "transform.");
        transform.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* g0, ggml_tensor_t* g1, ggml_tensor_t* g2, 
        ggml_tensor_t* h) {
        int H, W;
        ggml_tensor_t *g, *forget_gate, *update_gate, *new_value, *new_h;

        g0 = g16_conv.forward(ctx, g0);

        W = (int)g1->ne[0];
        H = (int)g1->ne[1];
        g1 = ggml_interpolate(ctx, g1, 0/*on W*/, W/2);
        g1 = ggml_interpolate(ctx, g1, 1/*on H*/, H/2);
        g1 = g8_conv.forward(ctx, g1);

        W = (int)g2->ne[0];
        H = (int)g2->ne[1];
        g2 = ggml_interpolate(ctx, g2, 0/*on W*/, W/4);
        g2 = ggml_interpolate(ctx, g2, 1/*on H*/, H/4);
        g2 = g4_conv.forward(ctx, g2);

        g = ggml_add(ctx, g0, g1);
        g = ggml_add(ctx, g, g2);
        g = ggml_concat(ctx, g, h, 2/*dim on channel*/);
        g = transform.forward(ctx, g);

        forget_gate = ggml_sigmoid(ctx, ggml_nn_slice(ctx, g, 2/*dim*/, 0, hidden_dim, 1/*step*/));
        update_gate = ggml_sigmoid(ctx, ggml_nn_slice(ctx, g, 2/*dim*/, hidden_dim, 2*hidden_dim, 1/*step*/));
        new_value = ggml_tanh(ctx, ggml_nn_slice(ctx, g, 2/*dim*/, 2*hidden_dim, 3*hidden_dim, 1/*step*/));

        g = ggml_dup(ctx, update_gate);
        g = ggml_constant(ctx, g, 1.0f);
        g = ggml_sub(ctx, g, update_gate);
        g = ggml_mul(ctx, h, g);

        forget_gate = ggml_mul(ctx, forget_gate, g);
        update_gate = ggml_mul(ctx, update_gate, new_value);

        new_h = ggml_add(ctx, forget_gate, update_gate);
    	return new_h;
    }
};

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

        int W = (int)x->ne[0];
        int H = (int)x->ne[1];
        int C = (int)x->ne[2];
        int B = (int)x->ne[3];

        avg = ggml_pool_2d(ctx, x, GGML_OP_POOL_AVG, H, W, H, W, (float)0.0, (float)0.0);
        avg = mlp_0.forward(ctx, avg);
        avg = mlp_1.forward(ctx, avg);
        avg = ggml_relu(ctx, avg);
        avg = mlp_3.forward(ctx, avg);

        max = ggml_pool_2d(ctx, x, GGML_OP_POOL_MAX, H, W, H, W, (float)0.0, (float)0.0);
        avg = mlp_0.forward(ctx, avg);
        avg = mlp_1.forward(ctx, avg);
        avg = ggml_relu(ctx, avg);
        avg = mlp_3.forward(ctx, avg);

        scale = ggml_add(ctx, avg, max);
        scale = ggml_sigmoid(ctx, scale);

        return ggml_mul(ctx, x, scale);
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

// class FeatureFusionBlock(nn.Module):
//     def __init__(self, g_in_dim):
//     # def __init__(self):
//         super().__init__()
//         x_in_dim = 1024
//         g_mid_dim = 512
//         g_out_dim = 512
//         self.block1 = GroupResBlock(x_in_dim + g_in_dim, g_mid_dim)
//         self.attention = CBAM(g_mid_dim)
//         self.block2 = GroupResBlock(g_mid_dim, g_out_dim)


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
        // B, C, H, W = g.size()
        // g = torch.cat([x.repeat(B, 1, 1, 1), g], dim = 1)
        // g = self.block1(g)
        // r = self.attention(g)
        // g = self.block2(g + r)
        // return g
        int B = g->ne[3];
        x = ggml_repeat_ext(ctx, x, 1, 1, 1, B);
        g = ggml_concat(ctx, x, g, 2/*dim on Channel*/);
        g = block1.forward(ctx, g);
        ggml_tensor_t *r = attention.forward(ctx, g);
        g = ggml_add(ctx, g, r);
        g = block2.forward(ctx, g);
        return g;
    }
};

struct Decoder {
    // network hparams
    int val_dim = 512;
    int hidden_dim = 64;

    // network params
    struct FeatureFusionBlock fuser;
    struct HiddenUpdater hidden_update;
    struct UpsampleBlock up_16_8;
    struct UpsampleBlock up_8_4;
    struct Conv2d pred;

    void create_weight_tensors(struct ggml_context* ctx) {
        // self.fuser = FeatureFusionBlock(val_dim+hidden_dim) 
        fuser.g_in_dim = val_dim + hidden_dim;
        fuser.create_weight_tensors(ctx);

        // self.hidden_update = HiddenUpdater([512, 256, 256+1], 256, hidden_dim) 
        hidden_update.create_weight_tensors(ctx);

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

        snprintf(s, sizeof(s), "%s%s", prefix, "hidden_update.");
        hidden_update.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "up_16_8.");
        up_16_8.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "up_8_4.");
        up_8_4.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "pred.");
        pred.setup_weight_names(s);
    }


    // def forward(self, f16, f8, f4, hidden_state, color_feature):
    //     g16 = self.fuser(f16, torch.cat([color_feature, hidden_state], dim=1))
    //     # tensor [g16] size: [2, 512, 35, 56], min: -89.383621, max: 14.023798, mean: -1.546733

    //     g8 = self.up_16_8(f8, g16)
    //     g4 = self.up_8_4(f4, g8)
    //     # tensor [g4] size: [2, 256, 140, 224], min: -34.172653, max: 25.263411, mean: -7.309633

    //     logits = self.pred(F.relu(g4))
    //     g4 = torch.cat([g4, logits], 1)
    //     hidden_state = self.hidden_update([g16, g8, g4], hidden_state)
    //     # tensor [hidden_state] size: [2, 64, 35, 56], min: -0.999481, max: 0.999002, mean: -0.085589
        
    //     logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
    //     logits = logits.permute(1, 0, 2, 3).contiguous() # (C, B, H, W) --> (B, C, H, W)
    //     # tensor [logits] size: [1, 2, 560, 896], min: -0.472656, max: 0.702148, mean: 0.024722
    //     predict_color_ab = torch.tanh(logits)

    //     return hidden_state, predict_color_ab

    std::vector<ggml_tensor_t *> forward(struct ggml_context* ctx, ggml_tensor_t* f16, ggml_tensor_t* f8, ggml_tensor_t* f4, 
            ggml_tensor_t* hidden_state, ggml_tensor_t* color_feature) {
        std::vector<ggml_tensor_t *> xlist;
        ggml_tensor_t *g16, *g8, *g4, *logits, *predict_color_ab;

        g16 = ggml_concat(ctx, color_feature, hidden_state, 2/*dim on channel*/);
        g16 = fuser.forward(ctx, f16, g16);
        g8 = up_16_8.forward(ctx, f8, g16);
        g4 = up_8_4.forward(ctx, f4, g8);
        logits = pred.forward(ctx, ggml_relu(ctx, g4));
        g4 = ggml_concat(ctx, g4, logits, 2/*dim on channenl*/);

        hidden_state = hidden_update.forward(ctx, g16, g8, g4, hidden_state);

        int W = (int)logits->ne[0];
        int H = (int)logits->ne[0];
        logits = ggml_interpolate(ctx, logits, 0/*on W*/, 4*W);
        logits = ggml_interpolate(ctx, logits, 1/*on H*/, 4*H);

        logits = ggml_cont(ctx, ggml_permute(ctx, logits, 0, 1, 3, 2)); // [W, H, B, C] -> [W, H, C, B]
        predict_color_ab = ggml_tanh(ctx, logits);

        xlist.push_back(hidden_state);
        xlist.push_back(predict_color_ab);

        return xlist;
    }
};

/*
 DWConv2d(
  (conv): Conv2d(1024, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1024, bias=False)
) */

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

    // def forward(self, x, size_2d):
    //     # tensor [x] size: [1960, 1, 1024], min: -9.280723, max: 4.028006, mean: -0.008225
    //     # size_2d -- (35, 36)
    //     h, w = size_2d
    //     n, bs, c = x.size() # 1, 1024
    //     x = x.view(h, w, bs, c).permute(2, 3, 0, 1) # [35, 36, 1, 1024] ==> [1, 1024, 35, 36]
    //     x = self.conv(x)
    //     # x = self.dropout(x)
    //     x = x.view(bs, c, h * w).permute(2, 0, 1) # [1, 1024, 1960] ==> [1960, 1, 1024]
    //     # tensor [x] size: [1960, 1, 1024], min: -6.485817, max: 5.726138, mean: -0.003478
    //     return x
    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x, int H, int W) {
        //# tensor [x] size: [1960, 1, 1024], min: -9.280723, max: 4.028006, mean: -0.008225
        int C = (int)x->ne[0];
        int B = (int)x->ne[1];
        x = ggml_reshape_4d(ctx, x, C, B, W, H);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)); // [C, B, W, H] --> [B, C, W, H]
        x = conv.forward(ctx, x); // Depthwise Conv2d
        x = ggml_reshape_3d(ctx, x, H*W, C, B);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3)); // [HW, C, B] -> [C, B, HW]
    	return x;
    }
};


/*
 LocalAttention(
  (relative_emb_k): Conv2d(64, 225, kernel_size=(1, 1), stride=(1, 1))
  (correlation_sampler): SpatialCorrelationSampler()
  (dw_conv): DWConv2d(
    (conv): Conv2d(1024, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1024, bias=False)
  )
  (projection): Linear(in_features=1024, out_features=1024, bias=True)
) */

struct LocalAttention {
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

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* q, ggml_tensor_t* k, ggml_tensor_t* v) {
        int W = (int)v->ne[0];
        int H = (int)v->ne[1];
        int C = (int)v->ne[2];
        int B = (int)v->ne[3];

        // # Scale
        q = ggml_scale(ctx, q, 8.0f); // sqrtf(d_att);

        ggml_tensor_t *qk = ggml_corr(ctx, q, k, window_size); // [225, 35, 56]
        qk = ggml_reshape_3d(ctx, qk, H*W, window_size * window_size, 1); // [1960, 225, 1]
        ggml_tensor_t *rel_emb = relative_emb_k.forward(ctx, q);
        rel_emb = ggml_reshape_3d(ctx, rel_emb, H * W, window_size * window_size, 1);
        qk = ggml_add(ctx, qk, rel_emb);

        ggml_tensor_t *local_attn = ggml_softmax(ctx, qk, 1/*dim*/); // [1960, 225, 1]
        local_attn = ggml_transpose(ctx, local_attn); // [225, 1960, 1]
        local_attn = ggml_reshape_1d(ctx, local_attn, -1);
        ggml_tensor_t *global_attn = ggml_global_attn(ctx, local_attn, H, W, max_dis); // [70x49, 35x56]

        // global_attn = global_attn.view(1, H * W, H + 2*MAX_DISP, W + 2*MAX_DISP)
        global_attn = ggml_reshape_3d(ctx, global_attn, W+2*max_dis, H+2*max_dis, H*W);
        global_attn = ggml_nn_slice(ctx, global_attn, 0/*on W*/, max_dis, W + max_dis, 1/*step*/);
        global_attn = ggml_nn_slice(ctx, global_attn, 1/*on H*/, max_dis, H + max_dis, 1/*step*/);
        global_attn = ggml_reshape_3d(ctx, global_attn, H*W, H*W, 1);

        v = ggml_reshape_4d(ctx, v, H*W, 2*hidden_dim, 1, -1);
        v = ggml_transpose(ctx, v);
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
        return output;
    }
};

/*
 HiddenReinforcer(
  (transform): Conv2d(576, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
) */

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

    // def forward(self, g, h):
    //     # tensor [g] size: [2, 512, 35, 56], min: -28.850132, max: 14.702946, mean: -0.759376
    //     # tensor [h] size: [2, 64, 35, 56], min: -4.463562, max: 4.321184, mean: 0.000787

    //     g = torch.cat([g, h], dim=1)
    //     # tensor [g] size: [2, 576, 35, 56], min: -28.850132, max: 14.702946, mean: -0.674914
    //     values = self.transform(g)
    //     forget_gate = torch.sigmoid(values[:, :self.hidden_dim])
    //     update_gate = torch.sigmoid(values[:, self.hidden_dim:self.hidden_dim*2])
    //     new_value = torch.tanh(values[:, self.hidden_dim*2:])

    //     new_h = forget_gate*h*(1-update_gate) + update_gate*new_value
    //     # tensor [new_h] size: [2, 64, 35, 56], min: -3.942453, max: 3.99185, mean: 0.111976
    //     return new_h
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

/*
 BasicBlock(
  (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
) */

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

        // def conv3x3(in_planes, out_planes, stride=1):
        //     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        //              padding=1, dilation=1, bias=False)

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
            downsample_conv.padding = { 1, 1 };
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
        // residual = x
        // out = self.conv1(x)
        // out = self.bn1(out)
        // out = self.relu(out)
        // out = self.conv2(out)
        // out = self.bn2(out)
        // if self.downsample is not None:
        //     residual = self.downsample(x)
        // out += residual
        // out = self.relu(out)
        // return out
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

        // snprintf(s, sizeof(s), "%s%s", prefix, "layer.0.");
        snprintf(s, sizeof(s), "%s%s", prefix, "0.");
        layers[0].setup_weight_names(s);

        for (int i = 1; i < blocks; i++) {
            // snprintf(s, sizeof(s), "%slayer.%d.", prefix, i);
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

// def __init__(self, value_dim, hidden_dim):
//     super().__init__()
//     assert value_dim == 512

//     # network = resnet.resnet18()
//     # self.conv1 = network.conv1
//     # self.bn1 = network.bn1
//     # self.maxpool = network.maxpool
//     # self.layer1 = network.layer1 # 1/4, 64
//     # self.layer2 = network.layer2 # 1/8, 128
//     # self.layer3 = network.layer3 # 1/16, 256

//     self.conv1 = nn.Conv2d(3 + 2, 64, kernel_size=7, stride=2, padding=3, bias=False)
//     self.bn1 = nn.BatchNorm2d(64)
//     self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
//     self.layer1 = resnet.make_basicblock_layer(64, 64, 2, stride=1)
//     self.layer2 = resnet.make_basicblock_layer(64, 128, 2, stride=2)
//     self.layer3 = resnet.make_basicblock_layer(128, 256, 2, stride=2)


struct ValueEncoder {
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

        // self.layer1 = resnet.make_basicblock_layer(64, 64, 2, stride=1)
        // self.layer2 = resnet.make_basicblock_layer(64, 128, 2, stride=2)
        // self.layer3 = resnet.make_basicblock_layer(128, 256, 2, stride=2)

        // self.fuser = FeatureFusionBlock(256)
        // self.hidden_reinforce = HiddenReinforcer()


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

    // def forward(self, image, f16, h16, ref_ab):
    //     # tensor [image] size: [1, 3, 560, 896], min: -5.011174, max: 5.030801, mean: 0.000676
    //     # tensor [f16] size: [1, 1024, 35, 56], min: 0.0, max: 2.263417, mean: 0.06662
    //     # tensor [h16] size: [2, 64, 35, 56], min: -4.67093, max: 4.75749, mean: 0.000638
    //     # tensor [ref_ab] size: [1, 2, 560, 896], min: -4.593725, max: 4.704186, mean: -0.002239
    //     s0 = ref_ab[:, 0:1, :, :]
    //     s1 = ref_ab[:, 1:2, :, :]
    //     ref_ba = torch.cat([s1, s0], dim = 1)
    //     # tensor [ref_ba] size: [1, 2, 560, 896], min: -4.593725, max: 4.704186, mean: -0.002239

    //     g = torch.cat([ref_ab, ref_ba], dim=0)
    //     B, C, H, W = g.size()
    //     g = torch.cat([image.repeat(B, 1, 1, 1), g], dim=1)

    //     g = self.conv1(g)
    //     g = self.bn1(g)     # 1/2, 64
    //     g = self.maxpool(g) # 1/4, 64
    //     g = F.relu(g) 

    //     g = self.layer1(g) # 1/4
    //     g = self.layer2(g) # 1/8
    //     g = self.layer3(g) # 1/16

    //     # handle dim problem raised by vit
    //     g = F.interpolate(g, f16.shape[2:], mode='bilinear', align_corners=False)

    //     g = self.fuser(f16, g)

    //     # tensor [g] size: [2, 512, 35, 56], min: -28.850132, max: 14.702946, mean: -0.759376
    //     # tensor [h16] size: [2, 64, 35, 56], min: -4.463562, max: 4.321184, mean: 0.000787
    //     h = self.hidden_reinforce(g, h16)
    //     # tensor [h] size: [2, 64, 35, 56], min: -4.807474, max: 4.858111, mean: 0.004066

    //     return g, h
    std::vector<ggml_tensor_t *> forward(struct ggml_context* ctx, ggml_tensor_t* image, ggml_tensor_t* f16, ggml_tensor_t* h16, 
        ggml_tensor_t* ref_ab) {
        std::vector<ggml_tensor_t *> xlist;
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

        xlist.push_back(g);
        xlist.push_back(h);

        return xlist;
    }
};

/*
 KeyProjection(
  (key_proj): Conv2d(1024, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (d_proj): Conv2d(1024, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (e_proj): Conv2d(1024, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
) */

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

    std::vector<ggml_tensor_t *> forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // key = self.key_proj(x)
        // shrinkage = self.d_proj(x)**2 + 1
        // selection = torch.sigmoid(self.e_proj(x))
        // return key, shrinkage, selection
        std::vector<ggml_tensor_t *>xlist;
        ggml_tensor_t *key = key_proj.forward(ctx, x);

        ggml_tensor_t *shrinkage = d_proj.forward(ctx, x);
        shrinkage = ggml_mul(ctx, shrinkage, shrinkage);
        shrinkage = ggml_add_constant(ctx, shrinkage, 1.0);

        ggml_tensor_t *selection = e_proj.forward(ctx, x);
        selection = ggml_sigmoid(ctx, selection);

        xlist.push_back(key);
        xlist.push_back(shrinkage);
        xlist.push_back(selection);

    	return xlist;
    }
};

/*
 CrossChannelAttention(
  (to_q): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
  (to_q_dw): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512)
  (to_k): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
  (to_k_dw): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512)
  (to_v): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
  (to_v_dw): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512)
  (to_out): Sequential(
    (0): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
  )
) */


// def __init__(self, dim, heads=8):
//     super().__init__()

//     self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

//     self.heads = heads
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

    // def forward(self, encoder, decoder):
    //     b, c, h, w = encoder.shape

    //     q = self.to_q_dw(self.to_q(encoder))
    //     k = self.to_k_dw(self.to_k(decoder))
    //     v = self.to_v_dw(self.to_v(decoder))
    //     # tensor [q] size: [1, 2048, 35, 56], min: -0.950127, max: 1.107969, mean: 0.006086
    //     # tensor [k] size: [1, 2048, 35, 56], min: -3.406554, max: 3.649786, mean: -0.084804
    //     # tensor [v] size: [1, 2048, 35, 56], min: -8.682275, max: 10.331948, mean: 0.076861


    //     # --------------------------------------------------------------------------------
    //     q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.heads)
    //     k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.heads)
    //     v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.heads)
    //     # q = q.view(b, -1, h*w).view(b, self.heads, -1, h*w)
    //     # k = k.view(b, -1, h*w).view(b, self.heads, -1, h*w)
    //     # v = v.view(b, -1, h*w).view(b, self.heads, -1, h*w)

    //     # tensor [q] size: [1, 8, 256, 1960], min: -0.950127, max: 1.107969, mean: 0.006086
    //     # tensor [k] size: [1, 8, 256, 1960], min: -3.406554, max: 3.649786, mean: -0.084804
    //     # tensor [v] size: [1, 8, 256, 1960], min: -8.682275, max: 10.331948, mean: 0.076861


    //     q = F.normalize(q, dim=-1)
    //     k = F.normalize(k, dim=-1)

    //     attn = (q @ k.transpose(-2, -1)) * self.temperature
    //     attn = attn.softmax(dim=-1)
    //     out = (attn @ v)

    //     # out2 = out.view(b, -1, h*w).view(b, -1, h, w)
    //     # tensor [out1] size: [1, 8, 256, 1960], min: -0.91977, max: 1.421521, mean: 0.080392
    //     out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.heads, h=h, w=w)
    //     # tensor [out2] size: [1, 2048, 35, 56], min: -0.91977, max: 1.421521, mean: 0.080392
    //     # todos.debug.output_var("|out - out2|", (out - out2).abs())

    //     return self.to_out(out)
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
        q = ggml_reshape_3d(ctx, q, H*W, -1, B);
        q = ggml_reshape_4d(ctx, q, H*W, -1, heads, B);
        k = ggml_reshape_3d(ctx, k, H*W, -1, B);
        k = ggml_reshape_4d(ctx, k, H*W, -1, heads, B);
        v = ggml_reshape_3d(ctx, v, H*W, -1, B);
        v = ggml_reshape_4d(ctx, v, H*W, -1, heads, B);
        // # tensor [q] size: [1, 8, 256, 1960], min: -0.950127, max: 1.107969, mean: 0.006086
        // # tensor [k] size: [1, 8, 256, 1960], min: -3.406554, max: 3.649786, mean: -0.084804
        // # tensor [v] size: [1, 8, 256, 1960], min: -8.682275, max: 10.331948, mean: 0.076861

        // q = F.normalize(q, dim=-1)
        // k = F.normalize(k, dim=-1)
        q = ggml_norm_ext(ctx, q, 0/*dim*/, 1e-5);
        k = ggml_norm_ext(ctx, k, 0/*dim*/, 1e-5);

        k = ggml_transpose(ctx, k);
        attn = ggml_nn_mul_mat(ctx, q, k);
        attn = ggml_mul(ctx, attn, temperature);
        attn = ggml_softmax(ctx, attn, 0/*dim*/);

        out = ggml_nn_mul_mat(ctx, attn, v);

        // out = out.view(b, -1, h*w).view(b, -1, h, w)
        out = ggml_reshape_3d(ctx, out, B, -1, H*W);
        out = ggml_reshape_4d(ctx, out, B, -1 , H, W);
        // # tensor [out1] size: [1, 8, 256, 1960], min: -0.91977, max: 1.421521, mean: 0.080392

        return out;
    }
};


struct Fuse {
    int in_feat;
    int out_feat;

    // network params
    struct Conv2d encode_enc;

    struct LayerNorm norm1;
    struct LayerNorm norm2;
    struct CrossChannelAttention crossattn;
    struct LayerNorm norm3;

    void create_weight_tensors(struct ggml_context* ctx) {
        encode_enc.in_channels = in_feat;
        encode_enc.out_channels = out_feat;
        encode_enc.kernel_size = {3, 3};
        encode_enc.stride = { 1, 1 };
        encode_enc.padding = { 1, 1 };
        encode_enc.create_weight_tensors(ctx);

        norm1.normalized_shape = out_feat;
        norm1.dim = 2; // dim on C
        norm1.create_weight_tensors(ctx);

        norm2.normalized_shape = out_feat;
        norm2.dim = 2; // dim on C
        norm2.create_weight_tensors(ctx);

        crossattn.dim = out_feat;
        crossattn.create_weight_tensors(ctx);

        norm3.normalized_shape = out_feat;
        norm3.dim = 2; // dim on C
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
        output = crossattn.forward(ctx, enc, dec);
        output = ggml_add(ctx, output, res);
        output = norm3.forward(ctx, output);
        output = ggml_relu(ctx, output);

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
        // x = self.fc1(x)
        // x = self.act(x)
        // x = self.fc2(x)
        // return x
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

/*
 Attention(
  (qkv): Linear(in_features=384, out_features=1152, bias=True)
  (proj): Linear(in_features=384, out_features=384, bias=True)
) */

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
        int C = (int)x->ne[0]; // 384
        int N = (int)x->ne[1]; // 2561
        int B = (int)x->ne[2];
        // # tensor [x] size: [1, 2561, 384], min: -7.700042, max: 5.273503, mean: 0.004173

        x = qkv.forward(ctx, x);
        // # tensor [x] size: [1, 2561, 1152], min: -11.863246, max: 11.457247, mean: 0.032135
        x = ggml_reshape_4d(ctx, x, C/num_heads, 3*num_heads, N, B);
        ggml_tensor_t *q = ggml_nn_slice(ctx, x, 1, 0*num_heads, 1*num_heads, 1/*step*/);
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)); // [64, 2561, 6, 1] --> [64, 6, 2561, 1]
        q = ggml_scale(ctx, q, scale);
        ggml_tensor_t *k = ggml_nn_slice(ctx, x, 1, 1*num_heads, 2*num_heads, 1/*step*/);
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)); // [64, 2561, 6, 1] --> [64, 6, 2561, 1]
        ggml_tensor_t *v = ggml_nn_slice(ctx, x, 1, 2*num_heads, 3*num_heads, 1/*step*/);
        v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3)); // [64, 2561, 6, 1] --> [64, 6, 2561, 1]

        // # tensor [k] size: [1, 6, 2561, 64], min: -11.863246, max: 11.457247, mean: 0.10765
        // attn = q @ k.transpose(-2, -1)
        // # tensor [attn] size: [1, 6, 2561, 2561], min: 0.342996, max: 53.578362, mean: 20.544596
        // attn = attn.softmax(dim=-1)

        // # tensor [attn@v] size: [1, 6, 2561, 64], min: -1.033432, max: 1.115259, mean: -0.009298
        // x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        // x = self.proj(x)
        // return x # [1, 2561, 384] ???
        k = ggml_transpose(ctx, k);
        ggml_tensor_t *attn = ggml_nn_mul_mat(ctx, q, k);
        attn = ggml_softmax(ctx, attn, 0/*dim*/);
        attn = ggml_nn_mul_mat(ctx, attn, v);

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
        // def attn_residual_func(x):
        //     return self.ls1(self.attn(self.norm1(x)))

        // def ffn_residual_func(x):
        //     return self.ls2(self.mlp(self.norm2(x)))

        // x = x + attn_residual_func(x)
        // x = x + ffn_residual_func(x)
        // return x
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

/*
 PatchEmbed(
  (proj): Conv2d(3, 384, kernel_size=(14, 14), stride=(14, 14))
  (norm): Identity()
) */

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
        int W = (int)x->ne[0];
        int H = (int)x->ne[1];
        int C = (int)x->ne[2];
        int B = (int)x->ne[3];

        x = proj.forward(ctx, x);
        x = ggml_reshape_3d(ctx, x, W*H, C, B);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)); // [HW, C, B] -> [C, HW, B]
    	return x;
    }
};


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

    // def interpolate_pos_encoding(self, x, H, W):
    //     # x.size() -- [1, 2561, 384], H -- 560, W -- 896
    //     B, NP, D = x.size() 
    //     N = self.pos_embed.shape[1] - 1
    //     if N == NP - 1 and W == H:
    //         return self.pos_embed

    //     pos_embed = self.pos_embed.float() # [1, 1370, 384]
    //     class_pos_embed = pos_embed[:, 0:1]
    //     patch_pos_embed = pos_embed[:, 1:]

    //     M = int(math.sqrt(N))  # Recover the number of patches in each dimension
    //     assert N == M * M
    //     NH = (H + self.patch_size - 1) // self.patch_size
    //     NW = (W + self.patch_size - 1) // self.patch_size

    //     # tensor [patch_pos_embed] size: [1, 1369, 384], min: -0.1611, max: 0.126807, mean: 8.3e-05
    //     patch_pos_embed = patch_pos_embed.reshape(1, M, M, D)  # (1, M, M, D) -- (1, 37, 37, 384)
    //     patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2) # [1. 37, 37, 384] --> [1, 384, 37, 37]

    //     patch_pos_embed = F.interpolate(patch_pos_embed, size=(NH, NW), mode="bicubic", antialias=False)
    //     # tensor [patch_pos_embed] size: [1, 384, 40, 64], min: -0.162149, max: 0.127178, mean: 8.2e-05
    //     patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, D)
    //     # tensor [patch_pos_embed] size: [1, 2560, 384], min: -0.162149, max: 0.127178, mean: 8.2e-05

    //     # class_pos_embed.size() -- [1, 384]
    //     return torch.cat((class_pos_embed, patch_pos_embed), dim=1).to(x.dtype)

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

        x = ggml_concat(ctx, class_pos_embed, patch_pos_embed, 1/*middle dim*/);

        return x;
    }

    // def forward(self, x):
    //     # tensor [x] size: [1, 3, 560, 896], min: -0.495599, max: 0.496816, mean: -0.109927
    //     B, C, H, W = x.shape

    //     x = self.patch_embed(x)
    //     # tensor [x] size: [1, 2560, 384], min: -0.818547, max: 0.587891, mean: -0.002679

    //     x = torch.cat((self.cls_token.expand(B, 1, -1), x), dim=1)
    //     # tensor [x] size: [1, 2561, 384], min: -0.818547, max: 0.587891, mean: -0.002678

    //     x = x + self.interpolate_pos_encoding(x, H, W)

    //     NH = (H + self.patch_size - 1) // self.patch_size
    //     NW = (W + self.patch_size - 1) // self.patch_size

    //     outputs = []
    //     for i, blk in enumerate(self.blocks):
    //         x = blk(x)
    //         if i in [8, 9, 10, 11]:
    //             out = self.norm(x)  # [1, 2561, 384]
    //             out = out[:, 1:, :] # [1, 2560, 384]
    //             # w // self.patch_size, h // self.patch_size === 40, 64
    //             out = out.reshape(B, NH, NW, -1).permute(0, 3, 1, 2).contiguous()
    //             # [1, 40, 60, 384] --> [1, 384, 40, 64]
    //             outputs.append(out)

    //     # outputs is list: len = 4
    //     #     tensor [item] size: [1, 384, 40, 64], min: -64.29377, max: 62.932507, mean: 0.046734
    //     #     tensor [item] size: [1, 384, 40, 64], min: -58.107525, max: 53.356197, mean: 0.016807
    //     #     tensor [item] size: [1, 384, 40, 64], min: -48.493, max: 43.823879, mean: 0.01582
    //     #     tensor [item] size: [1, 384, 40, 64], min: -22.330799, max: 15.610704, mean: 0.011709
    //     return outputs


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

        //     outputs = []
        //     for i, blk in enumerate(self.blocks):
        //         x = blk(x)
        //         if i in [8, 9, 10, 11]:
        //             out = self.norm(x)  # [1, 2561, 384]
        //             out = out[:, 1:, :] # [1, 2560, 384]
        //             # w // self.patch_size, h // self.patch_size === 40, 64
        //             out = out.reshape(B, NH, NW, -1).permute(0, 3, 1, 2).contiguous()
        //             # [1, 40, 64, 384] --> [1, 384, 40, 64]
        //             outputs.append(out)
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
        // tokens = self.backbone(x) #.get_intermediate_layers(x, n=[8, 9, 10, 11], reshape=True) # last n=4 [8, 9, 10, 11]
        // # tokens is tuple: len = 4
        // #     tensor [item] size: [1, 384, 40, 64], min: -64.093712, max: 65.633827, mean: 0.04372
        // #     tensor [item] size: [1, 384, 40, 64], min: -60.8563, max: 49.656631, mean: 0.003902
        // #     tensor [item] size: [1, 384, 40, 64], min: -46.128963, max: 40.135544, mean: 0.009809
        // #     tensor [item] size: [1, 384, 40, 64], min: -21.549391, max: 19.685974, mean: 0.007802
        // f16 = torch.cat(tokens, dim=1)
        // # tensor [f16] size: [1, 1536, 40, 64], min: -64.093712, max: 65.633827, mean: 0.016308
        // f16 = self.conv3(f16)
        // f16 = self.bn3(f16)
        // f16 = F.relu(f16)
        // old_size = (f16.shape[2], f16.shape[3])
        // new_size = (int(old_size[0]*14/16), int(old_size[1]*14/16))
        // f16 = F.interpolate(f16, size=new_size, mode='bilinear', align_corners=False) # scale_factor=3.5
        // # tensor [f16] size: [1, 1536, 35, 56], min: 0.0, max: 10.015625, mean: 0.865097
        // return f16
        std::vector<ggml_tensor_t *> tokens = backbone.forward(ctx, x);
        ggml_tensor_t *f16 = ggml_cat(ctx, 4, tokens[0], tokens[1], tokens[2], tokens[3], 2/*dim on C*/);
        f16 = conv3.forward(ctx, f16);
        f16 = bn3.forward(ctx, f16);
        f16 = ggml_relu(ctx, f16);
        int W = (int)f16->ne[0];
        int H = (int)f16->ne[0];
        W = (W * 14)/16;
        H = (H * 14)/16;
        f16 = ggml_interpolate(ctx, f16, 0 /*dim on W*/, W);
        f16 = ggml_interpolate(ctx, f16, 1 /*dim on H*/, H);
        return f16;
    }
};

/*
 Bottleneck(
  (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
  (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
  (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
) */
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
            downsample_conv.padding = { 1, 1 };
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


    // def forward(self, f):
    //     # tensor [f] size: [1, 3, 560, 896], min: -0.994517, max: 1.0, mean: -0.189531

    //     x = self.conv1(f) 
    //     x = self.bn1(x)
    //     x = F.relu(x)   # 1/2, 64
    //     x = self.maxpool(x)  # 1/4, 64
    //     f4 = self.res2(x)   # 1/4, 256
    //     f8 = self.layer2(f4) # 1/8, 512
    //     f16 = self.layer3(f8) # 1/16, 1024
    //     # tensor [f16] size: [1, 1024, 35, 56], min: 0.0, max: 2.109375, mean: 0.067145

    //     dino_f16 = self.network2(f) # 1/14, 384  ->   interp to 1/16
    //     # tensor [dino_f16] size: [1, 1536, 35, 56], min: 0.0, max: 10.015625, mean: 0.865097

    //     g16 = self.fuse1(dino_f16, f16)
    //     g8 = self.fuse2(self.upsample2(dino_f16), f8)
    //     g4 = self.fuse3(self.upsample4(dino_f16), f4)
    //     # tensor [g16] size: [1, 1024, 35, 56], min: 0.0, max: 2.594945, mean: 0.063114
    //     # tensor [g8] size: [1, 512, 70, 112], min: 0.0, max: 1.842727, mean: 0.090533
    //     # tensor [g4] size: [1, 256, 140, 224], min: 0.0, max: 6.625021, mean: 0.200046

    //     return g16, g8, g4

    std::vector<ggml_tensor_t *> forward(struct ggml_context* ctx, ggml_tensor_t* f) {
        ggml_tensor_t *x, *f16, *f8, *f4, *dino_f16, *g16, *g8, *g4;
        std::vector<ggml_tensor_t *> xlist;
        x = conv1.forward(ctx, f);
        x = bn1.forward(ctx, x);
        x = ggml_relu(ctx, x);
        x = maxpool.forward(ctx, x);
        f4 = res2.forward(ctx, x);
        f8 = layer2.forward(ctx, f4);
        f16 = layer3.forward(ctx, f8);
        int W = (int)f16->ne[0];
        int H = (int)f16->ne[1];

        dino_f16 = network2.forward(ctx, f);
        g16 = fuse1.forward(ctx, dino_f16, f16);

        // self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        g8 = ggml_interpolate(ctx, f16, 0/*on W*/, 2*W);
        g8 = ggml_interpolate(ctx, f16, 0/*on W*/, 2*W);
        g8 = fuse2.forward(ctx, g8, f8);

        // self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
        g4 = ggml_interpolate(ctx, f16, 0/*on W*/, 4*W);
        g4 = ggml_interpolate(ctx, f16, 0/*on W*/, 4*W);
        g4 = fuse3.forward(ctx, g4, f4);

        xlist.push_back(g16);
        xlist.push_back(g8);
        xlist.push_back(g8);

        return xlist;
    }
};

struct ColorMNet : GGMLNetwork {
    // network hparams
    int MAX_H = 1024;
    int MAX_W = 1024;
    int MAX_TIMES = 112;

    // network params
    struct DINOv2_v6 key_encoder;
    struct KeyProjection key_proj;
    struct ValueEncoder value_encoder;
    struct LocalAttention short_term_attn;
    struct Decoder decoder;

    // def __init__(self):
    //     super().__init__()
    //     self.MAX_H = 1024
    //     self.MAX_W = 1024
    //     self.MAX_TIMES = 112
    //     # ------------------------------------------------------------------------------
    //     self.key_dim = 64
    //     self.value_dim = 512
    //     self.hidden_dim = 64

    //     self.key_encoder = DINOv2_v6()
    //     self.key_proj = KeyProjection(1024, self.key_dim)

    //     self.value_encoder = ValueEncoder(self.value_dim, self.hidden_dim)
    //     self.short_term_attn = LocalAttention(d_qk=64, d_vu=512 * 2, d_att=64, max_dis=7)
    //     self.decoder = Decoder(self.value_dim, self.hidden_dim)


    void create_weight_tensors(struct ggml_context* ctx) {
        key_encoder.create_weight_tensors(ctx);

        key_proj.create_weight_tensors(ctx);

        value_encoder.create_weight_tensors(ctx);

        short_term_attn.create_weight_tensors(ctx);

        decoder.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "key_encoder.");
        key_encoder.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "key_proj.");
        key_proj.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "value_encoder.");
        value_encoder.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "short_term_attn.");
        short_term_attn.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "decoder.");
        decoder.setup_weight_names(s);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, int argc, ggml_tensor_t* argv[]) {
        GGML_UNUSED(argc);
        ggml_tensor_t *x = argv[0];

    	return ggml_dup(ctx, x);
    }
};


struct VideoColorNetwork {
    ColorMNet net;
    GGMLModel model;

    TENSOR *R1 = NULL;
    TENSOR *R2 = NULL;
    TENSOR *R3 = NULL;
    TENSOR *R4 = NULL;

    int init(int device) {
        // -----------------------------------------------------------------------------------------
        net.set_device(device);
        net.start_engine();
        net.dump();

        check_point(model.preload("models/video_color_f32.gguf") == RET_OK);
        load();

        return RET_OK;
    }

    int load() {
        return net.load_weight(&model, "");
    }

    TENSOR *forward(TENSOR *input_tensor) {
        TENSOR *argv[5];
        argv[0] = input_tensor ;
        argv[1] = R1;
        argv[2] = R2;
        argv[3] = R3;
        argv[4] = R4;

        TENSOR *R = net.engine_forward(ARRAY_SIZE(argv), argv);
        // Update R1/R2/R3/R4
        {
            TENSOR *x;

            x = net.get_output_tensor((char *)"R1");
            if (tensor_valid(x)) {
                tensor_destroy(R1);
                R1 = tensor_slice_chan(x, x->chan/2, x->chan);
                tensor_destroy(x);
                // tensor_show((char *)"R1", R1);
            }

            x = net.get_output_tensor((char *)"R2");
            if (tensor_valid(x)) {
                tensor_destroy(R2);
                R2 = tensor_slice_chan(x, x->chan/2, x->chan);
                tensor_destroy(x);
                // tensor_show((char *)"R2", R2);
            }

            x = net.get_output_tensor((char *)"R3");
            if (tensor_valid(x)) {
                tensor_destroy(R3);
                R3 = tensor_slice_chan(x, x->chan/2, x->chan);
                tensor_destroy(x);
                // tensor_show((char *)"R3", R3);
            }

            x = net.get_output_tensor((char *)"R4");
            if (tensor_valid(x)) {
                tensor_destroy(R4);
                R4 = tensor_slice_chan(x, x->chan/2, x->chan);
                tensor_destroy(x);
                // tensor_show((char *)"R4", R4);
            }
        }

        return R;
    }

    void exit() {
        tensor_destroy(R4);
        tensor_destroy(R3);
        tensor_destroy(R2);
        tensor_destroy(R1);

        model.clear();
        net.stop_engine();
    }
};


#endif // __COLORMNET__H__
