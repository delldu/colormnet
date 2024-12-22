#ifndef __COLORMNET__H__
#define __COLORMNET__H__
#include "ggml_engine.h"
#include "ggml_nn.h"

#pragma GCC diagnostic ignored "-Wformat-truncation"


/*
 GroupResBlock(
  (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
) */

struct GroupResBlock {
    // network hparams
    

    // network params
    struct Conv2d conv1;
    struct Conv2d conv2;


    void create_weight_tensors(struct ggml_context* ctx) {
        conv1.in_channels = 16;
        conv1.out_channels = 4;
        conv1.kernel_size = {1, 1};
        conv1.stride = { 1, 1 };
        conv1.padding = { 0, 0 };
        conv1.create_weight_tensors(ctx);

        conv2.in_channels = 16;
        conv2.out_channels = 4;
        conv2.kernel_size = {1, 1};
        conv2.stride = { 1, 1 };
        conv2.padding = { 0, 0 };
        conv2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv1.");
        conv1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv2.");
        conv2.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // please implement forward by your self, please !!!

        return x;
    }
};


/*
 UpsampleBlock(
  (skip_conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (out_conv): GroupResBlock(
    (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
) */

struct UpsampleBlock {
    // network hparams
    

    // network params
    struct Conv2d skip_conv;
    struct GroupResBlock out_conv;


    void create_weight_tensors(struct ggml_context* ctx) {
        skip_conv.in_channels = 16;
        skip_conv.out_channels = 4;
        skip_conv.kernel_size = {1, 1};
        skip_conv.stride = { 1, 1 };
        skip_conv.padding = { 0, 0 };
        skip_conv.create_weight_tensors(ctx);

        out_conv.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "skip_conv.");
        skip_conv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "out_conv.");
        out_conv.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 HiddenUpdater(
  (g16_conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
  (g8_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  (g4_conv): Conv2d(257, 256, kernel_size=(1, 1), stride=(1, 1))
  (transform): Conv2d(320, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
) */

struct HiddenUpdater {
    // network hparams
    int hidden_dim = 64;

    // network params
    struct Conv2d g16_conv;
    struct Conv2d g8_conv;
    struct Conv2d g4_conv;
    struct Conv2d transform;

    void create_weight_tensors(struct ggml_context* ctx) {
        g16_conv.in_channels = 16;
        g16_conv.out_channels = 4;
        g16_conv.kernel_size = {1, 1};
        g16_conv.stride = { 1, 1 };
        g16_conv.padding = { 0, 0 };
        g16_conv.create_weight_tensors(ctx);

        g8_conv.in_channels = 16;
        g8_conv.out_channels = 4;
        g8_conv.kernel_size = {1, 1};
        g8_conv.stride = { 1, 1 };
        g8_conv.padding = { 0, 0 };
        g8_conv.create_weight_tensors(ctx);

        g4_conv.in_channels = 16;
        g4_conv.out_channels = 4;
        g4_conv.kernel_size = {1, 1};
        g4_conv.stride = { 1, 1 };
        g4_conv.padding = { 0, 0 };
        g4_conv.create_weight_tensors(ctx);

        transform.in_channels = 16;
        transform.out_channels = 4;
        transform.kernel_size = {1, 1};
        transform.stride = { 1, 1 };
        transform.padding = { 0, 0 };
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

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

struct Decoder {
    // network hparams
    

    // network params
    struct FeatureFusionBlock fuser;
    struct HiddenUpdater hidden_update;
    struct UpsampleBlock up_16_8;
    struct UpsampleBlock up_8_4;
    struct Conv2d pred;


    void create_weight_tensors(struct ggml_context* ctx) {
        fuser.create_weight_tensors(ctx);
        hidden_update.create_weight_tensors(ctx);
        up_16_8.create_weight_tensors(ctx);
        up_8_4.create_weight_tensors(ctx);

        pred.in_channels = 16;
        pred.out_channels = 4;
        pred.kernel_size = {1, 1};
        pred.stride = { 1, 1 };
        pred.padding = { 0, 0 };
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

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 DWConv2d(
  (conv): Conv2d(1024, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1024, bias=False)
) */

struct DWConv2d {
    // network hparams
    // network params
    struct Conv2d conv;  // torch.float32, [1024, 1, 5, 5]


    void create_weight_tensors(struct ggml_context* ctx) {
        conv.in_channels = 16;
        conv.out_channels = 4;
        conv.kernel_size = {1, 1};
        conv.stride = { 1, 1 };
        conv.padding = { 0, 0 };
        conv.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.");
        conv.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 SpatialCorrelationSampler() */

struct SpatialCorrelationSampler {
    // network hparams
    int kernel_size = 1;
    int patch_size = 15;
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    int dilation_patch = 1;

    // network params
    void create_weight_tensors(struct ggml_context* ctx) {
        
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

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
    // network hparams
    int window_size = 15;
    int max_dis = 7;
    int hidden_dim = 1024;
    int d_att = 64;

    // network params
    struct Conv2d relative_emb_k;
    struct SpatialCorrelationSampler correlation_sampler;
    struct DWConv2d dw_conv;

    struct Linear projection;  // torch.float32, [1024, 1024] 


    void create_weight_tensors(struct ggml_context* ctx) {
        relative_emb_k.in_channels = 16;
        relative_emb_k.out_channels = 4;
        relative_emb_k.kernel_size = {1, 1};
        relative_emb_k.stride = { 1, 1 };
        relative_emb_k.padding = { 0, 0 };
        relative_emb_k.create_weight_tensors(ctx);

        correlation_sampler.create_weight_tensors(ctx);
        dw_conv.create_weight_tensors(ctx);

        projection.in_features = 1;
        projection.out_features = 1;
        projection.has_bias = true; // Fixed default
        projection.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "relative_emb_k.");
        relative_emb_k.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "correlation_sampler.");
        correlation_sampler.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "dw_conv.");
        dw_conv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "projection.");
        projection.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 HiddenReinforcer(
  (transform): Conv2d(576, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
) */

struct HiddenReinforcer {
    // network hparams
    int hidden_dim = 64;

    // network params
    struct Conv2d transform;

    void create_weight_tensors(struct ggml_context* ctx) {
        transform.in_channels = 16;
        transform.out_channels = 4;
        transform.kernel_size = {1, 1};
        transform.stride = { 1, 1 };
        transform.padding = { 0, 0 };
        transform.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "transform.");
        transform.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 BasicConv(
  (conv): Conv2d(2, 1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
) */

struct BasicConv {
    // network hparams
    

    // network params
    struct Conv2d conv;


    void create_weight_tensors(struct ggml_context* ctx) {
        conv.in_channels = 16;
        conv.out_channels = 4;
        conv.kernel_size = {1, 1};
        conv.stride = { 1, 1 };
        conv.padding = { 0, 0 };
        conv.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.");
        conv.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 ChannelPool() */

struct ChannelPool {
    // network hparams
    

    // network params
    


    void create_weight_tensors(struct ggml_context* ctx) {
        
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 SpatialGate(
  (compress): ChannelPool()
  (spatial): BasicConv(
    (conv): Conv2d(2, 1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
  )
) */

struct SpatialGate {
    // network hparams
    

    // network params
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

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 Flatten() */

struct Flatten {
    // network hparams
    

    // network params
    


    void create_weight_tensors(struct ggml_context* ctx) {
        
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 ChannelGate(
  (mlp): Sequential(
    (0): Flatten()
    (1): Linear(in_features=512, out_features=32, bias=True)
    (2): ReLU()
    (3): Linear(in_features=32, out_features=512, bias=True)
  )
) */

struct ChannelGate {
    // network hparams
    int gate_channels = 512;

    // network params
    struct Flatten mlp_0;

    struct Linear mlp_1;
    struct Linear mlp_3;


    void create_weight_tensors(struct ggml_context* ctx) {
        mlp_0.create_weight_tensors(ctx);


        mlp_1.in_features = 1;
        mlp_1.out_features = 1;
        mlp_1.has_bias = true; // Fixed default
        mlp_1.create_weight_tensors(ctx);

        mlp_3.in_features = 1;
        mlp_3.out_features = 1;
        mlp_3.has_bias = true; // Fixed default
        mlp_3.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "mlp.0.");
        mlp_0.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "mlp.1.");
        mlp_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "mlp.3.");
        mlp_3.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 CBAM(
  (ChannelGate): ChannelGate(
    (mlp): Sequential(
      (0): Flatten()
      (1): Linear(in_features=512, out_features=32, bias=True)
      (2): ReLU()
      (3): Linear(in_features=32, out_features=512, bias=True)
    )
  )
  (SpatialGate): SpatialGate(
    (compress): ChannelPool()
    (spatial): BasicConv(
      (conv): Conv2d(2, 1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
    )
  )
) */

struct CBAM {
    // network hparams
    

    // network params
    struct ChannelGate ChannelGate;
    struct SpatialGate SpatialGate;


    void create_weight_tensors(struct ggml_context* ctx) {
        ChannelGate.create_weight_tensors(ctx);
        SpatialGate.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "ChannelGate.");
        ChannelGate.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "SpatialGate.");
        SpatialGate.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};


struct FeatureFusionBlock {
    // network hparams
    

    // network params
    struct GroupResBlock block1;
    struct CBAM attention;
    struct GroupResBlock block2;


    void create_weight_tensors(struct ggml_context* ctx) {
        block1.create_weight_tensors(ctx);
        attention.create_weight_tensors(ctx);
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

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
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
    // network hparams
    int stride = 1;

    // network params
    struct Conv2d conv1;
    struct BatchNorm2d bn1;

    struct Conv2d conv2;
    struct BatchNorm2d bn2;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv1.in_channels = 16;
        conv1.out_channels = 4;
        conv1.kernel_size = {1, 1};
        conv1.stride = { 1, 1 };
        conv1.padding = { 0, 0 };
        conv1.create_weight_tensors(ctx);

        bn1.num_features = 1;
        bn1.create_weight_tensors(ctx);

        conv2.in_channels = 16;
        conv2.out_channels = 4;
        conv2.kernel_size = {1, 1};
        conv2.stride = { 1, 1 };
        conv2.padding = { 0, 0 };
        conv2.create_weight_tensors(ctx);

        bn2.num_features = 1;
        bn2.create_weight_tensors(ctx);
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
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};


struct ValueEncoder {
    // network hparams
    
    // network params
    struct Conv2d conv1;
    struct BatchNorm2d bn1;

    struct BasicBlock layer1_0;
    struct BasicBlock layer1_1;
    struct BasicBlock layer2_0;
    struct BasicBlock layer2_1;
    struct BasicBlock layer3_0;
    struct BasicBlock layer3_1;
    struct FeatureFusionBlock fuser;
    struct HiddenReinforcer hidden_reinforce;


    void create_weight_tensors(struct ggml_context* ctx) {
        conv1.in_channels = 16;
        conv1.out_channels = 4;
        conv1.kernel_size = {1, 1};
        conv1.stride = { 1, 1 };
        conv1.padding = { 0, 0 };
        conv1.create_weight_tensors(ctx);

        bn1.num_features = 1;
        bn1.create_weight_tensors(ctx);

        layer1_0.create_weight_tensors(ctx);
        layer1_1.create_weight_tensors(ctx);
        layer2_0.create_weight_tensors(ctx);
        layer2_1.create_weight_tensors(ctx);
        layer3_0.create_weight_tensors(ctx);
        layer3_1.create_weight_tensors(ctx);
        fuser.create_weight_tensors(ctx);
        hidden_reinforce.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv1.");
        conv1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "bn1.");
        bn1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "layer1.0.");
        layer1_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer1.1.");
        layer1_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer2.0.");
        layer2_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer2.1.");
        layer2_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer3.0.");
        layer3_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer3.1.");
        layer3_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "fuser.");
        fuser.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "hidden_reinforce.");
        hidden_reinforce.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 KeyProjection(
  (key_proj): Conv2d(1024, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (d_proj): Conv2d(1024, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (e_proj): Conv2d(1024, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
) */

struct KeyProjection {
    // network hparams
    

    // network params
    struct Conv2d key_proj;
    struct Conv2d d_proj;
    struct Conv2d e_proj;

    void create_weight_tensors(struct ggml_context* ctx) {
        key_proj.in_channels = 16;
        key_proj.out_channels = 4;
        key_proj.kernel_size = {1, 1};
        key_proj.stride = { 1, 1 };
        key_proj.padding = { 0, 0 };
        key_proj.create_weight_tensors(ctx);

        d_proj.in_channels = 16;
        d_proj.out_channels = 4;
        d_proj.kernel_size = {1, 1};
        d_proj.stride = { 1, 1 };
        d_proj.padding = { 0, 0 };
        d_proj.create_weight_tensors(ctx);

        e_proj.in_channels = 16;
        e_proj.out_channels = 4;
        e_proj.kernel_size = {1, 1};
        e_proj.stride = { 1, 1 };
        e_proj.padding = { 0, 0 };
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

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
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

struct CrossChannelAttention {
    // network hparams
    int heads = 8;

    // network params
    struct Conv2d to_q;
    struct Conv2d to_q_dw;
    struct Conv2d to_k;
    struct Conv2d to_k_dw;
    struct Conv2d to_v;
    struct Conv2d to_v_dw;
    struct Conv2d to_out;

    void create_weight_tensors(struct ggml_context* ctx) {
        to_q.in_channels = 16;
        to_q.out_channels = 4;
        to_q.kernel_size = {1, 1};
        to_q.stride = { 1, 1 };
        to_q.padding = { 0, 0 };
        to_q.create_weight_tensors(ctx);

        to_q_dw.in_channels = 16;
        to_q_dw.out_channels = 4;
        to_q_dw.kernel_size = {1, 1};
        to_q_dw.stride = { 1, 1 };
        to_q_dw.padding = { 0, 0 };
        to_q_dw.create_weight_tensors(ctx);

        to_k.in_channels = 16;
        to_k.out_channels = 4;
        to_k.kernel_size = {1, 1};
        to_k.stride = { 1, 1 };
        to_k.padding = { 0, 0 };
        to_k.create_weight_tensors(ctx);

        to_k_dw.in_channels = 16;
        to_k_dw.out_channels = 4;
        to_k_dw.kernel_size = {1, 1};
        to_k_dw.stride = { 1, 1 };
        to_k_dw.padding = { 0, 0 };
        to_k_dw.create_weight_tensors(ctx);

        to_v.in_channels = 16;
        to_v.out_channels = 4;
        to_v.kernel_size = {1, 1};
        to_v.stride = { 1, 1 };
        to_v.padding = { 0, 0 };
        to_v.create_weight_tensors(ctx);

        to_v_dw.in_channels = 16;
        to_v_dw.out_channels = 4;
        to_v_dw.kernel_size = {1, 1};
        to_v_dw.stride = { 1, 1 };
        to_v_dw.padding = { 0, 0 };
        to_v_dw.create_weight_tensors(ctx);

        to_out.in_channels = 16;
        to_out.out_channels = 4;
        to_out.kernel_size = {1, 1};
        to_out.stride = { 1, 1 };
        to_out.padding = { 0, 0 };
        to_out.create_weight_tensors(ctx);                                        
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

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

        snprintf(s, sizeof(s), "%s%s", prefix, "to_out.");
        to_out.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 LayerNorm2d() */

struct LayerNorm2d {
    // network hparams
    float eps = 1e-06;

    // network params
    


    void create_weight_tensors(struct ggml_context* ctx) {
        
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};


struct Fuse {
    // network hparams

    // network params
    struct Conv2d encode_enc;

    struct LayerNorm2d norm1;
    struct LayerNorm2d norm2;
    struct CrossChannelAttention crossattn;
    struct LayerNorm2d norm3;

    void create_weight_tensors(struct ggml_context* ctx) {
        encode_enc.in_channels = 16;
        encode_enc.out_channels = 4;
        encode_enc.kernel_size = {1, 1};
        encode_enc.stride = { 1, 1 };
        encode_enc.padding = { 0, 0 };
        encode_enc.create_weight_tensors(ctx);

        norm1.create_weight_tensors(ctx);
        norm2.create_weight_tensors(ctx);
        crossattn.create_weight_tensors(ctx);
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

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};


struct Mlp {
    // network hparams
    

    // network params
    struct Linear fc1;
    struct Linear fc2;


    void create_weight_tensors(struct ggml_context* ctx) {
        fc1.in_features = 1;
        fc1.out_features = 1;
        fc1.has_bias = true; // Fixed default
        fc1.create_weight_tensors(ctx);

        fc2.in_features = 1;
        fc2.out_features = 1;
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

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 LayerScale() */

struct LayerScale {
    // network hparams
    

    // network params
    


    void create_weight_tensors(struct ggml_context* ctx) {
        
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 Attention(
  (qkv): Linear(in_features=384, out_features=1152, bias=True)
  (proj): Linear(in_features=384, out_features=384, bias=True)
) */

struct Attention {
    // network hparams
    int num_heads = 6;
    float scale = 0.125;

    // network params
    struct Linear qkv;
    struct Linear proj;

    void create_weight_tensors(struct ggml_context* ctx) {
        qkv.in_features = 1;
        qkv.out_features = 1;
        qkv.has_bias = true; // Fixed default
        qkv.create_weight_tensors(ctx);

        proj.in_features = 1;
        proj.out_features = 1;
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

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};


struct NestedTensorBlock {
    // network hparams
    

    // network params
    struct LayerNorm norm1;
    struct Attention attn;
    struct LayerScale ls1;
    struct LayerNorm norm2;
    struct Mlp mlp;
    struct LayerScale ls2;


    void create_weight_tensors(struct ggml_context* ctx) {
        norm1.normalized_shape = 1;
        norm1.create_weight_tensors(ctx);

        attn.create_weight_tensors(ctx);
        ls1.create_weight_tensors(ctx);

        norm2.normalized_shape = 1;
        norm2.create_weight_tensors(ctx);

        mlp.create_weight_tensors(ctx);
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

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 PatchEmbed(
  (proj): Conv2d(3, 384, kernel_size=(14, 14), stride=(14, 14))
  (norm): Identity()
) */

struct PatchEmbed {
    // network hparams
    int num_patches = 1369;

    // network params
    struct Conv2d proj;

    void create_weight_tensors(struct ggml_context* ctx) {
        proj.in_channels = 16;
        proj.out_channels = 4;
        proj.kernel_size = {1, 1};
        proj.stride = { 1, 1 };
        proj.padding = { 0, 0 };
        proj.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "proj.");
        proj.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};


struct DinoVisionTransformer {
    // network hparams
    int num_tokens = 1;
    int patch_size = 14;

    // network params
    struct PatchEmbed patch_embed;
    struct NestedTensorBlock blocks_0;
    struct NestedTensorBlock blocks_1;
    struct NestedTensorBlock blocks_2;
    struct NestedTensorBlock blocks_3;
    struct NestedTensorBlock blocks_4;
    struct NestedTensorBlock blocks_5;
    struct NestedTensorBlock blocks_6;
    struct NestedTensorBlock blocks_7;
    struct NestedTensorBlock blocks_8;
    struct NestedTensorBlock blocks_9;
    struct NestedTensorBlock blocks_10;
    struct NestedTensorBlock blocks_11;

    struct LayerNorm norm;


    void create_weight_tensors(struct ggml_context* ctx) {
        patch_embed.create_weight_tensors(ctx);
        blocks_0.create_weight_tensors(ctx);
        blocks_1.create_weight_tensors(ctx);
        blocks_2.create_weight_tensors(ctx);
        blocks_3.create_weight_tensors(ctx);
        blocks_4.create_weight_tensors(ctx);
        blocks_5.create_weight_tensors(ctx);
        blocks_6.create_weight_tensors(ctx);
        blocks_7.create_weight_tensors(ctx);
        blocks_8.create_weight_tensors(ctx);
        blocks_9.create_weight_tensors(ctx);
        blocks_10.create_weight_tensors(ctx);
        blocks_11.create_weight_tensors(ctx);

        norm.normalized_shape = 1;
        norm.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "patch_embed.");
        patch_embed.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.0.");
        blocks_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.1.");
        blocks_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.2.");
        blocks_2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.3.");
        blocks_3.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.4.");
        blocks_4.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.5.");
        blocks_5.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.6.");
        blocks_6.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.7.");
        blocks_7.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.8.");
        blocks_8.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.9.");
        blocks_9.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.10.");
        blocks_10.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.11.");
        blocks_11.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "norm.");
        norm.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

struct Segmentor {
    // network hparams
    

    // network params
    struct DinoVisionTransformer backbone;
    struct Conv2d conv3;  // torch.float32, [1536, 1536, 1, 1] 
    struct BatchNorm2d bn3;


    void create_weight_tensors(struct ggml_context* ctx) {
        backbone.create_weight_tensors(ctx);

        conv3.in_channels = 16;
        conv3.out_channels = 4;
        conv3.kernel_size = {1, 1};
        conv3.stride = { 1, 1 };
        conv3.padding = { 0, 0 };
        conv3.create_weight_tensors(ctx);

        bn3.num_features = 1;
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

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
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
    // network hparams
    int stride = 1;

    // network params
    struct Conv2d conv1;
    struct BatchNorm2d bn1;
    struct Conv2d conv2;
    struct BatchNorm2d bn2;
    struct Conv2d conv3;
    struct BatchNorm2d bn3;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv1.in_channels = 16;
        conv1.out_channels = 4;
        conv1.kernel_size = {1, 1};
        conv1.stride = { 1, 1 };
        conv1.padding = { 0, 0 };
        conv1.create_weight_tensors(ctx);

        bn1.num_features = 1;
        bn1.create_weight_tensors(ctx);

        conv2.in_channels = 16;
        conv2.out_channels = 4;
        conv2.kernel_size = {1, 1};
        conv2.stride = { 1, 1 };
        conv2.padding = { 0, 0 };
        conv2.create_weight_tensors(ctx);

        bn2.num_features = 1;
        bn2.create_weight_tensors(ctx);

        conv3.in_channels = 16;
        conv3.out_channels = 4;
        conv3.kernel_size = {1, 1};
        conv3.stride = { 1, 1 };
        conv3.padding = { 0, 0 };
        conv3.create_weight_tensors(ctx);

        bn3.num_features = 1;
        bn3.create_weight_tensors(ctx);
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
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

struct DINOv2_v6 {
    // network hparams

    // network params
    struct Conv2d conv1;
    struct BatchNorm2d bn1;

    struct Bottleneck res2_0;
    struct Bottleneck res2_1;
    struct Bottleneck res2_2;
    struct Bottleneck layer2_0;
    struct Bottleneck layer2_1;
    struct Bottleneck layer2_2;
    struct Bottleneck layer2_3;
    struct Bottleneck layer3_0;
    struct Bottleneck layer3_1;
    struct Bottleneck layer3_2;
    struct Bottleneck layer3_3;
    struct Bottleneck layer3_4;
    struct Bottleneck layer3_5;
    struct Segmentor network2;

    struct Fuse fuse1;
    struct Fuse fuse2;
    struct Fuse fuse3;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv1.in_channels = 16;
        conv1.out_channels = 4;
        conv1.kernel_size = {1, 1};
        conv1.stride = { 1, 1 };
        conv1.padding = { 0, 0 };
        conv1.create_weight_tensors(ctx);

        bn1.num_features = 1;
        bn1.create_weight_tensors(ctx);

        res2_0.create_weight_tensors(ctx);
        res2_1.create_weight_tensors(ctx);
        res2_2.create_weight_tensors(ctx);
        layer2_0.create_weight_tensors(ctx);
        layer2_1.create_weight_tensors(ctx);
        layer2_2.create_weight_tensors(ctx);
        layer2_3.create_weight_tensors(ctx);
        layer3_0.create_weight_tensors(ctx);
        layer3_1.create_weight_tensors(ctx);
        layer3_2.create_weight_tensors(ctx);
        layer3_3.create_weight_tensors(ctx);
        layer3_4.create_weight_tensors(ctx);
        layer3_5.create_weight_tensors(ctx);
        network2.create_weight_tensors(ctx);
        fuse1.create_weight_tensors(ctx);
        fuse2.create_weight_tensors(ctx);
        fuse3.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv1.");
        conv1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "bn1.");
        bn1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "res2.0.");
        res2_0.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "res2.1.");
        res2_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "res2.2.");
        res2_2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "layer2.0.");
        layer2_0.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "layer2.1.");
        layer2_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer2.2.");

        layer2_2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer2.3.");

        layer2_3.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer3.0.");

        layer3_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer3.1.");

        layer3_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer3.2.");

        layer3_2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer3.3.");

        layer3_3.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer3.4.");

        layer3_4.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer3.5.");

        layer3_5.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "network2.");

        network2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "fuse1.");
        fuse1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "fuse2.");
        fuse2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "fuse3.");
        fuse3.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

struct ColorMNet {
    // network hparams
    int MAX_H = 1024;
    int MAX_W = 1024;
    int MAX_TIMES = 112;
    int key_dim = 64;
    int value_dim = 512;
    int hidden_dim = 64;

    // network params
    struct DINOv2_v6 key_encoder;
    struct KeyProjection key_proj;
    struct ValueEncoder value_encoder;
    struct LocalAttention short_term_attn;
    struct Decoder decoder;

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

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

#endif // __COLORMNET__H__
