#ifndef __VIDEO_XMEM_H__
#define __VIDEO_XMEM_H__
#include "ggml_engine.h"
#include "ggml_nn.h"

#pragma GCC diagnostic ignored "-Wformat-truncation"
// x = ggml_cont(ctx, x);
// ggml_set_name(x, "x");
// ggml_set_output(x);

int tensor_are_same_shape(TENSOR *t1, TENSOR *t2);
void tensor_debug_show(char *prompt, TENSOR* tensor);


struct XMemCache {
    // Ring buffer for KV cache ...
    int H = 32;
    int W = 32;
    int S = 8;

    TENSOR *k = NULL;
    TENSOR *s = NULL;
    TENSOR *v = NULL;
    int index = 0;
    int count = 0;

    void create(int H2, int W2, int S2) {
        H = H2; W = W2; S = S2;

        k = tensor_create(S, 1, 64, H * W);
        GGML_ASSERT(tensor_valid(k));
        s = tensor_create(S, 1, H * W, 1);
        GGML_ASSERT(tensor_valid(s));
        v = tensor_create(S, 2, 512, H * W);
        GGML_ASSERT(tensor_valid(v));

        index = 0;
        count = 0;
    }

    void destroy() {
        tensor_destroy(k);
        tensor_destroy(s);
        tensor_destroy(v);

        k = s = v = NULL;
        index = count = 0;
    }

    void dump(char *title) {
        syslog_info("%s", title);
        tensor_debug_show((char *)"  k", k);
        tensor_debug_show((char *)"  s", s);
        tensor_debug_show((char *)"  v", v);
    }

    void set(TENSOR *k2,  TENSOR *s2, TENSOR *v2) {
        // i = self.index % self.S
        // self.k[:, :, :, i:i+1] = k.view(1, 64, self.H*self.W, 1)
        // self.s[:, :, :, i:i+1] = s.view(1, self.H*self.W, 1, 1)
        // self.v[:, :, :, i:i+1] = v.view(2, 512, self.H*self.W, 1)
        // self.index = self.index + 1
        // if (self.count < self.S):
        //     self.count = self.count + 1
        float *start;
        
        int i = index % S;
        start = tensor_start_batch(k, i);
        memcpy(start, k2->data, 64 * H * W * sizeof(float));
        start = tensor_start_batch(s, i);
        memcpy(start, s2->data, 1 * H * W * sizeof(float));
        start = tensor_start_batch(v, i);
        memcpy(start, v2->data, 2 * 512 * H * W * sizeof(float));
        index++;
        if (count < S) {
            count++;
        }
    }
};

struct XMem : GGMLNetwork {
    int TOP_K = 30;

    void create_weight_tensors(struct ggml_context* ctx) {
        GGML_UNUSED(ctx);
    }

    void setup_weight_names(const char *prefix) {
        GGML_UNUSED(prefix);
    }

    ggml_tensor_t* get_similarity(struct ggml_context* ctx, ggml_tensor_t* mem_key, ggml_tensor_t* mem_shrinkage, 
        ggml_tensor_t* query_key, ggml_tensor_t* query_selection) {
        // # tensor [mem_key] size: [1, 64, 1960], min: -2.803887, max: 3.253764, mean: -0.159051
        // # tensor [mem_shrinkage] size: [1, 1960, 1], min: 23.295803, max: 45.933445, mean: 32.660492
        // # tensor [query_key] size: [1, 64, 1960], min: -2.80991, max: 3.298451, mean: -0.158671
        // # tensor [query_selection] size: [1, 64, 1960], min: 0.0, max: 0.905321, mean: 0.5004

        ggml_tensor_t *a_sq, *two_ab, *b_sq, *similarity;

        // mem_key = mem_key.transpose(1, 2)
        mem_key = ggml_cont(ctx, ggml_transpose(ctx, mem_key));
        // mem_key    f32 [64, 1960, 1, 1], mem_key (transposed)

        // a_sq = (mem_key.pow(2) @ query_selection)
        a_sq = ggml_mul(ctx, mem_key, mem_key);
        a_sq = ggml_nn_mul_mat(ctx, a_sq, query_selection);

        // two_ab = 2 * (mem_key @ (query_key * query_selection))
        two_ab = ggml_mul(ctx, query_key, query_selection);
        two_ab = ggml_nn_mul_mat(ctx, mem_key, two_ab);
        two_ab = ggml_scale(ctx, two_ab, 2.0f);

        // b_sq = (query_selection * query_key.pow(2)).sum(1, keepdim=True)
        b_sq = ggml_mul(ctx, query_key, query_key);
        b_sq = ggml_mul(ctx, query_selection, b_sq);
        int N = (int)b_sq->ne[1];
        b_sq = ggml_mean_ext(ctx, b_sq, 1 /*dim*/);
        b_sq = ggml_scale(ctx, b_sq, (float)N);

        // similarity = (-a_sq + two_ab - b_sq)
        similarity = ggml_sub(ctx, two_ab, b_sq);
        similarity = ggml_sub(ctx, similarity, a_sq);

        similarity = ggml_mul(ctx, similarity, mem_shrinkage);
        similarity = ggml_scale(ctx, similarity, 0.125f); // sqrt(64.0)

        return similarity;
    }


    ggml_tensor_t* do_softmax(struct ggml_context* ctx, ggml_tensor_t* similarity, int top_k) {
        // # tensor [similarity] size: [1, 1960, 1960], min: -87.945297, max: 0.012002, mean: -21.991934
        ggml_tensor_t *affinity, *topk_out, *index, *value;

        topk_out = ggml_topk(ctx, similarity, 1 /*dim*/, top_k);

        value = ggml_nn_slice(ctx, topk_out, 1 /*dim*/, 0, top_k, 1/*step*/);
        index = ggml_nn_slice(ctx, topk_out, 1 /*dim*/, top_k, 2*top_k, 1/*step*/);
        value = ggml_softmax(ctx, value, 1/*dim*/);
        // // debug
        // {
        //     value = ggml_cont(ctx, value);
        //     ggml_set_name(value, "SOFTMAX");
        //     ggml_set_output(value);            
        // }
        affinity = ggml_dup(ctx, similarity);
        affinity = ggml_constant(ctx, affinity, 0.0f);
        affinity = ggml_paste(ctx, affinity, 1 /*dim*/, index, value);

        // # tensor [affinity] size: [1, 1960, 1960], min: 0.0, max: 0.442188, mean: 0.000255
        return affinity;
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, int argc, ggml_tensor_t* argv[]) {
        GGML_ASSERT(argc == 5);
        ggml_tensor_t* q_key = argv[0];
        ggml_tensor_t* q_selection = argv[1];
        ggml_tensor_t* mem_key = argv[2];
        ggml_tensor_t* mem_shrinkage = argv[3];
        ggml_tensor_t* mem_value = argv[4];

        int HW, C, W, H;
        HW = (int)mem_key->ne[0];
        C = (int)mem_key->ne[3];

        // mem_key:  (HW, 64, 1, C) --> (HW, C, 64, 1) -> (HWC, 64, 1)
        mem_key = ggml_cont(ctx, ggml_permute(ctx, mem_key, 0, 2, 3, 1));
        mem_key = ggml_reshape_3d(ctx, mem_key, HW*C, 64, 1);

        // mem_shrinkage: (1, HW, 1, C) --> (1, HW, C, 1) -> (1, HWC, 1)
        mem_shrinkage = ggml_reshape_3d(ctx, mem_shrinkage, 1, HW * C, 1);

        // mem_value: (HW, 512, 2, C) --> (HW, C, 512, 2) -> (HWC, 512, 2)
        mem_value = ggml_cont(ctx, ggml_permute(ctx, mem_value, 0, 2, 3, 1));
        mem_value = ggml_reshape_3d(ctx, mem_value, HW*C, 512, 2);

        ggml_tensor_t *similarity, *affinity, *final_value;

        W = (int)q_key->ne[0];
        H = (int)q_key->ne[1];
        // int C = (int)q_key->ne[2];
        // int B = (int)q_key->ne[3];

        q_key = ggml_reshape_3d(ctx, q_key, H*W, 64, 1);
        q_selection = ggml_reshape_3d(ctx, q_selection, H*W, 64, 1);
        similarity = get_similarity(ctx, mem_key, mem_shrinkage, q_key, q_selection);
        // tensor [similarity] size: [1, 1960, 1960], min: -87.556335, max: -0.000922, mean: -19.917494
        affinity = do_softmax(ctx, similarity, TOP_K);
        // tensor [affinity] size: [1, 1960, 1960], min: 0.0, max: 0.983038, mean: 0.00051
        // // debug
        // {
        //     similarity = ggml_cont(ctx, similarity);
        //     ggml_set_name(similarity, "SIMILARITY");
        //     ggml_set_output(similarity);

        //     affinity = ggml_cont(ctx, affinity);
        //     ggml_set_name(affinity, "AFFINITY");
        //     ggml_set_output(affinity);
        // }
        final_value = ggml_nn_mul_mat(ctx, mem_value, affinity);
        final_value = ggml_reshape_4d(ctx, final_value, W, H, 512, 2);
        // tensor [final_value] size: [2, 512, 35, 56], min: -10.663672, max: 5.041441, mean: -0.008074

        return final_value;
    }
};


struct VideoXMemNetwork {
    int H = 32;
    int W = 32;
    int WORK_SIZE = 8;
    int LONG_SIZE = 8;

    TENSOR *sensory = NULL;
    TENSOR *lastkey = NULL;
    TENSOR *lastval = NULL;
    struct XMemCache workmem;
    struct XMemCache longmem;

    XMem net;

    // -----------------------------------------------------------------------------------------
    void alloc_cache(int H2, int W2, int S1, int S2) {
        GGML_ASSERT(sensory == NULL); // DO NOT create twice !!!

        H = H2; W = W2; WORK_SIZE = S1; LONG_SIZE = S2;
        sensory = tensor_create(2, 64, H, W);
        GGML_ASSERT(tensor_valid(sensory));

        lastkey = tensor_create(1, 64, H, W);
        GGML_ASSERT(tensor_valid(lastkey));

        lastval = tensor_create(2, 512, H, W);
        GGML_ASSERT(tensor_valid(lastval));

        workmem.create(H, W, WORK_SIZE);
        longmem.create(H, W, LONG_SIZE);
    }

    void free_cache() {
        tensor_destroy(sensory);
        tensor_destroy(lastkey);
        tensor_destroy(lastval);
        workmem.destroy();
        longmem.destroy();

        sensory = NULL; // LET cache could be re-created !!!
    }

    void dump() {
        if (sensory != NULL) {
            tensor_debug_show((char *)"hidden", sensory);
            tensor_debug_show((char *)"last_key", lastkey);
            tensor_debug_show((char *)"last_value", lastval);
            workmem.dump((char *)"work memory");
            longmem.dump((char *)"long memory");
        } else {
            syslog_info("xmem is empty.");
        }
    }

    // -----------------------------------------------------------------------------------------
    void set_work_memory(TENSOR *k, TENSOR *s, TENSOR *v, TENSOR *h) {
        GGML_ASSERT(sensory != NULL); // MAKE SURE xmem has been created !!!
        GGML_ASSERT(tensor_are_same_shape(k, lastkey));
        GGML_ASSERT(tensor_are_same_shape(v, lastval));
        GGML_ASSERT(tensor_are_same_shape(h, sensory));

        memcpy(lastkey->data, k->data, k->batch * k->chan * k->height * k->width * sizeof(float));
        memcpy(lastval->data, v->data, v->batch * v->chan * v->height * v->width * sizeof(float));
        memcpy(sensory->data, h->data, h->batch * h->chan * h->height * h->width * sizeof(float));

        workmem.set(k, s, v);
    }

    // -----------------------------------------------------------------------------------------
    void set_long_memory(TENSOR *k, TENSOR *s, TENSOR *v, TENSOR *h) {
        GGML_ASSERT(sensory != NULL); // MAKE SURE xmem has been created !!!

        GGML_ASSERT(tensor_are_same_shape(k, lastkey));
        GGML_ASSERT(tensor_are_same_shape(v, lastval));
        GGML_ASSERT(tensor_are_same_shape(h, sensory));

        memcpy(lastkey->data, k->data, k->batch * k->chan * k->height * k->width * sizeof(float));
        memcpy(lastval->data, v->data, v->batch * v->chan * v->height * v->width * sizeof(float));
        memcpy(sensory->data, h->data, h->batch * h->chan * h->height * h->width * sizeof(float));

        longmem.set(k, s, v);
    }

    // -----------------------------------------------------------------------------------------
    TENSOR *get_hidden() {
        return sensory;
    }

    // -----------------------------------------------------------------------------------------
    TENSOR *get_last_key() {
        return lastkey;
    }

    // -----------------------------------------------------------------------------------------
    TENSOR *get_last_value() {
        return lastval;
    }

    TENSOR *get_key() {
        int c, n;
        TENSOR *m_key;
        float *s_start, *d_start;

        c = workmem.count + longmem.count;
        if (c < 1) {
            GGML_ASSERT(c > 0);
            return NULL;
        }

        m_key = tensor_create(c, 1, 64, H * W);
        CHECK_TENSOR(m_key);

        n = 64 * H * W; // tensor elements
        if (workmem.count > 0) {
            s_start = tensor_start_batch(workmem.k, 0);
            d_start = tensor_start_batch(m_key, 0);
            memcpy(d_start, s_start, workmem.count * n * sizeof(float));
        }

        if (longmem.count > 0) {
            s_start = tensor_start_batch(longmem.k, 0);
            d_start = tensor_start_batch(m_key, workmem.count);
            memcpy(d_start, s_start, longmem.count * n * sizeof(float));
        }

        return m_key;
    }

    TENSOR *get_shrinkage() {
        int c, n;
        TENSOR *m_shrinkage;
        float *s_start, *d_start;

        c = workmem.count + longmem.count;
        if (c < 1) {
            GGML_ASSERT(c > 0);
            return NULL;
        }

        m_shrinkage = tensor_create(c, 1, H * W, 1);
        CHECK_TENSOR(m_shrinkage);

        n = H * W; // tensor elements
        if (workmem.count > 0) {
            s_start = tensor_start_batch(workmem.s, 0);
            d_start = tensor_start_batch(m_shrinkage, 0);
            memcpy(d_start, s_start, workmem.count * n * sizeof(float));
        }

        if (longmem.count > 0) {
            s_start = tensor_start_batch(longmem.s, 0);
            d_start = tensor_start_batch(m_shrinkage, workmem.count);
            memcpy(d_start, s_start, longmem.count * n * sizeof(float));
        }

        return m_shrinkage;
    }


    TENSOR *get_value() {
        int c, n;
        TENSOR *m_value;
        float *s_start, *d_start;

        c = workmem.count + longmem.count;
        if (c < 1) {
            GGML_ASSERT(c > 0);
            return NULL;
        }

        m_value = tensor_create(c, 2, 512, H * W);
        CHECK_TENSOR(m_value);
        n = 2 * 512 * H * W; // tensor elements

        if (workmem.count > 0) {
            s_start = tensor_start_batch(workmem.v, 0);
            d_start = tensor_start_batch(m_value, 0);
            memcpy(d_start, s_start, workmem.count * n * sizeof(float));
        }

        if (longmem.count > 0) {
            s_start = tensor_start_batch(longmem.v, 0);
            d_start = tensor_start_batch(m_value, workmem.count);
            memcpy(d_start, s_start, longmem.count * n * sizeof(float));
        }

        return m_value;
    }

    // -----------------------------------------------------------------------------------------
    int init(int device) {
        net.set_device(device);
        net.start_engine();
        // net.dump();

        return RET_OK;
    }

    // -----------------------------------------------------------------------------------------
    TENSOR *query_value(TENSOR *key, TENSOR *selection) {
        GGML_ASSERT(sensory != NULL); // MAKE SURE xmem has been created !!!

        TENSOR* mem_key = get_key();
        TENSOR* mem_shrinkage = get_shrinkage();
        TENSOR* mem_value = get_value();

        TENSOR *argv[5];
        argv[0] = key ;
        argv[1] = selection;
        argv[2] = mem_key;
        argv[3] = mem_shrinkage;
        argv[4] = mem_value;

        // tensor_debug_show("===> key", key);
        // tensor_debug_show("===> selection", selection);
        // tensor_debug_show("===> mem_key", mem_key);
        // tensor_debug_show("===> mem_shrinkage", mem_shrinkage);
        // tensor_debug_show("===> mem_value", mem_value);

        // tensor [q_key] size: [1, 64, 1960], min: -2.80991, max: 3.298451, mean: -0.158671
        // tensor [q_selection] size: [1, 64, 1960], min: 0.0, max: 0.905321, mean: 0.5004
        // tensor [mem_key] size: [1, 64, 1960], min: -2.803887, max: 3.253764, mean: -0.159051
        // tensor [mem_shrinkage] size: [1, 1960, 1], min: 23.295803, max: 45.933445, mean: 32.660492
        // tensor [mem_value] size: [2, 512, 1960], min: -11.473879, max: 5.339447, mean: -0.008071

        // XMem
        TENSOR *value = net.engine_forward(ARRAY_SIZE(argv), argv);
        // tensor_debug_show("===> value", value);

        // tensor [similarity] size: [1, 1960, 1960], min: -87.556335, max: -0.000922, mean: -19.917494
        // tensor [affinity] size: [1, 1960, 1960], min: 0.0, max: 0.983038, mean: 0.00051
        // tensor [final_value] size: [2, 512, 35, 56], min: -10.663672, max: 5.041441, mean: -0.008074
        // destroy mem_*
        {
            tensor_destroy(mem_key);
            tensor_destroy(mem_shrinkage);
            tensor_destroy(mem_value);
        }

        return value;
    }

    // -----------------------------------------------------------------------------------------
    void exit() {
        free_cache();
        net.stop_engine();
    }
};

#endif // __VIDEO_XMEM_H__
