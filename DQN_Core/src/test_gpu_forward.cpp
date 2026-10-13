// test_gpu_forward.cpp
// GPU forward 与 CPU forward 数值一致性测试
#include "core.h"
#include "gpu_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern "C" {
    int gpu_init(void);
    int gpu_nn_create(GpuNetHandle* out);
    void gpu_nn_destroy(GpuNetHandle* h);
    int gpu_nn_upload(GpuNetHandle h, const void* cpu_params, size_t size);
    int gpu_nn_forward(GpuNetHandle h, const float* input, float* output);
}

#define LOG(...) do { fprintf(stderr, __VA_ARGS__); fflush(stderr); } while(0)

int main() {
    LOG("=== GPU Forward 一致性测试 ===\n");
    if (gpu_init() != 0) { LOG("GPU init 失败\n"); return 1; }
    LOG("GPU init OK\n");

    NeuralNet* net = (NeuralNet*)malloc(sizeof(NeuralNet));
    init_net(net);
    LOG("init_net OK\n");

    GpuNetHandle gh = {0};
    if (gpu_nn_create(&gh) != 0) { LOG("gpu_nn_create 失败\n"); return 1; }
    LOG("gpu_nn_create OK\n");

    GpuNetParams* p = (GpuNetParams*)malloc(sizeof(GpuNetParams));
    memset(p, 0, sizeof(*p));
    memcpy(p->token_embed, net->token_embed, sizeof(p->token_embed));
    memcpy(p->pos_embed, net->pos_embed, sizeof(p->pos_embed));
    memcpy(p->w1, net->w1, sizeof(p->w1));
    memcpy(p->b1, net->b1, sizeof(p->b1));
    memcpy(p->w2, net->w2, sizeof(p->w2));
    memcpy(p->b2, net->b2, sizeof(p->b2));
    memcpy(p->w3, net->w3, sizeof(p->w3));
    memcpy(p->b3, net->b3, sizeof(p->b3));
    memcpy(p->w4, net->w4, sizeof(p->w4));
    memcpy(p->b4, net->b4, sizeof(p->b4));
    memcpy(p->attn_wq, net->attn_wq, sizeof(p->attn_wq));
    memcpy(p->attn_bq, net->attn_bq, sizeof(p->attn_bq));
    memcpy(p->attn_wk, net->attn_wk, sizeof(p->attn_wk));
    memcpy(p->attn_bk, net->attn_bk, sizeof(p->attn_bk));
    memcpy(p->attn_wv, net->attn_wv, sizeof(p->attn_wv));
    memcpy(p->attn_bv, net->attn_bv, sizeof(p->attn_bv));
    memcpy(p->attn_wo, net->attn_wo, sizeof(p->attn_wo));
    memcpy(p->attn_bo, net->attn_bo, sizeof(p->attn_bo));
    memcpy(p->pool_w, net->pool_w, sizeof(p->pool_w));
    memcpy(p->pool_b, net->pool_b, sizeof(p->pool_b));
    if (gpu_nn_upload(gh, p, sizeof(*p)) != 0) { LOG("gpu_nn_upload 失败\n"); return 1; }
    LOG("gpu_nn_upload OK\n");

    float input[CELL_COUNT + 1];
    srand(42);
    for (int i = 0; i < CELL_COUNT; i++) input[i] = (float)(rand() % 3 - 1);
    input[CELL_COUNT] = 0.0f;

    float cpu_out[CELL_COUNT];
    forward_cpu(net, input, cpu_out);
    LOG("CPU forward OK\n");

    float gpu_out[CELL_COUNT];
    if (gpu_nn_forward(gh, input, gpu_out) != 0) { LOG("gpu_nn_forward 失败\n"); return 1; }
    LOG("GPU forward OK\n");

    float max_err = 0.0f;
    for (int i = 0; i < CELL_COUNT; i++) {
        float err = fabsf(cpu_out[i] - gpu_out[i]);
        if (err > max_err) max_err = err;
    }
    LOG("首次输入最大误差: %.8f\n", max_err);

    float total_max = 0.0f;
    for (int t = 0; t < 100; t++) {
        for (int i = 0; i < CELL_COUNT; i++) input[i] = (float)(rand() % 3 - 1);
        input[CELL_COUNT] = (float)(rand() % 2);
        forward_cpu(net, input, cpu_out);
        gpu_nn_forward(gh, input, gpu_out);
        for (int i = 0; i < CELL_COUNT; i++) {
            float err = fabsf(cpu_out[i] - gpu_out[i]);
            if (err > total_max) total_max = err;
        }
    }
    LOG("100 次随机输入最大误差: %.8f\n", total_max);
    LOG("容差 1e-4: %s\n", total_max < 1e-4f ? "PASS" : "FAIL");

    gpu_nn_destroy(&gh);
    free(net);
    free(p);
    return (total_max < 1e-4f) ? 0 : 1;
}
