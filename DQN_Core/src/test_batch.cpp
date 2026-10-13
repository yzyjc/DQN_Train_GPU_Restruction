// test_batch.cpp - GPU 批量 train_step 最小测试
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
    int gpu_nn_train_step_batch(GpuNetHandle h, const float* states, const int* actions,
        const float* targets, int batch_size, float lr, float* out_loss);
}

#include <windows.h>
static LONG WINAPI crash_handler(EXCEPTION_POINTERS* ex) {
    char mod[256] = "?";
    HMODULE m = NULL;
    ULONG_PTR addr = ex->ContextRecord->Rip;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCSTR)addr, &m)) GetModuleFileNameA(m, mod, 256);
    FILE* f = fopen("batch_crash.txt", "a");
    if (f) { fprintf(f, "RIP=0x%llX mod=%s\n", (unsigned long long)addr, mod); fclose(f); }
    printf("[CRASH] RIP=0x%llX mod=%s\n", (unsigned long long)addr, mod);
    return EXCEPTION_EXECUTE_HANDLER;
}

int main() {
    SetUnhandledExceptionFilter(crash_handler);
    printf("=== GPU 批量 train_step 测试 ===\n");
    if (gpu_init() != 0) return 1;

    NeuralNet* net = (NeuralNet*)malloc(sizeof(NeuralNet));
    srand(42);
    init_net(net);

    GpuNetHandle gh = {0};
    if (gpu_nn_create(&gh) != 0) { printf("create 失败\n"); return 1; }

    GpuNetParams* p = (GpuNetParams*)malloc(sizeof(GpuNetParams));
    memset(p, 0, sizeof(*p));
    memcpy(p->w1, net->w1, sizeof(p->w1));
    memcpy(p->b1, net->b1, sizeof(p->b1));
    memcpy(p->w2, net->w2, sizeof(p->w2));
    memcpy(p->b2, net->b2, sizeof(p->b2));
    memcpy(p->w3, net->w3, sizeof(p->w3));
    memcpy(p->b3, net->b3, sizeof(p->b3));
    memcpy(p->w4, net->w4, sizeof(p->w4));
    memcpy(p->b4, net->b4, sizeof(p->b4));
    memcpy(p->token_embed, net->token_embed, sizeof(p->token_embed));
    memcpy(p->pos_embed, net->pos_embed, sizeof(p->pos_embed));
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
    gpu_nn_upload(gh, p, sizeof(*p));
    printf("upload OK\n");

    // 构造 32 样本 batch
    const int BS = 32;
    float* states = (float*)malloc(sizeof(float) * (CELL_COUNT + 1) * BS);
    int* actions = (int*)malloc(sizeof(int) * BS);
    float* targets = (float*)malloc(sizeof(float) * BS);
    for (int b = 0; b < BS; b++) {
        for (int i = 0; i < CELL_COUNT; i++) states[b * (CELL_COUNT + 1) + i] = (float)(rand() % 3 - 1);
        states[b * (CELL_COUNT + 1) + CELL_COUNT] = 0.0f;
        actions[b] = rand() % CELL_COUNT;
        targets[b] = ((rand() % 3) - 1) * 0.5f;
    }
    printf("batch 构造 OK, 开始批量训练...\n");
    fflush(stdout);

    float loss = 0.0f;
    int ret = gpu_nn_train_step_batch(gh, states, actions, targets, BS, 0.0005f, &loss);
    printf("gpu_nn_train_step_batch ret=%d loss=%.6f\n", ret, loss);

    // 再来一次（验证可重复）
    ret = gpu_nn_train_step_batch(gh, states, actions, targets, BS, 0.0005f, &loss);
    printf("第二次 ret=%d loss=%.6f\n", ret, loss);

    printf("批量训练 OK\n");
    gpu_nn_destroy(&gh);
    return 0;
}
