// test_gpu_train.cpp
// GPU train_step 与 CPU train_step 数值一致性测试
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
    int gpu_nn_download(GpuNetHandle h, void* cpu_params, size_t size);
    int gpu_nn_train_step(GpuNetHandle h, const float* state, int action, float target, float lr, float* out_loss);
}

extern float g_learning_rate;

// NeuralNet -> GpuNetParams 打包
static void pack(const NeuralNet* n, GpuNetParams* p) {
    memset(p, 0, sizeof(*p));
    memcpy(p->token_embed, n->token_embed, sizeof(p->token_embed));
    memcpy(p->pos_embed, n->pos_embed, sizeof(p->pos_embed));
    memcpy(p->w1, n->w1, sizeof(p->w1));
    memcpy(p->b1, n->b1, sizeof(p->b1));
    memcpy(p->w2, n->w2, sizeof(p->w2));
    memcpy(p->b2, n->b2, sizeof(p->b2));
    memcpy(p->w3, n->w3, sizeof(p->w3));
    memcpy(p->b3, n->b3, sizeof(p->b3));
    memcpy(p->w4, n->w4, sizeof(p->w4));
    memcpy(p->b4, n->b4, sizeof(p->b4));
    memcpy(p->attn_wq, n->attn_wq, sizeof(p->attn_wq));
    memcpy(p->attn_bq, n->attn_bq, sizeof(p->attn_bq));
    memcpy(p->attn_wk, n->attn_wk, sizeof(p->attn_wk));
    memcpy(p->attn_bk, n->attn_bk, sizeof(p->attn_bk));
    memcpy(p->attn_wv, n->attn_wv, sizeof(p->attn_wv));
    memcpy(p->attn_bv, n->attn_bv, sizeof(p->attn_bv));
    memcpy(p->attn_wo, n->attn_wo, sizeof(p->attn_wo));
    memcpy(p->attn_bo, n->attn_bo, sizeof(p->attn_bo));
    memcpy(p->pool_w, n->pool_w, sizeof(p->pool_w));
    memcpy(p->pool_b, n->pool_b, sizeof(p->pool_b));
}

// GpuNetParams -> NeuralNet 解包
static void unpack(const GpuNetParams* p, NeuralNet* n) {
    memcpy(n->token_embed, p->token_embed, sizeof(n->token_embed));
    memcpy(n->pos_embed, p->pos_embed, sizeof(n->pos_embed));
    memcpy(n->w1, p->w1, sizeof(n->w1));
    memcpy(n->b1, p->b1, sizeof(n->b1));
    memcpy(n->w2, p->w2, sizeof(n->w2));
    memcpy(n->b2, p->b2, sizeof(n->b2));
    memcpy(n->w3, p->w3, sizeof(n->w3));
    memcpy(n->b3, p->b3, sizeof(n->b3));
    memcpy(n->w4, p->w4, sizeof(n->w4));
    memcpy(n->b4, p->b4, sizeof(n->b4));
    memcpy(n->attn_wq, p->attn_wq, sizeof(n->attn_wq));
    memcpy(n->attn_bq, p->attn_bq, sizeof(n->attn_bq));
    memcpy(n->attn_wk, p->attn_wk, sizeof(n->attn_wk));
    memcpy(n->attn_bk, p->attn_bk, sizeof(n->attn_bk));
    memcpy(n->attn_wv, p->attn_wv, sizeof(n->attn_wv));
    memcpy(n->attn_bv, p->attn_bv, sizeof(n->attn_bv));
    memcpy(n->attn_wo, p->attn_wo, sizeof(n->attn_wo));
    memcpy(n->attn_bo, p->attn_bo, sizeof(n->attn_bo));
    memcpy(n->pool_w, p->pool_w, sizeof(n->pool_w));
    memcpy(n->pool_b, p->pool_b, sizeof(n->pool_b));
}

int main() {
    printf("=== GPU train_step 一致性测试 ===\n");
    if (gpu_init() != 0) { printf("GPU init 失败\n"); return 1; }

    g_learning_rate = 0.0005f;

    // 两个相同初始参数的网络
    NeuralNet* net_cpu = (NeuralNet*)malloc(sizeof(NeuralNet));
    NeuralNet* net_gpu = (NeuralNet*)malloc(sizeof(NeuralNet));
    srand(42);
    init_net(net_cpu);
    memcpy(net_gpu, net_cpu, sizeof(NeuralNet));

    // 创建 GPU 句柄并上传
    GpuNetHandle gh = {0};
    if (gpu_nn_create(&gh) != 0) { printf("gpu_nn_create 失败\n"); return 1; }
    GpuNetParams* p = (GpuNetParams*)malloc(sizeof(GpuNetParams));
    pack(net_gpu, p);
    GpuNetParams* initial = (GpuNetParams*)malloc(sizeof(GpuNetParams));
    memcpy(initial, p, sizeof(*initial));
    gpu_nn_upload(gh, p, sizeof(*p));

    // 测试多组 (state, action, target)
    float input[CELL_COUNT + 1];
    float total_max = 0.0f;

    // 诊断：第一步打印各层误差
    {
        for (int i = 0; i < CELL_COUNT; i++) input[i] = (float)(rand() % 3 - 1);
        input[CELL_COUNT] = (float)(rand() % 2);
        int action = rand() % CELL_COUNT;
        float target = ((rand() % 3) - 1) * 0.5f;
        float loss_cpu = train_step(net_cpu, input, action, target);
        float loss_gpu = 0.0f;
        gpu_nn_train_step(gh, input, action, target, g_learning_rate, &loss_gpu);
        gpu_nn_download(gh, p, sizeof(*p));
        unpack(p, net_gpu);
        int changed_columns = 0;
        for (int col = 0; col < GPU_CELL_COUNT * GPU_TOKEN_EMBED_DIM; ++col) {
            int changed = 0;
            for (int row = 0; row < GPU_FEATURE_DIM; ++row) {
                if (p->pool_w[row][col] != initial->pool_w[row][col]) {
                    changed = 1;
                    break;
                }
            }
            changed_columns += changed;
        }
        printf("GPU pool_w changed columns: %d/400 (col0=%d col16=%d col399=%d)\n",
            changed_columns,
            p->pool_w[0][0] != initial->pool_w[0][0],
            p->pool_w[0][16] != initial->pool_w[0][16],
            p->pool_w[0][399] != initial->pool_w[0][399]);
        if (changed_columns != GPU_CELL_COUNT * GPU_TOKEN_EMBED_DIM) {
            printf("FAIL: GPU backward/update did not reach all 400 pooling columns\n");
            return 1;
        }
        // 对比 net_cpu（CPU更新）和 net_gpu（GPU更新）逐层
        GpuNetParams* pc = (GpuNetParams*)malloc(sizeof(GpuNetParams));
        pack(net_cpu, pc);  // CPU 结果转 GpuNetParams
        #define CMP(field) { float m=0; const float*a=(const float*)pc->field; const float*b=(const float*)p->field; int n=sizeof(pc->field)/sizeof(float); for(int i=0;i<n;i++){float e=fabsf(a[i]-b[i]); if(e>m)m=e;} printf("  %s: %.6f\n", #field, m); }
        printf("单步 loss: cpu=%.6f gpu=%.6f\n", loss_cpu, loss_gpu);
        CMP(token_embed); CMP(pos_embed);
        CMP(w1); CMP(b1); CMP(w2); CMP(b2); CMP(w3); CMP(b3); CMP(w4); CMP(b4);
        CMP(attn_wq); CMP(attn_bq); CMP(attn_wk); CMP(attn_bk); CMP(attn_wv); CMP(attn_bv);
        CMP(attn_wo); CMP(attn_bo); CMP(pool_w); CMP(pool_b);
        #undef CMP
        free(pc);
        // 重置两个 net 为相同初始，继续 50 轮测试
        free(net_cpu); free(net_gpu); free(p); free(initial);
        net_cpu = (NeuralNet*)malloc(sizeof(NeuralNet));
        net_gpu = (NeuralNet*)malloc(sizeof(NeuralNet));
        srand(42); init_net(net_cpu); memcpy(net_gpu, net_cpu, sizeof(NeuralNet));
        gpu_nn_destroy(&gh);
        gpu_nn_create(&gh);
        p = (GpuNetParams*)malloc(sizeof(GpuNetParams));
        pack(net_gpu, p); gpu_nn_upload(gh, p, sizeof(*p));
    }

    for (int t = 0; t < 5; t++) {
        for (int i = 0; i < CELL_COUNT; i++) input[i] = (float)(rand() % 3 - 1);
        input[CELL_COUNT] = (float)(rand() % 2);
        int action = rand() % CELL_COUNT;
        float target = ((rand() % 3) - 1) * 0.5f;

        // CPU train_step
        float loss_cpu = train_step(net_cpu, input, action, target);

        // GPU train_step
        float loss_gpu = 0.0f;
        if (gpu_nn_train_step(gh, input, action, target, g_learning_rate, &loss_gpu) != 0) {
            printf("gpu_nn_train_step 失败\n"); return 1;
        }

        // 下载 GPU 参数
        gpu_nn_download(gh, p, sizeof(*p));
        unpack(p, net_gpu);

        // 对比 loss
        float loss_err = fabsf(loss_cpu - loss_gpu);
        if (loss_err > total_max) total_max = loss_err;

        // 对比参数（只对比 GpuNetParams 纯参数区，排除缓存字段假象）
        GpuNetParams* pc = (GpuNetParams*)malloc(sizeof(GpuNetParams));
        pack(net_cpu, pc);
        float* a = (float*)pc;
        float* b = (float*)p;
        int param_count = sizeof(GpuNetParams) / sizeof(float);
        float param_max = 0.0f;
        for (int i = 0; i < param_count; i++) {
            float err = fabsf(a[i] - b[i]);
            if (err > param_max) param_max = err;
            if (err > total_max) total_max = err;
        }
        free(pc);
        printf("  轮%d: loss_cpu=%.6f loss_gpu=%.6f loss_diff=%.2e param_max=%.2e\n",
            t + 1, loss_cpu, loss_gpu, loss_err, param_max);
    }

    printf("5 轮训练最大误差(loss+参数): %.8f\n", total_max);
    printf("容差 1e-3: %s\n", total_max < 1e-3f ? "PASS" : "FAIL");

    // GPU-only stability smoke test.  This exercises the real kernels without
    // waiting for the curriculum/pretraining state machine to graduate.
    float max_smoke_loss = 0.0f;
    for (int t = 0; t < 500; ++t) {
        for (int i = 0; i < CELL_COUNT; ++i)
            input[i] = (float)(rand() % 3 - 1);
        input[CELL_COUNT] = (float)(rand() % 2);
        int action = rand() % CELL_COUNT;
        float target = ((rand() % 2001) - 1000) / 1000.0f;
        float loss = 0.0f;
        if (gpu_nn_train_step(gh, input, action, target, g_learning_rate, &loss) != 0 ||
            !isfinite(loss)) {
            printf("FAIL: GPU smoke step %d returned invalid loss %.9g\n", t, loss);
            return 1;
        }
        if (loss > max_smoke_loss) max_smoke_loss = loss;
    }
    if (gpu_nn_download(gh, p, sizeof(*p)) != 0) {
        printf("FAIL: GPU smoke download failed\n");
        return 1;
    }
    int finite_params = 1;
    const float* values = (const float*)p;
    for (size_t i = 0; i < sizeof(*p) / sizeof(float); ++i) {
        if (!isfinite(values[i])) { finite_params = 0; break; }
    }
    printf("GPU 500-step smoke: %s (max_loss=%.6f)\n",
        finite_params ? "PASS" : "FAIL", max_smoke_loss);
    if (!finite_params) return 1;

    gpu_nn_destroy(&gh);
    free(p);
    free(net_cpu);
    free(net_gpu);
    return (total_max < 1e-3f) ? 0 : 1;
}
