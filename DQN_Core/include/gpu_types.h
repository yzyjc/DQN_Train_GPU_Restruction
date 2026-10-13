// gpu_types.h
// GPU 神经网络类型定义（纯 C，供 .hip 和 .cpp 共用）
// 维度常量与 core.h 保持一致（若 core.h 维度改动，需同步这里）

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 网络维度（与 core.h 的 NeuralNet 一致）
// ============================================================================
#define GPU_CELL_COUNT          25
#define GPU_FEATURE_DIM         128
#define GPU_HIDDEN1_SIZE        1024
#define GPU_HIDDEN2_SIZE        512
#define GPU_HIDDEN3_SIZE        256
#define GPU_TOKEN_EMBED_DIM     16
#define GPU_ATTENTION_HEADS     4
#define GPU_ATTENTION_EMBED     32
#define GPU_ATTENTION_HEAD_DIM  (GPU_ATTENTION_EMBED / GPU_ATTENTION_HEADS)  // 8
#define GPU_STATE_DIM           (GPU_CELL_COUNT + 1)  // 26
#define GPU_ACTION_COUNT        GPU_CELL_COUNT        // 25
#define GPU_MAX_BATCH           32                    // 批量训练上限（= BATCH_SIZE）

// ============================================================================
// GPU 驻留参数布局（只含可学习参数，不含缓存/激活）
// 顺序与 NeuralNet 字段对应，pack/unpack 在 neural_net.cpp 中实现
// ============================================================================
typedef struct {
    float token_embed[3][GPU_TOKEN_EMBED_DIM];
    float pos_embed[GPU_CELL_COUNT][GPU_TOKEN_EMBED_DIM];
    float w1[GPU_HIDDEN1_SIZE][GPU_FEATURE_DIM];          // FC1: 128 -> 1024
    float b1[GPU_HIDDEN1_SIZE];
    float w2[GPU_HIDDEN2_SIZE][GPU_HIDDEN1_SIZE];         // FC2: 1024 -> 512
    float b2[GPU_HIDDEN2_SIZE];
    float w3[GPU_HIDDEN3_SIZE][GPU_HIDDEN2_SIZE];         // FC3: 512 -> 256
    float b3[GPU_HIDDEN3_SIZE];
    float w4[GPU_CELL_COUNT][GPU_HIDDEN3_SIZE];           // FC4: 256 -> 25
    float b4[GPU_CELL_COUNT];
    float attn_wq[GPU_ATTENTION_EMBED][GPU_TOKEN_EMBED_DIM];
    float attn_bq[GPU_ATTENTION_EMBED];
    float attn_wk[GPU_ATTENTION_EMBED][GPU_TOKEN_EMBED_DIM];
    float attn_bk[GPU_ATTENTION_EMBED];
    float attn_wv[GPU_ATTENTION_EMBED][GPU_TOKEN_EMBED_DIM];
    float attn_bv[GPU_ATTENTION_EMBED];
    float attn_wo[GPU_TOKEN_EMBED_DIM][GPU_ATTENTION_EMBED];
    float attn_bo[GPU_TOKEN_EMBED_DIM];
    float pool_w[GPU_FEATURE_DIM][GPU_CELL_COUNT * GPU_TOKEN_EMBED_DIM];
    float pool_b[GPU_FEATURE_DIM];
} GpuNetParams;

// GPU 工作区（forward 中间激活 + 反向梯度，全部驻留 GPU）
typedef struct {
    float input[GPU_STATE_DIM];
    // Train in a private device copy; failed batches never publish partial SGD.
    GpuNetParams update_params;
    int invalid_params;
    float tokens[GPU_CELL_COUNT][GPU_TOKEN_EMBED_DIM];
    float q[GPU_CELL_COUNT][GPU_ATTENTION_EMBED];
    float k[GPU_CELL_COUNT][GPU_ATTENTION_EMBED];
    float v[GPU_CELL_COUNT][GPU_ATTENTION_EMBED];
    float scores[GPU_ATTENTION_HEADS][GPU_CELL_COUNT][GPU_CELL_COUNT];
    float head_out[GPU_ATTENTION_HEADS][GPU_CELL_COUNT][GPU_ATTENTION_HEAD_DIM];
    float attn_out[GPU_CELL_COUNT][GPU_TOKEN_EMBED_DIM];
    float pooled[GPU_CELL_COUNT * GPU_TOKEN_EMBED_DIM];   /* flat[400]：展平后的 Attention_O 输出 */
    float pooled_feat[GPU_FEATURE_DIM];
    float h1[GPU_HIDDEN1_SIZE];
    float h2[GPU_HIDDEN2_SIZE];
    float h3[GPU_HIDDEN3_SIZE];
    float output[GPU_CELL_COUNT];
    // ---- 反向传播梯度区 ----
    float delta_out[GPU_CELL_COUNT];                 // 输出层梯度（Huber）
    float delta_h3[GPU_HIDDEN3_SIZE];
    float delta_h2[GPU_HIDDEN2_SIZE];
    float delta_h1[GPU_HIDDEN1_SIZE];
    float dL_dpooled_feat[GPU_FEATURE_DIM];
    float dL_dpooled[GPU_CELL_COUNT * GPU_TOKEN_EMBED_DIM];   /* dL_dflat[400]：展平层梯度 */
    float dL_dattn_out[GPU_CELL_COUNT][GPU_TOKEN_EMBED_DIM];
    float dL_dq_global[GPU_CELL_COUNT][GPU_ATTENTION_EMBED]; // 每个位置的 Q/K/V 梯度
    float dL_dk_global[GPU_CELL_COUNT][GPU_ATTENTION_EMBED];
    float dL_dv_global[GPU_CELL_COUNT][GPU_ATTENTION_EMBED];
    float dL_dtokens[GPU_CELL_COUNT][GPU_TOKEN_EMBED_DIM];  // embedding 梯度
    float dL_dconcat[GPU_CELL_COUNT][GPU_ATTENTION_EMBED];  // attention 输入梯度
    float grad_scratch[GPU_ATTENTION_HEADS][GPU_CELL_COUNT][GPU_CELL_COUNT]; // softmax 梯度
    float token_col_sum[GPU_TOKEN_EMBED_DIM];        // sum_i tokens[i][j]
    float loss_host;                                 // 损失（D2H 拷贝用）
    float scale;                                     // 梯度裁剪缩放因子
    // ---- 批量训练区 ----
    float batch_states[GPU_MAX_BATCH][GPU_STATE_DIM];
    int   batch_actions[GPU_MAX_BATCH];
    float batch_targets[GPU_MAX_BATCH];
    float loss_buf[GPU_MAX_BATCH];
    float batch_output[GPU_MAX_BATCH][GPU_ACTION_COUNT];  // 批量 forward 输出
} GpuWork;

// GPU 网络句柄（.hip 内部持有设备指针；对 .cpp 为不透明）
typedef struct GpuNetHandle {
    void* params;   // 设备端 GpuNetParams
    void* work;     // 设备端 GpuWork
    int   is_valid; // 1 = 已分配
} GpuNetHandle;

// 参数打包缓冲区（CPU 侧，用于 upload/download 中转）
// 大小 = sizeof(GpuNetParams)，在 neural_net.cpp 中作为静态缓冲
extern const size_t gpu_net_params_size;

#ifdef __cplusplus
}
#endif
