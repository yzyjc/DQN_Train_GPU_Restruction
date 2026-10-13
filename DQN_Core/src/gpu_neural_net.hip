// gpu_neural_net.hip
// ROCm GPU 全驻留神经网络模块
// 编译: hipcc --offload-arch=gfx1151
//
// 目标: forward / train_step / soft_update 全部在 GPU 上执行，
// 参数、梯度、激活全部驻留 GPU 显存，训练热路径无 CPU<->GPU 往返。

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>
#include "gpu_types.h"
#include "network_input.h"
#include "gpu_kernel_timer.h"
#include "gpu_workload_profile.h"

// ============================================================================
// 设备常量（编译期确定，kernel 内使用）
// ============================================================================
#define D_CELL    GPU_CELL_COUNT
#define D_FEAT    GPU_FEATURE_DIM
#define D_H1      GPU_HIDDEN1_SIZE
#define D_H2      GPU_HIDDEN2_SIZE
#define D_H3      GPU_HIDDEN3_SIZE
#define D_TED     GPU_TOKEN_EMBED_DIM
#define D_HEADS   GPU_ATTENTION_HEADS
#define D_AEMB    GPU_ATTENTION_EMBED
#define D_AHD     GPU_ATTENTION_HEAD_DIM
#define D_STATE   GPU_STATE_DIM

/* Huber limits the output delta, but deep-layer outer products can still be
   very large when an activation spikes.  Clip the actual parameter gradient,
   not the parameter vector, so a single replay sample cannot poison a batch. */
#define D_GRAD_CLIP 1.0f
#define D_PARAM_ABS_LIMIT 100.0f
/* A finite Huber loss can legitimately exceed this value when an otherwise
   valid replay sample exposes an unusually large Q error.  Huber already
   bounds the output gradient, so this is a reporting cap, not a safety limit. */
#define D_LOSS_REJECT_LIMIT 100.0f

__device__ __forceinline__ float clipped_gradient(float value) {
    if (!__builtin_isfinite(value)) return value; /* validation must see NaN/Inf */
    return fminf(D_GRAD_CLIP, fmaxf(-D_GRAD_CLIP, value));
}

// ============================================================================
// Forward kernels（对照 neural_net.cpp forward_cpu）
// ============================================================================

// 1. Token embedding: tokens[i][d] = token_embed[state(i)][d] + pos_embed[i][d]
__global__ void kernel_token_embedding(
    const float* token_embed,   // [3][D_TED]
    const float* pos_embed,     // [D_CELL][D_TED]
    const float* input,         // [D_CELL]
    float* output)              // [D_CELL][D_TED]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < D_CELL) {
        int s = network_token_index(input, idx);
        if (s < 0) s = 0;
        if (s > 2) s = 2;
        for (int d = 0; d < D_TED; d++) {
            output[idx * D_TED + d] =
                token_embed[s * D_TED + d] + pos_embed[idx * D_TED + d];
        }
    }
}

// 2. QKV projection: q/k/v = W*x + b
__global__ void kernel_qkv_projection(
    const float* tokens,   // [D_CELL][D_TED]
    const float* wq, const float* bq,   // [D_AEMB][D_TED], [D_AEMB]
    const float* wk, const float* bk,
    const float* wv, const float* bv,
    float* q, float* k, float* v)       // [D_CELL][D_AEMB]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < D_CELL) {
        for (int d = 0; d < D_AEMB; d++) {
            float qv = bq[d], kv = bk[d], vv = bv[d];
            for (int j = 0; j < D_TED; j++) {
                float tok = tokens[idx * D_TED + j];
                qv += wq[d * D_TED + j] * tok;
                kv += wk[d * D_TED + j] * tok;
                vv += wv[d * D_TED + j] * tok;
            }
            q[idx * D_AEMB + d] = qv;
            k[idx * D_AEMB + d] = kv;
            v[idx * D_AEMB + d] = vv;
        }
    }
}

// 3. Multi-head attention: 每线程处理一个 (head, row i)
//    scores[i][j] = dot(qh[i], kh[j]) / sqrt(head_dim)，裁剪 ±20，softmax
//    head_out[h][i][d] = sum_j scores[i][j] * vh[j][d]
__global__ void kernel_attention(
    const float* q, const float* k, const float* v,  // [D_CELL][D_AEMB]
    float* head_out, float* scores)                   // softmax saved for backward
{
    int h = blockIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (h >= D_HEADS || i >= D_CELL) return;

    int start = h * D_AHD;
    const float inv_sqrt = 1.0f / sqrtf((float)D_AHD);

    // 计算该行的 scores（未归一化）
    float row[D_CELL];
    float max_val = -1e9f;
    for (int j = 0; j < D_CELL; j++) {
        float dot = 0.0f;
        for (int d = 0; d < D_AHD; d++) {
            dot += q[i * D_AEMB + start + d] * k[j * D_AEMB + start + d];
        }
        float s = dot * inv_sqrt;
        if (s > 20.0f) s = 20.0f;
        if (s < -20.0f) s = -20.0f;
        row[j] = s;
        if (s > max_val) max_val = s;
    }

    // softmax
    float sum = 0.0f;
    for (int j = 0; j < D_CELL; j++) {
        row[j] = expf(row[j] - max_val);
        sum += row[j];
    }
    if (sum > 0.0f) {
        for (int j = 0; j < D_CELL; j++) row[j] /= sum;
    } else {
        for (int j = 0; j < D_CELL; j++) row[j] = 1.0f / D_CELL;
    }

    for (int j = 0; j < D_CELL; j++)
        scores[(h * D_CELL + i) * D_CELL + j] = row[j];

    // head_out[h][i][d] = sum_j scores[i][j] * vh[j][d]
    for (int d = 0; d < D_AHD; d++) {
        float acc = 0.0f;
        for (int j = 0; j < D_CELL; j++) {
            acc += row[j] * v[j * D_AEMB + start + d];
        }
        head_out[h * D_CELL * D_AHD + i * D_AHD + d] = acc;
    }
}

// 4. 拼接 + 输出投影 + 残差: attn_out[i][d] = bo[d] + sum_j wo[d][j]*concat[i][j] + tokens[i][d]
__global__ void kernel_concat_proj(
    const float* head_out,  // [D_HEADS][D_CELL][D_AHD]
    const float* wo,        // [D_TED][D_AEMB]
    const float* bo,        // [D_TED]
    const float* tokens,    // [D_CELL][D_TED]
    float* attn_out)        // [D_CELL][D_TED]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= D_CELL) return;
    for (int d = 0; d < D_TED; d++) {
        float acc = bo[d];
        for (int h = 0; h < D_HEADS; h++) {
            int start = h * D_AHD;
            for (int dd = 0; dd < D_AHD; dd++) {
                acc += wo[d * D_AEMB + start + dd] *
                    head_out[h * D_CELL * D_AHD + idx * D_AHD + dd];
            }
        }
        attn_out[idx * D_TED + d] = acc + tokens[idx * D_TED + d];
    }
}

// 5. 展平: flat[cell * D_TED + d] = attn_out[cell][d]
__global__ void kernel_flatten(
    const float* attn_out,  // [D_CELL][D_TED]
    float* flat)            // [D_CELL * D_TED]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= D_CELL * D_TED) return;
    flat[idx] = attn_out[idx];
}

// 6. 可学习池化层: pooled_feat[i] = relu(pool_b[i] + sum_j pool_w[i][j]*flat[j])
//    j 遍历全 D_CELL*D_TED=400 维
__global__ void kernel_pool_layer(
    const float* pool_w,  // [D_FEAT][D_CELL*D_TED]
    const float* pool_b,  // [D_FEAT]
    const float* flat,    // [D_CELL * D_TED]
    float* pooled_feat)   // [D_FEAT]
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D_FEAT) return;
    float acc = pool_b[i];
    for (int j = 0; j < D_CELL * D_TED; j++) {
        acc += pool_w[i * (D_CELL * D_TED) + j] * flat[j];
    }
    pooled_feat[i] = (acc > 0.0f) ? acc : 0.0f;
}

// 7. 全连接层（通用）: y[i] = b[i] + sum_j w[i][j]*x[j]，可选 ReLU
__global__ void kernel_fc(
    const float* w,  // [out][in]
    const float* b,  // [out]
    const float* x,  // [in]
    float* y,        // [out]
    int in_size, int out_size, int apply_relu)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_size) return;
    float acc = b[i];
    for (int j = 0; j < in_size; j++) {
        acc += w[i * in_size + j] * x[j];
    }
    y[i] = (apply_relu && acc < 0.0f) ? 0.0f : acc;
}

// 8. Soft update: target = tau * source + (1 - tau) * target
__global__ void kernel_soft_update(float* target, const float* source, float tau, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        target[idx] = tau * source[idx] + (1.0f - tau) * target[idx];
    }
}

// ============================================================================
// 反向传播 kernels（对照 neural_net.cpp train_step）
// ============================================================================

// Huber 输出层 delta: delta_out[action] = Huber 梯度，loss 输出
__global__ void kernel_output_huber(
    const float* pred,   // [D_CELL]
    float target, int action, float huber_delta,
    float* delta_out,    // [D_CELL]
    float* loss_out)
{
    float error = pred[action] - target;
    float abs_error = fabsf(error);
    float loss;
    if (abs_error <= huber_delta) {
        loss = 0.5f * error * error;
        delta_out[action] = error;
    } else {
        loss = huber_delta * (abs_error - 0.5f * huber_delta);
        delta_out[action] = (error > 0.0f) ? huber_delta : -huber_delta;
    }
    *loss_out = loss;
}

// FC 层 delta 反向: delta_cur[j] = sum_i w[i][j]*delta_next[i]，乘 relu mask
__global__ void kernel_fc_delta(
    const float* w,          // [next][cur]
    const float* delta_next, // [next]
    const float* act_mask,   // [cur] 激活值（relu mask）
    float* delta_cur,        // [cur]
    int cur_size, int next_size)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cur_size) return;
    float sum = 0.0f;
    for (int i = 0; i < next_size; i++) {
        sum += w[i * cur_size + j] * delta_next[i];
    }
    delta_cur[j] = (act_mask[j] > 0.0f) ? sum : 0.0f;
}

// FC 权重更新: w[i][j] -= lr * delta_cur[i] * x_prev[j]
__global__ void kernel_fc_update_w(
    float* w,                // [cur][prev]
    const float* delta_cur,  // [cur]
    const float* x_prev,     // [prev]
    float lr, int cur_size, int prev_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = cur_size * prev_size;
    if (idx >= total) return;
    int i = idx / prev_size;
    int j = idx % prev_size;
    w[idx] -= lr * clipped_gradient(delta_cur[i] * x_prev[j]);
}

// 偏置更新: b[i] -= lr * delta[i]
__global__ void kernel_bias_update(
    float* b, const float* delta, float lr, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    b[i] -= lr * clipped_gradient(delta[i]);
}

// 池化层反向 delta: dL_dpooled_feat[j] = sum_i w1[i][j]*delta_h1[i]，乘 relu mask
__global__ void kernel_pool_backward(
    const float* w1,            // [H1][FEAT]
    const float* delta_h1,      // [H1]
    const float* pooled_feat,   // [FEAT] relu mask
    float* dL_dpooled_feat,     // [FEAT]
    int feat_size, int h1_size)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= feat_size) return;
    float sum = 0.0f;
    for (int i = 0; i < h1_size; i++) {
        sum += w1[i * feat_size + j] * delta_h1[i];
    }
    dL_dpooled_feat[j] = (pooled_feat[j] > 0.0f) ? sum : 0.0f;
}

// 池化权重更新: pool_w[i][j] -= lr*dL_dpooled_feat[i]*flat[j] (全 400 列)
__global__ void kernel_pool_update(
    float* pool_w,              // [FEAT][CELL*TED]
    const float* dL_dpooled_feat, // [FEAT]
    const float* flat,          // [CELL*TED]
    float lr, int feat_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = feat_size * (D_CELL * D_TED);
    if (idx >= total) return;
    int i = idx / (D_CELL * D_TED);
    int j = idx % (D_CELL * D_TED);
    pool_w[i * (D_CELL * D_TED) + j] -= lr *
        clipped_gradient(dL_dpooled_feat[i] * flat[j]);
}

// 展平层反向: dL_dflat[j] = sum_i dL_dpooled_feat[i]*pool_w[i][j]
// dL_dattn_out[cell][d] = dL_dflat[cell * D_TED + d]（无除法，展平直接映射）
__global__ void kernel_flatten_backward(
    const float* dL_dpooled_feat, // [FEAT]
    const float* pool_w,        // [FEAT][CELL*TED]
    float* dL_dflat,            // [CELL*TED]
    float* dL_dattn_out,        // [CELL][TED]
    int feat_size)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D_CELL * D_TED) return;
    float sum = 0.0f;
    for (int i = 0; i < feat_size; i++) {
        sum += dL_dpooled_feat[i] * pool_w[i * (D_CELL * D_TED) + j];
    }
    dL_dflat[j] = sum;
    dL_dattn_out[j] = sum;  // 展平层：1:1 映射
}

// attention 输出投影反向: dL_dtokens += dL_dattn_out, dL_dconcat = dL_dattn_out * wo^T,
//   更新 wo/bo（wo[d][j] -= lr*sum_i dL_dattn_out[i][d]*concat[i][j], bo[d] -= lr*sum_i dL_dattn_out[i][d]）
__global__ void kernel_attn_out_backward(
    const float* dL_dattn_out,  // [CELL][TED]
    const float* head_out,      // [HEADS][CELL][AHD]（用于重构 concat）
    const float* wo,            // [TED][AEMB]
    const float* bo,            // [TED]
    float* dL_dtokens,          // [CELL][TED]
    float* dL_dconcat,          // [CELL][AEMB]
    float lr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D_CELL) return;
    for (int d = 0; d < D_TED; d++) {
        // residual: dL_dtokens[i][d] += dL_dattn_out[i][d]
        float v = dL_dattn_out[i * D_TED + d];
        dL_dtokens[i * D_TED + d] += v;
        // dL_dconcat[i][j] += dL_dattn_out[i][d] * wo[d][j]
        for (int j = 0; j < D_AEMB; j++) {
            dL_dconcat[i * D_AEMB + j] += v * wo[d * D_AEMB + j];
        }
    }
    // 更新 wo/bo（对每个 d,j 累加所有 i）— 由单独 kernel 处理，此 kernel 只算梯度
    (void)bo;
}

// 更新 attn_wo/bo（需要所有 i 的 dL_dattn_out 和 concat）
__global__ void kernel_attn_wo_update(
    float* wo, float* bo,
    const float* dL_dattn_out,  // [CELL][TED]
    const float* head_out,      // [HEADS][CELL][AHD]
    float lr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // wo 是 [TED][AEMB]，bo 是 [TED]
    int total = D_TED * (D_AEMB + 1);  // wo 元素 + bo 元素
    if (idx >= total) return;
    int d = idx / (D_AEMB + 1);
    int r = idx % (D_AEMB + 1);
    if (r < D_AEMB) {
        int j = r;
        float grad = 0.0f;
        for (int i = 0; i < D_CELL; i++) {
            // concat[i][j] 从 head_out 重构
            float c = 0.0f;
            for (int h = 0; h < D_HEADS; h++) {
                int start = h * D_AHD;
                if (j >= start && j < start + D_AHD) {
                    c = head_out[h * D_CELL * D_AHD + i * D_AHD + (j - start)];
                }
            }
            grad += dL_dattn_out[i * D_TED + d] * c;
        }
        wo[d * D_AEMB + j] -= lr * clipped_gradient(grad);
    } else {
        // bo[d]
        float grad = 0.0f;
        for (int i = 0; i < D_CELL; i++) {
            grad += dL_dattn_out[i * D_TED + d];
        }
        bo[d] -= lr * clipped_gradient(grad);
    }
}

// 多头 attention 反向: 每线程处理一个 (head, row i)
//   第一遍算 ds 存 grad_scratch，第二遍算 softmax 梯度，原子累加全局
__global__ void kernel_attn_backward(
    const float* dL_dconcat,  // [CELL][AEMB]
    const float* scores,      // [HEADS][CELL][CELL] 归一化概率
    const float* q, const float* k, const float* v,  // [CELL][AEMB]
    float* dL_dq_global, float* dL_dk_global, float* dL_dv_global, // [CELL][AEMB]
    float* grad_scratch,      // [HEADS][CELL][CELL] 存 ds
    int cell_count)
{
    int h = blockIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (h >= D_HEADS || i >= cell_count) return;

    int start = h * D_AHD;
    float inv_sqrt = 1.0f / sqrtf((float)D_AHD);

    // 第一遍：dL_dscores[i][j] = sum_d dL_dhead[i][d] * v[j][d]，存 grad_scratch
    float row_ds[D_CELL];
    for (int j = 0; j < cell_count; j++) {
        float ds = 0.0f;
        for (int d = 0; d < D_AHD; d++) {
            ds += dL_dconcat[i * D_AEMB + start + d] * v[j * D_AEMB + start + d];
        }
        row_ds[j] = ds;
        grad_scratch[h * cell_count * cell_count + i * cell_count + j] = ds;
    }

    // 第二遍：softmax 梯度 + q/k 梯度 + v 梯度
    // sum_k = sum_j scores[i][j]*ds[i][j]
    float sum_k = 0.0f;
    for (int j = 0; j < cell_count; j++) {
        sum_k += scores[h * cell_count * cell_count + i * cell_count + j] * row_ds[j];
    }
    for (int j = 0; j < cell_count; j++) {
        float prob = scores[h * cell_count * cell_count + i * cell_count + j];
        float gs = prob * (row_ds[j] - sum_k);  // grad_softmax[i][j]
        float raw_score = 0.0f;
        for (int d = 0; d < D_AHD; ++d)
            raw_score += q[i * D_AEMB + start + d] * k[j * D_AEMB + start + d];
        raw_score *= inv_sqrt;
        if (raw_score <= -20.0f || raw_score >= 20.0f) gs = 0.0f;
        for (int d = 0; d < D_AHD; d++) {
            // dL_dqh[i][d] += gs * k[j][d] * inv_sqrt
            atomicAdd(&dL_dq_global[i * D_AEMB + start + d],
                gs * k[j * D_AEMB + start + d] * inv_sqrt);
            // dL_dkh[j][d] += gs * q[i][d] * inv_sqrt
            atomicAdd(&dL_dk_global[j * D_AEMB + start + d],
                gs * q[i * D_AEMB + start + d] * inv_sqrt);
            // dL_dvh[j][d] += dL_dhead[i][d] * scores[i][j]
            atomicAdd(&dL_dv_global[j * D_AEMB + start + d],
                dL_dconcat[i * D_AEMB + start + d] * prob);
        }
    }
}

// Q/K/V 对 token 输入的反向传播（必须在投影权重更新前读取旧权重）
__global__ void kernel_qkv_input_backward(
    const float* wq, const float* wk, const float* wv,
    const float* dL_dq, const float* dL_dk, const float* dL_dv,
    float* dL_dtokens)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= D_CELL * D_TED) return;
    int i = idx / D_TED, j = idx % D_TED;
    float grad = 0.0f;
    for (int d = 0; d < D_AEMB; d++) {
        grad += wq[d * D_TED + j] * dL_dq[i * D_AEMB + d];
        grad += wk[d * D_TED + j] * dL_dk[i * D_AEMB + d];
        grad += wv[d * D_TED + j] * dL_dv[i * D_AEMB + d];
    }
    dL_dtokens[idx] += grad;
}

// 更新 attention 投影权重：sum_i dQ[i,d]*token[i,j]，偏置为 sum_i dQ[i,d]
__global__ void kernel_attn_wqkv_update(
    float* wq, float* bq, float* wk, float* bk, float* wv, float* bv,
    const float* dL_dq_global, const float* dL_dk_global, const float* dL_dv_global,
    const float* tokens,
    float lr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = D_AEMB * (D_TED + 1);  // 权重 + 偏置
    if (idx >= total) return;
    int d = idx / (D_TED + 1);
    int r = idx % (D_TED + 1);
    if (r < D_TED) {
        int j = r;
        float gq = 0.0f, gk = 0.0f, gv = 0.0f;
        for (int i = 0; i < D_CELL; i++) {
            float tok = tokens[i * D_TED + j];
            gq += dL_dq_global[i * D_AEMB + d] * tok;
            gk += dL_dk_global[i * D_AEMB + d] * tok;
            gv += dL_dv_global[i * D_AEMB + d] * tok;
        }
        wq[d * D_TED + j] -= lr * clipped_gradient(gq);
        wk[d * D_TED + j] -= lr * clipped_gradient(gk);
        wv[d * D_TED + j] -= lr * clipped_gradient(gv);
    } else {
        float gq = 0.0f, gk = 0.0f, gv = 0.0f;
        for (int i = 0; i < D_CELL; i++) {
            gq += dL_dq_global[i * D_AEMB + d];
            gk += dL_dk_global[i * D_AEMB + d];
            gv += dL_dv_global[i * D_AEMB + d];
        }
        bq[d] -= lr * clipped_gradient(gq);
        bk[d] -= lr * clipped_gradient(gk);
        bv[d] -= lr * clipped_gradient(gv);
    }
}

// 计算 token 列和: token_col_sum[j] = sum_i tokens[i][j]
__global__ void kernel_token_col_sum(
    const float* tokens, float* token_col_sum)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D_TED) return;
    float sum = 0.0f;
    for (int i = 0; i < D_CELL; i++) sum += tokens[i * D_TED + j];
    token_col_sum[j] = sum;
}

// embedding 更新: token_embed[s][d] -= lr*dL_dtokens[i][d], pos_embed[i][d] -= lr*dL_dtokens[i][d]
__global__ void kernel_embedding_update(
    float* token_embed, float* pos_embed,
    const float* dL_dtokens,  // [CELL][TED]
    const float* input,       // [CELL]
    float lr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D_CELL) return;
    int s = network_token_index(input, i);
    if (s < 0) s = 0;
    if (s > 2) s = 2;
    for (int d = 0; d < D_TED; d++) {
        float g = clipped_gradient(dL_dtokens[i * D_TED + d]);
        atomicAdd(&token_embed[s * D_TED + d], -lr * g);
        pos_embed[i * D_TED + d] -= lr * g;
    }
}

// ============================================================================
// 设备句柄
// ============================================================================
typedef struct GpuNetDevice {
    GpuNetParams* d_params;  // 设备端参数
    GpuWork*      d_work;    // 设备端工作区
} GpuNetDevice;

// ============================================================================
// 前向计算（供 gpu_nn_forward 和 gpu_nn_train_step 复用）
// 填充 work 区全部激活（tokens, q/k/v, scores, head_out, attn_out, pooled, pooled_feat, h1-3, output）
// ============================================================================
static void forward_device(GpuWork* w, GpuNetParams* p) {
    int block = 256;
    DQN_PROFILE_LAUNCH(kernel_token_embedding, dim3((D_CELL + block - 1) / block), dim3(block), 0, 0,
        (float*)p->token_embed, (float*)p->pos_embed, w->input, (float*)w->tokens);
    DQN_PROFILE_LAUNCH(kernel_qkv_projection, dim3((D_CELL + block - 1) / block), dim3(block), 0, 0,
        (float*)w->tokens,
        (float*)p->attn_wq, (float*)p->attn_bq,
        (float*)p->attn_wk, (float*)p->attn_bk,
        (float*)p->attn_wv, (float*)p->attn_bv,
        (float*)w->q, (float*)w->k, (float*)w->v);
    dim3 attn_grid(D_HEADS, (D_CELL + 31) / 32);
    dim3 attn_block(1, 32);
    DQN_PROFILE_LAUNCH(kernel_attention, attn_grid, attn_block, 0, 0,
        (float*)w->q, (float*)w->k, (float*)w->v, (float*)w->head_out, (float*)w->scores);
    DQN_PROFILE_LAUNCH(kernel_concat_proj, dim3((D_CELL + block - 1) / block), dim3(block), 0, 0,
        (float*)w->head_out, (float*)p->attn_wo, (float*)p->attn_bo,
        (float*)w->tokens, (float*)w->attn_out);
    DQN_PROFILE_LAUNCH(kernel_flatten, dim3((D_CELL * D_TED + block - 1) / block), dim3(block), 0, 0,
        (float*)w->attn_out, (float*)w->pooled);
    DQN_PROFILE_LAUNCH(kernel_pool_layer, dim3((D_FEAT + block - 1) / block), dim3(block), 0, 0,
        (float*)p->pool_w, (float*)p->pool_b, (float*)w->pooled, (float*)w->pooled_feat);
    DQN_PROFILE_LAUNCH(kernel_fc, dim3((D_H1 + block - 1) / block), dim3(block), 0, 0,
        (float*)p->w1, (float*)p->b1, (float*)w->pooled_feat, (float*)w->h1, D_FEAT, D_H1, 1);
    DQN_PROFILE_LAUNCH(kernel_fc, dim3((D_H2 + block - 1) / block), dim3(block), 0, 0,
        (float*)p->w2, (float*)p->b2, (float*)w->h1, (float*)w->h2, D_H1, D_H2, 1);
    DQN_PROFILE_LAUNCH(kernel_fc, dim3((D_H3 + block - 1) / block), dim3(block), 0, 0,
        (float*)p->w3, (float*)p->b3, (float*)w->h2, (float*)w->h3, D_H2, D_H3, 1);
    DQN_PROFILE_LAUNCH(kernel_fc, dim3((D_CELL + block - 1) / block), dim3(block), 0, 0,
        (float*)p->w4, (float*)p->b4, (float*)w->h3, (float*)w->output, D_H3, D_CELL, 0);
}

// ============================================================================
// Host 导出函数（extern "C"）
// ============================================================================
extern "C" {

int gpu_init(void) {
    printf("[GPU] 初始化 HIP/ROCm...\n");

    hipError_t err = hipInit(0);
    if (err != hipSuccess) {
        printf("[GPU] hipInit 失败: %d (%s)\n", err, hipGetErrorString(err));
        return (int)err;
    }

    int count = 0;
    err = hipGetDeviceCount(&count);
    if (err != hipSuccess || count == 0) {
        printf("[GPU] 未检测到 GPU (count=%d, err=%d)\n", count, err);
        return -1;
    }

    hipDeviceProp_t prop;
    err = hipGetDeviceProperties(&prop, 0);
    if (err != hipSuccess) {
        printf("[GPU] 获取设备属性失败: %d\n", err);
        return (int)err;
    }

    printf("[GPU] 设备: %s\n", prop.name);
    printf("[GPU] 架构: %s\n", prop.gcnArchName);
    printf("[GPU] 显存: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    printf("[GPU] 计算单元: %d\n", prop.multiProcessorCount);

    err = hipSetDevice(0);
    if (err != hipSuccess) {
        printf("[GPU] hipSetDevice(0) 失败: %d\n", err);
        return (int)err;
    }

    printf("[GPU] HIP/ROCm 初始化成功\n");

    /* 初始化 kernel 计时器 (失败不阻断训练) */
    gpu_timer_init();

    return 0;
}

void gpu_shutdown(void) {
    gpu_timer_shutdown();
    hipDeviceReset();
}

// 创建 GPU 网络句柄（分配参数区 + 工作区）
int gpu_nn_create(GpuNetHandle* out) {
    if (!out) return -1;
    fprintf(stderr, "[gpu_nn_create] allocating params (%d bytes)...\n", (int)sizeof(GpuNetParams));
    hipError_t err;
    err = hipMalloc((void**)&out->params, sizeof(GpuNetParams));
    if (err != hipSuccess) { fprintf(stderr, "[gpu_nn_create] hipMalloc params failed: %d\n", err); return -2; }
    fprintf(stderr, "[gpu_nn_create] allocating work (%d bytes)...\n", (int)sizeof(GpuWork));
    err = hipMalloc((void**)&out->work, sizeof(GpuWork));
    if (err != hipSuccess) {
        fprintf(stderr, "[gpu_nn_create] hipMalloc work failed: %d\n", err);
        hipFree(out->params);
        out->params = NULL;
        return -3;
    }
    err = gpu_profile::memset(out->params, 0, sizeof(GpuNetParams));
    if (err == hipSuccess) err = gpu_profile::memset(out->work, 0, sizeof(GpuWork));
    if (err != hipSuccess) {
        fprintf(stderr, "[gpu_nn_create] hipMemset failed: %d\n", err);
        hipFree(out->params);
        hipFree(out->work);
        out->params = out->work = NULL;
        return -4;
    }
    out->is_valid = 1;
    fprintf(stderr, "[gpu_nn_create] OK\n");
    return 0;
}

// 销毁 GPU 网络句柄
void gpu_nn_destroy(GpuNetHandle* h) {
    if (!h) return;
    if (h->params) { hipFree(h->params); h->params = NULL; }
    if (h->work)   { hipFree(h->work);   h->work   = NULL; }
    h->is_valid = 0;
}

// 上传参数（cpu_params 是打包好的 GpuNetParams 主存镜像）
int gpu_nn_upload(GpuNetHandle h, const void* cpu_params, size_t size) {
    if (!h.is_valid || !cpu_params || size != sizeof(GpuNetParams)) return -1;
    gpu_timer_begin(GPU_KT_UPLOAD, 0);
    hipError_t err = gpu_profile::copy(h.params, cpu_params, size, hipMemcpyHostToDevice);
    if (err != hipSuccess) return -2;
    gpu_timer_end();
    return 0;
}

// 下载参数
int gpu_nn_download(GpuNetHandle h, void* cpu_params, size_t size) {
    if (!h.is_valid || !cpu_params || size != sizeof(GpuNetParams)) return -1;
    gpu_timer_begin(GPU_KT_DOWNLOAD, 0);
    hipError_t err = gpu_profile::copy(cpu_params, h.params, size, hipMemcpyDeviceToHost);
    if (err != hipSuccess) return -2;
    gpu_timer_end();
    return 0;
}

// GPU 前向传播: input[26] -> output[25]
int gpu_nn_forward(GpuNetHandle h, const float* input, float* output) {
    if (!h.is_valid || !input || !output) return -1;
    gpu_timer_begin(GPU_KT_FORWARD, 0);

    GpuWork* w = (GpuWork*)h.work;
    GpuNetParams* p = (GpuNetParams*)h.params;

    // 输入上传
    hipError_t err = gpu_profile::copy(w->input, input, sizeof(float) * D_STATE, hipMemcpyHostToDevice);
    if (err != hipSuccess) return -2;

    // 前向
    forward_device(w, p);

    err = hipGetLastError();
    if (err != hipSuccess) return -3;
    err = gpu_profile::sync();
    if (err != hipSuccess) return -4;

    // 输出下载
    err = gpu_profile::copy(output, w->output, sizeof(float) * D_CELL, hipMemcpyDeviceToHost);
    if (err != hipSuccess) return -5;

    gpu_timer_end();
    return 0;
}

// GPU 批量前向：states[32][26] -> out_qvals[32][25]（CPU 输出）
// 批量内异步，一次同步 + 一次下载
int gpu_nn_forward_batch(GpuNetHandle h, const float* states, int batch_size, float* out_qvals) {
    if (!h.is_valid || !states || !out_qvals) return -1;
    if (batch_size <= 0) return -3;
    gpu_timer_begin(GPU_KT_FORWARD_BATCH, (uint32_t)batch_size);
    if (batch_size > GPU_MAX_BATCH) batch_size = GPU_MAX_BATCH;

    GpuWork* w = (GpuWork*)h.work;
    GpuNetParams* p = (GpuNetParams*)h.params;

    hipError_t err = gpu_profile::copy(w->batch_states, states,
        sizeof(float) * D_STATE * batch_size, hipMemcpyHostToDevice);
    if (err != hipSuccess) return -2;

    for (int b = 0; b < batch_size; b++) {
        err = gpu_profile::copy(w->input, &w->batch_states[b], sizeof(float) * D_STATE, hipMemcpyDeviceToDevice);
        if (err != hipSuccess) return -4;
        forward_device(w, p);
        err = gpu_profile::copy(&w->batch_output[b], w->output, sizeof(float) * D_CELL, hipMemcpyDeviceToDevice);
        if (err != hipSuccess) return -5;
    }

    err = hipGetLastError();
    if (err != hipSuccess) return -6;
    err = gpu_profile::sync();
    if (err != hipSuccess) return -7;
    err = gpu_profile::copy(out_qvals, w->batch_output,
        sizeof(float) * D_CELL * batch_size, hipMemcpyDeviceToHost);
    if (err != hipSuccess) return -8;
    gpu_timer_end();
    return 0;
}

// GPU 训练步骤：前向 + 反向 + 权重更新全在 GPU，参数驻留
// state[26], action, target, lr -> out_loss
// 反向传播 + 权重更新（单样本和批量共用；forward 和 output_huber 在调用方先执行）
/* backward_update 分段计时 (extern "C" 统一符号名, 定义在 neural_net.cpp) */
extern "C" {
extern volatile double g_bwd_fc4_us, g_bwd_fc3_us, g_bwd_fc2_us, g_bwd_fc1_us;
extern volatile double g_bwd_pool_us, g_bwd_attn_us, g_bwd_embed_us;
extern volatile int g_bwd_count;
/* train_step_batch per-sample 开销分段 */
extern volatile double g_ts_setup_us, g_ts_memset_us, g_ts_fwd_us, g_ts_hubler_us, g_ts_bwd_us;
extern volatile double g_ts_validate_us;
extern volatile int g_ts_count;
}

static void backward_update(GpuWork* w, GpuNetParams* p, float lr) {
    gpu_module_profile::stage("backward_fc4");
    int block = 256;
    LARGE_INTEGER pf_bu, t0, t1;

    // FC4 反向 + 更新 (delta_h3, w4, b4)
    QueryPerformanceFrequency(&pf_bu);
    QueryPerformanceCounter(&t0);
    DQN_PROFILE_LAUNCH(kernel_fc_delta, dim3((D_H3 + block - 1) / block), dim3(block), 0, 0,
        (float*)p->w4, (float*)w->delta_out, (float*)w->h3, (float*)w->delta_h3, D_H3, D_CELL);
    DQN_PROFILE_LAUNCH(kernel_fc_update_w, dim3((D_CELL * D_H3 + block - 1) / block), dim3(block), 0, 0,
        (float*)p->w4, (float*)w->delta_out, (float*)w->h3, lr, D_CELL, D_H3);
    DQN_PROFILE_LAUNCH(kernel_bias_update, dim3((D_CELL + block - 1) / block), dim3(block), 0, 0,
        (float*)p->b4, (float*)w->delta_out, lr, D_CELL);
    gpu_profile::sync();
    QueryPerformanceCounter(&t1);
    g_bwd_fc4_us += (double)(t1.QuadPart - t0.QuadPart) / (double)pf_bu.QuadPart * 1e6;
    gpu_module_profile::stage("backward_fc3");

    // FC3 反向 + 更新 (delta_h2, w3, b3)
    QueryPerformanceCounter(&t0);
    DQN_PROFILE_LAUNCH(kernel_fc_delta, dim3((D_H2 + block - 1) / block), dim3(block), 0, 0,
        (float*)p->w3, (float*)w->delta_h3, (float*)w->h2, (float*)w->delta_h2, D_H2, D_H3);
    DQN_PROFILE_LAUNCH(kernel_fc_update_w, dim3((D_H3 * D_H2 + block - 1) / block), dim3(block), 0, 0,
        (float*)p->w3, (float*)w->delta_h3, (float*)w->h2, lr, D_H3, D_H2);
    DQN_PROFILE_LAUNCH(kernel_bias_update, dim3((D_H3 + block - 1) / block), dim3(block), 0, 0,
        (float*)p->b3, (float*)w->delta_h3, lr, D_H3);
    gpu_profile::sync();
    QueryPerformanceCounter(&t1);
    g_bwd_fc3_us += (double)(t1.QuadPart - t0.QuadPart) / (double)pf_bu.QuadPart * 1e6;
    gpu_module_profile::stage("backward_fc2");

    // FC2 反向 + 更新 (delta_h1, w2, b2)
    QueryPerformanceCounter(&t0);
    DQN_PROFILE_LAUNCH(kernel_fc_delta, dim3((D_H1 + block - 1) / block), dim3(block), 0, 0,
        (float*)p->w2, (float*)w->delta_h2, (float*)w->h1, (float*)w->delta_h1, D_H1, D_H2);
    DQN_PROFILE_LAUNCH(kernel_fc_update_w, dim3((D_H2 * D_H1 + block - 1) / block), dim3(block), 0, 0,
        (float*)p->w2, (float*)w->delta_h2, (float*)w->h1, lr, D_H2, D_H1);
    DQN_PROFILE_LAUNCH(kernel_bias_update, dim3((D_H2 + block - 1) / block), dim3(block), 0, 0,
        (float*)p->b2, (float*)w->delta_h2, lr, D_H2);
    gpu_profile::sync();
    QueryPerformanceCounter(&t1);
    g_bwd_fc2_us += (double)(t1.QuadPart - t0.QuadPart) / (double)pf_bu.QuadPart * 1e6;
    gpu_module_profile::stage("backward_fc1");

    // FC1 反向 + 更新 (pool_backward, w1, b1)
    QueryPerformanceCounter(&t0);
    DQN_PROFILE_LAUNCH(kernel_pool_backward, dim3((D_FEAT + block - 1) / block), dim3(block), 0, 0,
        (float*)p->w1, (float*)w->delta_h1, (float*)w->pooled_feat, (float*)w->dL_dpooled_feat, D_FEAT, D_H1);
    DQN_PROFILE_LAUNCH(kernel_fc_update_w, dim3((D_H1 * D_FEAT + block - 1) / block), dim3(block), 0, 0,
        (float*)p->w1, (float*)w->delta_h1, (float*)w->pooled_feat, lr, D_H1, D_FEAT);
    DQN_PROFILE_LAUNCH(kernel_bias_update, dim3((D_H1 + block - 1) / block), dim3(block), 0, 0,
        (float*)p->b1, (float*)w->delta_h1, lr, D_H1);
    gpu_profile::sync();
    QueryPerformanceCounter(&t1);
    g_bwd_fc1_us += (double)(t1.QuadPart - t0.QuadPart) / (double)pf_bu.QuadPart * 1e6;
    gpu_module_profile::stage("backward_pool");

    // Pool 层反向 + 更新 (flatten_backward, pool_update, pool_b)
    QueryPerformanceCounter(&t0);
    DQN_PROFILE_LAUNCH(kernel_flatten_backward, dim3((D_CELL * D_TED + block - 1) / block), dim3(block), 0, 0,
        (float*)w->dL_dpooled_feat, (float*)p->pool_w, (float*)w->dL_dpooled, (float*)w->dL_dattn_out, D_FEAT);
    DQN_PROFILE_LAUNCH(kernel_pool_update, dim3((D_FEAT * D_CELL * D_TED + block - 1) / block), dim3(block), 0, 0,
        (float*)p->pool_w, (float*)w->dL_dpooled_feat, (float*)w->pooled, lr, D_FEAT);
    DQN_PROFILE_LAUNCH(kernel_bias_update, dim3((D_FEAT + block - 1) / block), dim3(block), 0, 0,
        (float*)p->pool_b, (float*)w->dL_dpooled_feat, lr, D_FEAT);
    gpu_profile::sync();
    QueryPerformanceCounter(&t1);
    g_bwd_pool_us += (double)(t1.QuadPart - t0.QuadPart) / (double)pf_bu.QuadPart * 1e6;
    gpu_module_profile::stage("backward_attention");

    // Attention 反向 + 更新 (attn_out_backward, attn_wo_update, attn_backward, qkv_input_backward, attn_wqkv_update)
    QueryPerformanceCounter(&t0);
    DQN_PROFILE_LAUNCH(kernel_attn_out_backward, dim3((D_CELL + block - 1) / block), dim3(block), 0, 0,
        (float*)w->dL_dattn_out, (float*)w->head_out, (float*)p->attn_wo, (float*)p->attn_bo,
        (float*)w->dL_dtokens, (float*)w->dL_dconcat, lr);
    DQN_PROFILE_LAUNCH(kernel_attn_wo_update, dim3((D_TED * (D_AEMB + 1) + block - 1) / block), dim3(block), 0, 0,
        (float*)p->attn_wo, (float*)p->attn_bo, (float*)w->dL_dattn_out, (float*)w->head_out, lr);
    dim3 attn_b_grid(D_HEADS, (D_CELL + 31) / 32);
    dim3 attn_b_block(1, 32);
    DQN_PROFILE_LAUNCH(kernel_attn_backward, attn_b_grid, attn_b_block, 0, 0,
        (float*)w->dL_dconcat, (float*)w->scores,
        (float*)w->q, (float*)w->k, (float*)w->v,
        (float*)w->dL_dq_global, (float*)w->dL_dk_global, (float*)w->dL_dv_global,
        (float*)w->grad_scratch, D_CELL);
    DQN_PROFILE_LAUNCH(kernel_qkv_input_backward,
        dim3((D_CELL * D_TED + block - 1) / block), dim3(block), 0, 0,
        (float*)p->attn_wq, (float*)p->attn_wk, (float*)p->attn_wv,
        (float*)w->dL_dq_global, (float*)w->dL_dk_global, (float*)w->dL_dv_global,
        (float*)w->dL_dtokens);
    DQN_PROFILE_LAUNCH(kernel_attn_wqkv_update, dim3((D_AEMB * (D_TED + 1) + block - 1) / block), dim3(block), 0, 0,
        (float*)p->attn_wq, (float*)p->attn_bq,
        (float*)p->attn_wk, (float*)p->attn_bk,
        (float*)p->attn_wv, (float*)p->attn_bv,
        (float*)w->dL_dq_global, (float*)w->dL_dk_global, (float*)w->dL_dv_global,
        (float*)w->tokens, lr);
    gpu_profile::sync();
    QueryPerformanceCounter(&t1);
    g_bwd_attn_us += (double)(t1.QuadPart - t0.QuadPart) / (double)pf_bu.QuadPart * 1e6;
    gpu_module_profile::stage("backward_embedding");

    // Embedding 更新
    QueryPerformanceCounter(&t0);
    DQN_PROFILE_LAUNCH(kernel_embedding_update, dim3((D_CELL + block - 1) / block), dim3(block), 0, 0,
        (float*)p->token_embed, (float*)p->pos_embed,
        (float*)w->dL_dtokens, (float*)w->input, lr);
    gpu_profile::sync();
    QueryPerformanceCounter(&t1);
    g_bwd_embed_us += (double)(t1.QuadPart - t0.QuadPart) / (double)pf_bu.QuadPart * 1e6;

    g_bwd_count++;
}

__global__ void kernel_validate_params(const float* params, int count, int* invalid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (!__builtin_isfinite(params[i])) atomicMax(invalid, 2);
    else if (fabsf(params[i]) > D_PARAM_ABS_LIMIT) atomicMax(invalid, 1);
}

// GPU 训练步骤（单样本）：前向 + 反向 + 权重更新全在 GPU
int gpu_nn_train_step_batch(GpuNetHandle h, const float* states, const int* actions,
    const float* targets, int batch_size, float lr, float* out_loss);

int gpu_nn_train_step(GpuNetHandle h, const float* state, int action, float target, float lr, float* out_loss) {
    return gpu_nn_train_step_batch(h, state, &action, &target, 1, lr, out_loss);
}

// GPU 批量训练步骤：batch 内异步排队，末尾一次同步。
// 每个样本使用 lr/batch_size，使一次调用的总更新量等价于平均批梯度，
// 而不是无意中把配置学习率放大 batch_size 倍。
int gpu_nn_train_step_batch(GpuNetHandle h, const float* states, const int* actions,
    const float* targets, int batch_size, float lr, float* out_loss) {
    if (!h.is_valid || !states || !actions || !targets || !out_loss) return -1;
    if (batch_size <= 0 || batch_size > GPU_MAX_BATCH || !isfinite(lr) || lr <= 0 || lr > 1) return -3;
    gpu_module_profile::Batch module_profile(batch_size);
    gpu_profile::Batch profile(batch_size);
    gpu_timer_begin(GPU_KT_TRAIN_STEP_BATCH, (uint32_t)batch_size);
    for (int b = 0; b < batch_size; ++b) {
        if (actions[b] < 0 || actions[b] >= D_CELL || !isfinite(targets[b])) return -3;
        for (int i = 0; i < D_STATE; ++i)
            if (!isfinite(states[b * D_STATE + i])) return -3;
    }

    GpuWork* w = (GpuWork*)h.work;
    GpuNetParams* p = &w->update_params;

    /* per-sample 开销分段计时 */
    LARGE_INTEGER pf_ts, ts0, ts1;
    QueryPerformanceFrequency(&pf_ts);

    QueryPerformanceCounter(&ts0);
    if (gpu_profile::copy(p, h.params, sizeof(*p), hipMemcpyDeviceToDevice) != hipSuccess) return -4;
    hipError_t err = gpu_profile::copy(w->batch_states, states,
        sizeof(float) * D_STATE * batch_size, hipMemcpyHostToDevice);
    if (err != hipSuccess) return -4;
    QueryPerformanceCounter(&ts1);
    g_ts_setup_us += (double)(ts1.QuadPart - ts0.QuadPart) / (double)pf_ts.QuadPart * 1e6;

    const float eff_lr = lr / (float)batch_size;

    for (int b = 0; b < batch_size; b++) {
        gpu_profile::stage("input_copy" );
        /* D2D copy */
        QueryPerformanceCounter(&ts0);
        if (gpu_profile::copy(w->input, &w->batch_states[b], sizeof(float) * D_STATE,
            hipMemcpyDeviceToDevice) != hipSuccess) return -5;

        gpu_profile::stage("memset" );
        /* 6x hipMemset */
        if (gpu_profile::memset(w->delta_out, 0, sizeof(float) * D_CELL) != hipSuccess ||
            gpu_profile::memset(w->dL_dq_global, 0, sizeof(float) * D_CELL * D_AEMB) != hipSuccess ||
            gpu_profile::memset(w->dL_dk_global, 0, sizeof(float) * D_CELL * D_AEMB) != hipSuccess ||
            gpu_profile::memset(w->dL_dv_global, 0, sizeof(float) * D_CELL * D_AEMB) != hipSuccess ||
            gpu_profile::memset(w->dL_dtokens, 0, sizeof(float) * D_CELL * D_TED) != hipSuccess ||
            gpu_profile::memset(w->dL_dconcat, 0, sizeof(float) * D_CELL * D_AEMB) != hipSuccess) return -6;
        QueryPerformanceCounter(&ts1);
        g_ts_memset_us += (double)(ts1.QuadPart - ts0.QuadPart) / (double)pf_ts.QuadPart * 1e6;

        gpu_profile::stage("forward" );
        /* forward */
        QueryPerformanceCounter(&ts0);
        forward_device(w, p);
        QueryPerformanceCounter(&ts1);
        g_ts_fwd_us += (double)(ts1.QuadPart - ts0.QuadPart) / (double)pf_ts.QuadPart * 1e6;

        gpu_profile::stage("huber" );
        /* huber */
        QueryPerformanceCounter(&ts0);
        DQN_PROFILE_LAUNCH(kernel_output_huber, 1, 1, 0, 0,
            (float*)w->output, targets[b], actions[b], 0.5f,
            (float*)w->delta_out, (float*)&w->loss_buf[b]);
        QueryPerformanceCounter(&ts1);
        g_ts_hubler_us += (double)(ts1.QuadPart - ts0.QuadPart) / (double)pf_ts.QuadPart * 1e6;

        gpu_profile::stage("backward" );
        /* backward */
        QueryPerformanceCounter(&ts0);
        backward_update(w, p, eff_lr);
        QueryPerformanceCounter(&ts1);
        g_ts_bwd_us += (double)(ts1.QuadPart - ts0.QuadPart) / (double)pf_ts.QuadPart * 1e6;
    }

    gpu_profile::stage("validation" );
    /* Parameter gradients were clipped at their update sites.  Validate the
       whole staged parameter image before publishing it as the live network. */
    QueryPerformanceCounter(&ts0);
    if (gpu_profile::memset(&w->invalid_params, 0, sizeof(int)) != hipSuccess) return -8;
    const int param_count = sizeof(GpuNetParams) / sizeof(float);
    DQN_PROFILE_LAUNCH(kernel_validate_params, dim3((param_count + 255) / 256), dim3(256), 0, 0,
        (const float*)p, param_count, &w->invalid_params);
    err = hipGetLastError();
    if (err != hipSuccess) return -7;
    err = gpu_profile::sync();
    if (err != hipSuccess) return -8;
    float losses[GPU_MAX_BATCH];
    err = gpu_profile::copy(losses, w->loss_buf, sizeof(float) * batch_size, hipMemcpyDeviceToHost);
    if (err != hipSuccess) return -9;
    double loss_sum = 0.0;
    for (int b = 0; b < batch_size; b++) {
        if (!isfinite(losses[b]) || losses[b] < 0.0f) return -101;
        /* p is still a private staged image.  Reject the complete transaction
           before publishing any update if a caller missed per-sample filtering. */
        if (losses[b] > D_LOSS_REJECT_LIMIT) return -102;
        loss_sum += losses[b];
    }
    // Only publish a fully successful batch. -100 marks an indeterminate commit.
    int invalid = 0;
    if (gpu_profile::copy(&invalid, &w->invalid_params, sizeof(invalid), hipMemcpyDeviceToHost) != hipSuccess) return -9;
    if (invalid == 2) return -103; // NaN/Inf parameter: live parameters remain intact
    if (invalid == 1) return -104; // finite but unsafe magnitude
    QueryPerformanceCounter(&ts1);
    g_ts_validate_us += (double)(ts1.QuadPart - ts0.QuadPart) / (double)pf_ts.QuadPart * 1e6;
    g_ts_count++;
    gpu_module_profile::stage("commit");
    if (gpu_profile::copy(h.params, p, sizeof(*p), hipMemcpyDeviceToDevice) != hipSuccess) return -100;
    profile.committed=true;
    module_profile.commit();
    *out_loss = (float)(loss_sum / batch_size);
    gpu_timer_end();
    return 0;
}

// GPU soft update: target = tau * source + (1-tau) * target
int gpu_nn_soft_update(GpuNetHandle target, GpuNetHandle source, float tau) {
    if (!target.is_valid || !source.is_valid) return -1;
    gpu_timer_begin(GPU_KT_SOFT_UPDATE, 0);
    int n = sizeof(GpuNetParams) / sizeof(float);
    int block = 256;
    DQN_PROFILE_LAUNCH(kernel_soft_update, dim3((n + block - 1) / block), dim3(block), 0, 0,
        (float*)target.params, (const float*)source.params, tau, n);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) return -2;
    err = gpu_profile::sync();
    if (err != hipSuccess) return -3;
    gpu_timer_end();
    return 0;
}

} // extern "C"
