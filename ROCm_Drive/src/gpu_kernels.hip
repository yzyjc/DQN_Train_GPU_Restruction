// gpu_kernels.hip
// GPU-Resident Training Kernels
// Compiled with: hipcc --offload-arch=gfx1151
//
// NOTE: ROCm headers patched for MSVC 2026 compatibility
// (added _MSC_VER guards in __clang_cuda_math_forward_declares.h and __clang_hip_cmath.h)

#include <hip/hip_runtime.h>

// ============================================================================
// NeuralNet dimensions (must match DQN_Core)
// ============================================================================

#define BOARD_SIZE      5
#define CELL_COUNT      (BOARD_SIZE * BOARD_SIZE)
#define FEATURE_DIM     128
#define HIDDEN1_SIZE    1024
#define HIDDEN2_SIZE    512
#define HIDDEN3_SIZE    256
#define TOKEN_EMBED_DIM 16
#define ATTENTION_HEADS 4
#define ATTENTION_EMBED 32
#define ATTENTION_HEAD_DIM (ATTENTION_EMBED / ATTENTION_HEADS)

// ============================================================================
// Soft update kernel: target = tau * source + (1 - tau) * target
// ============================================================================

__global__ void kernel_soft_update(float* target, const float* source, float tau, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        target[idx] = tau * source[idx] + (1.0f - tau) * target[idx];
    }
}

// ============================================================================
// ReLU activation kernel
// ============================================================================

__global__ void kernel_relu(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = fmaxf(data[idx], 0.0f);
    }
}

// ============================================================================
// ReLU backward kernel (gradient mask)
// ============================================================================

__global__ void kernel_relu_backward(float* grad, const float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad[idx] *= (data[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

// ============================================================================
// Weight update kernel: w -= lr * grad
// ============================================================================

__global__ void kernel_weight_update(float* w, const float* grad, float lr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        w[idx] -= lr * grad[idx];
    }
}

// ============================================================================
// Huber loss kernel
// ============================================================================

__global__ void kernel_huber_loss(const float* pred, const float* target, float* loss, int n, float delta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float error = pred[idx] - target[idx];
        float abs_error = fabsf(error);
        if (abs_error <= delta) {
            loss[idx] = 0.5f * error * error;
        } else {
            loss[idx] = delta * (abs_error - 0.5f * delta);
        }
    }
}

// ============================================================================
// Token embedding kernel
// ============================================================================

__global__ void kernel_token_embedding(
    const float* token_embed,   // [3][TOKEN_EMBED_DIM]
    const float* pos_embed,     // [CELL_COUNT][TOKEN_EMBED_DIM]
    const float* input,         // [CELL_COUNT] (0, 1, or -1)
    float* output,              // [CELL_COUNT][TOKEN_EMBED_DIM]
    int cell_count,
    int token_embed_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < cell_count) {
        int state_idx = (int)(input[idx] + 1.5f);
        if (state_idx < 0) state_idx = 0;
        if (state_idx > 2) state_idx = 2;

        for (int d = 0; d < token_embed_dim; d++) {
            output[idx * token_embed_dim + d] =
                token_embed[state_idx * token_embed_dim + d] +
                pos_embed[idx * token_embed_dim + d];
        }
    }
}

// ============================================================================
// QKV projection kernel
// ============================================================================

__global__ void kernel_qkv_projection(
    const float* tokens,        // [CELL_COUNT][TOKEN_EMBED_DIM]
    const float* wq,            // [ATTENTION_EMBED][TOKEN_EMBED_DIM]
    const float* bq,            // [ATTENTION_EMBED]
    const float* wk,            // [ATTENTION_EMBED][TOKEN_EMBED_DIM]
    const float* bk,            // [ATTENTION_EMBED]
    const float* wv,            // [ATTENTION_EMBED][TOKEN_EMBED_DIM]
    const float* bv,            // [ATTENTION_EMBED]
    float* q,                   // [CELL_COUNT][ATTENTION_EMBED]
    float* k,                   // [CELL_COUNT][ATTENTION_EMBED]
    float* v,                   // [CELL_COUNT][ATTENTION_EMBED]
    int cell_count,
    int token_embed_dim,
    int attention_embed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < cell_count) {
        for (int d = 0; d < attention_embed; d++) {
            float q_val = bq[d];
            float k_val = bk[d];
            float v_val = bv[d];
            for (int j = 0; j < token_embed_dim; j++) {
                float token = tokens[idx * token_embed_dim + j];
                q_val += wq[d * token_embed_dim + j] * token;
                k_val += wk[d * token_embed_dim + j] * token;
                v_val += wv[d * token_embed_dim + j] * token;
            }
            q[idx * attention_embed + d] = q_val;
            k[idx * attention_embed + d] = k_val;
            v[idx * attention_embed + d] = v_val;
        }
    }
}

// ============================================================================
// Host wrapper functions (called from rocm_accelerator_v2.cpp)
// ============================================================================

extern "C" {

void launch_soft_update(float* d_target, const float* d_source, float tau, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    hipLaunchKernelGGL(kernel_soft_update, dim3(grid), dim3(block), 0, 0,
        d_target, d_source, tau, n);
}

void launch_relu(float* d_data, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    hipLaunchKernelGGL(kernel_relu, dim3(grid), dim3(block), 0, 0, d_data, n);
}

void launch_relu_backward(float* d_grad, const float* d_data, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    hipLaunchKernelGGL(kernel_relu_backward, dim3(grid), dim3(block), 0, 0,
        d_grad, d_data, n);
}

void launch_weight_update(float* d_w, const float* d_grad, float lr, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    hipLaunchKernelGGL(kernel_weight_update, dim3(grid), dim3(block), 0, 0,
        d_w, d_grad, lr, n);
}

void launch_huber_loss(const float* d_pred, const float* d_target, float* d_loss, int n, float delta) {
    int block = 256;
    int grid = (n + block - 1) / block;
    hipLaunchKernelGGL(kernel_huber_loss, dim3(grid), dim3(block), 0, 0,
        d_pred, d_target, d_loss, n, delta);
}

void launch_token_embedding(
    const float* d_token_embed, const float* d_pos_embed,
    const float* d_input, float* d_output,
    int cell_count, int token_embed_dim)
{
    int block = 256;
    int grid = (cell_count + block - 1) / block;
    hipLaunchKernelGGL(kernel_token_embedding, dim3(grid), dim3(block), 0, 0,
        d_token_embed, d_pos_embed, d_input, d_output,
        cell_count, token_embed_dim);
}

void launch_qkv_projection(
    const float* d_tokens,
    const float* d_wq, const float* d_bq,
    const float* d_wk, const float* d_bk,
    const float* d_wv, const float* d_bv,
    float* d_q, float* d_k, float* d_v,
    int cell_count, int token_embed_dim, int attention_embed)
{
    int block = 256;
    int grid = (cell_count + block - 1) / block;
    hipLaunchKernelGGL(kernel_qkv_projection, dim3(grid), dim3(block), 0, 0,
        d_tokens, d_wq, d_bq, d_wk, d_bk, d_wv, d_bv,
        d_q, d_k, d_v, cell_count, token_embed_dim, attention_embed);
}

} // extern "C"
