// accelerator_cpu_v2.cpp
// CPU Fallback for Accelerator v2 Interface
// Handwritten matrix multiplication, no GPU dependency

#define _CRT_SECURE_NO_WARNINGS
#include "accelerator_v2.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// Internal context
struct AccelV2Context {
    int initialized;
};

struct GpuMemoryBlock {
    void* data;
    size_t size;
};

struct GpuNeuralNetCtx {
    void* weights;
    size_t weights_size;
};

struct GpuReplayBufferCtx {
    int capacity;
    int state_dim;
    int count;
    int head;
    float* states;
    int* actions;
    float* rewards;
    float* next_states;
    int* dones;
};

// ============================================================================
// Handwritten matrix operations
// ============================================================================

static void cpu_sgemv(int m, int n, float alpha, const float* A, int lda,
    const float* x, int incx, float beta, float* y, int incy)
{
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) sum += A[i * lda + j] * x[j * incx];
        y[i * incy] = alpha * sum + beta * y[i * incy];
    }
}

static void cpu_sger(int m, int n, float alpha, const float* x, int incx,
    const float* y, int incy, float* A, int lda)
{
    for (int i = 0; i < m; i++) {
        float xv = alpha * x[i * incx];
        for (int j = 0; j < n; j++) A[i * lda + j] += xv * y[j * incy];
    }
}

static void cpu_sgemm(int m, int n, int k, float alpha, const float* A, int lda,
    const float* B, int ldb, float beta, float* C, int ldc)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) sum += A[i * lda + p] * B[p * ldb + j];
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}

// ============================================================================
// Device Management
// ============================================================================

AccelV2Status accelv2_init(AccelV2Handle* out_accel) {
    if (!out_accel) return ACCEL_V2_ERROR_INVALID_PARAM;
    AccelV2Context* ctx = (AccelV2Context*)malloc(sizeof(AccelV2Context));
    if (!ctx) return ACCEL_V2_ERROR_OUT_OF_MEMORY;
    ctx->initialized = 1;
    *out_accel = (AccelV2Handle)ctx;
    printf("[ACCEL-V2] CPU fallback initialized\n");
    return ACCEL_V2_SUCCESS;
}

AccelV2Status accelv2_cleanup(AccelV2Handle accel) {
    if (!accel) return ACCEL_V2_ERROR_INVALID_PARAM;
    free(accel);
    return ACCEL_V2_SUCCESS;
}

AccelV2Status accelv2_get_device_name(AccelV2Handle accel, char* name, int name_size) {
    if (!accel || !name) return ACCEL_V2_ERROR_INVALID_PARAM;
    strncpy(name, "CPU (Handwritten)", name_size - 1);
    name[name_size - 1] = '\0';
    return ACCEL_V2_SUCCESS;
}

AccelV2Status accelv2_get_device_memory(AccelV2Handle accel, uint64_t* total_mem, uint64_t* free_mem) {
    if (!accel || !total_mem || !free_mem) return ACCEL_V2_ERROR_INVALID_PARAM;
    *total_mem = 0;
    *free_mem = 0;
    return ACCEL_V2_SUCCESS;
}

AccelV2Status accelv2_sync(AccelV2Handle accel) {
    if (!accel) return ACCEL_V2_ERROR_INVALID_PARAM;
    return ACCEL_V2_SUCCESS;
}

// ============================================================================
// GPU Memory Management (CPU: just malloc/free)
// ============================================================================

AccelV2Status accelv2_malloc(AccelV2Handle accel, size_t size, GpuMemory* out_mem) {
    if (!accel || !out_mem || size == 0) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuMemoryBlock* blk = (GpuMemoryBlock*)malloc(sizeof(GpuMemoryBlock));
    if (!blk) return ACCEL_V2_ERROR_OUT_OF_MEMORY;
    blk->data = malloc(size);
    if (!blk->data) { free(blk); return ACCEL_V2_ERROR_OUT_OF_MEMORY; }
    blk->size = size;
    *out_mem = (GpuMemory)blk;
    return ACCEL_V2_SUCCESS;
}

AccelV2Status accelv2_free(AccelV2Handle accel, GpuMemory mem) {
    if (!accel || !mem) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuMemoryBlock* blk = (GpuMemoryBlock*)mem;
    free(blk->data);
    free(blk);
    return ACCEL_V2_SUCCESS;
}

AccelV2Status accelv2_upload(AccelV2Handle accel, GpuMemory dst, const void* src, size_t size) {
    if (!accel || !dst || !src) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuMemoryBlock* blk = (GpuMemoryBlock*)dst;
    if (size > blk->size) return ACCEL_V2_ERROR_INVALID_PARAM;
    memcpy(blk->data, src, size);
    return ACCEL_V2_SUCCESS;
}

AccelV2Status accelv2_download(AccelV2Handle accel, void* dst, GpuMemory src, size_t size) {
    if (!accel || !dst || !src) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuMemoryBlock* blk = (GpuMemoryBlock*)src;
    if (size > blk->size) return ACCEL_V2_ERROR_INVALID_PARAM;
    memcpy(dst, blk->data, size);
    return ACCEL_V2_SUCCESS;
}

AccelV2Status accelv2_memcpy(AccelV2Handle accel, GpuMemory dst, GpuMemory src, size_t size) {
    if (!accel || !dst || !src) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuMemoryBlock* d = (GpuMemoryBlock*)dst;
    GpuMemoryBlock* s = (GpuMemoryBlock*)src;
    if (size > d->size || size > s->size) return ACCEL_V2_ERROR_INVALID_PARAM;
    memcpy(d->data, s->data, size);
    return ACCEL_V2_SUCCESS;
}

// ============================================================================
// Matrix Operations
// ============================================================================

AccelV2Status accelv2_sgemv(AccelV2Handle accel, AccelV2Layout layout, AccelV2Transpose trans,
    int m, int n, float alpha, GpuMemory A, int lda, GpuMemory x, int incx,
    float beta, GpuMemory y, int incy)
{
    if (!accel || !A || !x || !y) return ACCEL_V2_ERROR_INVALID_PARAM;
    if (layout != ACCEL_V2_ROW_MAJOR) return ACCEL_V2_ERROR_NOT_SUPPORTED;

    GpuMemoryBlock* bA = (GpuMemoryBlock*)A;
    GpuMemoryBlock* bx = (GpuMemoryBlock*)x;
    GpuMemoryBlock* by = (GpuMemoryBlock*)y;

    if (trans == ACCEL_V2_NO_TRANS) {
        cpu_sgemv(m, n, alpha, (float*)bA->data, lda, (float*)bx->data, incx, beta, (float*)by->data, incy);
    } else {
        // Transposed: y = alpha * A^T * x + beta * y
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int i = 0; i < m; i++) sum += ((float*)bA->data)[i * lda + j] * ((float*)bx->data)[i * incx];
            ((float*)by->data)[j * incy] = alpha * sum + beta * ((float*)by->data)[j * incy];
        }
    }
    return ACCEL_V2_SUCCESS;
}

AccelV2Status accelv2_sger(AccelV2Handle accel, AccelV2Layout layout,
    int m, int n, float alpha, GpuMemory x, int incx, GpuMemory y, int incy, GpuMemory A, int lda)
{
    if (!accel || !x || !y || !A) return ACCEL_V2_ERROR_INVALID_PARAM;
    if (layout != ACCEL_V2_ROW_MAJOR) return ACCEL_V2_ERROR_NOT_SUPPORTED;

    cpu_sger(m, n, alpha, (float*)((GpuMemoryBlock*)x)->data, incx,
             (float*)((GpuMemoryBlock*)y)->data, incy,
             (float*)((GpuMemoryBlock*)A)->data, lda);
    return ACCEL_V2_SUCCESS;
}

AccelV2Status accelv2_sgemm(AccelV2Handle accel, AccelV2Layout layout,
    AccelV2Transpose transA, AccelV2Transpose transB,
    int m, int n, int k, float alpha, GpuMemory A, int lda, GpuMemory B, int ldb,
    float beta, GpuMemory C, int ldc)
{
    if (!accel || !A || !B || !C) return ACCEL_V2_ERROR_INVALID_PARAM;
    if (layout != ACCEL_V2_ROW_MAJOR) return ACCEL_V2_ERROR_NOT_SUPPORTED;
    if (transA != ACCEL_V2_NO_TRANS || transB != ACCEL_V2_NO_TRANS) return ACCEL_V2_ERROR_NOT_SUPPORTED;

    cpu_sgemm(m, n, k, alpha,
        (float*)((GpuMemoryBlock*)A)->data, lda,
        (float*)((GpuMemoryBlock*)B)->data, ldb,
        beta, (float*)((GpuMemoryBlock*)C)->data, ldc);
    return ACCEL_V2_SUCCESS;
}

// ============================================================================
// NeuralNet Operations
// ============================================================================

AccelV2Status accelv2_nn_create(AccelV2Handle accel, GpuNeuralNetHandle* out_handle) {
    if (!accel || !out_handle) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuNeuralNetCtx* nn = (GpuNeuralNetCtx*)malloc(sizeof(GpuNeuralNetCtx));
    if (!nn) return ACCEL_V2_ERROR_OUT_OF_MEMORY;
    nn->weights = NULL;
    nn->weights_size = 0;
    *out_handle = (GpuNeuralNetHandle)nn;
    return ACCEL_V2_SUCCESS;
}

AccelV2Status accelv2_nn_destroy(AccelV2Handle accel, GpuNeuralNetHandle handle) {
    if (!accel || !handle) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuNeuralNetCtx* nn = (GpuNeuralNetCtx*)handle;
    if (nn->weights) free(nn->weights);
    free(nn);
    return ACCEL_V2_SUCCESS;
}

AccelV2Status accelv2_nn_upload(AccelV2Handle accel, GpuNeuralNetHandle handle, const void* cpu_nn, size_t size) {
    if (!accel || !handle || !cpu_nn) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuNeuralNetCtx* nn = (GpuNeuralNetCtx*)handle;
    if (nn->weights) free(nn->weights);
    nn->weights = malloc(size);
    if (!nn->weights) return ACCEL_V2_ERROR_OUT_OF_MEMORY;
    memcpy(nn->weights, cpu_nn, size);
    nn->weights_size = size;
    return ACCEL_V2_SUCCESS;
}

AccelV2Status accelv2_nn_download(AccelV2Handle accel, GpuNeuralNetHandle handle, void* cpu_nn, size_t size) {
    if (!accel || !handle || !cpu_nn) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuNeuralNetCtx* nn = (GpuNeuralNetCtx*)handle;
    if (!nn->weights || size > nn->weights_size) return ACCEL_V2_ERROR_INVALID_PARAM;
    memcpy(cpu_nn, nn->weights, size);
    return ACCEL_V2_SUCCESS;
}

AccelV2Status accelv2_nn_copy(AccelV2Handle accel, GpuNeuralNetHandle dst, GpuNeuralNetHandle src) {
    if (!accel || !dst || !src) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuNeuralNetCtx* d = (GpuNeuralNetCtx*)dst;
    GpuNeuralNetCtx* s = (GpuNeuralNetCtx*)src;
    if (!s->weights || s->weights_size == 0) return ACCEL_V2_ERROR_INVALID_PARAM;
    if (d->weights) free(d->weights);
    d->weights = malloc(s->weights_size);
    if (!d->weights) return ACCEL_V2_ERROR_OUT_OF_MEMORY;
    memcpy(d->weights, s->weights, s->weights_size);
    d->weights_size = s->weights_size;
    return ACCEL_V2_SUCCESS;
}

AccelV2Status accelv2_nn_soft_update(AccelV2Handle accel, GpuNeuralNetHandle target, GpuNeuralNetHandle source, float tau) {
    if (!accel || !target || !source) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuNeuralNetCtx* t = (GpuNeuralNetCtx*)target;
    GpuNeuralNetCtx* s = (GpuNeuralNetCtx*)source;
    if (!t->weights || !s->weights || t->weights_size != s->weights_size) return ACCEL_V2_ERROR_INVALID_PARAM;

    float* tw = (float*)t->weights;
    const float* sw = (const float*)s->weights;
    int n = (int)(t->weights_size / sizeof(float));
    for (int i = 0; i < n; i++) {
        tw[i] = tau * sw[i] + (1.0f - tau) * tw[i];
    }
    return ACCEL_V2_SUCCESS;
}

AccelV2Status accelv2_nn_forward(AccelV2Handle accel, GpuNeuralNetHandle net, GpuMemory input, int batch_size, GpuMemory output) {
    // CPU fallback: use existing forward_cpu from neural_net.cpp
    return ACCEL_V2_ERROR_NOT_SUPPORTED;
}

AccelV2Status accelv2_nn_train_step(AccelV2Handle accel, GpuNeuralNetHandle net, GpuNeuralNetHandle target_net, GpuMemory states, GpuMemory actions, GpuMemory rewards, GpuMemory next_states, GpuMemory dones, int batch_size, float gamma, float learning_rate, GpuMemory out_loss) {
    // CPU fallback: use existing train_step from neural_net.cpp
    return ACCEL_V2_ERROR_NOT_SUPPORTED;
}

// ============================================================================
// Replay Buffer
// ============================================================================

AccelV2Status accelv2_rb_create(AccelV2Handle accel, GpuReplayBufferHandle* out_handle, int capacity, int state_dim) {
    if (!accel || !out_handle || capacity <= 0 || state_dim <= 0) return ACCEL_V2_ERROR_INVALID_PARAM;

    GpuReplayBufferCtx* rb = (GpuReplayBufferCtx*)malloc(sizeof(GpuReplayBufferCtx));
    if (!rb) return ACCEL_V2_ERROR_OUT_OF_MEMORY;

    rb->capacity = capacity;
    rb->state_dim = state_dim;
    rb->count = 0;
    rb->head = 0;
    rb->states = (float*)malloc(capacity * state_dim * sizeof(float));
    rb->actions = (int*)malloc(capacity * sizeof(int));
    rb->rewards = (float*)malloc(capacity * sizeof(float));
    rb->next_states = (float*)malloc(capacity * state_dim * sizeof(float));
    rb->dones = (int*)malloc(capacity * sizeof(int));

    if (!rb->states || !rb->actions || !rb->rewards || !rb->next_states || !rb->dones) {
        free(rb->states); free(rb->actions); free(rb->rewards);
        free(rb->next_states); free(rb->dones); free(rb);
        return ACCEL_V2_ERROR_OUT_OF_MEMORY;
    }

    *out_handle = (GpuReplayBufferHandle)rb;
    return ACCEL_V2_SUCCESS;
}

AccelV2Status accelv2_rb_destroy(AccelV2Handle accel, GpuReplayBufferHandle handle) {
    if (!accel || !handle) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuReplayBufferCtx* rb = (GpuReplayBufferCtx*)handle;
    free(rb->states); free(rb->actions); free(rb->rewards);
    free(rb->next_states); free(rb->dones); free(rb);
    return ACCEL_V2_SUCCESS;
}

AccelV2Status accelv2_rb_store(AccelV2Handle accel, GpuReplayBufferHandle handle,
    const float* state, int action, float reward, const float* next_state, int done)
{
    if (!accel || !handle || !state || !next_state) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuReplayBufferCtx* rb = (GpuReplayBufferCtx*)handle;
    int idx = rb->head;

    memcpy(rb->states + idx * rb->state_dim, state, rb->state_dim * sizeof(float));
    rb->actions[idx] = action;
    rb->rewards[idx] = reward;
    memcpy(rb->next_states + idx * rb->state_dim, next_state, rb->state_dim * sizeof(float));
    rb->dones[idx] = done;

    rb->head = (rb->head + 1) % rb->capacity;
    if (rb->count < rb->capacity) rb->count++;
    return ACCEL_V2_SUCCESS;
}

AccelV2Status accelv2_rb_sample(AccelV2Handle accel, GpuReplayBufferHandle handle,
    int batch_size, GpuMemory out_states, GpuMemory out_actions,
    GpuMemory out_rewards, GpuMemory out_next_states, GpuMemory out_dones)
{
    if (!accel || !handle) return ACCEL_V2_ERROR_INVALID_PARAM;
    if (!out_states || !out_actions || !out_rewards || !out_next_states || !out_dones)
        return ACCEL_V2_ERROR_INVALID_PARAM;

    GpuReplayBufferCtx* rb = (GpuReplayBufferCtx*)handle;
    if (rb->count < batch_size) return ACCEL_V2_ERROR_INVALID_PARAM;

    GpuMemoryBlock* bs = (GpuMemoryBlock*)out_states;
    GpuMemoryBlock* ba = (GpuMemoryBlock*)out_actions;
    GpuMemoryBlock* br = (GpuMemoryBlock*)out_rewards;
    GpuMemoryBlock* bns = (GpuMemoryBlock*)out_next_states;
    GpuMemoryBlock* bd = (GpuMemoryBlock*)out_dones;

    int start = (rb->head - batch_size + rb->capacity) % rb->capacity;
    for (int i = 0; i < batch_size; i++) {
        int idx = (start + i) % rb->capacity;
        memcpy((float*)bs->data + i * rb->state_dim, rb->states + idx * rb->state_dim, rb->state_dim * sizeof(float));
        ((int*)ba->data)[i] = rb->actions[idx];
        ((float*)br->data)[i] = rb->rewards[idx];
        memcpy((float*)bns->data + i * rb->state_dim, rb->next_states + idx * rb->state_dim, rb->state_dim * sizeof(float));
        ((int*)bd->data)[i] = rb->dones[idx];
    }

    return ACCEL_V2_SUCCESS;
}
