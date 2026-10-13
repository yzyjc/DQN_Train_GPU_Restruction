// rocm_accelerator_v2.cpp
// ROCm v2 Implementation using hipBLAS

#include "rocm_accelerator_v2.h"
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <windows.h>

// GPU log file
static FILE* g_gpu_log = NULL;

static void gpu_log(const char* fmt, ...) {
    if (!g_gpu_log) {
        // Try multiple paths for the log file
        g_gpu_log = fopen("gpu_accel.log", "w");
        if (!g_gpu_log) {
            // Try temp directory
            char temp_path[MAX_PATH];
            GetTempPathA(MAX_PATH, temp_path);
            strcat_s(temp_path, "gpu_accel.log");
            g_gpu_log = fopen(temp_path, "w");
        }
        if (!g_gpu_log) return;
        fprintf(g_gpu_log, "=== GPU Accelerator Log ===\n");
        fprintf(g_gpu_log, "Working directory: ");
        char cwd[MAX_PATH];
        if (GetCurrentDirectoryA(MAX_PATH, cwd)) {
            fprintf(g_gpu_log, "%s\n", cwd);
        }
        fflush(g_gpu_log);
    }
    va_list args;
    va_start(args, fmt);
    vfprintf(g_gpu_log, fmt, args);
    va_end(args);
    fflush(g_gpu_log);
}

// Internal GPU memory block
struct GpuMemBlock {
    void* ptr;
    size_t size;
};

struct GpuNeuralNetCtx {
    void* d_weights;  // device pointer
    size_t weights_size;
};

struct GpuReplayBufferCtx {
    int capacity;
    int state_dim;
    int count;
    int head;
    void* d_states;
    void* d_actions;
    void* d_rewards;
    void* d_next_states;
    void* d_dones;
};

static AccelV2Context* get_ctx(AccelV2Handle h) { return (AccelV2Context*)h; }

// ============================================================================
// Device Management
// ============================================================================

AccelV2Status rocm_v2_init(AccelV2Handle* out_accel) {
    if (!out_accel) return ACCEL_V2_ERROR_INVALID_PARAM;

    // HIP must be initialized from the main process before loading this DLL
    // Try hipSetDevice first - if hipInit was called from main process, this will work
    hipError_t err = hipSetDevice(0);
    gpu_log("[INIT] hipSetDevice(0) returned: %d (%s)\n", err, hipGetErrorString(err));

    if (err != hipSuccess) {
        // hipInit wasn't called from main process - try here (may fail in DLL context)
        gpu_log("[INIT] hipSetDevice failed, trying hipInit from DLL...\n");
        hipError_t init_err = hipInit(0);
        gpu_log("[INIT] hipInit(0) returned: %d (%s)\n", init_err, hipGetErrorString(init_err));
        if (init_err == hipSuccess) {
            err = hipSetDevice(0);
            gpu_log("[INIT] hipSetDevice(0) after init: %d\n", err);
        }
        if (err != hipSuccess) {
            printf("[ROCm-V2] HIP initialization failed in DLL context\n");
            printf("[ROCm-V2] hipInit must be called from main process before loading DLL\n");
            return ACCEL_V2_ERROR_DEVICE_NOT_FOUND;
        }
    }

    AccelV2Context* ctx = (AccelV2Context*)malloc(sizeof(AccelV2Context));
    if (!ctx) return ACCEL_V2_ERROR_OUT_OF_MEMORY;
    memset(ctx, 0, sizeof(AccelV2Context));

    // Skip stream creation for now (hangs on some systems)
    ctx->stream = NULL;

    // Skip hipBLAS for now (crashes on some systems)
    ctx->blas_handle = NULL;
    gpu_log("[INIT] hipBLAS skipped (debug mode)\n");

    ctx->initialized = 1;
    ctx->device_id = 0;

    char name[256];
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    strncpy(name, prop.name, 255);

    gpu_log("[INIT] Device: %s\n", name);
    gpu_log("[INIT] Memory: %llu MB\n", prop.totalGlobalMem / (1024 * 1024));
    gpu_log("[INIT] Compute Units: %d\n", prop.multiProcessorCount);
    gpu_log("[INIT] hipBLAS handle created\n");

    printf("[ROCm-V2] Initialized: %s\n", name);
    *out_accel = (AccelV2Handle)ctx;
    return ACCEL_V2_SUCCESS;
}

AccelV2Status rocm_v2_cleanup(AccelV2Handle accel) {
    if (!accel) return ACCEL_V2_ERROR_INVALID_PARAM;
    AccelV2Context* ctx = get_ctx(accel);
    if (ctx->blas_handle) hipblasDestroy((hipblasHandle_t)ctx->blas_handle);
    free(ctx);
    return ACCEL_V2_SUCCESS;
}

AccelV2Status rocm_v2_get_device_name(AccelV2Handle accel, char* name, int name_size) {
    if (!accel || !name) return ACCEL_V2_ERROR_INVALID_PARAM;
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    strncpy(name, prop.name, name_size - 1);
    name[name_size - 1] = '\0';
    return ACCEL_V2_SUCCESS;
}

AccelV2Status rocm_v2_get_device_memory(AccelV2Handle accel, uint64_t* total_mem, uint64_t* free_mem) {
    if (!accel || !total_mem || !free_mem) return ACCEL_V2_ERROR_INVALID_PARAM;
    hipMemGetInfo(free_mem, total_mem);
    return ACCEL_V2_SUCCESS;
}

AccelV2Status rocm_v2_sync(AccelV2Handle accel) {
    if (!accel) return ACCEL_V2_ERROR_INVALID_PARAM;
    AccelV2Context* ctx = get_ctx(accel);
    // Skip sync if no stream (using default stream)
    if (ctx->stream) {
        hipError_t err = hipStreamSynchronize((hipStream_t)ctx->stream);
        return (err == hipSuccess) ? ACCEL_V2_SUCCESS : ACCEL_V2_ERROR_SYNC_FAILED;
    }
    hipDeviceSynchronize();
    return ACCEL_V2_SUCCESS;
}

// ============================================================================
// Memory Management
// ============================================================================

AccelV2Status rocm_v2_malloc(AccelV2Handle accel, size_t size, GpuMemory* out_mem) {
    if (!accel || !out_mem || size == 0) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuMemBlock* blk = (GpuMemBlock*)malloc(sizeof(GpuMemBlock));
    if (!blk) return ACCEL_V2_ERROR_OUT_OF_MEMORY;
    hipError_t err = hipMalloc(&blk->ptr, size);
    if (err != hipSuccess) { free(blk); return ACCEL_V2_ERROR_OUT_OF_MEMORY; }
    blk->size = size;
    *out_mem = (GpuMemory)blk;
    return ACCEL_V2_SUCCESS;
}

AccelV2Status rocm_v2_free(AccelV2Handle accel, GpuMemory mem) {
    if (!accel || !mem) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuMemBlock* blk = (GpuMemBlock*)mem;
    hipFree(blk->ptr);
    free(blk);
    return ACCEL_V2_SUCCESS;
}

AccelV2Status rocm_v2_upload(AccelV2Handle accel, GpuMemory dst, const void* src, size_t size) {
    if (!accel || !dst || !src) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuMemBlock* blk = (GpuMemBlock*)dst;
    hipError_t err = hipMemcpy(blk->ptr, src, size, hipMemcpyHostToDevice);
    return (err == hipSuccess) ? ACCEL_V2_SUCCESS : ACCEL_V2_ERROR_SYNC_FAILED;
}

AccelV2Status rocm_v2_download(AccelV2Handle accel, void* dst, GpuMemory src, size_t size) {
    if (!accel || !dst || !src) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuMemBlock* blk = (GpuMemBlock*)src;
    hipError_t err = hipMemcpy(dst, blk->ptr, size, hipMemcpyDeviceToHost);
    return (err == hipSuccess) ? ACCEL_V2_SUCCESS : ACCEL_V2_ERROR_SYNC_FAILED;
}

AccelV2Status rocm_v2_memcpy(AccelV2Handle accel, GpuMemory dst, GpuMemory src, size_t size) {
    if (!accel || !dst || !src) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuMemBlock* d = (GpuMemBlock*)dst;
    GpuMemBlock* s = (GpuMemBlock*)src;
    hipError_t err = hipMemcpy(d->ptr, s->ptr, size, hipMemcpyDeviceToDevice);
    return (err == hipSuccess) ? ACCEL_V2_SUCCESS : ACCEL_V2_ERROR_SYNC_FAILED;
}

// ============================================================================
// Matrix Operations (hipBLAS)
// ============================================================================

AccelV2Status rocm_v2_sgemv(AccelV2Handle accel, AccelV2Layout layout, AccelV2Transpose trans,
    int m, int n, float alpha, GpuMemory A, int lda, GpuMemory x, int incx,
    float beta, GpuMemory y, int incy)
{
    if (!accel || !A || !x || !y) return ACCEL_V2_ERROR_INVALID_PARAM;
    AccelV2Context* ctx = get_ctx(accel);
    GpuMemBlock* bA = (GpuMemBlock*)A;
    GpuMemBlock* bx = (GpuMemBlock*)x;
    GpuMemBlock* by = (GpuMemBlock*)y;

    gpu_log("[SGEMV] m=%d n=%d alpha=%.4f beta=%.4f trans=%d\n", m, n, alpha, beta, trans);

    hipblasOperation_t op = (trans == ACCEL_V2_NO_TRANS) ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasStatus_t s = hipblasSgemv((hipblasHandle_t)ctx->blas_handle, op,
        n, m, &alpha, (float*)bA->ptr, lda, (float*)bx->ptr, incx, &beta, (float*)by->ptr, incy);

    if (s != HIPBLAS_STATUS_SUCCESS) {
        gpu_log("[SGEMV] hipblasSgemv failed: %d\n", s);
        return ACCEL_V2_ERROR_KERNEL_LAUNCH_FAILED;
    }
    return ACCEL_V2_SUCCESS;
}

AccelV2Status rocm_v2_sger(AccelV2Handle accel, AccelV2Layout layout,
    int m, int n, float alpha, GpuMemory x, int incx, GpuMemory y, int incy, GpuMemory A, int lda)
{
    if (!accel || !x || !y || !A) return ACCEL_V2_ERROR_INVALID_PARAM;
    AccelV2Context* ctx = get_ctx(accel);

    hipblasStatus_t s = hipblasSger((hipblasHandle_t)ctx->blas_handle,
        n, m, &alpha, (float*)((GpuMemBlock*)y)->ptr, incy,
        (float*)((GpuMemBlock*)x)->ptr, incx, (float*)((GpuMemBlock*)A)->ptr, lda);
    return (s == HIPBLAS_STATUS_SUCCESS) ? ACCEL_V2_SUCCESS : ACCEL_V2_ERROR_KERNEL_LAUNCH_FAILED;
}

AccelV2Status rocm_v2_sgemm(AccelV2Handle accel, AccelV2Layout layout,
    AccelV2Transpose transA, AccelV2Transpose transB,
    int m, int n, int k, float alpha, GpuMemory A, int lda, GpuMemory B, int ldb,
    float beta, GpuMemory C, int ldc)
{
    if (!accel || !A || !B || !C) return ACCEL_V2_ERROR_INVALID_PARAM;
    AccelV2Context* ctx = get_ctx(accel);

    hipblasOperation_t opA = (transA == ACCEL_V2_NO_TRANS) ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t opB = (transB == ACCEL_V2_NO_TRANS) ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasStatus_t s = hipblasSgemm((hipblasHandle_t)ctx->blas_handle,
        opB, opA, n, m, k, &alpha,
        (float*)((GpuMemBlock*)B)->ptr, ldb,
        (float*)((GpuMemBlock*)A)->ptr, lda,
        &beta, (float*)((GpuMemBlock*)C)->ptr, ldc);
    return (s == HIPBLAS_STATUS_SUCCESS) ? ACCEL_V2_SUCCESS : ACCEL_V2_ERROR_KERNEL_LAUNCH_FAILED;
}

// ============================================================================
// NeuralNet Operations (stubs - need GPU kernels for forward/backward)
// ============================================================================

AccelV2Status rocm_v2_nn_create(AccelV2Handle accel, GpuNeuralNetHandle* out_handle) {
    if (!accel || !out_handle) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuNeuralNetCtx* nn = (GpuNeuralNetCtx*)malloc(sizeof(GpuNeuralNetCtx));
    if (!nn) return ACCEL_V2_ERROR_OUT_OF_MEMORY;
    memset(nn, 0, sizeof(GpuNeuralNetCtx));
    *out_handle = (GpuNeuralNetHandle)nn;
    return ACCEL_V2_SUCCESS;
}

AccelV2Status rocm_v2_nn_destroy(AccelV2Handle accel, GpuNeuralNetHandle handle) {
    if (!accel || !handle) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuNeuralNetCtx* nn = (GpuNeuralNetCtx*)handle;
    if (nn->d_weights) hipFree(nn->d_weights);
    free(nn);
    return ACCEL_V2_SUCCESS;
}

AccelV2Status rocm_v2_nn_upload(AccelV2Handle accel, GpuNeuralNetHandle handle, const void* cpu_nn, size_t size) {
    if (!accel || !handle || !cpu_nn) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuNeuralNetCtx* nn = (GpuNeuralNetCtx*)handle;
    if (nn->d_weights) hipFree(nn->d_weights);
    hipError_t err = hipMalloc(&nn->d_weights, size);
    if (err != hipSuccess) {
        gpu_log("[NN_UPLOAD] hipMalloc failed: %d\n", err);
        return ACCEL_V2_ERROR_OUT_OF_MEMORY;
    }
    err = hipMemcpy(nn->d_weights, cpu_nn, size, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        gpu_log("[NN_UPLOAD] hipMemcpy failed: %d\n", err);
        return ACCEL_V2_ERROR_SYNC_FAILED;
    }
    nn->weights_size = size;
    gpu_log("[NN_UPLOAD] Uploaded %zu bytes to GPU\n", size);
    return ACCEL_V2_SUCCESS;
}

AccelV2Status rocm_v2_nn_download(AccelV2Handle accel, GpuNeuralNetHandle handle, void* cpu_nn, size_t size) {
    if (!accel || !handle || !cpu_nn) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuNeuralNetCtx* nn = (GpuNeuralNetCtx*)handle;
    if (!nn->d_weights || size > nn->weights_size) return ACCEL_V2_ERROR_INVALID_PARAM;
    hipError_t err = hipMemcpy(cpu_nn, nn->d_weights, size, hipMemcpyDeviceToHost);
    return (err == hipSuccess) ? ACCEL_V2_SUCCESS : ACCEL_V2_ERROR_SYNC_FAILED;
}

AccelV2Status rocm_v2_nn_copy(AccelV2Handle accel, GpuNeuralNetHandle dst, GpuNeuralNetHandle src) {
    if (!accel || !dst || !src) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuNeuralNetCtx* d = (GpuNeuralNetCtx*)dst;
    GpuNeuralNetCtx* s = (GpuNeuralNetCtx*)src;
    if (!s->d_weights || s->weights_size == 0) return ACCEL_V2_ERROR_INVALID_PARAM;
    if (d->d_weights) hipFree(d->d_weights);
    hipError_t err = hipMalloc(&d->d_weights, s->weights_size);
    if (err != hipSuccess) return ACCEL_V2_ERROR_OUT_OF_MEMORY;
    err = hipMemcpy(d->d_weights, s->d_weights, s->weights_size, hipMemcpyDeviceToDevice);
    if (err != hipSuccess) return ACCEL_V2_ERROR_SYNC_FAILED;
    d->weights_size = s->weights_size;
    return ACCEL_V2_SUCCESS;
}

AccelV2Status rocm_v2_nn_soft_update(AccelV2Handle accel, GpuNeuralNetHandle target, GpuNeuralNetHandle source, float tau) {
    if (!accel || !target || !source) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuNeuralNetCtx* t = (GpuNeuralNetCtx*)target;
    GpuNeuralNetCtx* s = (GpuNeuralNetCtx*)source;
    if (!t->d_weights || !s->d_weights || t->weights_size != s->weights_size) return ACCEL_V2_ERROR_INVALID_PARAM;

    gpu_log("[SOFT_UPDATE] tau=%.4f, size=%zu bytes\n", tau, t->weights_size);

    // CPU fallback for soft update (TODO: GPU kernel)
    size_t n = t->weights_size / sizeof(float);
    float* tmp = (float*)malloc(t->weights_size);
    if (!tmp) return ACCEL_V2_ERROR_OUT_OF_MEMORY;

    hipMemcpy(tmp, t->d_weights, t->weights_size, hipMemcpyDeviceToHost);
    float* src_host = (float*)malloc(s->weights_size);
    if (!src_host) { free(tmp); return ACCEL_V2_ERROR_OUT_OF_MEMORY; }
    hipMemcpy(src_host, s->d_weights, s->weights_size, hipMemcpyDeviceToHost);

    for (size_t i = 0; i < n; i++) {
        tmp[i] = tau * src_host[i] + (1.0f - tau) * tmp[i];
    }

    hipMemcpy(t->d_weights, tmp, t->weights_size, hipMemcpyHostToDevice);
    free(tmp);
    free(src_host);
    return ACCEL_V2_SUCCESS;
}

AccelV2Status rocm_v2_nn_forward(AccelV2Handle accel, GpuNeuralNetHandle net, GpuMemory input, int batch_size, GpuMemory output) {
    // TODO: Implement with GPU kernels
    return ACCEL_V2_ERROR_NOT_SUPPORTED;
}

AccelV2Status rocm_v2_nn_train_step(AccelV2Handle accel, GpuNeuralNetHandle net, GpuNeuralNetHandle target_net, GpuMemory states, GpuMemory actions, GpuMemory rewards, GpuMemory next_states, GpuMemory dones, int batch_size, float gamma, float learning_rate, GpuMemory out_loss) {
    // TODO: Implement with GPU kernels
    return ACCEL_V2_ERROR_NOT_SUPPORTED;
}

// ============================================================================
// Replay Buffer (GPU-resident)
// ============================================================================

AccelV2Status rocm_v2_rb_create(AccelV2Handle accel, GpuReplayBufferHandle* out_handle, int capacity, int state_dim) {
    if (!accel || !out_handle || capacity <= 0 || state_dim <= 0) return ACCEL_V2_ERROR_INVALID_PARAM;

    GpuReplayBufferCtx* rb = (GpuReplayBufferCtx*)malloc(sizeof(GpuReplayBufferCtx));
    if (!rb) return ACCEL_V2_ERROR_OUT_OF_MEMORY;
    memset(rb, 0, sizeof(GpuReplayBufferCtx));

    rb->capacity = capacity;
    rb->state_dim = state_dim;

    hipMalloc(&rb->d_states, capacity * state_dim * sizeof(float));
    hipMalloc(&rb->d_actions, capacity * sizeof(int));
    hipMalloc(&rb->d_rewards, capacity * sizeof(float));
    hipMalloc(&rb->d_next_states, capacity * state_dim * sizeof(float));
    hipMalloc(&rb->d_dones, capacity * sizeof(int));

    if (!rb->d_states || !rb->d_actions || !rb->d_rewards || !rb->d_next_states || !rb->d_dones) {
        hipFree(rb->d_states); hipFree(rb->d_actions); hipFree(rb->d_rewards);
        hipFree(rb->d_next_states); hipFree(rb->d_dones); free(rb);
        return ACCEL_V2_ERROR_OUT_OF_MEMORY;
    }

    *out_handle = (GpuReplayBufferHandle)rb;
    printf("[ROCm-V2] Replay buffer created: %d x %d on GPU\n", capacity, state_dim);
    return ACCEL_V2_SUCCESS;
}

AccelV2Status rocm_v2_rb_destroy(AccelV2Handle accel, GpuReplayBufferHandle handle) {
    if (!accel || !handle) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuReplayBufferCtx* rb = (GpuReplayBufferCtx*)handle;
    hipFree(rb->d_states); hipFree(rb->d_actions); hipFree(rb->d_rewards);
    hipFree(rb->d_next_states); hipFree(rb->d_dones); free(rb);
    return ACCEL_V2_SUCCESS;
}

AccelV2Status rocm_v2_rb_store(AccelV2Handle accel, GpuReplayBufferHandle handle,
    const float* state, int action, float reward, const float* next_state, int done)
{
    if (!accel || !handle || !state || !next_state) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuReplayBufferCtx* rb = (GpuReplayBufferCtx*)handle;
    int idx = rb->head;

    // Upload single experience to GPU
    size_t state_bytes = rb->state_dim * sizeof(float);
    hipMemcpy((char*)rb->d_states + idx * state_bytes, state, state_bytes, hipMemcpyHostToDevice);
    hipMemcpy((int*)rb->d_actions + idx, &action, sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy((float*)rb->d_rewards + idx, &reward, sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy((char*)rb->d_next_states + idx * state_bytes, next_state, state_bytes, hipMemcpyHostToDevice);
    hipMemcpy((int*)rb->d_dones + idx, &done, sizeof(int), hipMemcpyHostToDevice);

    rb->head = (rb->head + 1) % rb->capacity;
    if (rb->count < rb->capacity) rb->count++;
    return ACCEL_V2_SUCCESS;
}

AccelV2Status rocm_v2_rb_sample(AccelV2Handle accel, GpuReplayBufferHandle handle,
    int batch_size, GpuMemory out_states, GpuMemory out_actions,
    GpuMemory out_rewards, GpuMemory out_next_states, GpuMemory out_dones)
{
    if (!accel || !handle) return ACCEL_V2_ERROR_INVALID_PARAM;
    GpuReplayBufferCtx* rb = (GpuReplayBufferCtx*)handle;
    if (rb->count < batch_size) return ACCEL_V2_ERROR_INVALID_PARAM;

    // Sequential sample (TODO: random sampling on GPU)
    size_t state_bytes = rb->state_dim * sizeof(float);
    int start = (rb->head - batch_size + rb->capacity) % rb->capacity;

    GpuMemBlock* bs = (GpuMemBlock*)out_states;
    GpuMemBlock* ba = (GpuMemBlock*)out_actions;
    GpuMemBlock* br = (GpuMemBlock*)out_rewards;
    GpuMemBlock* bns = (GpuMemBlock*)out_next_states;
    GpuMemBlock* bd = (GpuMemBlock*)out_dones;

    for (int i = 0; i < batch_size; i++) {
        int idx = (start + i) % rb->capacity;
        hipMemcpy((char*)bs->ptr + i * state_bytes, (char*)rb->d_states + idx * state_bytes, state_bytes, hipMemcpyDeviceToDevice);
        hipMemcpy((int*)ba->ptr + i, (int*)rb->d_actions + idx, sizeof(int), hipMemcpyDeviceToDevice);
        hipMemcpy((float*)br->ptr + i, (float*)rb->d_rewards + idx, sizeof(float), hipMemcpyDeviceToDevice);
        hipMemcpy((char*)bns->ptr + i * state_bytes, (char*)rb->d_next_states + idx * state_bytes, state_bytes, hipMemcpyDeviceToDevice);
        hipMemcpy((int*)bd->ptr + i, (int*)rb->d_dones + idx, sizeof(int), hipMemcpyDeviceToDevice);
    }

    return ACCEL_V2_SUCCESS;
}
