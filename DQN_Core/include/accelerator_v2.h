#pragma once

// DQN Accelerator Interface v2 - GPU-Resident Training
// Standalone header - does NOT include accelerator.h
// Uses GpuMemory handles instead of raw void* pointers

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Types (independent from v1)
// ============================================================================

typedef enum {
    ACCEL_V2_SUCCESS = 0,
    ACCEL_V2_ERROR_INVALID_PARAM,
    ACCEL_V2_ERROR_OUT_OF_MEMORY,
    ACCEL_V2_ERROR_DEVICE_NOT_FOUND,
    ACCEL_V2_ERROR_INITIALIZATION_FAILED,
    ACCEL_V2_ERROR_KERNEL_LAUNCH_FAILED,
    ACCEL_V2_ERROR_SYNC_FAILED,
    ACCEL_V2_ERROR_NOT_SUPPORTED
} AccelV2Status;

typedef enum {
    ACCEL_V2_ROW_MAJOR = 0,
    ACCEL_V2_COL_MAJOR
} AccelV2Layout;

typedef enum {
    ACCEL_V2_NO_TRANS = 0,
    ACCEL_V2_TRANS = 1
} AccelV2Transpose;

// Opaque handles
typedef void* AccelV2Handle;       // Accelerator device context
typedef void* GpuMemory;           // GPU memory buffer
typedef void* GpuNeuralNetHandle;  // NeuralNet on GPU
typedef void* GpuReplayBufferHandle;

// ============================================================================
// Device Management
// ============================================================================

AccelV2Status accelv2_init(AccelV2Handle* out_accel);
AccelV2Status accelv2_cleanup(AccelV2Handle accel);
AccelV2Status accelv2_get_device_name(AccelV2Handle accel, char* name, int name_size);
AccelV2Status accelv2_get_device_memory(AccelV2Handle accel, uint64_t* total_mem, uint64_t* free_mem);
AccelV2Status accelv2_sync(AccelV2Handle accel);

// ============================================================================
// GPU Memory Management
// ============================================================================

AccelV2Status accelv2_malloc(AccelV2Handle accel, size_t size, GpuMemory* out_mem);
AccelV2Status accelv2_free(AccelV2Handle accel, GpuMemory mem);
AccelV2Status accelv2_upload(AccelV2Handle accel, GpuMemory dst, const void* src, size_t size);
AccelV2Status accelv2_download(AccelV2Handle accel, void* dst, GpuMemory src, size_t size);
AccelV2Status accelv2_memcpy(AccelV2Handle accel, GpuMemory dst, GpuMemory src, size_t size);

// ============================================================================
// Matrix Operations
// ============================================================================

AccelV2Status accelv2_sgemv(AccelV2Handle accel,
    AccelV2Layout layout, AccelV2Transpose trans,
    int m, int n, float alpha,
    GpuMemory A, int lda,
    GpuMemory x, int incx,
    float beta, GpuMemory y, int incy);

AccelV2Status accelv2_sger(AccelV2Handle accel,
    AccelV2Layout layout,
    int m, int n, float alpha,
    GpuMemory x, int incx,
    GpuMemory y, int incy,
    GpuMemory A, int lda);

AccelV2Status accelv2_sgemm(AccelV2Handle accel,
    AccelV2Layout layout, AccelV2Transpose transA, AccelV2Transpose transB,
    int m, int n, int k, float alpha,
    GpuMemory A, int lda,
    GpuMemory B, int ldb,
    float beta, GpuMemory C, int ldc);

// ============================================================================
// NeuralNet Operations
// ============================================================================

AccelV2Status accelv2_nn_create(AccelV2Handle accel, GpuNeuralNetHandle* out_handle);
AccelV2Status accelv2_nn_destroy(AccelV2Handle accel, GpuNeuralNetHandle handle);
AccelV2Status accelv2_nn_upload(AccelV2Handle accel, GpuNeuralNetHandle handle, const void* cpu_nn, size_t size);
AccelV2Status accelv2_nn_download(AccelV2Handle accel, GpuNeuralNetHandle handle, void* cpu_nn, size_t size);
AccelV2Status accelv2_nn_copy(AccelV2Handle accel, GpuNeuralNetHandle dst, GpuNeuralNetHandle src);
AccelV2Status accelv2_nn_soft_update(AccelV2Handle accel, GpuNeuralNetHandle target, GpuNeuralNetHandle source, float tau);
AccelV2Status accelv2_nn_forward(AccelV2Handle accel, GpuNeuralNetHandle net, GpuMemory input, int batch_size, GpuMemory output);
AccelV2Status accelv2_nn_train_step(AccelV2Handle accel, GpuNeuralNetHandle net, GpuNeuralNetHandle target_net, GpuMemory states, GpuMemory actions, GpuMemory rewards, GpuMemory next_states, GpuMemory dones, int batch_size, float gamma, float learning_rate, GpuMemory out_loss);

// ============================================================================
// Replay Buffer
// ============================================================================

AccelV2Status accelv2_rb_create(AccelV2Handle accel, GpuReplayBufferHandle* out_handle, int capacity, int state_dim);
AccelV2Status accelv2_rb_destroy(AccelV2Handle accel, GpuReplayBufferHandle handle);
AccelV2Status accelv2_rb_store(AccelV2Handle accel, GpuReplayBufferHandle handle, const float* state, int action, float reward, const float* next_state, int done);
AccelV2Status accelv2_rb_sample(AccelV2Handle accel, GpuReplayBufferHandle handle, int batch_size, GpuMemory out_states, GpuMemory out_actions, GpuMemory out_rewards, GpuMemory out_next_states, GpuMemory out_dones);

#ifdef __cplusplus
}
#endif
