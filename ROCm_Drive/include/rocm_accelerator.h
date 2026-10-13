#pragma once

// ROCm Accelerator Implementation
// Uses HIP for AMD GPU acceleration

#include <stdint.h>
#include <stddef.h>

// Accelerator types (same as DQN_Core/accelerator.h)
typedef enum {
    ACCEL_TYPE_AUTO = 0,
    ACCEL_TYPE_CPU,
    ACCEL_TYPE_VULKAN,
    ACCEL_TYPE_ROCM,
    ACCEL_TYPE_CUDA
} AccelType;

typedef enum {
    ACCEL_SUCCESS = 0,
    ACCEL_ERROR_INVALID_PARAM,
    ACCEL_ERROR_OUT_OF_MEMORY,
    ACCEL_ERROR_DEVICE_NOT_FOUND,
    ACCEL_ERROR_INITIALIZATION_FAILED,
    ACCEL_ERROR_KERNEL_LAUNCH_FAILED,
    ACCEL_ERROR_SYNC_FAILED,
    ACCEL_ERROR_NOT_SUPPORTED,
    ACCEL_ERROR_DLL_LOAD_FAILED,
    ACCEL_ERROR_FUNCTION_NOT_FOUND
} AccelStatus;

typedef enum {
    ACCEL_ROW_MAJOR = 0,
    ACCEL_COL_MAJOR
} AccelLayout;

typedef enum {
    ACCEL_NO_TRANS = 0,
    ACCEL_TRANS = 1
} AccelTranspose;

typedef struct {
    char name[256];
    char vendor[64];
    AccelType type;
    uint64_t global_mem;
    uint32_t compute_units;
    uint32_t max_workgroup;
    uint32_t api_version;
    int is_discrete;
    int is_available;
} AccelDeviceInfo;

typedef void* AcceleratorHandle;

#ifdef __cplusplus
extern "C" {
#endif

// ROCm-specific initialization
AccelStatus rocm_init(AccelType type, AcceleratorHandle* out_accel);
AccelStatus rocm_cleanup(AcceleratorHandle accel);

// Device detection
AccelStatus rocm_detect_devices(AccelDeviceInfo* devices, int max_count, int* out_count);
AccelStatus rocm_get_device_info(AcceleratorHandle accel, AccelDeviceInfo* info);

// Memory operations
AccelStatus rocm_malloc(AcceleratorHandle accel, size_t size, void** out_ptr);
AccelStatus rocm_free(AcceleratorHandle accel, void* ptr);
AccelStatus rocm_upload(AcceleratorHandle accel, void* dst, const void* src, size_t size);
AccelStatus rocm_download(AcceleratorHandle accel, void* dst, const void* src, size_t size);

// Matrix operations
AccelStatus rocm_sgemv(AcceleratorHandle accel, AccelLayout layout, AccelTranspose trans,
    int m, int n, float alpha, const float* A, int lda, const float* x, int incx,
    float beta, float* y, int incy);

AccelStatus rocm_sger(AcceleratorHandle accel, AccelLayout layout,
    int m, int n, float alpha, const float* x, int incx, const float* y, int incy,
    float* A, int lda);

AccelStatus rocm_sgemm(AcceleratorHandle accel, AccelLayout layout,
    AccelTranspose transA, AccelTranspose transB,
    int m, int n, int k, float alpha, const float* A, int lda,
    const float* B, int ldb, float beta, float* C, int ldc);

AccelStatus rocm_batch_sgemv(AcceleratorHandle accel, AccelLayout layout,
    AccelTranspose trans, int m, int n, int batch_size, float alpha,
    const float* A, int lda, const float* X, int ldX, float beta, float* Y, int ldY);

AccelStatus rocm_sync(AcceleratorHandle accel);

#ifdef __cplusplus
}
#endif
