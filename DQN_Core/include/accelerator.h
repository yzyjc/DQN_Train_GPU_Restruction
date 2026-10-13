#pragma once

// DQN Unified Accelerator Interface v1.0

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Accelerator types
typedef enum {
    ACCEL_TYPE_AUTO = 0,
    ACCEL_TYPE_CPU,
    ACCEL_TYPE_VULKAN,
    ACCEL_TYPE_ROCM,
    ACCEL_TYPE_CUDA
} AccelType;

// Status codes
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

// Matrix layout
typedef enum {
    ACCEL_ROW_MAJOR = 0,
    ACCEL_COL_MAJOR
} AccelLayout;

// Transpose options
typedef enum {
    ACCEL_NO_TRANS = 0,
    ACCEL_TRANS = 1
} AccelTranspose;

// Device info structure
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

// Opaque handle type
typedef void* AcceleratorHandle;

// Function pointer types for DLL exports
typedef AccelStatus (*PFN_accel_init)(AccelType type, AcceleratorHandle* out_accel);
typedef AccelStatus (*PFN_accel_cleanup)(AcceleratorHandle accel);
typedef AccelStatus (*PFN_accel_get_device_info)(AcceleratorHandle accel, AccelDeviceInfo* info);
typedef AccelStatus (*PFN_accel_detect_devices)(AccelDeviceInfo* devices, int max_count, int* out_count);
typedef AccelStatus (*PFN_accel_malloc)(AcceleratorHandle accel, size_t size, void** out_ptr);
typedef AccelStatus (*PFN_accel_free)(AcceleratorHandle accel, void* ptr);
typedef AccelStatus (*PFN_accel_upload)(AcceleratorHandle accel, void* dst, const void* src, size_t size);
typedef AccelStatus (*PFN_accel_download)(AcceleratorHandle accel, void* dst, const void* src, size_t size);
typedef AccelStatus (*PFN_accel_sgemv)(AcceleratorHandle accel, AccelLayout layout, AccelTranspose trans, int m, int n, float alpha, const float* A, int lda, const float* x, int incx, float beta, float* y, int incy);
typedef AccelStatus (*PFN_accel_sger)(AcceleratorHandle accel, AccelLayout layout, int m, int n, float alpha, const float* x, int incx, const float* y, int incy, float* A, int lda);
typedef AccelStatus (*PFN_accel_sgemm)(AcceleratorHandle accel, AccelLayout layout, AccelTranspose transA, AccelTranspose transB, int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc);
typedef AccelStatus (*PFN_accel_batch_sgemv)(AcceleratorHandle accel, AccelLayout layout, AccelTranspose trans, int m, int n, int batch_size, float alpha, const float* A, int lda, const float* X, int ldX, float beta, float* Y, int ldY);
typedef AccelStatus (*PFN_accel_sync)(AcceleratorHandle accel);

// Interface functions (static library implementation)
AccelStatus accel_init(AccelType type, AcceleratorHandle* out_accel);
AccelStatus accel_cleanup(AcceleratorHandle accel);
AccelStatus accel_get_device_info(AcceleratorHandle accel, AccelDeviceInfo* info);
AccelStatus accel_detect_devices(AccelDeviceInfo* devices, int max_count, int* out_count);
const char* accel_get_type_name(AccelType type);
const char* accel_get_error_string(AccelStatus status);
AccelStatus accel_malloc(AcceleratorHandle accel, size_t size, void** out_ptr);
AccelStatus accel_free(AcceleratorHandle accel, void* ptr);
AccelStatus accel_upload(AcceleratorHandle accel, void* dst, const void* src, size_t size);
AccelStatus accel_download(AcceleratorHandle accel, void* dst, const void* src, size_t size);
AccelStatus accel_sgemv(AcceleratorHandle accel, AccelLayout layout, AccelTranspose trans, int m, int n, float alpha, const float* A, int lda, const float* x, int incx, float beta, float* y, int incy);
AccelStatus accel_sger(AcceleratorHandle accel, AccelLayout layout, int m, int n, float alpha, const float* x, int incx, const float* y, int incy, float* A, int lda);
AccelStatus accel_sgemm(AcceleratorHandle accel, AccelLayout layout, AccelTranspose transA, AccelTranspose transB, int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc);
AccelStatus accel_batch_sgemv(AcceleratorHandle accel, AccelLayout layout, AccelTranspose trans, int m, int n, int batch_size, float alpha, const float* A, int lda, const float* X, int ldX, float beta, float* Y, int ldY);
AccelStatus accel_sync(AcceleratorHandle accel);

#ifdef __cplusplus
}
#endif
