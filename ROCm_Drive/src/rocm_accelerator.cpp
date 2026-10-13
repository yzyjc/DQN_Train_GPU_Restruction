// rocm_accelerator.cpp
// ROCm Accelerator Implementation using rocBLAS

#include "rocm_accelerator.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Check if HIP SDK is available
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#define HIP_AVAILABLE 1
#else
#define HIP_AVAILABLE 0
typedef int hipError_t;
typedef void* hipStream_t;
typedef void* hipblasHandle_t;
#define hipSuccess 0
#endif

// Internal context structure
struct AcceleratorContext {
    AccelType type;
    AccelDeviceInfo device_info;
    int initialized;
    int device_id;
    hipStream_t stream;
    hipblasHandle_t blas_handle;
};

// Helper function
static AcceleratorContext* get_context(AcceleratorHandle handle) {
    return (AcceleratorContext*)handle;
}

// Initialize ROCm accelerator
AccelStatus rocm_init(AccelType type, AcceleratorHandle* out_accel) {
    if (!out_accel) return ACCEL_ERROR_INVALID_PARAM;

#if !HIP_AVAILABLE
    return ACCEL_ERROR_NOT_SUPPORTED;
#else
    printf("[ROCM] Initializing ROCm accelerator...\n");

    // Check if HIP is available
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
        printf("[ROCM] No HIP devices found\n");
        return ACCEL_ERROR_DEVICE_NOT_FOUND;
    }
    printf("[ROCM] Found %d HIP devices\n", device_count);

    // Create context
    AcceleratorContext* ctx = (AcceleratorContext*)malloc(sizeof(AcceleratorContext));
    if (!ctx) return ACCEL_ERROR_OUT_OF_MEMORY;
    memset(ctx, 0, sizeof(AcceleratorContext));

    // Select first available device
    err = hipSetDevice(0);
    if (err != hipSuccess) {
        printf("[ROCM] Failed to set device\n");
        free(ctx);
        return ACCEL_ERROR_INITIALIZATION_FAILED;
    }

    // Get device properties
    hipDeviceProp_t prop;
    err = hipGetDeviceProperties(&prop, 0);
    if (err != hipSuccess) {
        printf("[ROCM] Failed to get device properties\n");
        free(ctx);
        return ACCEL_ERROR_INITIALIZATION_FAILED;
    }

    // Fill device info
    AccelDeviceInfo* info = &ctx->device_info;
    strncpy_s(info->name, prop.name, sizeof(info->name) - 1);
    strcpy_s(info->vendor, "AMD");
    info->type = ACCEL_TYPE_ROCM;
    info->global_mem = prop.totalGlobalMem;
    info->compute_units = prop.multiProcessorCount;
    info->max_workgroup = prop.maxThreadsPerBlock;
    info->api_version = prop.major * 1000 + prop.minor * 100;
    info->is_discrete = (prop.totalGlobalMem > 1ULL * 1024 * 1024 * 1024);
    info->is_available = 1;

    printf("[ROCM] Device: %s\n", info->name);
    printf("[ROCM] Memory: %llu MB\n", info->global_mem / (1024 * 1024));
    printf("[ROCM] Compute Units: %u\n", info->compute_units);

    // Create stream
    printf("[ROCM] Creating HIP stream...\n");
    hipStream_t hip_stream;
    err = hipStreamCreate(&hip_stream);
    if (err != hipSuccess) {
        printf("[ROCM] Failed to create stream (error %d)\n", err);
        free(ctx);
        return ACCEL_ERROR_INITIALIZATION_FAILED;
    }
    ctx->stream = hip_stream;
    printf("[ROCM] Stream created successfully\n");

    // Create rocBLAS handle
    printf("[ROCM] Creating rocBLAS handle...\n");
    hipblasHandle_t blas_handle;
    hipblasStatus_t blas_status = hipblasCreate(&blas_handle);
    if (blas_status != HIPBLAS_STATUS_SUCCESS) {
        printf("[ROCM] Failed to create rocBLAS handle (error %d)\n", blas_status);
        hipStreamDestroy(hip_stream);
        free(ctx);
        return ACCEL_ERROR_INITIALIZATION_FAILED;
    }
    ctx->blas_handle = blas_handle;

    // Set stream for rocBLAS
    hipblasSetStream(blas_handle, hip_stream);
    printf("[ROCM] rocBLAS handle created successfully\n");

    ctx->type = ACCEL_TYPE_ROCM;
    ctx->device_id = 0;
    ctx->initialized = 1;

    printf("[ROCM] Initialization complete!\n");
    *out_accel = (AcceleratorHandle)ctx;
    return ACCEL_SUCCESS;
#endif
}

// Cleanup
AccelStatus rocm_cleanup(AcceleratorHandle accel) {
    if (!accel) return ACCEL_ERROR_INVALID_PARAM;

    AcceleratorContext* ctx = get_context(accel);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;

#if HIP_AVAILABLE
    if (ctx->blas_handle) {
        hipblasDestroy(ctx->blas_handle);
    }
    if (ctx->stream) {
        hipStreamDestroy((hipStream_t)ctx->stream);
    }
#endif

    free(ctx);
    return ACCEL_SUCCESS;
}

// Get device info
AccelStatus rocm_get_device_info(AcceleratorHandle accel, AccelDeviceInfo* info) {
    if (!accel || !info) return ACCEL_ERROR_INVALID_PARAM;
    AcceleratorContext* ctx = get_context(accel);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;
    *info = ctx->device_info;
    return ACCEL_SUCCESS;
}

// Device detection
AccelStatus rocm_detect_devices(AccelDeviceInfo* devices, int max_count, int* out_count) {
    if (!devices || !out_count || max_count < 1) return ACCEL_ERROR_INVALID_PARAM;

#if !HIP_AVAILABLE
    *out_count = 0;
    return ACCEL_ERROR_NOT_SUPPORTED;
#else
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
        *out_count = 0;
        return ACCEL_ERROR_DEVICE_NOT_FOUND;
    }

    if (device_count > max_count) device_count = max_count;

    for (int i = 0; i < device_count; i++) {
        hipDeviceProp_t prop;
        err = hipGetDeviceProperties(&prop, i);
        if (err != hipSuccess) continue;

        AccelDeviceInfo* info = &devices[i];
        memset(info, 0, sizeof(AccelDeviceInfo));
        strncpy_s(info->name, prop.name, sizeof(info->name) - 1);
        strcpy_s(info->vendor, "AMD");
        info->type = ACCEL_TYPE_ROCM;
        info->global_mem = prop.totalGlobalMem;
        info->compute_units = prop.multiProcessorCount;
        info->max_workgroup = prop.maxThreadsPerBlock;
        info->api_version = prop.major * 1000 + prop.minor * 100;
        info->is_discrete = (prop.totalGlobalMem > 1ULL * 1024 * 1024 * 1024);
        info->is_available = 1;
    }

    *out_count = device_count;
    return ACCEL_SUCCESS;
#endif
}

// Memory allocation
AccelStatus rocm_malloc(AcceleratorHandle accel, size_t size, void** out_ptr) {
    if (!accel || !out_ptr || size == 0) return ACCEL_ERROR_INVALID_PARAM;
    AcceleratorContext* ctx = get_context(accel);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;

#if !HIP_AVAILABLE
    return ACCEL_ERROR_NOT_SUPPORTED;
#else
    hipError_t err = hipMalloc(out_ptr, size);
    if (err != hipSuccess) return ACCEL_ERROR_OUT_OF_MEMORY;
    return ACCEL_SUCCESS;
#endif
}

// Memory free
AccelStatus rocm_free(AcceleratorHandle accel, void* ptr) {
    if (!accel || !ptr) return ACCEL_ERROR_INVALID_PARAM;
    AcceleratorContext* ctx = get_context(accel);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;

#if !HIP_AVAILABLE
    return ACCEL_ERROR_NOT_SUPPORTED;
#else
    hipError_t err = hipFree(ptr);
    if (err != hipSuccess) return ACCEL_ERROR_INVALID_PARAM;
    return ACCEL_SUCCESS;
#endif
}

// Upload to GPU
AccelStatus rocm_upload(AcceleratorHandle accel, void* dst, const void* src, size_t size) {
    if (!accel || !dst || !src) return ACCEL_ERROR_INVALID_PARAM;
    AcceleratorContext* ctx = get_context(accel);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;

#if !HIP_AVAILABLE
    return ACCEL_ERROR_NOT_SUPPORTED;
#else
    hipError_t err = hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
    if (err != hipSuccess) return ACCEL_ERROR_SYNC_FAILED;
    return ACCEL_SUCCESS;
#endif
}

// Download from GPU
AccelStatus rocm_download(AcceleratorHandle accel, void* dst, const void* src, size_t size) {
    if (!accel || !dst || !src) return ACCEL_ERROR_INVALID_PARAM;
    AcceleratorContext* ctx = get_context(accel);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;

#if !HIP_AVAILABLE
    return ACCEL_ERROR_NOT_SUPPORTED;
#else
    hipError_t err = hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
    if (err != hipSuccess) return ACCEL_ERROR_SYNC_FAILED;
    return ACCEL_SUCCESS;
#endif
}

// Synchronize
AccelStatus rocm_sync(AcceleratorHandle accel) {
    if (!accel) return ACCEL_ERROR_INVALID_PARAM;
    AcceleratorContext* ctx = get_context(accel);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;

#if !HIP_AVAILABLE
    return ACCEL_ERROR_NOT_SUPPORTED;
#else
    hipError_t err = hipStreamSynchronize((hipStream_t)ctx->stream);
    if (err != hipSuccess) return ACCEL_ERROR_SYNC_FAILED;
    return ACCEL_SUCCESS;
#endif
}

// Matrix-vector multiply using rocBLAS
AccelStatus rocm_sgemv(AcceleratorHandle accel, AccelLayout layout, AccelTranspose trans,
    int m, int n, float alpha, const float* A, int lda, const float* x, int incx,
    float beta, float* y, int incy)
{
    if (!accel || !A || !x || !y) return ACCEL_ERROR_INVALID_PARAM;
    AcceleratorContext* ctx = get_context(accel);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;
    if (m <= 0 || n <= 0) return ACCEL_ERROR_INVALID_PARAM;
    if (layout != ACCEL_ROW_MAJOR) return ACCEL_ERROR_NOT_SUPPORTED;

#if !HIP_AVAILABLE
    return ACCEL_ERROR_NOT_SUPPORTED;
#else
    // Allocate device memory
    float *d_A, *d_x, *d_y;
    size_t size_A = (size_t)m * lda * sizeof(float);
    size_t size_x = (size_t)(trans == ACCEL_NO_TRANS ? n : m) * incx * sizeof(float);
    size_t size_y = (size_t)(trans == ACCEL_NO_TRANS ? m : n) * incy * sizeof(float);

    hipMalloc(&d_A, size_A);
    hipMalloc(&d_x, size_x);
    hipMalloc(&d_y, size_y);

    // Upload data
    hipMemcpy(d_A, A, size_A, hipMemcpyHostToDevice);
    hipMemcpy(d_x, x, size_x, hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, size_y, hipMemcpyHostToDevice);

    // Use rocBLAS for matrix-vector multiply
    // Note: rocBLAS uses column-major, so we swap m and n for row-major
    hipblasOperation_t op = (trans == ACCEL_NO_TRANS) ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasStatus_t status = hipblasSgemv((hipblasHandle_t)ctx->blas_handle,
        op, n, m, &alpha, d_A, lda, d_x, incx, &beta, d_y, incy);

    // Download result
    hipMemcpy(y, d_y, size_y, hipMemcpyDeviceToHost);

    // Cleanup
    hipFree(d_A);
    hipFree(d_x);
    hipFree(d_y);

    return (status == HIPBLAS_STATUS_SUCCESS) ? ACCEL_SUCCESS : ACCEL_ERROR_KERNEL_LAUNCH_FAILED;
#endif
}

// Outer product using rocBLAS
AccelStatus rocm_sger(AcceleratorHandle accel, AccelLayout layout,
    int m, int n, float alpha, const float* x, int incx, const float* y, int incy,
    float* A, int lda)
{
    if (!accel || !x || !y || !A) return ACCEL_ERROR_INVALID_PARAM;
    AcceleratorContext* ctx = get_context(accel);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;
    if (m <= 0 || n <= 0) return ACCEL_ERROR_INVALID_PARAM;
    if (layout != ACCEL_ROW_MAJOR) return ACCEL_ERROR_NOT_SUPPORTED;

#if !HIP_AVAILABLE
    return ACCEL_ERROR_NOT_SUPPORTED;
#else
    // Allocate device memory
    float *d_A, *d_x, *d_y;
    size_t size_A = (size_t)m * lda * sizeof(float);
    size_t size_x = (size_t)m * incx * sizeof(float);
    size_t size_y = (size_t)n * incy * sizeof(float);

    hipMalloc(&d_A, size_A);
    hipMalloc(&d_x, size_x);
    hipMalloc(&d_y, size_y);

    // Upload data
    hipMemcpy(d_A, A, size_A, hipMemcpyHostToDevice);
    hipMemcpy(d_x, x, size_x, hipMemcpyHostToDevice);
    hipMemcpy(d_y, y, size_y, hipMemcpyHostToDevice);

    // Use rocBLAS for outer product
    hipblasStatus_t status = hipblasSger((hipblasHandle_t)ctx->blas_handle,
        n, m, &alpha, d_y, incy, d_x, incx, d_A, lda);

    // Download result
    hipMemcpy(A, d_A, size_A, hipMemcpyDeviceToHost);

    // Cleanup
    hipFree(d_A);
    hipFree(d_x);
    hipFree(d_y);

    return (status == HIPBLAS_STATUS_SUCCESS) ? ACCEL_SUCCESS : ACCEL_ERROR_KERNEL_LAUNCH_FAILED;
#endif
}

// Matrix multiply using rocBLAS
AccelStatus rocm_sgemm(AcceleratorHandle accel, AccelLayout layout,
    AccelTranspose transA, AccelTranspose transB,
    int m, int n, int k, float alpha, const float* A, int lda,
    const float* B, int ldb, float beta, float* C, int ldc)
{
    if (!accel || !A || !B || !C) return ACCEL_ERROR_INVALID_PARAM;
    AcceleratorContext* ctx = get_context(accel);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;
    if (m <= 0 || n <= 0 || k <= 0) return ACCEL_ERROR_INVALID_PARAM;
    if (layout != ACCEL_ROW_MAJOR) return ACCEL_ERROR_NOT_SUPPORTED;

#if !HIP_AVAILABLE
    return ACCEL_ERROR_NOT_SUPPORTED;
#else
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    size_t size_A = (size_t)m * lda * sizeof(float);
    size_t size_B = (size_t)k * ldb * sizeof(float);
    size_t size_C = (size_t)m * ldc * sizeof(float);

    hipMalloc(&d_A, size_A);
    hipMalloc(&d_B, size_B);
    hipMalloc(&d_C, size_C);

    // Upload data
    hipMemcpy(d_A, A, size_A, hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, size_B, hipMemcpyHostToDevice);
    hipMemcpy(d_C, C, size_C, hipMemcpyHostToDevice);

    // Use rocBLAS for matrix multiply
    // Row-major: C = A * B → Column-major: C^T = B^T * A^T
    hipblasOperation_t opA = (transA == ACCEL_NO_TRANS) ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t opB = (transB == ACCEL_NO_TRANS) ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasStatus_t status = hipblasSgemm((hipblasHandle_t)ctx->blas_handle,
        opB, opA, n, m, k, &alpha, d_B, ldb, d_A, lda, &beta, d_C, ldc);

    // Download result
    hipMemcpy(C, d_C, size_C, hipMemcpyDeviceToHost);

    // Cleanup
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    return (status == HIPBLAS_STATUS_SUCCESS) ? ACCEL_SUCCESS : ACCEL_ERROR_KERNEL_LAUNCH_FAILED;
#endif
}

// Batch matrix-vector multiply
AccelStatus rocm_batch_sgemv(AcceleratorHandle accel, AccelLayout layout,
    AccelTranspose trans, int m, int n, int batch_size, float alpha,
    const float* A, int lda, const float* X, int ldX, float beta, float* Y, int ldY)
{
    if (!accel || !A || !X || !Y) return ACCEL_ERROR_INVALID_PARAM;
    AcceleratorContext* ctx = get_context(accel);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;
    if (m <= 0 || n <= 0 || batch_size <= 0) return ACCEL_ERROR_INVALID_PARAM;
    if (layout != ACCEL_ROW_MAJOR) return ACCEL_ERROR_NOT_SUPPORTED;
    if (trans != ACCEL_NO_TRANS) return ACCEL_ERROR_NOT_SUPPORTED;

    // Process each batch
    for (int b = 0; b < batch_size; b++) {
        const float* x_batch = X + b * ldX;
        float* y_batch = Y + b * ldY;
        AccelStatus status = rocm_sgemv(accel, layout, trans,
            m, n, alpha, A, lda, x_batch, 1, beta, y_batch, 1);
        if (status != ACCEL_SUCCESS) return status;
    }

    return ACCEL_SUCCESS;
}
