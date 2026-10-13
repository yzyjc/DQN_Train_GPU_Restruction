// CPU Accelerator Implementation
// Handwritten matrix operations, no OpenBLAS dependency

#include "pch.h"
#include "accelerator.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Internal context structure
struct AcceleratorContext {
    AccelType type;
    AccelDeviceInfo device_info;
    int initialized;
};

// Helper function to get context from handle
static AcceleratorContext* get_context(AcceleratorHandle handle) {
    return (AcceleratorContext*)handle;
}

// Matrix-vector multiply: y = alpha * A * x + beta * y (row major)
static void cpu_sgemv_row_major(int m, int n, float alpha, const float* A, int lda, const float* x, int incx, float beta, float* y, int incy) {
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        const float* row = A + i * lda;
        for (int j = 0; j < n; j++) {
            sum += row[j] * x[j * incx];
        }
        y[i * incy] = alpha * sum + beta * y[i * incy];
    }
}

// Matrix-vector multiply: y = alpha * A^T * x + beta * y (row major)
static void cpu_sgemv_row_major_trans(int m, int n, float alpha, const float* A, int lda, const float* x, int incx, float beta, float* y, int incy) {
    for (int j = 0; j < n; j++) {
        float sum = 0.0f;
        for (int i = 0; i < m; i++) {
            sum += A[i * lda + j] * x[i * incx];
        }
        y[j * incy] = alpha * sum + beta * y[j * incy];
    }
}

// Outer product: A = alpha * x * y^T + A (row major)
static void cpu_sger_row_major(int m, int n, float alpha, const float* x, int incx, const float* y, int incy, float* A, int lda) {
    for (int i = 0; i < m; i++) {
        float x_val = alpha * x[i * incx];
        for (int j = 0; j < n; j++) {
            A[i * lda + j] += x_val * y[j * incy];
        }
    }
}

// Matrix multiply: C = alpha * A * B + beta * C (row major)
static void cpu_sgemm_row_major(int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i * lda + p] * B[p * ldb + j];
            }
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}

// Batch matrix-vector multiply
static void cpu_batch_sgemv_row_major(int m, int n, int batch_size, float alpha, const float* A, int lda, const float* X, int ldX, float beta, float* Y, int ldY) {
    for (int b = 0; b < batch_size; b++) {
        const float* x = X + b * ldX;
        float* y = Y + b * ldY;
        cpu_sgemv_row_major(m, n, alpha, A, lda, x, 1, beta, y, 1);
    }
}

// Interface implementation

extern "C" {

AccelStatus accel_init(AccelType type, AcceleratorHandle* out_accel) {
    if (!out_accel) return ACCEL_ERROR_INVALID_PARAM;
    if (type == ACCEL_TYPE_AUTO) type = ACCEL_TYPE_CPU;
    if (type != ACCEL_TYPE_CPU) return ACCEL_ERROR_NOT_SUPPORTED;

    AcceleratorContext* ctx = (AcceleratorContext*)malloc(sizeof(AcceleratorContext));
    if (!ctx) return ACCEL_ERROR_OUT_OF_MEMORY;

    memset(ctx, 0, sizeof(AcceleratorContext));
    ctx->type = ACCEL_TYPE_CPU;
    ctx->initialized = 1;

    AccelDeviceInfo* info = &ctx->device_info;
    strcpy_s(info->name, "CPU (Handwritten MatMul)");
    strcpy_s(info->vendor, "Fallback");
    info->type = ACCEL_TYPE_CPU;
    info->global_mem = 0;
    info->compute_units = 1;
    info->max_workgroup = 1;
    info->api_version = 1;
    info->is_discrete = 0;
    info->is_available = 1;

    *out_accel = (AcceleratorHandle)ctx;
    return ACCEL_SUCCESS;
}

AccelStatus accel_cleanup(AcceleratorHandle accel) {
    if (!accel) return ACCEL_ERROR_INVALID_PARAM;
    AcceleratorContext* ctx = get_context(accel);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;
    free(ctx);
    return ACCEL_SUCCESS;
}

AccelStatus accel_get_device_info(AcceleratorHandle accel, AccelDeviceInfo* info) {
    if (!accel || !info) return ACCEL_ERROR_INVALID_PARAM;
    AcceleratorContext* ctx = get_context(accel);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;
    *info = ctx->device_info;
    return ACCEL_SUCCESS;
}

AccelStatus accel_detect_devices(AccelDeviceInfo* devices, int max_count, int* out_count) {
    if (!devices || !out_count || max_count < 1) return ACCEL_ERROR_INVALID_PARAM;
    AccelDeviceInfo* info = &devices[0];
    strcpy_s(info->name, "CPU (Handwritten MatMul)");
    strcpy_s(info->vendor, "Fallback");
    info->type = ACCEL_TYPE_CPU;
    info->global_mem = 0;
    info->compute_units = 1;
    info->max_workgroup = 1;
    info->api_version = 1;
    info->is_discrete = 0;
    info->is_available = 1;
    *out_count = 1;
    return ACCEL_SUCCESS;
}

const char* accel_get_type_name(AccelType type) {
    switch (type) {
    case ACCEL_TYPE_AUTO: return "Auto";
    case ACCEL_TYPE_CPU: return "CPU";
    case ACCEL_TYPE_VULKAN: return "Vulkan";
    case ACCEL_TYPE_ROCM: return "ROCm";
    case ACCEL_TYPE_CUDA: return "CUDA";
    default: return "Unknown";
    }
}

const char* accel_get_error_string(AccelStatus status) {
    switch (status) {
    case ACCEL_SUCCESS: return "Success";
    case ACCEL_ERROR_INVALID_PARAM: return "Invalid parameter";
    case ACCEL_ERROR_OUT_OF_MEMORY: return "Out of memory";
    case ACCEL_ERROR_DEVICE_NOT_FOUND: return "Device not found";
    case ACCEL_ERROR_INITIALIZATION_FAILED: return "Initialization failed";
    case ACCEL_ERROR_KERNEL_LAUNCH_FAILED: return "Kernel launch failed";
    case ACCEL_ERROR_SYNC_FAILED: return "Sync failed";
    case ACCEL_ERROR_NOT_SUPPORTED: return "Not supported";
    case ACCEL_ERROR_DLL_LOAD_FAILED: return "DLL load failed";
    case ACCEL_ERROR_FUNCTION_NOT_FOUND: return "Function not found";
    default: return "Unknown error";
    }
}

AccelStatus accel_malloc(AcceleratorHandle accel, size_t size, void** out_ptr) {
    if (!accel || !out_ptr || size == 0) return ACCEL_ERROR_INVALID_PARAM;
    AcceleratorContext* ctx = get_context(accel);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;
    *out_ptr = malloc(size);
    if (!*out_ptr) return ACCEL_ERROR_OUT_OF_MEMORY;
    return ACCEL_SUCCESS;
}

AccelStatus accel_free(AcceleratorHandle accel, void* ptr) {
    if (!accel || !ptr) return ACCEL_ERROR_INVALID_PARAM;
    AcceleratorContext* ctx = get_context(accel);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;
    free(ptr);
    return ACCEL_SUCCESS;
}

AccelStatus accel_upload(AcceleratorHandle accel, void* dst, const void* src, size_t size) {
    if (!accel || !dst || !src) return ACCEL_ERROR_INVALID_PARAM;
    AcceleratorContext* ctx = get_context(accel);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;
    memcpy(dst, src, size);
    return ACCEL_SUCCESS;
}

AccelStatus accel_download(AcceleratorHandle accel, void* dst, const void* src, size_t size) {
    if (!accel || !dst || !src) return ACCEL_ERROR_INVALID_PARAM;
    AcceleratorContext* ctx = get_context(accel);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;
    memcpy(dst, src, size);
    return ACCEL_SUCCESS;
}

AccelStatus accel_sgemv(AcceleratorHandle accel, AccelLayout layout, AccelTranspose trans, int m, int n, float alpha, const float* A, int lda, const float* x, int incx, float beta, float* y, int incy) {
    if (!accel || !A || !x || !y) return ACCEL_ERROR_INVALID_PARAM;
    AcceleratorContext* ctx = get_context(accel);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;
    if (m <= 0 || n <= 0) return ACCEL_ERROR_INVALID_PARAM;
    if (layout != ACCEL_ROW_MAJOR) return ACCEL_ERROR_NOT_SUPPORTED;

    if (trans == ACCEL_NO_TRANS) {
        cpu_sgemv_row_major(m, n, alpha, A, lda, x, incx, beta, y, incy);
    } else {
        cpu_sgemv_row_major_trans(m, n, alpha, A, lda, x, incx, beta, y, incy);
    }
    return ACCEL_SUCCESS;
}

AccelStatus accel_sger(AcceleratorHandle accel, AccelLayout layout, int m, int n, float alpha, const float* x, int incx, const float* y, int incy, float* A, int lda) {
    if (!accel || !x || !y || !A) return ACCEL_ERROR_INVALID_PARAM;
    AcceleratorContext* ctx = get_context(accel);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;
    if (m <= 0 || n <= 0) return ACCEL_ERROR_INVALID_PARAM;
    if (layout != ACCEL_ROW_MAJOR) return ACCEL_ERROR_NOT_SUPPORTED;
    cpu_sger_row_major(m, n, alpha, x, incx, y, incy, A, lda);
    return ACCEL_SUCCESS;
}

AccelStatus accel_sgemm(AcceleratorHandle accel, AccelLayout layout, AccelTranspose transA, AccelTranspose transB, int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
    if (!accel || !A || !B || !C) return ACCEL_ERROR_INVALID_PARAM;
    AcceleratorContext* ctx = get_context(accel);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;
    if (m <= 0 || n <= 0 || k <= 0) return ACCEL_ERROR_INVALID_PARAM;
    if (layout != ACCEL_ROW_MAJOR) return ACCEL_ERROR_NOT_SUPPORTED;
    if (transA != ACCEL_NO_TRANS || transB != ACCEL_NO_TRANS) return ACCEL_ERROR_NOT_SUPPORTED;
    cpu_sgemm_row_major(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return ACCEL_SUCCESS;
}

AccelStatus accel_batch_sgemv(AcceleratorHandle accel, AccelLayout layout, AccelTranspose trans, int m, int n, int batch_size, float alpha, const float* A, int lda, const float* X, int ldX, float beta, float* Y, int ldY) {
    if (!accel || !A || !X || !Y) return ACCEL_ERROR_INVALID_PARAM;
    AcceleratorContext* ctx = get_context(accel);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;
    if (m <= 0 || n <= 0 || batch_size <= 0) return ACCEL_ERROR_INVALID_PARAM;
    if (layout != ACCEL_ROW_MAJOR) return ACCEL_ERROR_NOT_SUPPORTED;
    if (trans != ACCEL_NO_TRANS) return ACCEL_ERROR_NOT_SUPPORTED;
    cpu_batch_sgemv_row_major(m, n, batch_size, alpha, A, lda, X, ldX, beta, Y, ldY);
    return ACCEL_SUCCESS;
}

AccelStatus accel_sync(AcceleratorHandle accel) {
    if (!accel) return ACCEL_ERROR_INVALID_PARAM;
    AcceleratorContext* ctx = get_context(accel);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;
    return ACCEL_SUCCESS;
}

} // extern "C"
