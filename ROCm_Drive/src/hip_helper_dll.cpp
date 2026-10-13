// hip_helper_dll.cpp
// HIP Helper DLL - compiled with hipcc
// All HIP operations happen inside this DLL

#include "hip_helper_dll.h"
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

// Global state
static hipblasHandle_t g_blas_handle = nullptr;
static hipStream_t g_stream = nullptr;
static int g_initialized = 0;
static FILE* g_log = nullptr;

static void log_msg(const char* fmt, ...) {
    if (!g_log) {
        g_log = fopen("hip_helper.log", "w");
        if (!g_log) return;
        fprintf(g_log, "=== HIP Helper Log ===\n");
    }
    va_list args;
    va_start(args, fmt);
    vfprintf(g_log, fmt, args);
    va_end(args);
    fflush(g_log);
}

// ============================================================================
// Initialization / Cleanup
// ============================================================================

__declspec(dllexport) int hip_helper_init(void) {
    if (g_initialized) return 0;

    log_msg("[INIT] Initializing HIP...\n");

    hipError_t err = hipInit(0);
    log_msg("[INIT] hipInit(0) = %d (%s)\n", err, hipGetErrorString(err));
    if (err != hipSuccess) return (int)err;

    err = hipSetDevice(0);
    log_msg("[INIT] hipSetDevice(0) = %d\n", err);
    if (err != hipSuccess) return (int)err;

    err = hipStreamCreate(&g_stream);
    log_msg("[INIT] hipStreamCreate = %d\n", err);
    if (err != hipSuccess) return (int)err;

    hipblasStatus_t bs = hipblasCreate(&g_blas_handle);
    log_msg("[INIT] hipblasCreate = %d\n", bs);
    if (bs != HIPBLAS_STATUS_SUCCESS) return -1;

    hipblasSetStream(g_blas_handle, g_stream);

    g_initialized = 1;
    log_msg("[INIT] HIP initialized successfully\n");
    return 0;
}

__declspec(dllexport) __declspec(dllexport) __declspec(dllexport) void hip_helper_cleanup(void) {
    if (g_blas_handle) { hipblasDestroy(g_blas_handle); g_blas_handle = nullptr; }
    if (g_stream) { hipStreamDestroy(g_stream); g_stream = nullptr; }
    g_initialized = 0;
    if (g_log) { fclose(g_log); g_log = nullptr; }
}

// ============================================================================
// Device Query
// ============================================================================

__declspec(dllexport) int hip_helper_get_name(char* name, int size) {
    if (!name || size <= 0) return -1;
    hipDeviceProp_t prop;
    hipError_t err = hipGetDeviceProperties(&prop, 0);
    if (err != hipSuccess) return (int)err;
    strncpy(name, prop.name, size - 1);
    name[size - 1] = 0;
    return 0;
}

__declspec(dllexport) int hip_helper_get_mem(size_t* total, size_t* free_mem) {
    if (!total || !free_mem) return -1;
    return (int)hipMemGetInfo(free_mem, total);
}

// ============================================================================
// Memory Management
// ============================================================================

__declspec(dllexport) int hip_helper_malloc(void** ptr, size_t size) {
    if (!ptr || size == 0) return -1;
    hipError_t err = hipMalloc(ptr, size);
    log_msg("[MALLOC] size=%zu, ptr=%p, err=%d\n", size, *ptr, err);
    return (int)err;
}

__declspec(dllexport) int hip_helper_free(void* ptr) {
    if (!ptr) return 0;
    return (int)hipFree(ptr);
}

__declspec(dllexport) int hip_helper_h2d(void* dst, const void* src, size_t size) {
    if (!dst || !src || size == 0) return -1;
    return (int)hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
}

__declspec(dllexport) int hip_helper_d2h(void* dst, const void* src, size_t size) {
    if (!dst || !src || size == 0) return -1;
    return (int)hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
}

__declspec(dllexport) int hip_helper_d2d(void* dst, const void* src, size_t size) {
    if (!dst || !src || size == 0) return -1;
    return (int)hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice);
}

// ============================================================================
// Matrix Operations (hipBLAS)
// ============================================================================

__declspec(dllexport) int hip_helper_sgemv(int m, int n, float alpha,
    const float* d_A, int lda, const float* d_x, int incx,
    float beta, float* d_y, int incy)
{
    if (!g_initialized || !g_blas_handle) return -1;
    // Row-major A[m][n] = Column-major A^T[n][m]
    // y = alpha * A * x + beta * y  →  y = alpha * A^T * x + beta * y (col-major)
    hipblasStatus_t s = hipblasSgemv(g_blas_handle, HIPBLAS_OP_T,
        n, m, &alpha, d_A, lda, d_x, incx, &beta, d_y, incy);
    return (int)s;
}

__declspec(dllexport) int hip_helper_sgemv_t(int m, int n, float alpha,
    const float* d_A, int lda, const float* d_x, int incx,
    float beta, float* d_y, int incy)
{
    if (!g_initialized || !g_blas_handle) return -1;
    // Row-major A^T = Column-major A (no trans)
    hipblasStatus_t s = hipblasSgemv(g_blas_handle, HIPBLAS_OP_N,
        n, m, &alpha, d_A, lda, d_x, incx, &beta, d_y, incy);
    return (int)s;
}

__declspec(dllexport) int hip_helper_sger(int m, int n, float alpha,
    const float* d_x, int incx, const float* d_y, int incy,
    float* d_A, int lda)
{
    if (!g_initialized || !g_blas_handle) return -1;
    // Row-major: A += alpha * x * y^T  →  Column-major: A^T += alpha * y * x^T
    hipblasStatus_t s = hipblasSger(g_blas_handle,
        n, m, &alpha, d_y, incy, d_x, incx, d_A, lda);
    return (int)s;
}

__declspec(dllexport) int hip_helper_sgemm(int m, int n, int k, float alpha,
    const float* d_A, int lda, const float* d_B, int ldb,
    float beta, float* d_C, int ldc)
{
    if (!g_initialized || !g_blas_handle) return -1;
    // Row-major C=A*B → Column-major C^T = B^T * A^T
    hipblasStatus_t s = hipblasSgemm(g_blas_handle,
        HIPBLAS_OP_T, HIPBLAS_OP_T, n, m, k,
        &alpha, d_B, ldb, d_A, lda, &beta, d_C, ldc);
    return (int)s;
}

// ============================================================================
// GPU Kernels
// ============================================================================

__global__ void kernel_soft_update(float* target, const float* source, float tau, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        target[idx] = tau * source[idx] + (1.0f - tau) * target[idx];
    }
}

__declspec(dllexport) int hip_helper_soft_update(float* d_target, const float* d_source, float tau, int n) {
    if (!g_initialized || !d_target || !d_source || n <= 0) return -1;
    int block = 256;
    int grid = (n + block - 1) / block;
    hipLaunchKernelGGL(kernel_soft_update, dim3(grid), dim3(block), 0, g_stream,
        d_target, d_source, tau, n);
    return (int)hipStreamSynchronize(g_stream);
}

__declspec(dllexport) int hip_helper_sync(void) {
    if (!g_initialized) return -1;
    return (int)hipStreamSynchronize(g_stream);
}
