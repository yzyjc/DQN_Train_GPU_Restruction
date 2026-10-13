// hip_helper_dll.h
// HIP Helper DLL - compiled with hipcc
// All HIP operations happen inside this DLL

#ifndef HIP_HELPER_DLL_H
#define HIP_HELPER_DLL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize HIP and create BLAS handle
// Returns 0 on success, HIP error code on failure
int hip_helper_init(void);

// Get device name
int hip_helper_get_name(char* name, int size);

// Get device memory info
int hip_helper_get_mem(size_t* total, size_t* free_mem);

// Allocate GPU memory
int hip_helper_malloc(void** ptr, size_t size);

// Free GPU memory
int hip_helper_free(void* ptr);

// Copy host to device
int hip_helper_h2d(void* dst, const void* src, size_t size);

// Copy device to host
int hip_helper_d2h(void* dst, const void* src, size_t size);

// Copy device to device
int hip_helper_d2d(void* dst, const void* src, size_t size);

// Matrix-vector multiply: y = alpha * A * x + beta * y (row-major, no trans)
int hip_helper_sgemv(int m, int n, float alpha,
    const float* d_A, int lda, const float* d_x, int incx,
    float beta, float* d_y, int incy);

// Matrix-vector multiply: y = alpha * A^T * x + beta * y (row-major)
int hip_helper_sgemv_t(int m, int n, float alpha,
    const float* d_A, int lda, const float* d_x, int incx,
    float beta, float* d_y, int incy);

// Outer product: A += alpha * x * y^T (row-major)
int hip_helper_sger(int m, int n, float alpha,
    const float* d_x, int incx, const float* d_y, int incy,
    float* d_A, int lda);

// Matrix multiply: C = alpha * A * B + beta * C (row-major)
int hip_helper_sgemm(int m, int n, int k, float alpha,
    const float* d_A, int lda, const float* d_B, int ldb,
    float beta, float* d_C, int ldc);

// Soft update: target = tau * source + (1-tau) * target
int hip_helper_soft_update(float* d_target, const float* d_source, float tau, int n);

// Synchronize device
int hip_helper_sync(void);

// Cleanup
void hip_helper_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif // HIP_HELPER_DLL_H
