// matrix_ops.hip
// HIP kernels for matrix operations
// Must be compiled with hipcc

#include <hip/hip_runtime.h>

// Matrix-vector multiply kernel: y = alpha * A * x + beta * y
__global__ void sgemv_kernel(int m, int n, float alpha,
    const float* A, int lda, const float* x, int incx,
    float beta, float* y, int incy)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        float sum = 0.0f;
        const float* A_row = A + row * lda;
        for (int j = 0; j < n; j++) {
            sum += A_row[j] * x[j * incx];
        }
        y[row * incy] = alpha * sum + beta * y[row * incy];
    }
}

// Matrix-vector multiply transpose kernel: y = alpha * A^T * x + beta * y
__global__ void sgemv_trans_kernel(int m, int n, float alpha,
    const float* A, int lda, const float* x, int incx,
    float beta, float* y, int incy)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        float sum = 0.0f;
        for (int i = 0; i < m; i++) {
            sum += A[i * lda + col] * x[i * incx];
        }
        y[col * incy] = alpha * sum + beta * y[col * incy];
    }
}

// Outer product kernel: A = alpha * x * y^T + A
__global__ void sger_kernel(int m, int n, float alpha,
    const float* x, int incx, const float* y, int incy,
    float* A, int lda)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        A[row * lda + col] += alpha * x[row * incx] * y[col * incy];
    }
}

// Matrix multiply kernel: C = alpha * A * B + beta * C
__global__ void sgemm_kernel(int m, int n, int k, float alpha,
    const float* A, int lda, const float* B, int ldb,
    float beta, float* C, int ldc)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int p = 0; p < k; p++) {
            sum += A[row * lda + p] * B[p * ldb + col];
        }
        C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
    }
}

// Wrapper functions called from host code
extern "C" {

void launch_sgemv(int m, int n, float alpha,
    const float* d_A, int lda, const float* d_x, int incx,
    float beta, float* d_y, int incy, int transpose)
{
    int block_size = 256;
    int grid_size = (transpose ? n : m + block_size - 1) / block_size;

    if (transpose) {
        hipLaunchKernelGGL(sgemv_trans_kernel,
            dim3(grid_size), dim3(block_size), 0, 0,
            m, n, alpha, d_A, lda, d_x, incx, beta, d_y, incy);
    } else {
        hipLaunchKernelGGL(sgemv_kernel,
            dim3(grid_size), dim3(block_size), 0, 0,
            m, n, alpha, d_A, lda, d_x, incx, beta, d_y, incy);
    }
}

void launch_sger(int m, int n, float alpha,
    const float* d_x, int incx, const float* d_y, int incy,
    float* d_A, int lda)
{
    dim3 block_size(16, 16);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);

    hipLaunchKernelGGL(sger_kernel,
        grid_size, block_size, 0, 0,
        m, n, alpha, d_x, incx, d_y, incy, d_A, lda);
}

void launch_sgemm(int m, int n, int k, float alpha,
    const float* d_A, int lda, const float* d_B, int ldb,
    float beta, float* d_C, int ldc)
{
    dim3 block_size(16, 16);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);

    hipLaunchKernelGGL(sgemm_kernel,
        grid_size, block_size, 0, 0,
        m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
}

} // extern "C"
