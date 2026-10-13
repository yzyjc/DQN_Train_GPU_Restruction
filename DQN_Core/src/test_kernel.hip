// test_kernel.hip
// 最小 kernel 执行测试：验证 gfx1151 device 代码能否真正运行
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void kernel_add_one(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += 1.0f;
}

int main() {
    fprintf(stderr, "hipInit...\n");
    hipError_t e = hipInit(0);
    fprintf(stderr, "hipInit=%d\n", e);
    if (e != hipSuccess) return 1;

    int count = 0;
    hipGetDeviceCount(&count);
    fprintf(stderr, "devices=%d\n", count);
    if (count == 0) return 1;

    hipSetDevice(0);

    float* d = 0;
    e = hipMalloc(&d, 1024 * sizeof(float));
    fprintf(stderr, "hipMalloc=%d\n", e);
    if (e != hipSuccess) return 1;

    // 初始化 host 数据并上传，避免依赖 hipMemset
    float init[1024];
    for (int i = 0; i < 1024; i++) init[i] = 0.0f;
    e = hipMemcpy(d, init, sizeof(init), hipMemcpyHostToDevice);
    fprintf(stderr, "hipMemcpy H2D=%d\n", e);
    if (e != hipSuccess) return 1;

    fprintf(stderr, "launch kernel_add_one...\n");
    hipLaunchKernelGGL(kernel_add_one, dim3(4), dim3(256), 0, 0, d, 1024);
    hipError_t le = hipGetLastError();
    fprintf(stderr, "launch error=%d\n", le);

    hipDeviceSynchronize();
    hipError_t se = hipGetLastError();
    fprintf(stderr, "sync error=%d\n", se);

    float host[4];
    e = hipMemcpy(host, d, 4 * sizeof(float), hipMemcpyDeviceToHost);
    fprintf(stderr, "hipMemcpy D2H=%d, host[0..3]=%.1f %.1f %.1f %.1f\n", e, host[0], host[1], host[2], host[3]);

    hipFree(d);
    fprintf(stderr, "PASS\n");
    return 0;
}
