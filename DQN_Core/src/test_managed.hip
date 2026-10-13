// test_managed.hip - hipMallocManaged 可行性测试
// 若 managed memory 工作，host 与 GPU 共享指针，可做 GPU-resident 且免 hipMemcpy
#include <hip/hip_runtime.h>
#include <stdio.h>

#define LOG(...) do { fprintf(stderr, __VA_ARGS__); fflush(stderr); } while(0)

__global__ void kernel_add_one(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += 1.0f;
}

int main() {
    LOG("=== hipMallocManaged 测试 ===\n");
    hipError_t e = hipInit(0);
    LOG("hipInit=%d\n", e);
    if (e != hipSuccess) return 1;
    hipSetDevice(0);

    // 分配 managed memory
    float* p = 0;
    e = hipMallocManaged(&p, 256 * sizeof(float));
    LOG("hipMallocManaged=%d ptr=%p\n", e, p);
    if (e != hipSuccess || !p) return 1;

    // host 直接写
    LOG("host 写 p...\n");
    for (int i = 0; i < 256; i++) p[i] = 0.0f;
    LOG("host 写 OK\n");

    // kernel 修改
    LOG("launch kernel...\n");
    hipLaunchKernelGGL(kernel_add_one, dim3(1), dim3(256), 0, 0, p, 256);
    e = hipGetLastError();
    LOG("launch err=%d (%s)\n", e, hipGetErrorString(e));

    e = hipDeviceSynchronize();
    LOG("sync err=%d (%s)\n", e, hipGetErrorString(e));

    // host 读验证
    LOG("host 读 p...\n");
    float sum = 0.0f;
    for (int i = 0; i < 256; i++) sum += p[i];
    LOG("sum=%.0f (期望 256)\n", sum);

    hipFree(p);
    LOG("=== 完成 ===\n");
    return (sum == 256.0f) ? 0 : 1;
}
