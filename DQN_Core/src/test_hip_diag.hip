// test_hip_diag.hip - 细粒度 HIP 诊断
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <string.h>

#define LOG(...) do { fprintf(stderr, __VA_ARGS__); fflush(stderr); } while(0)

__global__ void kernel_noop(float* d) { d[threadIdx.x] = 1.0f; }

int main() {
    LOG("=== HIP 细粒度诊断 ===\n");
    LOG("hipInit...\n");
    hipError_t e = hipInit(0);
    LOG("hipInit=%d (%s)\n", e, hipGetErrorString(e));
    if (e != hipSuccess) return 1;

    int count = 0;
    hipGetDeviceCount(&count);
    LOG("devices=%d\n", count);
    if (count == 0) return 1;

    e = hipSetDevice(0);
    LOG("hipSetDevice=%d\n", e);

    hipDeviceProp_t prop;
    e = hipGetDeviceProperties(&prop, 0);
    LOG("device=%s arch=%s\n", prop.name, prop.gcnArchName);

    // 测试 managed memory 支持
    int supports_managed = 0;
    e = hipDeviceGetAttribute(&supports_managed, hipDeviceAttributeManagedMemory, 0);
    LOG("ManagedMemory attr=%d err=%d\n", supports_managed, e);

    int supports_concurrent = 0;
    e = hipDeviceGetAttribute(&supports_concurrent, hipDeviceAttributeConcurrentKernels, 0);
    LOG("ConcurrentKernels attr=%d err=%d\n", supports_concurrent, e);

    // hipMalloc 小 + 大
    float* small = 0;
    e = hipMalloc(&small, 1024);
    LOG("hipMalloc(1KB)=%d ptr=%p\n", e, small);

    float* big = 0;
    e = hipMalloc(&big, 4 * 1024 * 1024);
    LOG("hipMalloc(4MB)=%d ptr=%p\n", e, big);

    // 尝试 host 直接写 device 指针（UMA 上应可访问）
    if (small) {
        LOG("host 直接写 device 指针 small[0]...\n");
        small[0] = 42.0f;
        LOG("host 写 OK, small[0]=%.1f\n", small[0]);
    }

    // hipMemset
    if (big) {
        LOG("hipMemset(4MB)...\n");
        e = hipMemset(big, 0, 4 * 1024 * 1024);
        LOG("hipMemset=%d (%s)\n", e, hipGetErrorString(e));
    }

    // hipMemcpy H2D
    float host[16];
    for (int i = 0; i < 16; i++) host[i] = (float)i;
    LOG("hipMemcpy H2D 16 floats...\n");
    e = hipMemcpy(small, host, 16 * sizeof(float), hipMemcpyHostToDevice);
    LOG("hipMemcpy H2D=%d (%s)\n", e, hipGetErrorString(e));

    // hipMemcpy D2H
    float back[16] = {0};
    LOG("hipMemcpy D2H 16 floats...\n");
    e = hipMemcpy(back, small, 16 * sizeof(float), hipMemcpyDeviceToHost);
    LOG("hipMemcpy D2H=%d (%s)\n", e, hipGetErrorString(e));
    LOG("back[0..3]=%.1f %.1f %.1f %.1f\n", back[0], back[1], back[2], back[3]);

    // 自定义 kernel
    LOG("launch kernel_noop...\n");
    hipLaunchKernelGGL(kernel_noop, dim3(1), dim3(16), 0, 0, small);
    e = hipGetLastError();
    LOG("launch err=%d (%s)\n", e, hipGetErrorString(e));
    LOG("hipDeviceSynchronize...\n");
    e = hipDeviceSynchronize();
    LOG("sync err=%d (%s)\n", e, hipGetErrorString(e));

    hipFree(small);
    hipFree(big);
    LOG("=== 诊断完成 ===\n");
    return 0;
}
