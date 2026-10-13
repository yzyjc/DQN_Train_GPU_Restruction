// test_hip_minimal.cpp
// Minimal HIP diagnostic for gfx1151 APU
// Compile: D:\AMD_ROCm\bin\hipcc.exe test_hip_minimal.cpp -o test_hip_minimal.exe

#include <hip/hip_runtime.h>
#include <stdio.h>

int main() {
    printf("=== HIP Minimal Diagnostic ===\n");

    // Test 1: hipInit
    printf("\n[1] hipInit(0)...\n");
    hipError_t err = hipInit(0);
    printf("    hipInit = %d\n", err);
    printf("    message = %s\n", hipGetErrorString(err));

    // Test 2: hipGetDeviceCount
    printf("\n[2] hipGetDeviceCount...\n");
    int count = 0;
    err = hipGetDeviceCount(&count);
    printf("    hipGetDeviceCount = %d\n", err);
    printf("    message = %s\n", hipGetErrorString(err));
    printf("    count = %d\n", count);

    // Test 3: hipGetDeviceProperties (if count > 0)
    if (count > 0) {
        printf("\n[3] hipGetDeviceProperties(0)...\n");
        hipDeviceProp_t prop;
        err = hipGetDeviceProperties(&prop, 0);
        printf("    hipGetDeviceProperties = %d\n", err);
        printf("    message = %s\n", hipGetErrorString(err));
        printf("    name = %s\n", prop.name);
        printf("    gcnArchName = %s\n", prop.gcnArchName);
        printf("    globalMem = %zu MB\n", prop.totalGlobalMem / (1024*1024));
        printf("    computeUnits = %d\n", prop.multiProcessorCount);
    }

    // Test 4: hipSetDevice (even if count=0, try it)
    printf("\n[4] hipSetDevice(0)...\n");
    err = hipSetDevice(0);
    printf("    hipSetDevice = %d\n", err);
    printf("    message = %s\n", hipGetErrorString(err));

    // Test 5: hipMalloc (even if previous failed)
    printf("\n[5] hipMalloc(1024)...\n");
    void* ptr = NULL;
    err = hipMalloc(&ptr, 1024);
    printf("    hipMalloc = %d\n", err);
    printf("    message = %s\n", hipGetErrorString(err));
    printf("    ptr = %p\n", ptr);
    if (err == hipSuccess) {
        hipFree(ptr);
        printf("    hipFree: OK\n");
    }

    printf("\n=== Done ===\n");
    return 0;
}
