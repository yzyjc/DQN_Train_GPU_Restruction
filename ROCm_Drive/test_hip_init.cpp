// test_hip_init.cpp
// Minimal HIP initialization test for gfx1151 APU
// Compile with: hipcc test_hip_init.cpp -o test_hip_init.exe

#include <hip/hip_runtime.h>
#include <stdio.h>

int main() {
    printf("=== HIP Initialization Test ===\n");
    printf("HIP Runtime Version: %d\n", HIP_VERSION_MAJOR);

    // Test 1: hipInit
    printf("\n[Test 1] hipInit(0)...\n");
    hipError_t err = hipInit(0);
    printf("  Result: %d\n", err);
    printf("  Message: %s\n", hipGetErrorString(err));

    // Test 2: hipGetDeviceCount
    printf("\n[Test 2] hipGetDeviceCount...\n");
    int count = 0;
    err = hipGetDeviceCount(&count);
    printf("  Result: %d\n", err);
    printf("  Message: %s\n", hipGetErrorString(err));
    printf("  Count: %d\n", count);

    // Test 3: hipGetDeviceProperties
    if (count > 0) {
        printf("\n[Test 3] hipGetDeviceProperties(0)...\n");
        hipDeviceProp_t prop;
        err = hipGetDeviceProperties(&prop, 0);
        printf("  Result: %d\n", err);
        printf("  Message: %s\n", hipGetErrorString(err));
        printf("  Name: %s\n", prop.name);
        printf("  gcnArchName: %s\n", prop.gcnArchName);
        printf("  GlobalMem: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
        printf("  Compute Units: %d\n", prop.multiProcessorCount);
        printf("  Is APU: %s\n", (prop.totalGlobalMem < 2ULL * 1024 * 1024 * 1024) ? "Yes" : "No");
    }

    // Test 4: hipSetDevice
    if (count > 0) {
        printf("\n[Test 4] hipSetDevice(0)...\n");
        err = hipSetDevice(0);
        printf("  Result: %d\n", err);
        printf("  Message: %s\n", hipGetErrorString(err));
    }

    // Test 5: hipMalloc
    if (count > 0) {
        printf("\n[Test 5] hipMalloc(1024)...\n");
        void* ptr = NULL;
        err = hipMalloc(&ptr, 1024);
        printf("  Result: %d\n", err);
        printf("  Message: %s\n", hipGetErrorString(err));
        printf("  Pointer: %p\n", ptr);
        if (err == hipSuccess && ptr) {
            hipFree(ptr);
            printf("  hipFree: OK\n");
        }
    }

    printf("\n=== Test Complete ===\n");
    return 0;
}
