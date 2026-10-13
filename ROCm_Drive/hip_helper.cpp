// hip_helper.cpp
// HIP Helper - compiled with hipcc, initializes HIP runtime
// This file should be compiled with: hipcc hip_helper.cpp -o hip_helper.exe
// The training program can call this to verify HIP works

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Export functions for DLL usage
extern "C" {

// Initialize HIP and return device info
int hip_helper_init() {
    hipError_t err = hipInit(0);
    if (err != hipSuccess) {
        printf("[HIP Helper] hipInit failed: %d (%s)\n", err, hipGetErrorString(err));
        return -1;
    }

    int count = 0;
    err = hipGetDeviceCount(&count);
    if (err != hipSuccess || count == 0) {
        printf("[HIP Helper] No devices found\n");
        return -1;
    }

    hipDeviceProp_t prop;
    err = hipGetDeviceProperties(&prop, 0);
    if (err != hipSuccess) {
        printf("[HIP Helper] Failed to get device properties\n");
        return -1;
    }

    printf("[HIP Helper] Device: %s\n", prop.name);
    printf("[HIP Helper] gcnArchName: %s\n", prop.gcnArchName);
    printf("[HIP Helper] GlobalMem: %zu MB\n", prop.totalGlobalMem / (1024*1024));
    printf("[HIP Helper] ComputeUnits: %d\n", prop.multiProcessorCount);

    return 0;
}

// Allocate GPU memory
int hip_helper_malloc(void** ptr, size_t size) {
    return hipMalloc(ptr, size);
}

// Free GPU memory
int hip_helper_free(void* ptr) {
    return hipFree(ptr);
}

// Copy host to device
int hip_helper_memcpy_h2d(void* dst, const void* src, size_t size) {
    return hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
}

// Copy device to host
int hip_helper_memcpy_d2h(void* dst, const void* src, size_t size) {
    return hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
}

// Synchronize
int hip_helper_sync() {
    return hipDeviceSynchronize();
}

} // extern "C"
