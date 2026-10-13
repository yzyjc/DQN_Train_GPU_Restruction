// hip_init_test.cpp - Direct HIP initialization test
// This file is compiled with hipcc and linked into the training program
// It provides a function that can be called from MSVC-compiled code

#include <hip/hip_runtime.h>
#include <stdio.h>

extern "C" {

// This function is compiled with hipcc and linked into the main executable
// It should work because hipcc sets up the HIP runtime properly
int hip_init_from_main(void) {
    printf("[HIP Direct] Calling hipInit(0)...\n");
    hipError_t err = hipInit(0);
    printf("[HIP Direct] hipInit(0) = %d (%s)\n", err, hipGetErrorString(err));

    if (err != hipSuccess) {
        return (int)err;
    }

    int count = 0;
    err = hipGetDeviceCount(&count);
    printf("[HIP Direct] hipGetDeviceCount = %d, count=%d\n", err, count);

    if (err != hipSuccess || count == 0) {
        return -1;
    }

    hipDeviceProp_t prop;
    err = hipGetDeviceProperties(&prop, 0);
    printf("[HIP Direct] Device: %s\n", prop.name);

    err = hipSetDevice(0);
    printf("[HIP Direct] hipSetDevice(0) = %d\n", err);

    return (int)err;
}

} // extern "C"
