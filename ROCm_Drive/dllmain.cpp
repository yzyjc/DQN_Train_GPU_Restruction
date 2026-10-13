// dllmain.cpp
// ROCm Drive DLL entry point

#include <windows.h>
#include "rocm_accelerator.h"

// DLL entry point
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

// Function table structure for DLL exports
typedef struct {
    AccelStatus (*init)(AccelType type, AcceleratorHandle* out_accel);
    AccelStatus (*cleanup)(AcceleratorHandle accel);
    AccelStatus (*get_device_info)(AcceleratorHandle accel, AccelDeviceInfo* info);
    AccelStatus (*detect_devices)(AccelDeviceInfo* devices, int max_count, int* out_count);
    AccelStatus (*malloc)(AcceleratorHandle accel, size_t size, void** out_ptr);
    AccelStatus (*free)(AcceleratorHandle accel, void* ptr);
    AccelStatus (*upload)(AcceleratorHandle accel, void* dst, const void* src, size_t size);
    AccelStatus (*download)(AcceleratorHandle accel, void* dst, const void* src, size_t size);
    AccelStatus (*sgemv)(AcceleratorHandle accel, AccelLayout layout, AccelTranspose trans,
        int m, int n, float alpha, const float* A, int lda, const float* x, int incx,
        float beta, float* y, int incy);
    AccelStatus (*sger)(AcceleratorHandle accel, AccelLayout layout,
        int m, int n, float alpha, const float* x, int incx, const float* y, int incy,
        float* A, int lda);
    AccelStatus (*sgemm)(AcceleratorHandle accel, AccelLayout layout,
        AccelTranspose transA, AccelTranspose transB,
        int m, int n, int k, float alpha, const float* A, int lda,
        const float* B, int ldb, float beta, float* C, int ldc);
    AccelStatus (*batch_sgemv)(AcceleratorHandle accel, AccelLayout layout,
        AccelTranspose trans, int m, int n, int batch_size, float alpha,
        const float* A, int lda, const float* X, int ldX, float beta, float* Y, int ldY);
    AccelStatus (*sync)(AcceleratorHandle accel);
} AccelFunctionTable;

// Function table instance
static AccelFunctionTable g_func_table = {
    rocm_init,
    rocm_cleanup,
    rocm_get_device_info,
    rocm_detect_devices,
    rocm_malloc,
    rocm_free,
    rocm_upload,
    rocm_download,
    rocm_sgemv,
    rocm_sger,
    rocm_sgemm,
    rocm_batch_sgemv,
    rocm_sync
};

// Export function table
extern "C" __declspec(dllexport) const AccelFunctionTable* GetAccelFunctionTable() {
    return &g_func_table;
}

// Export individual functions with standard names (no prefix)
extern "C" {
    __declspec(dllexport) AccelStatus accel_init(AccelType type, AcceleratorHandle* out_accel) {
        return rocm_init(type, out_accel);
    }

    __declspec(dllexport) AccelStatus accel_cleanup(AcceleratorHandle accel) {
        return rocm_cleanup(accel);
    }

    __declspec(dllexport) AccelStatus accel_get_device_info(AcceleratorHandle accel, AccelDeviceInfo* info) {
        return rocm_get_device_info(accel, info);
    }

    __declspec(dllexport) AccelStatus accel_detect_devices(AccelDeviceInfo* devices, int max_count, int* out_count) {
        return rocm_detect_devices(devices, max_count, out_count);
    }

    __declspec(dllexport) AccelStatus accel_malloc(AcceleratorHandle accel, size_t size, void** out_ptr) {
        return rocm_malloc(accel, size, out_ptr);
    }

    __declspec(dllexport) AccelStatus accel_free(AcceleratorHandle accel, void* ptr) {
        return rocm_free(accel, ptr);
    }

    __declspec(dllexport) AccelStatus accel_upload(AcceleratorHandle accel, void* dst, const void* src, size_t size) {
        return rocm_upload(accel, dst, src, size);
    }

    __declspec(dllexport) AccelStatus accel_download(AcceleratorHandle accel, void* dst, const void* src, size_t size) {
        return rocm_download(accel, dst, src, size);
    }

    __declspec(dllexport) AccelStatus accel_sgemv(AcceleratorHandle accel,
        AccelLayout layout, AccelTranspose trans,
        int m, int n, float alpha,
        const float* A, int lda,
        const float* x, int incx,
        float beta, float* y, int incy)
    {
        return rocm_sgemv(accel, layout, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }

    __declspec(dllexport) AccelStatus accel_sger(AcceleratorHandle accel,
        AccelLayout layout,
        int m, int n, float alpha,
        const float* x, int incx,
        const float* y, int incy,
        float* A, int lda)
    {
        return rocm_sger(accel, layout, m, n, alpha, x, incx, y, incy, A, lda);
    }

    __declspec(dllexport) AccelStatus accel_sgemm(AcceleratorHandle accel,
        AccelLayout layout, AccelTranspose transA, AccelTranspose transB,
        int m, int n, int k, float alpha,
        const float* A, int lda,
        const float* B, int ldb,
        float beta, float* C, int ldc)
    {
        return rocm_sgemm(accel, layout, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    __declspec(dllexport) AccelStatus accel_batch_sgemv(AcceleratorHandle accel,
        AccelLayout layout, AccelTranspose trans,
        int m, int n, int batch_size, float alpha,
        const float* A, int lda,
        const float* X, int ldX,
        float beta, float* Y, int ldY)
    {
        return rocm_batch_sgemv(accel, layout, trans, m, n, batch_size, alpha, A, lda, X, ldX, beta, Y, ldY);
    }

    __declspec(dllexport) AccelStatus accel_sync(AcceleratorHandle accel) {
        return rocm_sync(accel);
    }
}
