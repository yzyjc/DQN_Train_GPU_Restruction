// accelerator_manager.cpp
// Accelerator Manager - Auto-detect and load optimal accelerator

#include "pch.h"
#include "accelerator_manager.h"
#include <windows.h>
#include <stdlib.h>
#include <string.h>

// DLL function signatures
typedef AccelStatus (*PFN_accel_init)(AccelType type, AcceleratorHandle* out_accel);
typedef AccelStatus (*PFN_accel_cleanup)(AcceleratorHandle accel);
typedef AccelStatus (*PFN_accel_detect_devices)(AccelDeviceInfo* devices, int max_count, int* out_count);

// Manager context
struct AccelManagerContext {
    HMODULE dll_handles[4];  // CPU, Vulkan, ROCm, CUDA
    AccelType loaded_types[4];
    int loaded_count;
    int initialized;
};

// Helper
static AccelManagerContext* get_manager_context(AccelManagerHandle handle) {
    return (AccelManagerContext*)handle;
}

// Try to load DLL
static HMODULE try_load_dll(const char* dll_name) {
    return LoadLibraryA(dll_name);
}

// Get function from DLL
static void* get_dll_func(HMODULE dll, const char* func_name) {
    if (!dll) return NULL;
    return (void*)GetProcAddress(dll, func_name);
}

// Initialize manager
AccelStatus accel_manager_init(AccelManagerHandle* out_manager) {
    if (!out_manager) return ACCEL_ERROR_INVALID_PARAM;

    AccelManagerContext* ctx = (AccelManagerContext*)malloc(sizeof(AccelManagerContext));
    if (!ctx) return ACCEL_ERROR_OUT_OF_MEMORY;

    memset(ctx, 0, sizeof(AccelManagerContext));
    ctx->initialized = 1;

    // Try to load accelerator DLLs
    // Order: ROCm (AMD), Vulkan (cross-platform), CPU (fallback)

    // Try ROCm first (best for AMD GPUs)
    ctx->dll_handles[0] = try_load_dll("rocm_drive.dll");
    if (ctx->dll_handles[0]) {
        ctx->loaded_types[ctx->loaded_count] = ACCEL_TYPE_ROCM;
        ctx->loaded_count++;
        printf("[ACCEL] Loaded rocm_drive.dll\n");
    } else {
        printf("[ACCEL] rocm_drive.dll not found (error %lu)\n", GetLastError());
    }

    // Try Vulkan (cross-platform)
    ctx->dll_handles[1] = try_load_dll("vulkan_drive.dll");
    if (ctx->dll_handles[1]) {
        ctx->loaded_types[ctx->loaded_count] = ACCEL_TYPE_VULKAN;
        ctx->loaded_count++;
        printf("[ACCEL] Loaded vulkan_drive.dll\n");
    }

    // CPU is always available (statically linked)
    ctx->loaded_types[ctx->loaded_count] = ACCEL_TYPE_CPU;
    ctx->loaded_count++;

    *out_manager = (AccelManagerHandle)ctx;
    return ACCEL_SUCCESS;
}

// Cleanup manager
AccelStatus accel_manager_cleanup(AccelManagerHandle manager) {
    if (!manager) return ACCEL_ERROR_INVALID_PARAM;

    AccelManagerContext* ctx = get_manager_context(manager);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;

    // Free DLL handles
    for (int i = 0; i < 4; i++) {
        if (ctx->dll_handles[i]) {
            FreeLibrary(ctx->dll_handles[i]);
            ctx->dll_handles[i] = NULL;
        }
    }

    free(ctx);
    return ACCEL_SUCCESS;
}

// Auto-detect and create best accelerator
AccelStatus accel_manager_create_auto(AccelManagerHandle manager, AcceleratorHandle* out_accel) {
    if (!manager || !out_accel) return ACCEL_ERROR_INVALID_PARAM;

    AccelManagerContext* ctx = get_manager_context(manager);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;

    printf("[ACCEL] Trying to create accelerator...\n");

    // Try each loaded accelerator in order
    for (int i = 0; i < ctx->loaded_count; i++) {
        AccelType type = ctx->loaded_types[i];

        // Skip CPU for auto-detection (we want GPU if available)
        if (type == ACCEL_TYPE_CPU) {
            printf("[ACCEL] Skipping CPU (fallback)\n");
            continue;
        }

        printf("[ACCEL] Trying %s...\n", accel_get_type_name(type));

        // Find the DLL for this type
        HMODULE dll = NULL;
        if (type == ACCEL_TYPE_ROCM) dll = ctx->dll_handles[0];
        else if (type == ACCEL_TYPE_VULKAN) dll = ctx->dll_handles[1];

        if (!dll) {
            printf("[ACCEL] DLL not loaded for %s\n", accel_get_type_name(type));
            continue;
        }

        // Try to initialize
        PFN_accel_init pfn_init = (PFN_accel_init)get_dll_func(dll, "accel_init");
        PFN_accel_detect_devices pfn_detect = (PFN_accel_detect_devices)get_dll_func(dll, "accel_detect_devices");

        if (!pfn_init || !pfn_detect) {
            printf("[ACCEL] Failed to get functions from DLL (init=%p, detect=%p)\n", pfn_init, pfn_detect);
            continue;
        }

        printf("[ACCEL] Found functions, detecting devices...\n");

        // Check if devices are available
        AccelDeviceInfo devices[4];
        int device_count = 0;
        AccelStatus status = pfn_detect(devices, 4, &device_count);
        printf("[ACCEL] Device detection result: %s, count=%d\n",
            accel_get_error_string(status), device_count);

        if (status != ACCEL_SUCCESS || device_count == 0) continue;

        printf("[ACCEL] Found %d devices, trying to initialize...\n", device_count);

        // Try to create accelerator
        AcceleratorHandle accel = NULL;
        status = pfn_init(type, &accel);
        printf("[ACCEL] Init result: %s, handle=%p\n",
            accel_get_error_string(status), accel);

        if (status == ACCEL_SUCCESS && accel) {
            printf("[ACCEL] Successfully created %s accelerator\n", accel_get_type_name(type));
            *out_accel = accel;
            return ACCEL_SUCCESS;
        }
    }

    // Fallback to CPU (statically linked)
    printf("[ACCEL] No GPU available, falling back to CPU\n");
    return accel_init(ACCEL_TYPE_CPU, out_accel);
}

// Create specific accelerator type
AccelStatus accel_manager_create(AccelManagerHandle manager, AccelType type, AcceleratorHandle* out_accel) {
    if (!manager || !out_accel) return ACCEL_ERROR_INVALID_PARAM;

    AccelManagerContext* ctx = get_manager_context(manager);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;

    // CPU is always available
    if (type == ACCEL_TYPE_CPU) {
        return accel_init(ACCEL_TYPE_CPU, out_accel);
    }

    // Find the DLL for this type
    HMODULE dll = NULL;
    if (type == ACCEL_TYPE_ROCM) dll = ctx->dll_handles[0];
    else if (type == ACCEL_TYPE_VULKAN) dll = ctx->dll_handles[1];

    if (!dll) return ACCEL_ERROR_NOT_SUPPORTED;

    // Get init function
    PFN_accel_init pfn_init = (PFN_accel_init)get_dll_func(dll, "accel_init");
    if (!pfn_init) return ACCEL_ERROR_FUNCTION_NOT_FOUND;

    return pfn_init(type, out_accel);
}

// Get available accelerator types
AccelStatus accel_manager_get_available(AccelManagerHandle manager, AccelType* types, int max_count, int* out_count) {
    if (!manager || !types || !out_count) return ACCEL_ERROR_INVALID_PARAM;

    AccelManagerContext* ctx = get_manager_context(manager);
    if (!ctx->initialized) return ACCEL_ERROR_INVALID_PARAM;

    int count = 0;
    for (int i = 0; i < ctx->loaded_count && count < max_count; i++) {
        types[count++] = ctx->loaded_types[i];
    }

    *out_count = count;
    return ACCEL_SUCCESS;
}

// Get accelerator name
const char* accel_manager_get_name(AccelType type) {
    return accel_get_type_name(type);
}
