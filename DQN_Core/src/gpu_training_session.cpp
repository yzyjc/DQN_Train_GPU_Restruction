#define _CRT_SECURE_NO_WARNINGS
#include "gpu_training_session.h"
#include "accelerator_v2.h"
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// DLL function pointer types (mirrors accelerator_v2.h)
// ============================================================================
typedef AccelV2Status (*PFN_accelv2_init)(AccelV2Handle* out_accel);
typedef AccelV2Status (*PFN_accelv2_cleanup)(AccelV2Handle accel);
typedef AccelV2Status (*PFN_accelv2_get_device_name)(AccelV2Handle accel, char* name, int name_size);
typedef AccelV2Status (*PFN_accelv2_get_device_memory)(AccelV2Handle accel, uint64_t* total, uint64_t* free_mem);
typedef AccelV2Status (*PFN_accelv2_malloc)(AccelV2Handle accel, size_t size, GpuMemory* out_mem);
typedef AccelV2Status (*PFN_accelv2_free)(AccelV2Handle accel, GpuMemory mem);
typedef AccelV2Status (*PFN_accelv2_upload)(AccelV2Handle accel, GpuMemory dst, const void* src, size_t size);
typedef AccelV2Status (*PFN_accelv2_download)(AccelV2Handle accel, void* dst, GpuMemory src, size_t size);
typedef AccelV2Status (*PFN_accelv2_nn_create)(AccelV2Handle accel, GpuNeuralNetHandle* out_handle);
typedef AccelV2Status (*PFN_accelv2_nn_destroy)(AccelV2Handle accel, GpuNeuralNetHandle handle);
typedef AccelV2Status (*PFN_accelv2_nn_upload)(AccelV2Handle accel, GpuNeuralNetHandle handle, const void* cpu_nn, size_t size);
typedef AccelV2Status (*PFN_accelv2_nn_download)(AccelV2Handle accel, GpuNeuralNetHandle handle, void* cpu_nn, size_t size);
typedef AccelV2Status (*PFN_accelv2_nn_copy)(AccelV2Handle accel, GpuNeuralNetHandle dst, GpuNeuralNetHandle src);
typedef AccelV2Status (*PFN_accelv2_nn_soft_update)(AccelV2Handle accel, GpuNeuralNetHandle target, GpuNeuralNetHandle source, float tau);
typedef AccelV2Status (*PFN_accelv2_nn_forward)(AccelV2Handle accel, GpuNeuralNetHandle net, GpuMemory input, int batch_size, GpuMemory output);

// ============================================================================
// Internal session state
// ============================================================================
struct GpuTrainingSession {
    // DLL handle
    HMODULE dll;
    char backend[32];     // "rocm", "vulkan", or "cpu"
    int is_gpu_active;

    // v2 accelerator handle
    AccelV2Handle accel;

    // GPU NeuralNet handles
    GpuNeuralNetHandle gpu_q_net;
    GpuNeuralNetHandle gpu_target_net;

    // Function pointers
    PFN_accelv2_init              pfn_init;
    PFN_accelv2_cleanup           pfn_cleanup;
    PFN_accelv2_get_device_name   pfn_get_device_name;
    PFN_accelv2_nn_create         pfn_nn_create;
    PFN_accelv2_nn_destroy        pfn_nn_destroy;
    PFN_accelv2_nn_upload         pfn_nn_upload;
    PFN_accelv2_nn_download       pfn_nn_download;
    PFN_accelv2_nn_copy           pfn_nn_copy;
    PFN_accelv2_nn_soft_update    pfn_nn_soft_update;
    PFN_accelv2_nn_forward        pfn_nn_forward;

    // Statistics
    int stat_gpu_soft_updates;
    int stat_gpu_uploads;
    int stat_gpu_downloads;
    int stat_gpu_forwards;
    int stat_cpu_forwards;

    // Device info
    char device_name[256];
};

// ============================================================================
// Internal helpers
// ============================================================================

static void session_log(GpuTrainingSession* s, const char* fmt, ...) {
    FILE* f = fopen("gpu_training.log", "a");
    if (!f) return;
    va_list args;
    va_start(args, fmt);
    vfprintf(f, fmt, args);
    va_end(args);
    fclose(f);
}

// Try to load a DLL and resolve all v2 function pointers.
// Returns 1 on success (all required pointers resolved), 0 on failure.
static int try_load_backend(GpuTrainingSession* s, const char* dll_name, const char* backend_name) {
    HMODULE dll = LoadLibraryA(dll_name);
    if (!dll) return 0;

    // Resolve required functions
    auto pfn_init    = (PFN_accelv2_init)GetProcAddress(dll, "accelv2_init");
    auto pfn_cleanup = (PFN_accelv2_cleanup)GetProcAddress(dll, "accelv2_cleanup");
    auto pfn_nn_create  = (PFN_accelv2_nn_create)GetProcAddress(dll, "accelv2_nn_create");
    auto pfn_nn_destroy = (PFN_accelv2_nn_destroy)GetProcAddress(dll, "accelv2_nn_destroy");
    auto pfn_nn_upload  = (PFN_accelv2_nn_upload)GetProcAddress(dll, "accelv2_nn_upload");
    auto pfn_nn_download = (PFN_accelv2_nn_download)GetProcAddress(dll, "accelv2_nn_download");
    auto pfn_nn_copy     = (PFN_accelv2_nn_copy)GetProcAddress(dll, "accelv2_nn_copy");
    auto pfn_nn_soft_update = (PFN_accelv2_nn_soft_update)GetProcAddress(dll, "accelv2_nn_soft_update");

    // These are required
    if (!pfn_init || !pfn_cleanup || !pfn_nn_create || !pfn_nn_destroy ||
        !pfn_nn_upload || !pfn_nn_download || !pfn_nn_copy || !pfn_nn_soft_update) {
        printf("[GPU_SESSION] %s: missing required functions\n", dll_name);
        FreeLibrary(dll);
        return 0;
    }

    // Try to initialize
    AccelV2Handle accel = NULL;
    AccelV2Status status = pfn_init(&accel);
    if (status != ACCEL_V2_SUCCESS || !accel) {
        printf("[GPU_SESSION] %s: init failed (error %d)\n", dll_name, status);
        session_log(s, "[GPU_SESSION] %s: init failed (error %d)\n", dll_name, status);
        FreeLibrary(dll);
        return 0;
    }

    // Get device name
    char device_name[256] = {0};
    auto pfn_get_name = (PFN_accelv2_get_device_name)GetProcAddress(dll, "accelv2_get_device_name");
    if (pfn_get_name) {
        pfn_get_name(accel, device_name, 256);
    }

    // Create NeuralNet handles
    GpuNeuralNetHandle q_net = NULL, target_net = NULL;
    status = pfn_nn_create(accel, &q_net);
    if (status != ACCEL_V2_SUCCESS) {
        printf("[GPU_SESSION] %s: nn_create q_net failed (error %d)\n", dll_name, status);
        session_log(s, "[GPU_SESSION] %s: nn_create q_net failed (error %d)\n", dll_name, status);
        pfn_cleanup(accel);
        FreeLibrary(dll);
        return 0;
    }
    status = pfn_nn_create(accel, &target_net);
    if (status != ACCEL_V2_SUCCESS) {
        printf("[GPU_SESSION] %s: nn_create target_net failed (error %d)\n", dll_name, status);
        session_log(s, "[GPU_SESSION] %s: nn_create target_net failed (error %d)\n", dll_name, status);
        pfn_nn_destroy(accel, q_net);
        pfn_cleanup(accel);
        FreeLibrary(dll);
        return 0;
    }

    // Success - store everything
    s->dll = dll;
    s->accel = accel;
    s->gpu_q_net = q_net;
    s->gpu_target_net = target_net;
    s->is_gpu_active = 1;
    strncpy(s->backend, backend_name, sizeof(s->backend) - 1);
    strncpy(s->device_name, device_name, sizeof(s->device_name) - 1);

    s->pfn_init = pfn_init;
    s->pfn_cleanup = pfn_cleanup;
    s->pfn_get_device_name = pfn_get_name;
    s->pfn_nn_create = pfn_nn_create;
    s->pfn_nn_destroy = pfn_nn_destroy;
    s->pfn_nn_upload = pfn_nn_upload;
    s->pfn_nn_download = pfn_nn_download;
    s->pfn_nn_copy = pfn_nn_copy;
    s->pfn_nn_soft_update = pfn_nn_soft_update;
    s->pfn_nn_forward = (PFN_accelv2_nn_forward)GetProcAddress(dll, "accelv2_nn_forward");

    return 1;
}

// ============================================================================
// Public API
// ============================================================================

GpuTrainingSession* gpu_session_create(void) {
    GpuTrainingSession* s = (GpuTrainingSession*)calloc(1, sizeof(GpuTrainingSession));
    if (!s) return NULL;  // Should never happen

    printf("[GPU_SESSION] Initializing...\n");
    session_log(s, "=== GPU Training Session Created ===\n");

    // Try backends in order: ROCm → Vulkan → CPU fallback
    if (try_load_backend(s, "rocm_drive.dll", "rocm")) {
        printf("[GPU_SESSION] ✓ ROCm GPU active: %s\n", s->device_name);
        session_log(s, "[INIT] ROCm active: %s\n", s->device_name);
        return s;
    }
    if (try_load_backend(s, "vulkan_drive.dll", "vulkan")) {
        printf("[GPU_SESSION] ✓ Vulkan GPU active: %s\n", s->device_name);
        session_log(s, "[INIT] Vulkan active: %s\n", s->device_name);
        return s;
    }

    // CPU fallback
    printf("[GPU_SESSION] No GPU available, using CPU fallback\n");
    session_log(s, "[INIT] CPU fallback (no GPU DLL)\n");
    strcpy(s->backend, "cpu");
    s->is_gpu_active = 0;
    return s;
}

void gpu_session_destroy(GpuTrainingSession* session) {
    if (!session) return;

    if (session->is_gpu_active && session->accel) {
        // Destroy GPU NeuralNet handles
        if (session->gpu_q_net && session->pfn_nn_destroy) {
            session->pfn_nn_destroy(session->accel, session->gpu_q_net);
        }
        if (session->gpu_target_net && session->pfn_nn_destroy) {
            session->pfn_nn_destroy(session->accel, session->gpu_target_net);
        }
        // Cleanup accelerator
        if (session->pfn_cleanup) {
            session->pfn_cleanup(session->accel);
        }
    }

    if (session->dll) {
        FreeLibrary(session->dll);
    }

    session_log(session, "[CLEANUP] Session destroyed (soft_updates=%d, uploads=%d, downloads=%d)\n",
        session->stat_gpu_soft_updates, session->stat_gpu_uploads, session->stat_gpu_downloads);

    free(session);
}

// ============================================================================
// Query
// ============================================================================

int gpu_session_is_gpu_active(GpuTrainingSession* session) {
    return session ? session->is_gpu_active : 0;
}

void gpu_session_get_device_name(GpuTrainingSession* session, char* name, int name_size) {
    if (!session || !name || name_size <= 0) return;
    strncpy(name, session->device_name, name_size - 1);
    name[name_size - 1] = '\0';
}

const char* gpu_session_get_backend_name(GpuTrainingSession* session) {
    if (!session) return "cpu";
    return session->backend;
}

// ============================================================================
// NeuralNet Operations
// ============================================================================

GpuSessionStatus gpu_session_upload(GpuTrainingSession* session,
    const void* q_net, const void* target_net, size_t nn_size)
{
    if (!session) return GPU_SESSION_ERROR_INVALID_PARAM;

    if (!session->is_gpu_active) return GPU_SESSION_SUCCESS;  // CPU: nothing to do

    AccelV2Status st;
    st = session->pfn_nn_upload(session->accel, session->gpu_q_net, q_net, nn_size);
    if (st != ACCEL_V2_SUCCESS) {
        session_log(session, "[UPLOAD] q_net failed: %d\n", st);
        return GPU_SESSION_ERROR_INIT_FAILED;
    }
    st = session->pfn_nn_upload(session->accel, session->gpu_target_net, target_net, nn_size);
    if (st != ACCEL_V2_SUCCESS) {
        session_log(session, "[UPLOAD] target_net failed: %d\n", st);
        return GPU_SESSION_ERROR_INIT_FAILED;
    }

    session->stat_gpu_uploads++;
    session_log(session, "[UPLOAD] OK (%zu bytes each)\n", nn_size);
    return GPU_SESSION_SUCCESS;
}

GpuSessionStatus gpu_session_download(GpuTrainingSession* session,
    void* q_net, void* target_net, size_t nn_size)
{
    if (!session) return GPU_SESSION_ERROR_INVALID_PARAM;

    if (!session->is_gpu_active) return GPU_SESSION_SUCCESS;  // CPU: nothing to do

    AccelV2Status st;
    st = session->pfn_nn_download(session->accel, session->gpu_q_net, q_net, nn_size);
    if (st != ACCEL_V2_SUCCESS) {
        session_log(session, "[DOWNLOAD] q_net failed: %d\n", st);
        return GPU_SESSION_ERROR_INIT_FAILED;
    }
    st = session->pfn_nn_download(session->accel, session->gpu_target_net, target_net, nn_size);
    if (st != ACCEL_V2_SUCCESS) {
        session_log(session, "[DOWNLOAD] target_net failed: %d\n", st);
        return GPU_SESSION_ERROR_INIT_FAILED;
    }

    session->stat_gpu_downloads++;
    return GPU_SESSION_SUCCESS;
}

GpuSessionStatus gpu_session_soft_update(GpuTrainingSession* session, float tau) {
    if (!session) return GPU_SESSION_ERROR_INVALID_PARAM;

    if (!session->is_gpu_active) return GPU_SESSION_ERROR_NO_GPU;  // Caller should do CPU update

    AccelV2Status st = session->pfn_nn_soft_update(
        session->accel, session->gpu_target_net, session->gpu_q_net, tau);
    if (st != ACCEL_V2_SUCCESS) {
        session_log(session, "[SOFT_UPDATE] failed: %d\n", st);
        return GPU_SESSION_ERROR_INIT_FAILED;
    }

    session->stat_gpu_soft_updates++;
    return GPU_SESSION_SUCCESS;
}

GpuSessionStatus gpu_session_copy_target(GpuTrainingSession* session) {
    if (!session) return GPU_SESSION_ERROR_INVALID_PARAM;

    if (!session->is_gpu_active) return GPU_SESSION_ERROR_NO_GPU;

    AccelV2Status st = session->pfn_nn_copy(
        session->accel, session->gpu_target_net, session->gpu_q_net);
    if (st != ACCEL_V2_SUCCESS) {
        return GPU_SESSION_ERROR_INIT_FAILED;
    }
    return GPU_SESSION_SUCCESS;
}

GpuSessionStatus gpu_session_forward(GpuTrainingSession* session,
    const float* input, int state_dim, float* output, int action_count)
{
    if (!session) return GPU_SESSION_ERROR_INVALID_PARAM;

    if (!session->is_gpu_active || !session->pfn_nn_forward) {
        session->stat_cpu_forwards++;
        return GPU_SESSION_ERROR_NOT_SUPPORTED;  // Caller should use CPU forward
    }

    // TODO: Implement GPU forward pass when DLL supports it
    session->stat_cpu_forwards++;
    return GPU_SESSION_ERROR_NOT_SUPPORTED;
}

// ============================================================================
// Statistics
// ============================================================================

void gpu_session_get_stats(GpuTrainingSession* session, GpuSessionStats* stats) {
    if (!session || !stats) return;
    memset(stats, 0, sizeof(GpuSessionStats));
    stats->gpu_soft_updates = session->stat_gpu_soft_updates;
    stats->gpu_uploads = session->stat_gpu_uploads;
    stats->gpu_downloads = session->stat_gpu_downloads;
    stats->gpu_forwards = session->stat_gpu_forwards;
    stats->cpu_forwards = session->stat_cpu_forwards;
    stats->is_gpu_active = session->is_gpu_active;
    strncpy(stats->backend, session->backend, sizeof(stats->backend) - 1);
    strncpy(stats->device_name, session->device_name, sizeof(stats->device_name) - 1);
}

void gpu_session_log_stats(GpuTrainingSession* session) {
    if (!session) return;
    session_log(session, "[STATS] gpu=%s backend=%s soft_updates=%d uploads=%d downloads=%d fwd_gpu=%d fwd_cpu=%d\n",
        session->is_gpu_active ? "active" : "cpu_fallback",
        session->backend,
        session->stat_gpu_soft_updates,
        session->stat_gpu_uploads,
        session->stat_gpu_downloads,
        session->stat_gpu_forwards,
        session->stat_cpu_forwards);
}
