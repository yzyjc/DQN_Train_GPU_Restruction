#pragma once

// GPU Training Session - High-level abstraction layer
// Hides all DLL loading, function pointers, and GPU handle management.
// Main program just calls simple methods; CPU fallback is automatic.

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle
typedef struct GpuTrainingSession GpuTrainingSession;

// Status codes
typedef enum {
    GPU_SESSION_SUCCESS = 0,
    GPU_SESSION_ERROR_NO_GPU,        // No GPU available, using CPU fallback
    GPU_SESSION_ERROR_DLL_NOT_FOUND,  // No accelerator DLL found
    GPU_SESSION_ERROR_INIT_FAILED,    // GPU initialization failed
    GPU_SESSION_ERROR_INVALID_PARAM,
    GPU_SESSION_ERROR_NOT_SUPPORTED
} GpuSessionStatus;

// ============================================================================
// Lifecycle
// ============================================================================

// Create session: auto-detects best GPU DLL, initializes device.
// Always succeeds - falls back to CPU if no GPU available.
// Returns session pointer (never NULL).
GpuTrainingSession* gpu_session_create(void);

// Destroy session: releases GPU resources, frees DLL.
void gpu_session_destroy(GpuTrainingSession* session);

// ============================================================================
// Query
// ============================================================================

// Is GPU acceleration active? (0 = CPU fallback, 1 = GPU)
int gpu_session_is_gpu_active(GpuTrainingSession* session);

// Get device name (e.g. "AMD Radeon 8060S"). Returns empty string if CPU.
void gpu_session_get_device_name(GpuTrainingSession* session, char* name, int name_size);

// Get backend name: "rocm", "vulkan", or "cpu"
const char* gpu_session_get_backend_name(GpuTrainingSession* session);

// ============================================================================
// NeuralNet Operations
// ============================================================================

// Upload NeuralNet weights to GPU. Both q_net and target_net.
// If GPU not active, silently succeeds (CPU handles it).
GpuSessionStatus gpu_session_upload(GpuTrainingSession* session,
    const void* q_net, const void* target_net, size_t nn_size);

// Download NeuralNet weights from GPU to CPU.
// If GPU not active, silently succeeds.
GpuSessionStatus gpu_session_download(GpuTrainingSession* session,
    void* q_net, void* target_net, size_t nn_size);

// GPU-side soft update: target = tau * source + (1 - tau) * target
// If GPU not active, returns error (caller should do CPU soft update).
GpuSessionStatus gpu_session_soft_update(GpuTrainingSession* session, float tau);

// Copy q_net → target_net on GPU.
GpuSessionStatus gpu_session_copy_target(GpuTrainingSession* session);

// ============================================================================
// GPU-resident forward pass (future: not yet implemented)
// ============================================================================

// Forward pass on GPU. Returns NOT_SUPPORTED if DLL doesn't implement it.
// input: float[state_dim], output: float[action_count]
GpuSessionStatus gpu_session_forward(GpuTrainingSession* session,
    const float* input, int state_dim, float* output, int action_count);

// ============================================================================
// Statistics
// ============================================================================

typedef struct {
    int gpu_soft_updates;     // Number of GPU soft updates performed
    int gpu_uploads;          // Number of weight uploads
    int gpu_downloads;        // Number of weight downloads
    int gpu_forwards;         // Number of GPU forward passes
    int cpu_forwards;         // Number of CPU forward passes (fallback)
    int is_gpu_active;        // 1 if GPU is active
    char backend[32];         // "rocm", "vulkan", or "cpu"
    char device_name[256];    // GPU device name
} GpuSessionStats;

void gpu_session_get_stats(GpuTrainingSession* session, GpuSessionStats* stats);
void gpu_session_log_stats(GpuTrainingSession* session);  // Writes to gpu_training.log

#ifdef __cplusplus
}
#endif
