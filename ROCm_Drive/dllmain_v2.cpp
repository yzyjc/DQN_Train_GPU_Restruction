// dllmain_v2.cpp
// ROCm Drive DLL v2 - Exports accelv2_* functions

#include <windows.h>
#include "rocm_accelerator_v2.h"

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    return TRUE;
}

// Forward declarations from rocm_accelerator_v2.cpp
extern "C" {
    AccelV2Status rocm_v2_init(AccelV2Handle* out_accel);
    AccelV2Status rocm_v2_cleanup(AccelV2Handle accel);
    AccelV2Status rocm_v2_get_device_name(AccelV2Handle accel, char* name, int name_size);
    AccelV2Status rocm_v2_get_device_memory(AccelV2Handle accel, uint64_t* total_mem, uint64_t* free_mem);
    AccelV2Status rocm_v2_sync(AccelV2Handle accel);
    AccelV2Status rocm_v2_malloc(AccelV2Handle accel, size_t size, GpuMemory* out_mem);
    AccelV2Status rocm_v2_free(AccelV2Handle accel, GpuMemory mem);
    AccelV2Status rocm_v2_upload(AccelV2Handle accel, GpuMemory dst, const void* src, size_t size);
    AccelV2Status rocm_v2_download(AccelV2Handle accel, void* dst, GpuMemory src, size_t size);
    AccelV2Status rocm_v2_memcpy(AccelV2Handle accel, GpuMemory dst, GpuMemory src, size_t size);
    AccelV2Status rocm_v2_sgemv(AccelV2Handle accel, AccelV2Layout layout, AccelV2Transpose trans, int m, int n, float alpha, GpuMemory A, int lda, GpuMemory x, int incx, float beta, GpuMemory y, int incy);
    AccelV2Status rocm_v2_sger(AccelV2Handle accel, AccelV2Layout layout, int m, int n, float alpha, GpuMemory x, int incx, GpuMemory y, int incy, GpuMemory A, int lda);
    AccelV2Status rocm_v2_sgemm(AccelV2Handle accel, AccelV2Layout layout, AccelV2Transpose transA, AccelV2Transpose transB, int m, int n, int k, float alpha, GpuMemory A, int lda, GpuMemory B, int ldb, float beta, GpuMemory C, int ldc);
    AccelV2Status rocm_v2_nn_create(AccelV2Handle accel, GpuNeuralNetHandle* out_handle);
    AccelV2Status rocm_v2_nn_destroy(AccelV2Handle accel, GpuNeuralNetHandle handle);
    AccelV2Status rocm_v2_nn_upload(AccelV2Handle accel, GpuNeuralNetHandle handle, const void* cpu_nn, size_t size);
    AccelV2Status rocm_v2_nn_download(AccelV2Handle accel, GpuNeuralNetHandle handle, void* cpu_nn, size_t size);
    AccelV2Status rocm_v2_nn_copy(AccelV2Handle accel, GpuNeuralNetHandle dst, GpuNeuralNetHandle src);
    AccelV2Status rocm_v2_nn_soft_update(AccelV2Handle accel, GpuNeuralNetHandle target, GpuNeuralNetHandle source, float tau);
    AccelV2Status rocm_v2_nn_forward(AccelV2Handle accel, GpuNeuralNetHandle net, GpuMemory input, int batch_size, GpuMemory output);
    AccelV2Status rocm_v2_nn_train_step(AccelV2Handle accel, GpuNeuralNetHandle net, GpuNeuralNetHandle target_net, GpuMemory states, GpuMemory actions, GpuMemory rewards, GpuMemory next_states, GpuMemory dones, int batch_size, float gamma, float learning_rate, GpuMemory out_loss);
    AccelV2Status rocm_v2_rb_create(AccelV2Handle accel, GpuReplayBufferHandle* out_handle, int capacity, int state_dim);
    AccelV2Status rocm_v2_rb_destroy(AccelV2Handle accel, GpuReplayBufferHandle handle);
    AccelV2Status rocm_v2_rb_store(AccelV2Handle accel, GpuReplayBufferHandle handle, const float* state, int action, float reward, const float* next_state, int done);
    AccelV2Status rocm_v2_rb_sample(AccelV2Handle accel, GpuReplayBufferHandle handle, int batch_size, GpuMemory out_states, GpuMemory out_actions, GpuMemory out_rewards, GpuMemory out_next_states, GpuMemory out_dones);
}

// Export wrappers (accelv2_* -> rocm_v2_*)
extern "C" {
    __declspec(dllexport) AccelV2Status accelv2_init(AccelV2Handle* out_accel) { return rocm_v2_init(out_accel); }
    __declspec(dllexport) AccelV2Status accelv2_cleanup(AccelV2Handle accel) { return rocm_v2_cleanup(accel); }
    __declspec(dllexport) AccelV2Status accelv2_get_device_name(AccelV2Handle accel, char* name, int name_size) { return rocm_v2_get_device_name(accel, name, name_size); }
    __declspec(dllexport) AccelV2Status accelv2_get_device_memory(AccelV2Handle accel, uint64_t* total_mem, uint64_t* free_mem) { return rocm_v2_get_device_memory(accel, total_mem, free_mem); }
    __declspec(dllexport) AccelV2Status accelv2_sync(AccelV2Handle accel) { return rocm_v2_sync(accel); }
    __declspec(dllexport) AccelV2Status accelv2_malloc(AccelV2Handle accel, size_t size, GpuMemory* out_mem) { return rocm_v2_malloc(accel, size, out_mem); }
    __declspec(dllexport) AccelV2Status accelv2_free(AccelV2Handle accel, GpuMemory mem) { return rocm_v2_free(accel, mem); }
    __declspec(dllexport) AccelV2Status accelv2_upload(AccelV2Handle accel, GpuMemory dst, const void* src, size_t size) { return rocm_v2_upload(accel, dst, src, size); }
    __declspec(dllexport) AccelV2Status accelv2_download(AccelV2Handle accel, void* dst, GpuMemory src, size_t size) { return rocm_v2_download(accel, dst, src, size); }
    __declspec(dllexport) AccelV2Status accelv2_memcpy(AccelV2Handle accel, GpuMemory dst, GpuMemory src, size_t size) { return rocm_v2_memcpy(accel, dst, src, size); }
    __declspec(dllexport) AccelV2Status accelv2_sgemv(AccelV2Handle accel, AccelV2Layout layout, AccelV2Transpose trans, int m, int n, float alpha, GpuMemory A, int lda, GpuMemory x, int incx, float beta, GpuMemory y, int incy) { return rocm_v2_sgemv(accel, layout, trans, m, n, alpha, A, lda, x, incx, beta, y, incy); }
    __declspec(dllexport) AccelV2Status accelv2_sger(AccelV2Handle accel, AccelV2Layout layout, int m, int n, float alpha, GpuMemory x, int incx, GpuMemory y, int incy, GpuMemory A, int lda) { return rocm_v2_sger(accel, layout, m, n, alpha, x, incx, y, incy, A, lda); }
    __declspec(dllexport) AccelV2Status accelv2_sgemm(AccelV2Handle accel, AccelV2Layout layout, AccelV2Transpose transA, AccelV2Transpose transB, int m, int n, int k, float alpha, GpuMemory A, int lda, GpuMemory B, int ldb, float beta, GpuMemory C, int ldc) { return rocm_v2_sgemm(accel, layout, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); }
    __declspec(dllexport) AccelV2Status accelv2_nn_create(AccelV2Handle accel, GpuNeuralNetHandle* out_handle) { return rocm_v2_nn_create(accel, out_handle); }
    __declspec(dllexport) AccelV2Status accelv2_nn_destroy(AccelV2Handle accel, GpuNeuralNetHandle handle) { return rocm_v2_nn_destroy(accel, handle); }
    __declspec(dllexport) AccelV2Status accelv2_nn_upload(AccelV2Handle accel, GpuNeuralNetHandle handle, const void* cpu_nn, size_t size) { return rocm_v2_nn_upload(accel, handle, cpu_nn, size); }
    __declspec(dllexport) AccelV2Status accelv2_nn_download(AccelV2Handle accel, GpuNeuralNetHandle handle, void* cpu_nn, size_t size) { return rocm_v2_nn_download(accel, handle, cpu_nn, size); }
    __declspec(dllexport) AccelV2Status accelv2_nn_copy(AccelV2Handle accel, GpuNeuralNetHandle dst, GpuNeuralNetHandle src) { return rocm_v2_nn_copy(accel, dst, src); }
    __declspec(dllexport) AccelV2Status accelv2_nn_soft_update(AccelV2Handle accel, GpuNeuralNetHandle target, GpuNeuralNetHandle source, float tau) { return rocm_v2_nn_soft_update(accel, target, source, tau); }
    __declspec(dllexport) AccelV2Status accelv2_nn_forward(AccelV2Handle accel, GpuNeuralNetHandle net, GpuMemory input, int batch_size, GpuMemory output) { return rocm_v2_nn_forward(accel, net, input, batch_size, output); }
    __declspec(dllexport) AccelV2Status accelv2_nn_train_step(AccelV2Handle accel, GpuNeuralNetHandle net, GpuNeuralNetHandle target_net, GpuMemory states, GpuMemory actions, GpuMemory rewards, GpuMemory next_states, GpuMemory dones, int batch_size, float gamma, float learning_rate, GpuMemory out_loss) { return rocm_v2_nn_train_step(accel, net, target_net, states, actions, rewards, next_states, dones, batch_size, gamma, learning_rate, out_loss); }
    __declspec(dllexport) AccelV2Status accelv2_rb_create(AccelV2Handle accel, GpuReplayBufferHandle* out_handle, int capacity, int state_dim) { return rocm_v2_rb_create(accel, out_handle, capacity, state_dim); }
    __declspec(dllexport) AccelV2Status accelv2_rb_destroy(AccelV2Handle accel, GpuReplayBufferHandle handle) { return rocm_v2_rb_destroy(accel, handle); }
    __declspec(dllexport) AccelV2Status accelv2_rb_store(AccelV2Handle accel, GpuReplayBufferHandle handle, const float* state, int action, float reward, const float* next_state, int done) { return rocm_v2_rb_store(accel, handle, state, action, reward, next_state, done); }
    __declspec(dllexport) AccelV2Status accelv2_rb_sample(AccelV2Handle accel, GpuReplayBufferHandle handle, int batch_size, GpuMemory out_states, GpuMemory out_actions, GpuMemory out_rewards, GpuMemory out_next_states, GpuMemory out_dones) { return rocm_v2_rb_sample(accel, handle, batch_size, out_states, out_actions, out_rewards, out_next_states, out_dones); }
}
