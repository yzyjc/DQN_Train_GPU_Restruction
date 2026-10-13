// gpu_stubs.cpp - 纯 CPU 推理时 gpu_nn_* 的空实现
// TUI 不做 GPU 训练，但 neural_net.cpp 无条件引用这些符号，提供空实现消除链接依赖。
// forward_cpu 内部 g_gpu_forward_active=0 时永远走 CPU 本体，不会调用这里。
#include "gpu_types.h"
#include <stddef.h>

extern "C" {
int  gpu_nn_create(GpuNetHandle* out) { if (out) out->is_valid = 0; return -1; }
void gpu_nn_destroy(GpuNetHandle* h) { (void)h; }
int  gpu_nn_upload(GpuNetHandle h, const void* p, size_t s) { (void)h; (void)p; (void)s; return -1; }
int  gpu_nn_download(GpuNetHandle h, void* p, size_t s) { (void)h; (void)p; (void)s; return -1; }
int  gpu_nn_forward(GpuNetHandle h, const float* i, float* o) { (void)h; (void)i; (void)o; return -1; }
int  gpu_nn_forward_batch(GpuNetHandle h, const float* s, int b, float* o) { (void)h; (void)s; (void)b; (void)o; return -1; }
int  gpu_nn_train_step(GpuNetHandle h, const float* s, int a, float t, float lr, float* l) { (void)h; (void)s; (void)a; (void)t; (void)lr; (void)l; return -1; }
int  gpu_nn_train_step_batch(GpuNetHandle h, const float* s, const int* a, const float* t, int b, float lr, float* l) { (void)h; (void)s; (void)a; (void)t; (void)b; (void)lr; (void)l; return -1; }
int  gpu_nn_soft_update(GpuNetHandle t, GpuNetHandle s, float tau) { (void)t; (void)s; (void)tau; return -1; }
}
