// GPU 运行库加载取证探针（隔离诊断，不属于生产程序）
//
// 目的：本机同时存在
//   C:\Windows\SYSTEM32\amdhip64_7.dll  (2026-05-05)
//   D:\AMD_ROCm\bin\amdhip64_7.dll      (2026-05-15)
// 两个不同版本。默认加载顺序会先命中 SYSTEM32，导致 GPU 训练测试在
// gpu_nn_create 内 0xC0000005。本探针不猜测，而是：
//   1) 打印 amdhip64_7.dll / amd_comgr0713.dll / rocblas.dll 的实际加载路径；
//   2) 逐步调用 gpu_init -> gpu_nn_create -> upload -> forward -> 释放，
//      走到哪一步崩就说明问题定位在哪一层。
#include "core.h"
#include "gpu_types.h"
#include <new>

extern "C" int gpu_init(void);
extern "C" void gpu_shutdown(void);
extern "C" int gpu_nn_create(GpuNetHandle* out);
extern "C" void gpu_nn_destroy(GpuNetHandle* h);
extern "C" int gpu_nn_upload(GpuNetHandle h, const void* cpu_params, size_t size);
extern "C" int gpu_nn_forward(GpuNetHandle h, const float* input, float* output);
extern "C" int gpu_nn_download(GpuNetHandle h, void* cpu_params, size_t size);

/* 与 neural_net.cpp 的 pack_params 同布局；诊断程序，不进入生产链接。
   GPU 侧参数区只接受这个尺寸（3394340），传 sizeof(NeuralNet)（3400612，
   含前向缓存字段）会被尺寸守卫正确拒绝 —— 那不是 GPU 故障。 */
static void pack_params_local(const NeuralNet* net, GpuNetParams* p) {
    memcpy(p->token_embed, net->token_embed, sizeof(p->token_embed));
    memcpy(p->pos_embed, net->pos_embed, sizeof(p->pos_embed));
    memcpy(p->w1, net->w1, sizeof(p->w1));
    memcpy(p->b1, net->b1, sizeof(p->b1));
    memcpy(p->w2, net->w2, sizeof(p->w2));
    memcpy(p->b2, net->b2, sizeof(p->b2));
    memcpy(p->w3, net->w3, sizeof(p->w3));
    memcpy(p->b3, net->b3, sizeof(p->b3));
    memcpy(p->w4, net->w4, sizeof(p->w4));
    memcpy(p->b4, net->b4, sizeof(p->b4));
    memcpy(p->attn_wq, net->attn_wq, sizeof(p->attn_wq));
    memcpy(p->attn_bq, net->attn_bq, sizeof(p->attn_bq));
    memcpy(p->attn_wk, net->attn_wk, sizeof(p->attn_wk));
    memcpy(p->attn_bk, net->attn_bk, sizeof(p->attn_bk));
    memcpy(p->attn_wv, net->attn_wv, sizeof(p->attn_wv));
    memcpy(p->attn_bv, net->attn_bv, sizeof(p->attn_bv));
    memcpy(p->attn_wo, net->attn_wo, sizeof(p->attn_wo));
    memcpy(p->attn_bo, net->attn_bo, sizeof(p->attn_bo));
    memcpy(p->pool_w, net->pool_w, sizeof(p->pool_w));
    memcpy(p->pool_b, net->pool_b, sizeof(p->pool_b));
}

static void report_loaded_module(const char* name) {
    HMODULE h = GetModuleHandleA(name);
    if (!h) { printf("[MODULE] %-24s NOT LOADED\n", name); fflush(stdout); return; }
    char path[MAX_PATH] = { 0 };
    const DWORD n = GetModuleFileNameA(h, path, MAX_PATH);
    printf("[MODULE] %-24s base=0x%p path=%s (len=%lu)\n", name, (void*)h, path, (unsigned long)n);
    /* 崩溃可能发生在下一行；取证必须先落盘，否则会像 output.txt 那样全部丢失。 */
    fflush(stdout);
}

static void dump_amd_modules(void) {
    report_loaded_module("amdhip64_7.dll");
    report_loaded_module("amdhip64.dll");
    report_loaded_module("amd_comgr0713.dll");
    report_loaded_module("rocblas.dll");
    report_loaded_module("MIOpen.dll");
}

int main() {
    printf("=== GPU runtime load probe ===\n");
    printf("CWD       : ");
    { char cwd[MAX_PATH] = {0}; GetCurrentDirectoryA(MAX_PATH, cwd); printf("%s\n", cwd); }
    printf("This exe  : ");
    { char exe[MAX_PATH] = {0}; GetModuleFileNameA(NULL, exe, MAX_PATH); printf("%s\n", exe); }
    printf("PATH      : %s\n", getenv("PATH") ? getenv("PATH") : "(null)");
    printf("--- modules before gpu_init ---\n");
    dump_amd_modules();

    printf("--- step 1: gpu_init ---\n");
    const int rc_init = gpu_init();
    printf("gpu_init returned %d\n", rc_init);
    dump_amd_modules();

    printf("--- step 2: gpu_nn_create ---\n");
    GpuNetHandle h = { 0 };
    const int rc_create = gpu_nn_create(&h);
    printf("gpu_nn_create returned %d valid=%d\n", rc_create, h.is_valid);
    dump_amd_modules();

    if (rc_create == 0) {
        printf("--- step 3: upload GpuNetParams (device parameter area) ---\n");
        NeuralNet* net = (NeuralNet*)calloc(1, sizeof(NeuralNet));
        GpuNetParams* params = (GpuNetParams*)calloc(1, sizeof(GpuNetParams));
        printf("host NeuralNet=%zu bytes; device GpuNetParams=%zu bytes\n",
            sizeof(NeuralNet), sizeof(GpuNetParams));
        fflush(stdout);
        const int rc_bad = gpu_nn_upload(h, net, sizeof(NeuralNet));
        printf("gpu_nn_upload(sizeof(NeuralNet)) returned %d  <- 预期被尺寸守卫拒绝，非 GPU 故障\n", rc_bad);
        fflush(stdout);
        pack_params_local(net, params);
        const int rc_up = gpu_nn_upload(h, params, sizeof(GpuNetParams));
        printf("gpu_nn_upload(sizeof(GpuNetParams)) returned %d\n", rc_up);
        fflush(stdout);

        printf("--- step 4: forward ---\n");
        float in[CELL_COUNT + 1] = { 0 };
        float out[CELL_COUNT] = { 0 };
        fflush(stdout);
        const int rc_fw = gpu_nn_forward(h, in, out);
        printf("gpu_nn_forward returned %d out[0]=%g\n", rc_fw, (double)out[0]);
        fflush(stdout);

        printf("--- step 5: download back into GpuNetParams ---\n");
        memset(params, 0, sizeof(*params));
        fflush(stdout);
        const int rc_dn = gpu_nn_download(h, params, sizeof(GpuNetParams));
        printf("gpu_nn_download returned %d\n", rc_dn);
        fflush(stdout);
        free(params);
        free(net);
        gpu_nn_destroy(&h);
        printf("destroyed handle\n");
        fflush(stdout);
    }
    gpu_shutdown();
    printf("=== probe finished without crash ===\n");
    return 0;
}
