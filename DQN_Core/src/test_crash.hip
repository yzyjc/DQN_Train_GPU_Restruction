// test_crash.hip - 捕获 0xC0000005 崩溃位置
// 用 SetUnhandledExceptionFilter 记录 RIP，判断崩在哪个模块/函数
#include <hip/hip_runtime.h>
#include <windows.h>
#include <stdio.h>

static void write_crash_info(const char* where, ULONG_PTR addr) {
    // 解析 RIP 所在模块
    char mod_name[MAX_PATH] = "unknown";
    HMODULE mod = NULL;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCSTR)addr, &mod)) {
        GetModuleFileNameA(mod, mod_name, MAX_PATH);
    }
    FILE* f = fopen("crash_loc.txt", "a");
    if (f) {
        fprintf(f, "[%s] RIP=0x%llX module=%s\n", where, (unsigned long long)addr, mod_name);
        fclose(f);
    }
    printf("[CRASH] %s at 0x%llX module=%s\n", where, (unsigned long long)addr, mod_name);
}

static LONG WINAPI crash_handler(EXCEPTION_POINTERS* ex) {
    write_crash_info("UnhandledException", ex->ContextRecord->Rip);
    return EXCEPTION_EXECUTE_HANDLER;
}

int main() {
    SetUnhandledExceptionFilter(crash_handler);
    DeleteFileA("crash_loc.txt");

    printf("[1] hipInit...\n"); hipInit(0);
    int count = 0; hipGetDeviceCount(&count);
    printf("[1] devices=%d\n", count);
    hipSetDevice(0);

    float* d = 0;
    hipError_t e = hipMalloc(&d, 1024 * sizeof(float));
    printf("[2] hipMalloc=%d ptr=%p\n", e, d);

    float h[1024];
    for (int i = 0; i < 1024; i++) h[i] = 0.0f;

    printf("[3] hipMemcpy H2D...\n");
    fflush(stdout);
    e = hipMemcpy(d, h, 1024 * sizeof(float), hipMemcpyHostToDevice);
    printf("[3] hipMemcpy=%d\n", e);

    printf("[4] hipMemset...\n");
    fflush(stdout);
    e = hipMemset(d, 0, 1024 * sizeof(float));
    printf("[4] hipMemset=%d\n", e);

    printf("[5] DONE - no crash\n");
    return 0;
}
