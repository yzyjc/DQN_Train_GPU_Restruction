// test_hip_helper.cpp - Test hip_helper.dll from MSVC-compiled program
#include <windows.h>
#include <stdio.h>

typedef int (*PFN_hip_helper_init)(void);
typedef int (*PFN_hip_helper_get_name)(char* name, int size);
typedef int (*PFN_hip_helper_malloc)(void** ptr, size_t size);
typedef int (*PFN_hip_helper_free)(void* ptr);

int main() {
    printf("Loading hip_helper.dll...\n");
    HMODULE dll = LoadLibraryA("hip_helper.dll");
    if (!dll) {
        printf("Failed to load DLL: %lu\n", GetLastError());
        return 1;
    }
    printf("DLL loaded OK\n");

    auto pfn_init = (PFN_hip_helper_init)GetProcAddress(dll, "hip_helper_init");
    auto pfn_get_name = (PFN_hip_helper_get_name)GetProcAddress(dll, "hip_helper_get_name");
    auto pfn_malloc = (PFN_hip_helper_malloc)GetProcAddress(dll, "hip_helper_malloc");
    auto pfn_free = (PFN_hip_helper_free)GetProcAddress(dll, "hip_helper_free");

    if (!pfn_init) { printf("hip_helper_init not found\n"); return 1; }

    printf("Calling hip_helper_init...\n");
    int result = pfn_init();
    printf("hip_helper_init returned: %d\n", result);

    if (result == 0) {
        char name[256] = {0};
        pfn_get_name(name, 256);
        printf("Device: %s\n", name);

        void* ptr = NULL;
        int err = pfn_malloc(&ptr, 1024);
        printf("hip_helper_malloc: err=%d, ptr=%p\n", err, ptr);
        if (err == 0 && ptr) {
            pfn_free(ptr);
            printf("hip_helper_free: OK\n");
        }
        printf("GPU is working!\n");
    } else {
        printf("HIP initialization failed: %d\n", result);
    }

    FreeLibrary(dll);
    return 0;
}
