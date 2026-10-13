/* gpu_detector.cpp — GPU 设备探测
   使用 HIP + DXGI 查询 GPU 设备属性, 填充 GpuDeviceInfo。 */

#include "gpu_telemetry_protocol.h"
#include <stdio.h>
#include <string.h>

/* HIP (host-side API only, 不含 C++ 模板, cl.exe 可编) */
#include <hip/hip_runtime_api.h>

/* DXGI */
#include <dxgi.h>
#pragma comment(lib, "dxgi.lib")

/* ========== HIP 设备探测 ========== */

static int detect_hip(GpuDeviceInfo* info) {
    hipError_t err = hipInit(0);
    if (err != hipSuccess) {
        printf("[GPU_Detector] hipInit 失败: %d (%s)\n", err, hipGetErrorString(err));
        return -1;
    }

    int count = 0;
    err = hipGetDeviceCount(&count);
    if (err != hipSuccess || count == 0) {
        printf("[GPU_Detector] 未检测到 HIP 设备 (count=%d, err=%d)\n", count, err);
        return -1;
    }

    hipDeviceProp_t prop;
    err = hipGetDeviceProperties(&prop, 0);
    if (err != hipSuccess) {
        printf("[GPU_Detector] hipGetDeviceProperties 失败: %d\n", err);
        return -1;
    }

    /* 基本属性 */
    strncpy(info->device_name, prop.name, GPU_TELEM_DEVICE_NAME_LEN - 1);
    info->device_name[GPU_TELEM_DEVICE_NAME_LEN - 1] = '\0';
    strncpy(info->arch_name, prop.gcnArchName, GPU_TELEM_ARCH_NAME_LEN - 1);
    info->arch_name[GPU_TELEM_ARCH_NAME_LEN - 1] = '\0';

    info->total_vram_bytes = prop.totalGlobalMem;
    info->compute_units = (uint32_t)prop.multiProcessorCount;
    info->clock_mhz = (uint32_t)(prop.clockRate / 1000);  /* kHz -> MHz */
    info->memory_clock_mhz = (uint32_t)(prop.memoryClockRate / 1000);
    info->memory_bus_width = (uint32_t)prop.memoryBusWidth;
    info->l2_cache_size = (uint32_t)prop.l2CacheSize;
    info->max_shared_per_cu = 0;  /* maxSharedMemoryPerMultiprocessor 在当前 HIP 版本中不可用 */
    info->warp_size = (uint32_t)prop.warpSize;

    /* 理论显存带宽: memoryClockRate(kHz) * 2(DDR) * busWidth(bits) / 8(bytes) / 1e6(GB/s)
       memoryClockRate 已经是 kHz, 所以: kHz * 2 * bits / 8 / 1e6 = GB/s */
    info->theoretical_bw_gb_s = (float)prop.memoryClockRate * 2.0f *
                                 (float)prop.memoryBusWidth / 8.0f / 1e6f;

    printf("[GPU_Detector] HIP 设备: %s (%s)\n", info->device_name, info->arch_name);
    printf("[GPU_Detector] HIP totalGlobalMem: %llu MB (注意: APU 上此值包含共享内存, 不准确)\n",
           (unsigned long long)(info->total_vram_bytes / (1024 * 1024)));
    printf("[GPU_Detector] CU: %u, 时钟: %u MHz\n",
           info->compute_units, info->clock_mhz);
    printf("[GPU_Detector] 显存时钟: %u MHz, 位宽: %u bit, 理论带宽: %.1f GB/s\n",
           info->memory_clock_mhz, info->memory_bus_width, info->theoretical_bw_gb_s);
    printf("[GPU_Detector] L2 缓存: %u KB, 每 CU 共享内存: %u KB, Warp: %u\n",
           info->l2_cache_size / 1024, info->max_shared_per_cu / 1024, info->warp_size);

    return 0;
}

/* ========== DXGI 设备探测 ========== */

static int detect_dxgi(GpuDeviceInfo* info) {
    IDXGIFactory1* factory = NULL;
    HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory);
    if (FAILED(hr)) {
        printf("[GPU_Detector] CreateDXGIFactory1 失败: 0x%08lX\n", hr);
        return -1;
    }

    IDXGIAdapter1* adapter = NULL;
    /* 枚举所有 adapter, 找 AMD (0x1002) 或最大显存的 */
    IDXGIAdapter1* best_adapter = NULL;
    size_t best_vram = 0;

    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) == S_OK; i++) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);

        size_t vram = (size_t)desc.DedicatedVideoMemory;
        int is_amd = (desc.VendorId == 0x1002);

        printf("[GPU_Detector] DXGI Adapter %u: %ls (VRAM %llu MB, Vendor 0x%04X)\n",
               i, desc.Description,
               (unsigned long long)(vram / (1024 * 1024)), desc.VendorId);

        /* 优先 AMD 且有实际 VRAM 的, 否则选最大显存 */
        if (!best_adapter ||
            (is_amd && vram > 0 && (best_vram == 0 || best_vram < vram)) ||
            (!is_amd && best_vram == 0 && vram > 0) ||
            (vram > best_vram)) {
            if (best_adapter) best_adapter->Release();
            best_adapter = adapter;
            best_adapter->AddRef();
            best_vram = vram;
            info->vendor_id = desc.VendorId;
            info->device_id = desc.DeviceId;
            info->dedicated_vram_bytes = desc.DedicatedVideoMemory;
            info->shared_vram_bytes = desc.SharedSystemMemory;
        }
        adapter->Release();
    }

    if (best_adapter) {
        DXGI_ADAPTER_DESC1 desc;
        best_adapter->GetDesc1(&desc);
        printf("[GPU_Detector] 选定 DXGI Adapter: %ls\n", desc.Description);
        printf("[GPU_Detector] Dedicated: %llu MB, Shared: %llu MB\n",
               (unsigned long long)(desc.DedicatedVideoMemory / (1024 * 1024)),
               (unsigned long long)(desc.SharedSystemMemory / (1024 * 1024)));
        best_adapter->Release();
    } else {
        printf("[GPU_Detector] 未找到 DXGI adapter\n");
    }

    factory->Release();
    return best_adapter ? 0 : -1;
}

/* ========== 公共 API ========== */

int gpu_detector_init(GpuDeviceInfo* info) {
    if (!info) return -1;
    memset(info, 0, sizeof(GpuDeviceInfo));

    /* 默认值: 不可用的字段设为 -1 或 0 */
    info->theoretical_bw_gb_s = 0.0f;

    int hip_ok = detect_hip(info);
    int dxgi_ok = detect_dxgi(info);

    if (hip_ok != 0 && dxgi_ok != 0) {
        printf("[GPU_Detector] HIP 和 DXGI 均未检测到 GPU\n");
        return -1;
    }

    /* 如果 HIP 失败但 DXGI 成功, 用 DXGI 数据填充基本信息 */
    if (hip_ok != 0 && dxgi_ok == 0) {
        printf("[GPU_Detector] HIP 不可用, 仅使用 DXGI 信息\n");
    }

    return 0;
}

void gpu_detector_print(const GpuDeviceInfo* info) {
    if (!info) return;
    printf("\n========== GPU 设备信息 ==========\n");
    printf("设备: %s\n", info->device_name);
    printf("架构: %s\n", info->arch_name);
    printf("显存: %llu MB (Dedicated: %llu MB, Shared: %llu MB)\n",
           (unsigned long long)(info->total_vram_bytes / (1024 * 1024)),
           (unsigned long long)(info->dedicated_vram_bytes / (1024 * 1024)),
           (unsigned long long)(info->shared_vram_bytes / (1024 * 1024)));
    printf("计算单元: %u\n", info->compute_units);
    printf("GPU 时钟: %u MHz\n", info->clock_mhz);
    printf("显存时钟: %u MHz, 位宽: %u bit\n", info->memory_clock_mhz, info->memory_bus_width);
    printf("理论带宽: %.1f GB/s\n", info->theoretical_bw_gb_s);
    printf("L2 缓存: %u KB\n", info->l2_cache_size / 1024);
    printf("每 CU 共享内存: %u KB\n", info->max_shared_per_cu / 1024);
    printf("Warp 大小: %u\n", info->warp_size);
    printf("PCI: %04X:%04X\n", info->vendor_id, info->device_id);
    printf("==================================\n\n");
}