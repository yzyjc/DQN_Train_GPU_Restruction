// GPT6-Astra. Keep ADLX ownership on the sampler thread, release interfaces first.
#include "gpu_sensors.h"
#include "../third_party/adlx/SDK/ADLXHelper/Windows/Cpp/ADLXHelper.cpp"
#include "../third_party/adlx/SDK/Platform/Windows/WinAPIs.cpp"
#include "../third_party/adlx/SDK/Include/IPerformanceMonitoring.h"
#include <cmath>
#include <cstring>
#include <new>
using namespace adlx;
struct Sensors {
    ADLXHelper helper;
    IADLXGPUPtr gpu;
    IADLXPerformanceMonitoringServicesPtr perf;
    IADLXGPUMetricsSupportPtr support;
    bool temp=false, power=false, clock=false, memory_clock=false, hotspot=false;
};
void* gpu_sensors_create(const char* expected_name) {
    auto s=new(std::nothrow) Sensors;
    if(!s)return nullptr;
    bool ok=false; printf("[ADLX] initialize\n");
    if(ADLX_SUCCEEDED(s->helper.Initialize())) {
        printf("[ADLX] initialized\n"); IADLXGPUListPtr list;
        auto system=s->helper.GetSystemServices();
        // Until PCI matching is implemented, reject multi-GPU ambiguity.
        if(ADLX_SUCCEEDED(system->GetGPUs(&list)) && list->Size()==1 &&
           ADLX_SUCCEEDED(list->At(list->Begin(),&s->gpu))) {
            printf("[ADLX] selected\n"); const char* name=nullptr;
            if(ADLX_SUCCEEDED(s->gpu->Name(&name)) && name && expected_name &&
               strcmp(name,expected_name)==0 &&
               ADLX_SUCCEEDED(system->GetPerformanceMonitoringServices(&s->perf)) &&
               ADLX_SUCCEEDED(s->perf->GetSupportedGPUMetrics(s->gpu,&s->support))) {
                adlx_bool yes=false;
                s->temp=ADLX_SUCCEEDED(s->support->IsSupportedGPUTemperature(&yes)) && yes;
                yes=false;s->power=ADLX_SUCCEEDED(s->support->IsSupportedGPUPower(&yes)) && yes;
                yes=false;s->clock=ADLX_SUCCEEDED(s->support->IsSupportedGPUClockSpeed(&yes)) && yes;
                yes=false;s->memory_clock=ADLX_SUCCEEDED(s->support->IsSupportedGPUVRAMClockSpeed(&yes)) && yes;
                yes=false;s->hotspot=ADLX_SUCCEEDED(s->support->IsSupportedGPUHotspotTemperature(&yes)) && yes;
                ok=true;
                printf("[ADLX] %s temperature=%d power=%d clock=%d memory_clock=%d hotspot=%d\n",
                       name,s->temp,s->power,s->clock,s->memory_clock,s->hotspot);
            }
        }
    }
    if(!ok){printf("[ADLX] releasing unavailable backend\n");delete s;printf("[ADLX] unavailable or ambiguous device; sensors remain N/A\n");return nullptr;}
    return s;
}
void gpu_sensors_sample(void* ptr,GpuTelemetrySample* sample) {
    sample->gpu_temp_c=sample->gpu_power_w=sample->gpu_clock_mhz=-1;
    sample->gpu_memory_clock_mhz=sample->gpu_hotspot_temp_c=-1;
    sample->_reserved=GPU_TELEM_SENSOR_EXT_MAGIC;
    auto s=static_cast<Sensors*>(ptr);if(!s)return;
    IADLXGPUMetricsPtr m;
    if(!ADLX_SUCCEEDED(s->perf->GetCurrentGPUMetrics(s->gpu,&m)))return;
    adlx_double value=-1;adlx_int clock=-1;
    if(s->temp && ADLX_SUCCEEDED(m->GPUTemperature(&value)) && std::isfinite(value) && value>=0 && value<=150)
        sample->gpu_temp_c=(float)value;
    value=-1;
    if(s->power && ADLX_SUCCEEDED(m->GPUPower(&value)) && std::isfinite(value) && value>=0 && value<=2000)
        sample->gpu_power_w=(float)value;
    if(s->clock && ADLX_SUCCEEDED(m->GPUClockSpeed(&clock)) && clock>=0 && clock<=10000)
        sample->gpu_clock_mhz=(float)clock;
    if(s->memory_clock && ADLX_SUCCEEDED(m->GPUVRAMClockSpeed(&clock)) && clock>=0 && clock<=10000)
        sample->gpu_memory_clock_mhz=(float)clock;
    value=-1;
    if(s->hotspot && ADLX_SUCCEEDED(m->GPUHotspotTemperature(&value)) && std::isfinite(value) && value>=0 && value<=200)
        sample->gpu_hotspot_temp_c=(float)value;
}
void gpu_sensors_destroy(void* ptr){delete static_cast<Sensors*>(ptr);}
